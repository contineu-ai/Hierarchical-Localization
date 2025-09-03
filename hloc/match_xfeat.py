import matplotlib
matplotlib.use('Agg')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "../../accelerated_features"))
from modules.xfeat import XFeat


import argparse
import pprint
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from . import logger
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval
from .ransac import *
import numpy as np
import time
from scipy.spatial import cKDTree
import math
from collections import defaultdict
import gc
import threading
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Precompute constants
PI = math.pi
PI_2 = PI / 2

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy available - GPU coordinate conversion enabled")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - using CPU coordinate conversion")

def _as_numpy(x):
    # Accept torch / numpy / cupy and return numpy.ndarray
    try:
        import cupy as cp  # will be available if CUPY_AVAILABLE
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _as_torch(x, device, dtype=torch.float32):
    # Accept torch / numpy / cupy and return torch.Tensor on `device`
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype, non_blocking=True)
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            x = cp.asnumpy(x)
    except Exception:
        pass
    if isinstance(x, np.ndarray):
        # ensure contiguous to avoid weird views
        x = np.ascontiguousarray(x)
        return torch.from_numpy(x).to(device=device, dtype=dtype, non_blocking=True)
    # fallback for lists/tuples
    return torch.tensor(x, device=device, dtype=dtype)

def _get_mkpts(matches, device):
    """Return (mkpts0, mkpts1) as torch.Tensors, or (None, None) if absent/empty."""
    if not matches:
        return None, None
    m0 = matches[0] or {}
    mk0 = m0.get("mkpts_0", None)
    mk1 = m0.get("mkpts_1", None)
    if mk0 is None or mk1 is None:
        return None, None
    # empty?
    try:
        if len(mk0) == 0 or len(mk1) == 0:
            return None, None
    except Exception:
        return None, None
    # normalize to tensors
    mk0 = _as_torch(mk0, device=device)
    mk1 = _as_torch(mk1, device=device)
    return mk0, mk1

def worker_init(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._fd_q = None
    dataset._fd_r = None

if CUPY_AVAILABLE:
    def identify_dicemap_face_batch_gpu(x_dice_cp, y_dice_cp, face_size):
        """CuPy-accelerated face identification"""
        rows = (y_dice_cp // face_size).astype(cp.int32)
        cols = (x_dice_cp // face_size).astype(cp.int32)
        
        # Face mapping: (row, col) -> face_id
        # Order matches GPU_Convert: F(0), R(1), B(2), L(3), U(4), D(5)
        face_map = cp.full((3, 4), -1, dtype=cp.int32)
        face_map[0, 1] = 4  # U
        face_map[1, 0] = 3  # L
        face_map[1, 1] = 0  # F
        face_map[1, 2] = 1  # R
        face_map[1, 3] = 2  # B
        face_map[2, 1] = 5  # D
        
        # Clamp indices and map
        rows = cp.clip(rows, 0, 2)
        cols = cp.clip(cols, 0, 3)
        return face_map[rows, cols]

    def cubemap_to_equirectangular_batch_gpu(faces_cp, x_cp, y_cp, face_size, eq_width, eq_height):
        """CuPy-accelerated batch coordinate conversion using GPU_Convert math"""
        # Normalize to [-0.5, 0.5] range (matching GPU_Convert's xyzcube)
        x_norm = (x_cp / face_size) - 0.5
        y_norm = (y_cp / face_size) - 0.5
        
        # Initialize vectors
        vecs = cp.zeros((len(faces_cp), 3), dtype=cp.float32)
        
        # Face-specific vector assignments matching GPU_Convert's xyzcube
        mask_f = (faces_cp == 0)  # Front (z = 0.5)
        vecs[mask_f, 0] = x_norm[mask_f]
        vecs[mask_f, 1] = -y_norm[mask_f]
        vecs[mask_f, 2] = 0.5
        
        mask_r = (faces_cp == 1)  # Right (x = 0.5)
        vecs[mask_r, 0] = 0.5
        vecs[mask_r, 1] = -y_norm[mask_r]
        vecs[mask_r, 2] = -x_norm[mask_r]
        
        mask_b = (faces_cp == 2)  # Back (z = -0.5)
        vecs[mask_b, 0] = -x_norm[mask_b]
        vecs[mask_b, 1] = -y_norm[mask_b]
        vecs[mask_b, 2] = -0.5
        
        mask_l = (faces_cp == 3)  # Left (x = -0.5)
        vecs[mask_l, 0] = -0.5
        vecs[mask_l, 1] = -y_norm[mask_l]
        vecs[mask_l, 2] = x_norm[mask_l]
        
        mask_u = (faces_cp == 4)  # Up (y = 0.5)
        vecs[mask_u, 0] = x_norm[mask_u]
        vecs[mask_u, 1] = 0.5
        vecs[mask_u, 2] = y_norm[mask_u]
        
        mask_d = (faces_cp == 5)  # Down (y = -0.5)
        vecs[mask_d, 0] = x_norm[mask_d]
        vecs[mask_d, 1] = -0.5
        vecs[mask_d, 2] = -y_norm[mask_d]
        
        # Normalize vectors (matching GPU_Convert's xyz2uv)
        norms = cp.sqrt(vecs[:, 0]**2 + vecs[:, 1]**2 + vecs[:, 2]**2)
        norms = cp.maximum(norms, 1e-9)  # Prevent division by zero
        vecs = vecs / norms[:, cp.newaxis]
        
        # Spherical coordinates using GPU_Convert's approach
        # longitude (u) = arctan2(x, z)
        # latitude (v) = arcsin(y)
        u = cp.arctan2(vecs[:, 0], vecs[:, 2])
        v = cp.arcsin(cp.clip(vecs[:, 1], -1, 1))
        
        # Convert to equirectangular coordinates (matching GPU_Convert's uv2coor)
        coor_x = (u / (2 * cp.pi) + 0.5) * eq_width - 0.5
        coor_y = (-v / cp.pi + 0.5) * eq_height - 0.5
        
        # Return as (x, y) coordinates
        return cp.stack((coor_x, coor_y), axis=1)

@torch.jit.script
def identify_dicemap_face_batch_torch(x_dice: torch.Tensor, y_dice: torch.Tensor, face_size: float) -> torch.Tensor:
    """GPU-accelerated face identification."""
    rows = (y_dice // face_size).long()
    cols = (x_dice // face_size).long()
    
    # Create face mapping as tensor
    # Order matches GPU_Convert: F(0), R(1), B(2), L(3), U(4), D(5)
    face_map = torch.zeros((3, 4), dtype=torch.long, device=x_dice.device)
    face_map[0, 1] = 4  # U
    face_map[1, 0] = 3  # L
    face_map[1, 1] = 0  # F
    face_map[1, 2] = 1  # R
    face_map[1, 3] = 2  # B
    face_map[2, 1] = 5  # D
    
    # Clamp indices to valid range
    rows = torch.clamp(rows, 0, 2)
    cols = torch.clamp(cols, 0, 3)
    
    return face_map[rows, cols]

import math # Make sure math is imported

@torch.jit.script
def cubemap_to_equirectangular_torch(faces: torch.Tensor, x: torch.Tensor, y: torch.Tensor, 
                                     cubemap_size: float, eq_width: float, eq_height: float) -> torch.Tensor:
    """Vectorized GPU cubemap to equirectangular conversion using GPU_Convert math."""
    # Normalize to [-0.5, 0.5] range (matching GPU_Convert's xyzcube)
    x_norm = (x / cubemap_size) - 0.5
    y_norm = (y / cubemap_size) - 0.5
    
    # Initialize vectors
    vecs = torch.zeros((len(faces), 3), device=x.device, dtype=x.dtype)
    
    # Face 0: F (front, z = 0.5)
    mask_f = (faces == 0)
    vecs[mask_f, 0] = x_norm[mask_f]
    vecs[mask_f, 1] = -y_norm[mask_f]
    vecs[mask_f, 2] = 0.5
    
    # Face 1: R (right, x = 0.5)
    mask_r = (faces == 1)
    vecs[mask_r, 0] = 0.5
    vecs[mask_r, 1] = -y_norm[mask_r]
    vecs[mask_r, 2] = -x_norm[mask_r]
    
    # Face 2: B (back, z = -0.5)
    mask_b = (faces == 2)
    vecs[mask_b, 0] = -x_norm[mask_b]
    vecs[mask_b, 1] = -y_norm[mask_b]
    vecs[mask_b, 2] = -0.5
    
    # Face 3: L (left, x = -0.5)
    mask_l = (faces == 3)
    vecs[mask_l, 0] = -0.5
    vecs[mask_l, 1] = -y_norm[mask_l]
    vecs[mask_l, 2] = x_norm[mask_l]
    
    # Face 4: U (up, y = 0.5)
    mask_u = (faces == 4)
    vecs[mask_u, 0] = x_norm[mask_u]
    vecs[mask_u, 1] = 0.5
    vecs[mask_u, 2] = y_norm[mask_u]
    
    # Face 5: D (down, y = -0.5)
    mask_d = (faces == 5)
    vecs[mask_d, 0] = x_norm[mask_d]
    vecs[mask_d, 1] = -0.5
    vecs[mask_d, 2] = -y_norm[mask_d]
    
    # Normalize vectors (matching GPU_Convert's xyz2uv)
    norms = torch.sqrt(vecs[:, 0]**2 + vecs[:, 1]**2 + vecs[:, 2]**2)
    norms = torch.clamp(norms, min=1e-9)  # Prevent division by zero
    vecs = vecs / norms.unsqueeze(1)
    
    # Convert to spherical coordinates using GPU_Convert's approach
    # longitude (u) = arctan2(x, z)
    # latitude (v) = arcsin(y)
    u = torch.atan2(vecs[:, 0], vecs[:, 2])
    v = torch.arcsin(torch.clamp(vecs[:, 1], -1, 1))
    
    # Convert to equirectangular pixel coordinates (matching GPU_Convert's uv2coor)
    coor_x = (u / (2 * math.pi) + 0.5) * eq_width - 0.5
    coor_y = (-v / math.pi + 0.5) * eq_height - 0.5
    
    return torch.stack((coor_x, coor_y), dim=1)

class BatchedFeaturePairsDataset(Dataset):
    """Optimized dataset with intelligent batching and padding"""
    
    def __init__(self, pairs, feature_path_q, feature_path_r, max_keypoints=4096, enable_caching=True):
        self.pairs = pairs
        self.feature_path_q = feature_path_q  
        self.feature_path_r = feature_path_r
        self.max_keypoints = max_keypoints
        self.enable_caching = enable_caching
        
        # Persistent file handles for better I/O performance
        self.fd_q = None
        self.fd_r = None
        
        # Cache for frequently accessed data
        self._cache = {} if enable_caching else None
        self._cache_size_limit = 50  # Limit cache size
        
        # Pre-analyze keypoint distributions for better batching
        self._analyze_keypoint_distributions()
        
    def _lazy_init(self):
        """Open HDF5 files only inside a worker process"""
        if self._fd_q is None:
            self._fd_q = h5py.File(self.feature_path_q, "r", swmr=True, libver="latest")
        if self._fd_r is None:
            self._fd_r = h5py.File(self.feature_path_r, "r", swmr=True, libver="latest")

    def __del__(self):
        try:
            if self._fd_q is not None:
                self._fd_q.close()
            if self._fd_r is not None:
                self._fd_r.close()
        except Exception:
            pass
    
    def _analyze_keypoint_distributions(self):
        """Analyze keypoint counts across dataset for optimal batching"""
        print("Analyzing keypoint distributions...")
        self.keypoint_stats = {'counts0': [], 'counts1': [], 'image_sizes': set()}
        
        # Sample 100 pairs for statistics
        sample_size = min(100, len(self.pairs))
        sample_indices = np.linspace(0, len(self.pairs)-1, sample_size, dtype=int)
        
        for idx in sample_indices:
            try:
                name0, name1 = self.pairs[idx]
                
                if name0 in self.fd_q:
                    kpts0 = self.fd_q[name0].get('dicemap_keypoints', self.fd_q[name0].get('keypoints', np.array([])))
                    self.keypoint_stats['counts0'].append(len(kpts0))
                
                if name1 in self.fd_r:
                    kpts1 = self.fd_r[name1].get('dicemap_keypoints', self.fd_r[name1].get('keypoints', np.array([])))
                    self.keypoint_stats['counts1'].append(len(kpts1))
                    
                    img_size = self.fd_r[name1].get('image_size', np.array([7680, 3840]))
                    self.keypoint_stats['image_sizes'].add(tuple(img_size))
            except:
                continue
        
        if self.keypoint_stats['counts0']:
            avg_kpts = np.mean(self.keypoint_stats['counts0'] + self.keypoint_stats['counts1'])
            max_kpts = max(self.keypoint_stats['counts0'] + self.keypoint_stats['counts1'])
            print(f"Keypoint analysis: avg={avg_kpts:.0f}, max={max_kpts}, sizes={len(self.keypoint_stats['image_sizes'])}")
    
    def _load_feature_data(self, name, file_handle, suffix):
        """Optimized feature loading with caching"""
        cache_key = f"{name}_{suffix}"

        if self._cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        if name not in file_handle:
            return self._create_empty_data(suffix)

        grp = file_handle[name]
        data = {}        
        # Load essential data efficiently
        essential_keys = ['dicemap_keypoints', 'dicemap_descriptors', 'dicemap_scores', 
                         'keypoints', 'descriptors', 'scores', 'image_size']
        
        for key in essential_keys:
            if key in grp:
                data[f"{key}{suffix}"] = torch.from_numpy(grp[key][:]).float()
            elif key == 'image_size':
                # Default image size
                data[f"image_size{suffix}"] = torch.tensor([7680, 3840], dtype=torch.long)
        
        # Add empty image tensor
# Add empty image tensor
        img_size = data.get(f"image_size{suffix}", torch.tensor([7680, 3840]))
        if isinstance(img_size, torch.Tensor):
            size = img_size.flip(0).tolist()   # e.g. [H, W]
        else:
            size = list(img_size[::-1])        # fallback if it's numpy or tuple
        data[f"image{suffix}"] = torch.empty((1, *map(int, size)))
        
        # Cache if enabled and under limit
        if self._cache and len(self._cache) < self._cache_size_limit:
            self._cache[cache_key] = {k: v.clone() for k, v in data.items()}
        
        return data
    
    def _create_empty_data(self, suffix):
        """Create empty data structure for missing entries"""
        return {
            f"dicemap_keypoints{suffix}": torch.zeros((0, 2), dtype=torch.float32),
            f"dicemap_descriptors{suffix}": torch.zeros((0, 64), dtype=torch.float32),
            f"dicemap_scores{suffix}": torch.zeros((0,), dtype=torch.float32),
            f"keypoints{suffix}": torch.zeros((0, 2), dtype=torch.float32),
            f"descriptors{suffix}": torch.zeros((0, 64), dtype=torch.float32),
            f"scores{suffix}": torch.zeros((0,), dtype=torch.float32),
            f"image_size{suffix}": torch.tensor([7680, 3840], dtype=torch.long),
            f"image{suffix}": torch.empty((1, 3840, 7680))
        }
    
    def __getitem__(self, idx):
        self._lazy_init()  # ensures handles exist in this worker
        name0, name1 = self.pairs[idx]

        data0 = self._load_feature_data(name0, self._fd_q, "0")
        data1 = self._load_feature_data(name1, self._fd_r, "1")

        data = {**data0, **data1}
        data['pair_name'] = names_to_pair(name0, name1)
        data['pair_idx'] = idx
        return data
        
    def __len__(self):
        return len(self.pairs)

def smart_collate_fn(batch):
    """Intelligent collation with adaptive padding"""
    if len(batch) == 1:
        return batch[0]
    
    # Find maximum keypoint counts for padding
    max_kpts_0 = max(len(item.get('dicemap_keypoints0', torch.empty(0, 2))) for item in batch)
    max_kpts_1 = max(len(item.get('dicemap_keypoints1', torch.empty(0, 2))) for item in batch)
    max_eq_kpts_0 = max(len(item.get('keypoints0', torch.empty(0, 2))) for item in batch)
    max_eq_kpts_1 = max(len(item.get('keypoints1', torch.empty(0, 2))) for item in batch)
    
    # Limit maximum keypoints to prevent memory issues
    max_kpts_0 = min(max_kpts_0, 4096)
    max_kpts_1 = min(max_kpts_1, 4096)
    max_eq_kpts_0 = min(max_eq_kpts_0, 4096)
    max_eq_kpts_1 = min(max_eq_kpts_1, 4096)
    
    collated_batch = defaultdict(list)
    
    for item in batch:
        for key, value in item.items():
            if 'keypoints' in key and isinstance(value, torch.Tensor) and value.dim() >= 2:
                # Pad keypoints
                max_kpts = max_kpts_0 if '0' in key else max_kpts_1
                if 'dicemap' not in key:  # equirectangular keypoints
                    max_kpts = max_eq_kpts_0 if '0' in key else max_eq_kpts_1
                
                current_kpts = len(value)
                if current_kpts < max_kpts:
                    padding = torch.zeros((max_kpts - current_kpts, value.shape[1]), dtype=value.dtype)
                    padded_value = torch.cat([value, padding], dim=0)
                else:
                    padded_value = value[:max_kpts]
                collated_batch[key].append(padded_value)
                
            elif 'descriptors' in key and isinstance(value, torch.Tensor) and value.dim() >= 2:
                # Pad descriptors
                max_kpts = max_kpts_0 if '0' in key else max_kpts_1
                if 'dicemap' not in key:
                    max_kpts = max_eq_kpts_0 if '0' in key else max_eq_kpts_1
                
                current_kpts = len(value)
                if current_kpts < max_kpts:
                    padding = torch.zeros((max_kpts - current_kpts, value.shape[1]), dtype=value.dtype)
                    padded_value = torch.cat([value, padding], dim=0)
                else:
                    padded_value = value[:max_kpts]
                collated_batch[key].append(padded_value)
                
            elif 'scores' in key and isinstance(value, torch.Tensor) and value.dim() >= 1:
                # Pad scores
                max_kpts = max_kpts_0 if '0' in key else max_kpts_1
                if 'dicemap' not in key:
                    max_kpts = max_eq_kpts_0 if '0' in key else max_eq_kpts_1
                
                current_kpts = len(value)
                if current_kpts < max_kpts:
                    padding = torch.zeros((max_kpts - current_kpts,), dtype=value.dtype)
                    padded_value = torch.cat([value, padding], dim=0)
                else:
                    padded_value = value[:max_kpts]
                collated_batch[key].append(padded_value)
                
            else:
                # Non-padded data (metadata, etc.)
                collated_batch[key].append(value)
    
    # Stack everything
    final_batch = {}
    for key, value_list in collated_batch.items():
        if isinstance(value_list[0], torch.Tensor):
            final_batch[key] = torch.stack(value_list)
        else:
            final_batch[key] = value_list
    
    # Add keypoint count metadata for unpadding later
    final_batch['original_kpts_0'] = [len(item.get('dicemap_keypoints0', torch.empty(0, 2))) for item in batch]
    final_batch['original_kpts_1'] = [len(item.get('dicemap_keypoints1', torch.empty(0, 2))) for item in batch]
    final_batch['original_eq_kpts_0'] = [len(item.get('keypoints0', torch.empty(0, 2))) for item in batch]
    final_batch['original_eq_kpts_1'] = [len(item.get('keypoints1', torch.empty(0, 2))) for item in batch]
    
    return final_batch

class UltraBatchProcessor:
    """Ultra-optimized batch processor with memory management and GPU acceleration"""
    
    def __init__(self, xfeat, device, batch_size=16, max_memory_gb=8.0):
        self.xfeat = xfeat
        self.device = device
        self.batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        
        # Performance tracking
        self.processing_stats = {
            'batch_times': [],
            'coordinate_times': [],
            'matching_times': [],
            'memory_peaks': []
        }
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        
        # Compile XFeat if supported
        if hasattr(torch, 'compile'):
            try:
                self.xfeat.forward = torch.compile(self.xfeat.forward, mode='max-autotune')

                print("XFeat compiled for optimal performance")
            except:
                pass
    
    def _get_memory_usage(self):
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / 1024**2
        return 0
    
    def _estimate_batch_memory_usage(self, batch_data):
        """Estimate memory usage for a batch"""
        if not torch.cuda.is_available():
            return 0
        
        # Rough estimation based on tensor sizes
        memory_est = 0
        for key, tensor in batch_data.items():
            if isinstance(tensor, torch.Tensor):
                memory_est += tensor.numel() * tensor.element_size()
        
        return memory_est / 1024**2  # Convert to MB
    
    def process_batch_ultra_optimized(self, batch_data, pair_names):
        """Ultra-optimized batch processing with GPU acceleration"""
        batch_start_time = time.time()
        initial_memory = self._get_memory_usage()
        
        try:
            batch_size = len(pair_names)
            results = []
            
            # Move all data to GPU efficiently
            gpu_batch = {}
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor) and not key.startswith('image') and not key.startswith('original'):
                    gpu_batch[key] = value.to(self.device, non_blocking=True)
                else:
                    gpu_batch[key] = value
            
            # Get image dimensions (assuming all images in batch have same size)
            img_size0 = gpu_batch.get('image_size0')
            if img_size0 is not None:
                if isinstance(img_size0, torch.Tensor):
                    if img_size0.dim() > 1:
                        eq_width = int(img_size0[0, 0].item())
                        eq_height = int(img_size0[0, 1].item()) if img_size0.shape[1] > 1 else eq_width // 2
                    else:
                        eq_width = int(img_size0[0].item())
                        eq_height = int(img_size0[1].item()) if len(img_size0) > 1 else eq_width // 2
                else:
                    eq_width, eq_height = 7680, 3840
            else:
                eq_width, eq_height = 7680, 3840
            
            face_size = eq_width // 4
            
            # Process each pair in the batch
            for b in range(batch_size):
                try:
                    # Extract data for this pair
                    dicemap_kpts0 = gpu_batch['dicemap_keypoints0'][b]
                    dicemap_kpts1 = gpu_batch['dicemap_keypoints1'][b]
                    dicemap_desc0 = gpu_batch['dicemap_descriptors0'][b]
                    dicemap_desc1 = gpu_batch['dicemap_descriptors1'][b]
                    dicemap_scores0 = gpu_batch['dicemap_scores0'][b]
                    dicemap_scores1 = gpu_batch['dicemap_scores1'][b]
                    
                    overall_kpts0 = gpu_batch['keypoints0'][b]
                    overall_kpts1 = gpu_batch['keypoints1'][b]
                    
                    # Get original keypoint counts (before padding)
                    orig_kpts0 = batch_data['original_kpts_0'][b]
                    orig_kpts1 = batch_data['original_kpts_1'][b]
                    orig_eq_kpts0 = batch_data['original_eq_kpts_0'][b]
                    orig_eq_kpts1 = batch_data['original_eq_kpts_1'][b]
                    
                    # Trim to original sizes
                    dicemap_kpts0 = dicemap_kpts0[:orig_kpts0]
                    dicemap_kpts1 = dicemap_kpts1[:orig_kpts1]
                    dicemap_desc0 = dicemap_desc0[:orig_kpts0]
                    dicemap_desc1 = dicemap_desc1[:orig_kpts1]
                    dicemap_scores0 = dicemap_scores0[:orig_kpts0]
                    dicemap_scores1 = dicemap_scores1[:orig_kpts1]
                    
                    overall_kpts0 = overall_kpts0[:orig_eq_kpts0]
                    overall_kpts1 = overall_kpts1[:orig_eq_kpts1]
                    
                    # Skip if no keypoints
                    if len(dicemap_kpts0) == 0 or len(dicemap_kpts1) == 0:
                        results.append(self._create_empty_result(pair_names[b], orig_eq_kpts0))
                        continue
                    
                    # Prepare XFeat inputs
                    output0 = {
                        'keypoints': dicemap_kpts0,
                        'descriptors': dicemap_desc0,
                        'scores': dicemap_scores0,
                        'image_size': (face_size * 4, face_size * 3)
                    }
                    output1 = {
                        'keypoints': dicemap_kpts1,
                        'descriptors': dicemap_desc1,
                        'scores': dicemap_scores1,
                        'image_size': (face_size * 4, face_size * 3)
                    }
                    
                    # XFeat matching with mixed precision
                    match_start = time.time()
                    with torch.cuda.amp.autocast():
                        matches = self.xfeat.batch_match_lighterglue([output0], [output1])
                    self.processing_stats['matching_times'].append(time.time() - match_start)
                    
                    # Process matches with GPU acceleration
                    coord_start = time.time()
                    final_matches = self._process_matches_gpu(
                        matches, overall_kpts0, overall_kpts1, 
                        face_size, eq_width, eq_height, orig_eq_kpts0
                    )
                    self.processing_stats['coordinate_times'].append(time.time() - coord_start)
                    
                    results.append((pair_names[b], final_matches))
                    
                except Exception as e:
                    logger.warning(f"Error processing pair {pair_names[b]}: {e}")
                    results.append(self._create_empty_result(pair_names[b], orig_eq_kpts0))
            
            # Update statistics
            total_time = time.time() - batch_start_time
            peak_memory = self._get_memory_usage()
            self.processing_stats['batch_times'].append(total_time)
            self.processing_stats['memory_peaks'].append(peak_memory - initial_memory)
            
            return results
            
        except torch.cuda.OutOfMemoryError:
            print(f"OOM in batch processing (size={len(pair_names)}), falling back to individual processing")
            torch.cuda.empty_cache()
            
            # Process individually as fallback
            return self._process_individually_fallback(batch_data, pair_names)
    
    def _process_matches_gpu(self, matches, overall_kpts0, overall_kpts1,
                            face_size, eq_width, eq_height, num_eq_kpts0):
        # SAFELY extract matches once; avoid using undeclared locals
        mkpts0_dice, mkpts1_dice = _get_mkpts(matches, self.device)
        if mkpts0_dice is None:
            return self._create_empty_matches(num_eq_kpts0)

        # Route to CuPy or Torch path; both paths will re-normalize types as needed
        use_cupy = CUPY_AVAILABLE
        try:
            # Prefer CuPy only if it will actually help
            use_cupy = CUPY_AVAILABLE and (mkpts0_dice.shape[0] > 100)
        except Exception:
            use_cupy = False

        if use_cupy:
            return self._process_matches_cupy(
                mkpts0_dice, mkpts1_dice, overall_kpts0, overall_kpts1,
                face_size, eq_width, eq_height, num_eq_kpts0
            )
        else:
            return self._process_matches_torch(
                mkpts0_dice, mkpts1_dice, overall_kpts0, overall_kpts1,
                face_size, eq_width, eq_height, num_eq_kpts0
            )

    def _process_matches_cupy(self, mkpts0_dice, mkpts1_dice, overall_kpts0, overall_kpts1, 
                             face_size, eq_width, eq_height, num_eq_kpts0):
        """CuPy-accelerated match processing"""
        # Convert to CuPy
        mkpts0_cp = cp.asarray(_as_numpy(mkpts0_dice))
        mkpts1_cp = cp.asarray(_as_numpy(mkpts1_dice))
        
        # Identify faces on GPU
        faces0 = identify_dicemap_face_batch_gpu(mkpts0_cp[:, 0], mkpts0_cp[:, 1], face_size)
        faces1 = identify_dicemap_face_batch_gpu(mkpts1_cp[:, 0], mkpts1_cp[:, 1], face_size)
        
        # Get face-relative coordinates
        face_x0 = mkpts0_cp[:, 0] - (mkpts0_cp[:, 0] // face_size) * face_size
        face_y0 = mkpts0_cp[:, 1] - (mkpts0_cp[:, 1] // face_size) * face_size
        face_x1 = mkpts1_cp[:, 0] - (mkpts1_cp[:, 0] // face_size) * face_size
        face_y1 = mkpts1_cp[:, 1] - (mkpts1_cp[:, 1] // face_size) * face_size
        
        # Convert to equirectangular coordinates on GPU
        equirect0 = cubemap_to_equirectangular_batch_gpu(faces0, face_x0, face_y0, face_size, eq_width, eq_height)
        equirect1 = cubemap_to_equirectangular_batch_gpu(faces1, face_x1, face_y1, face_size, eq_width, eq_height)
        
        # Filter valid coordinates
        valid_mask = (~cp.isnan(equirect0).any(axis=1)) & (~cp.isnan(equirect1).any(axis=1))
        valid_equirect0 = equirect0[valid_mask]
        valid_equirect1 = equirect1[valid_mask]
        
        # Convert back to CPU for kd-tree operations
        if len(valid_equirect0) > 0:
            equirect0_cpu = _as_numpy(valid_equirect0)
            equirect1_cpu = _as_numpy(valid_equirect1)
            overall_kpts0_cpu = _as_numpy(overall_kpts0)
            overall_kpts1_cpu = _as_numpy(overall_kpts1)
            
            return self._create_final_matches(equirect0_cpu, equirect1_cpu, overall_kpts0_cpu, overall_kpts1_cpu, num_eq_kpts0)
        else:
            return self._create_empty_matches(num_eq_kpts0)
    
    def _process_matches_torch(self, mkpts0_dice, mkpts1_dice, overall_kpts0, overall_kpts1,
                              face_size, eq_width, eq_height, num_eq_kpts0):
        """PyTorch-based match processing"""
        # Identify faces
        faces0 = identify_dicemap_face_batch_torch(mkpts0_dice[:, 0], mkpts0_dice[:, 1], float(face_size))
        faces1 = identify_dicemap_face_batch_torch(mkpts1_dice[:, 0], mkpts1_dice[:, 1], float(face_size))
        
        # Get face-relative coordinates
        face_x0 = mkpts0_dice[:, 0] - torch.floor(mkpts0_dice[:, 0] / face_size) * face_size
        face_y0 = mkpts0_dice[:, 1] - torch.floor(mkpts0_dice[:, 1] / face_size) * face_size
        face_x1 = mkpts1_dice[:, 0] - torch.floor(mkpts1_dice[:, 0] / face_size) * face_size
        face_y1 = mkpts1_dice[:, 1] - torch.floor(mkpts1_dice[:, 1] / face_size) * face_size
        
        # Convert coordinates
        equirect0 = cubemap_to_equirectangular_torch(faces0, face_x0, face_y0, float(face_size), float(eq_width), float(eq_height))
        equirect1 = cubemap_to_equirectangular_torch(faces1, face_x1, face_y1, float(face_size), float(eq_width), float(eq_height))
        
        # Filter valid coordinates
        valid_mask = (~torch.isnan(equirect0).any(dim=1)) & (~torch.isnan(equirect1).any(dim=1))
        valid_equirect0 = equirect0[valid_mask]
        valid_equirect1 = equirect1[valid_mask]
        
        if len(valid_equirect0) > 0:
            equirect0_cpu = valid_equirect0.cpu().numpy()
            equirect1_cpu = valid_equirect1.cpu().numpy()
            overall_kpts0_cpu = overall_kpts0.cpu().numpy()
            overall_kpts1_cpu = overall_kpts1.cpu().numpy()
            
            return self._create_final_matches(equirect0_cpu, equirect1_cpu, overall_kpts0_cpu, overall_kpts1_cpu, num_eq_kpts0)
        else:
            return self._create_empty_matches(num_eq_kpts0)
    
    def _create_final_matches(self, equirect0, equirect1, overall_kpts0, overall_kpts1, num_eq_kpts0):
        """Create final matches using optimized nearest neighbor search"""
        matches0 = np.full(num_eq_kpts0, -1, dtype=np.int64)
        scores0 = np.zeros(num_eq_kpts0, dtype=np.float32)
        
        if len(equirect0) > 0 and len(overall_kpts0) > 0 and len(overall_kpts1) > 0:
            # Use optimized kd-tree search
            tree0 = cKDTree(overall_kpts0.astype(np.float32))
            tree1 = cKDTree(overall_kpts1.astype(np.float32))
            
            # Find nearest neighbors with distance threshold
            distances0, indices0 = tree0.query(equirect0.astype(np.float32), k=1)
            distances1, indices1 = tree1.query(equirect1.astype(np.float32), k=1)
            
            # Apply distance threshold and create matches
            distance_threshold = 10.0  # pixels
            valid_matches = distances0 < distance_threshold
            
            if np.any(valid_matches):
                valid_indices0 = indices0[valid_matches]
                valid_indices1 = indices1[valid_matches]
                matches0[valid_indices0] = valid_indices1
                scores0[valid_indices0] = 1.0 - (distances0[valid_matches] / distance_threshold)
        
        return {
            'matches0': torch.from_numpy(matches0),
            'matching_scores0': torch.from_numpy(scores0)
        }
    
    def _create_empty_matches(self, num_keypoints):
        """Create empty match result"""
        return {
            'matches0': torch.full((num_keypoints,), -1, dtype=torch.int64),
            'matching_scores0': torch.zeros(num_keypoints, dtype=torch.float32)
        }
    
    def _create_empty_result(self, pair_name, num_keypoints):
        """Create empty result for failed processing"""
        return (pair_name, self._create_empty_matches(max(num_keypoints, 1)))
    
    def _process_individually_fallback(self, batch_data, pair_names):
        """Fallback to individual processing on OOM"""
        results = []
        batch_size = len(pair_names)
        
        for b in range(batch_size):
            try:
                # Extract single pair data
                single_data = {}
                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor) and value.dim() > 1:
                        single_data[key] = value[b]
                    elif isinstance(value, list):
                        single_data[key] = [value[b]]
                    else:
                        single_data[key] = value
                
                # Process single pair
                single_result = self.process_batch_ultra_optimized(single_data, [pair_names[b]])
                results.extend(single_result)
                
            except Exception as e:
                logger.warning(f"Fallback processing failed for {pair_names[b]}: {e}")
                results.append(self._create_empty_result(pair_names[b], 1))
        
        return results
    
    def print_stats(self):
        """Print performance statistics"""
        if not self.processing_stats['batch_times']:
            return
        
        print("\n=== Batch Processing Statistics ===")
        print(f"Average batch time: {np.mean(self.processing_stats['batch_times']):.3f}s")
        print(f"Average matching time: {np.mean(self.processing_stats['matching_times']):.3f}s")
        print(f"Average coordinate time: {np.mean(self.processing_stats['coordinate_times']):.3f}s")
        print(f"Peak memory usage: {max(self.processing_stats['memory_peaks']):.1f}MB")

class HighThroughputWriter:
    """High-throughput HDF5 writer with batched operations"""
    
    def __init__(self, match_path, num_workers=6, write_batch_size=64):
        self.match_path = match_path
        self.write_batch_size = write_batch_size
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.pending_writes = []
        self.futures = []
        self.lock = threading.Lock()
    
    def put(self, results):
        """Queue results for writing"""
        with self.lock:
            if isinstance(results, list):
                self.pending_writes.extend(results)
            else:
                self.pending_writes.append(results)
            
            # Write when batch is ready
            if len(self.pending_writes) >= self.write_batch_size:
                self._submit_write_batch()
    
    def _submit_write_batch(self):
        """Submit a batch for writing"""
        if not self.pending_writes:
            return
        
        write_batch = self.pending_writes[:self.write_batch_size]
        self.pending_writes = self.pending_writes[self.write_batch_size:]
        
        future = self.executor.submit(self._write_batch_optimized, write_batch)
        self.futures.append(future)
    
    def _write_batch_optimized(self, batch_results):
        """Optimized batch writing with error recovery"""
        try:
            with h5py.File(str(self.match_path), "a", libver="latest") as fd:
                for pair_name, pred in batch_results:
                    try:
                        # Remove existing data
                        if pair_name in fd:
                            del fd[pair_name]
                        
                        # Create group and write data
                        grp = fd.create_group(pair_name)
                        
                        # Convert to appropriate datatypes for storage
                        matches = pred["matches0"].cpu().numpy().astype(np.int32)
                        scores = pred["matching_scores0"].cpu().numpy().astype(np.float32)
                        
                        # Write with compression
                        grp.create_dataset("matches0", data=matches, compression="gzip", compression_opts=6)
                        grp.create_dataset("matching_scores0", data=scores, compression="gzip", compression_opts=6)
                        
                    except Exception as e:
                        logger.warning(f"Failed to write {pair_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Batch write error: {e}")
    
    def join(self):
        """Wait for all writes to complete"""
        # Submit ALL remaining writes in batches
        with self.lock:
            while self.pending_writes:
                batch_size = min(len(self.pending_writes), self.write_batch_size)
                write_batch = self.pending_writes[:batch_size]
                self.pending_writes = self.pending_writes[batch_size:]
                
                if write_batch:  # Only submit if there's something to write
                    future = self.executor.submit(self._write_batch_optimized, write_batch)
                    self.futures.append(future)
        
        # Wait for completion
        for future in self.futures:
            try:
                future.result(timeout=60)  # 60 second timeout per batch
            except Exception as e:
                logger.error(f"Write future failed: {e}")
        
        self.executor.shutdown(wait=True)

def identify_dicemap_face_batch(x_dice_batch, y_dice_batch, face_size):
    """CPU fallback for face identification"""
    # Order matches GPU_Convert: F(0), R(1), B(2), L(3), U(4), D(5)
    face_positions = {
        (0, 1): 4,  # U
        (1, 0): 3,  # L
        (1, 1): 0,  # F
        (1, 2): 1,  # R
        (1, 3): 2,  # B
        (2, 1): 5   # D
    }
    
    rows = (y_dice_batch // face_size).astype(int)
    cols = (x_dice_batch // face_size).astype(int)
    
    faces = []
    for row, col in zip(rows, cols):
        faces.append(face_positions.get((row, col), -1))
    
    return np.array(faces)

def dicemap_to_equirectangular_uv_batch_optimized(faces, x_dice_batch, y_dice_batch, face_size, eq_width, eq_height):
    """CPU fallback for coordinate conversion using GPU_Convert math"""
    results = []
    
    for face_id, x_dice, y_dice in zip(faces, x_dice_batch, y_dice_batch):
        if face_id < 0:
            results.append([np.nan, np.nan])
            continue
        
        # Get face-relative coordinates
        row = int(y_dice // face_size)
        col = int(x_dice // face_size)
        x_face = x_dice - col * face_size
        y_face = y_dice - row * face_size
        
        # Use the cubemap_to_equirectangular_uv function with GPU_Convert math
        u_eq, v_eq = cubemap_to_equirectangular_uv(face_id, x_face, y_face, face_size, eq_width, eq_height)
        results.append([u_eq, v_eq])
    
    return np.array(results)

def cubemap_to_equirectangular_uv(face_id, x, y, cubemap_size=1920, eq_width=7680, eq_height=3840):
    """CPU fallback coordinate conversion using GPU_Convert math"""
    # Normalize to [-0.5, 0.5] range (matching GPU_Convert's xyzcube)
    x_norm = (x / cubemap_size) - 0.5
    y_norm = (y / cubemap_size) - 0.5
    
    # Face-specific vector assignments matching GPU_Convert's xyzcube
    if face_id == 0:  # F (front, z = 0.5)
        vec = [x_norm, -y_norm, 0.5]
    elif face_id == 1:  # R (right, x = 0.5)
        vec = [0.5, -y_norm, -x_norm]
    elif face_id == 2:  # B (back, z = -0.5)
        vec = [-x_norm, -y_norm, -0.5]
    elif face_id == 3:  # L (left, x = -0.5)
        vec = [-0.5, -y_norm, x_norm]
    elif face_id == 4:  # U (up, y = 0.5)
        vec = [x_norm, 0.5, y_norm]
    elif face_id == 5:  # D (down, y = -0.5)
        vec = [x_norm, -0.5, -y_norm]
    else:
        return np.nan, np.nan
    
    # Normalize vector (matching GPU_Convert's xyz2uv)
    vec = np.array(vec)
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        norm = 1e-9
    vec = vec / norm
    
    # Spherical coordinates using GPU_Convert's approach
    # longitude (u) = arctan2(x, z)
    # latitude (v) = arcsin(y)
    u = np.arctan2(vec[0], vec[2])
    v = np.arcsin(np.clip(vec[1], -1, 1))
    
    # Convert to equirectangular pixel coordinates (matching GPU_Convert's uv2coor)
    coor_x = (u / (2 * np.pi) + 0.5) * eq_width - 0.5
    coor_y = (-v / np.pi + 0.5) * eq_height - 0.5
    
    return coor_x, coor_y

def identify_dicemap_face(x_dice, y_dice, face_size):
    """Original single point version for compatibility."""
    face_positions = {
        (0, 1): "U", (1, 0): "L", (1, 1): "F",
        (1, 2): "R", (1, 3): "B", (2, 1): "D"
    }
    row = int(y_dice // face_size)
    col = int(x_dice // face_size)
    return face_positions.get((row, col), None)

@torch.no_grad()
def match_from_paths_batched_optimized(
    conf: Dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
    use_dicemap: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    max_keypoints: int = 4096,
) -> Path:
    """Ultra-optimized matching with intelligent batching and GPU acceleration"""
    
    logger.info(f"Batched matching pipeline:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Max keypoints: {max_keypoints}")
    logger.info(f"  Workers: {num_workers}")
    logger.info(f"  Use dicemap: {use_dicemap}")
    
    # Validation
    for path, name in [(feature_path_q, "query"), (feature_path_ref, "reference")]:
        if not path.exists():
            raise FileNotFoundError(f"{name.title()} features not found: {path}")
    
    match_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Load and filter pairs
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    
    if len(pairs) == 0:
        logger.info("All pairs already processed.")
        return match_path
    
    print(f"Processing {len(pairs)} unique pairs...")
    
    # Setup device and components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize XFeat with optimizations
    xfeat = XFeat()
    
    # Enable model optimizations
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            xfeat = torch.compile(xfeat, mode='reduce-overhead')
            print("XFeat compiled for optimal performance")
        except Exception as e:
            print(f"Model compilation failed: {e}")
    
    # Setup dataset with intelligent batching
    dataset = BatchedFeaturePairsDataset(
        pairs, feature_path_q, feature_path_ref, 
        max_keypoints=max_keypoints, enable_caching=True
    )
    
    # Determine optimal batch size based on available memory
    if device.type == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
        # Adaptive batch sizing based on GPU memory
        if gpu_memory_gb >= 16:
            adaptive_batch_size = min(batch_size, 256)
        elif gpu_memory_gb >= 8:
            adaptive_batch_size = min(batch_size, 32)
        else:
            adaptive_batch_size = min(batch_size, 16)
        print(f"Adaptive batch size: {adaptive_batch_size} (GPU memory: {gpu_memory_gb:.1f}GB)")
    else:
        adaptive_batch_size = min(batch_size, 8)
    
    # Setup data loader with optimizations
    loader = DataLoader(
        dataset,
        batch_size=adaptive_batch_size,
        shuffle=False,
        num_workers=min(num_workers, mp.cpu_count()),
        worker_init_fn=worker_init,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
        collate_fn=smart_collate_fn,
        drop_last=False
    )
    
    # Initialize processors
    batch_processor = UltraBatchProcessor(xfeat, device, adaptive_batch_size)
    writer = HighThroughputWriter(match_path, num_workers=6, write_batch_size=128)
    
    # Main processing loop
    try:
        total_batches = len(loader)
        
        for batch_idx, batch_data in enumerate(tqdm(loader, desc="Processing batches")):
            try:
                # Extract pair names
                if isinstance(batch_data.get('pair_name'), list):
                    pair_names = batch_data['pair_name']
                else:
                    # Single item
                    pair_names = [batch_data['pair_name']]
                    # Convert single item to batch format
                    batch_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 else [v] 
                                 for k, v in batch_data.items()}
                
                # Process batch
                batch_results = batch_processor.process_batch_ultra_optimized(batch_data, pair_names)
                
                # Queue for writing
                if batch_results:
                    writer.put(batch_results)
                
                # Memory management
                if batch_idx % 10 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Create empty results for failed batch
                if 'pair_name' in batch_data:
                    pair_names = batch_data['pair_name'] if isinstance(batch_data['pair_name'], list) else [batch_data['pair_name']]
                    empty_results = [(name, batch_processor._create_empty_matches(1)) for name in pair_names]
                    writer.put(empty_results)
                continue
    
    finally:
        # Cleanup
        writer.join()
        batch_processor.print_stats()
        
        if hasattr(dataset, '__del__'):
            dataset.__del__()
        
        # Final memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    logger.info("Batched matching completed successfully.")
    return match_path

def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    """Optimized pair filtering with set operations and better I/O"""
    # Remove duplicates using set operations
    pairs_set = set()
    for i, j in pairs_all:
        # Add both orderings to check, but only keep one
        key = (min(i, j), max(i, j))
        pairs_set.add(key)
    
    pairs = [(i, j) for i, j in pairs_set]
    
    if match_path is not None and match_path.exists():
        try:
            with h5py.File(str(match_path), "r", libver="latest") as fd:
                existing_keys = set(fd.keys())
                
                filtered_pairs = []
                for i, j in pairs:
                    # Check all possible naming conventions
                    possible_names = [
                        names_to_pair(i, j),
                        names_to_pair(j, i),
                        names_to_pair_old(i, j),
                        names_to_pair_old(j, i)
                    ]
                    
                    # Only keep if none of the names exist
                    if not any(name in existing_keys for name in possible_names):
                        filtered_pairs.append((i, j))
                
                print(f"Filtered: {len(pairs)} -> {len(filtered_pairs)} new pairs")
                return filtered_pairs
        except Exception as e:
            logger.warning(f"Error reading existing matches: {e}")
            return pairs
    
    return pairs

# Legacy compatibility classes and functions
class FeaturePairsDataset(torch.utils.data.Dataset):
    """Original dataset class for backward compatibility"""
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, "r") as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k + "0"] = torch.from_numpy(v.__array__()).float()
            data["image0"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        with h5py.File(self.feature_path_r, "r") as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k + "1"] = torch.from_numpy(v.__array__()).float()
            data["image1"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        return data

    def __len__(self):
        return len(self.pairs)

class WorkQueue:
    """Original work queue for backward compatibility"""
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,)) for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)

def batch_writer_fn(batch_results, match_path):
    """Optimized batch writer for multiple pairs at once."""
    try:
        with h5py.File(str(match_path), "a", libver="latest") as fd:
            for pair, pred in batch_results:
                if pair in fd:
                    del fd[pair]
                grp = fd.create_group(pair)
                matches = pred["matches0"].cpu().short().numpy()
                grp.create_dataset("matches0", data=matches)
                if "matching_scores0" in pred:
                    scores = pred["matching_scores0"].cpu().half().numpy()
                    grp.create_dataset("matching_scores0", data=scores)
    except Exception as e:
        print(f"Error in batch_writer_fn: {str(e)}")
        import traceback
        traceback.print_exc()

def writer_fn(inp, match_path):
    """Original writer function for compatibility."""
    try:
        pair, pred = inp
        with h5py.File(str(match_path), "a", libver="latest") as fd:
            if pair in fd:
                del fd[pair]
            grp = fd.create_group(pair)
            matches = pred["matches0"].cpu().short().numpy()
            grp.create_dataset("matches0", data=matches)
            if "matching_scores0" in pred:
                scores = pred["matching_scores0"].cpu().half().numpy()
                grp.create_dataset("matching_scores0", data=scores)
    except Exception as e:
        print(f"Error in writer_fn: {str(e)}")
        import traceback
        traceback.print_exc()

def process_single_pair_with_batched_xfeat(data, pair_name, xfeat, device, verbose=False):
    """Process a single pair with robust tensor handling (legacy compatibility)"""
    try:
        overall_keypoints0 = data["keypoints0"].squeeze().cpu().numpy()
        overall_keypoints1 = data["keypoints1"].squeeze().cpu().numpy()

        # Robust image size extraction
        image_size0 = data.get("image_size0")
        
        # Handle various tensor shapes
        if isinstance(image_size0, torch.Tensor):
            image_size0 = image_size0.cpu().numpy()
            if image_size0.ndim == 0:
                eq_width = int(image_size0.item())
                eq_height = eq_width // 2
            elif image_size0.ndim == 1:
                eq_width = int(image_size0[0])
                eq_height = int(image_size0[1]) if len(image_size0) > 1 else eq_width // 2
            elif image_size0.ndim == 2:
                eq_width = int(image_size0[0, 0])
                eq_height = int(image_size0[0, 1]) if image_size0.shape[1] > 1 else eq_width // 2
            else:
                eq_width = int(image_size0.flatten()[0])
                eq_height = int(image_size0.flatten()[1]) if image_size0.size > 1 else eq_width // 2
        else:
            print (pair_name)
            # Fallback - assume standard equirectangular dimensions
            eq_width, eq_height = 7680, 3840
        
        face_size = eq_width // 4
        
        # Prepare data for XFeat matching
        output0 = {
            'keypoints': data["dicemap_keypoints0"].squeeze(),
            'descriptors': data["dicemap_descriptors0"].squeeze(),
            'scores': data["dicemap_scores0"].squeeze(),
            'image_size': (face_size * 4, face_size * 3)
        }
        output1 = {
            'keypoints': data["dicemap_keypoints1"].squeeze(),
            'descriptors': data["dicemap_descriptors1"].squeeze(),
            'scores': data["dicemap_scores1"].squeeze(),
            'image_size': (face_size * 4, face_size * 3)
        }
        
        # XFeat matching
        res = xfeat.batch_match_lighterglue([output0], [output1])
        
        equirect_coords1 = []
        equirect_coords2 = []
        
        if len(res) > 0 and len(res[0]["mkpts_0"]) > 0:
            mkpts0_dice = res[0]["mkpts_0"]
            mkpts1_dice = res[0]["mkpts_1"]
            
            # Convert coordinates with batch processing
            faces0 = identify_dicemap_face_batch(mkpts0_dice[:, 0], mkpts0_dice[:, 1], face_size)
            faces1 = identify_dicemap_face_batch(mkpts1_dice[:, 0], mkpts1_dice[:, 1], face_size)
            
            equirect_batch0 = dicemap_to_equirectangular_uv_batch_optimized(
                faces0, mkpts0_dice[:, 0], mkpts0_dice[:, 1], face_size, eq_width, eq_height
            )
            equirect_batch1 = dicemap_to_equirectangular_uv_batch_optimized(
                faces1, mkpts1_dice[:, 0], mkpts1_dice[:, 1], face_size, eq_width, eq_height
            )
            
            # Filter valid conversions
            valid_mask = (~np.isnan(equirect_batch0).any(axis=1)) & (~np.isnan(equirect_batch1).any(axis=1))
            equirect_coords1 = equirect_batch0[valid_mask]
            equirect_coords2 = equirect_batch1[valid_mask]
        
        # Create matches using nearest neighbor search
        num_keypoints = overall_keypoints0.shape[0]
        matches0 = -np.ones(num_keypoints, dtype=np.int64)
        matching_scores0 = np.zeros(num_keypoints, dtype=np.float32)
        
        if len(equirect_coords1) > 0:
            tree0 = cKDTree(overall_keypoints0.astype('float32'))
            tree1 = cKDTree(overall_keypoints1.astype('float32'))
            
            _, indices0 = tree0.query(equirect_coords1.astype('float32'), k=1)
            _, indices1 = tree1.query(equirect_coords2.astype('float32'), k=1)
            
            matches0[indices0] = indices1
        
        pred = {
            'matches0': torch.from_numpy(matches0),
            'matching_scores0': torch.from_numpy(matching_scores0),
        }
        
        return (pair_name, pred)
        
    except Exception as e:
        if verbose:
            print(f"Error processing {pair_name}: {e}")
            import traceback
            traceback.print_exc()
        
        # Return empty result for failed pairs
        num_keypoints = len(data.get("keypoints0", torch.empty(0)).squeeze())
        matches0 = -np.ones(max(num_keypoints, 1), dtype=np.int64)
        matching_scores0 = np.zeros(max(num_keypoints, 1), dtype=np.float32)
        
        pred = {
            'matches0': torch.from_numpy(matches0),
            'matching_scores0': torch.from_numpy(matching_scores0),
        }
        return (pair_name, pred)

class BatchProcessor:
    """Original batch processor for compatibility"""
    def __init__(self, xfeat, batch_size=8):
        self.xfeat = xfeat
        self.batch_size = batch_size
        self.reset()
    
    def reset(self):
        self.pending_data = []
        self.pending_pairs = []
    
    def add(self, data, pair_name, device):
        """Add processed data to the pending batch."""
        # Move data to device
        data = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) and not k.startswith("image") else v 
                for k, v in data.items()}
        
        self.pending_data.append(data)
        self.pending_pairs.append(pair_name)
    
    def is_ready(self):
        return len(self.pending_data) >= self.batch_size
    
    def is_empty(self):
        return len(self.pending_data) == 0
    
    def process_and_reset(self, device):
        """Process accumulated batch and reset."""
        if self.is_empty():
            return []
        
        results = []
        
        # Process each pair but batch the XFeat operations
        for i, (data, pair_name) in enumerate(zip(self.pending_data, self.pending_pairs)):
            try:
                result = process_single_pair_with_batched_xfeat(data, pair_name, self.xfeat, device, i == 0)
                results.append(result)
            except Exception as e:
                print(f"Error processing {pair_name}: {e}")
                continue
        
        self.reset()
        return results

# Legacy functions for backward compatibility
@torch.no_grad()
def match_from_paths_optimized(
    conf: Dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
    use_dicemap: bool = True,
    batch_size: int = 8,
) -> Path:
    """Legacy optimized implementation"""
    logger.info(
        "Matching local features with optimized batching (batch_size={})\nConfiguration:\n{}".format(
            batch_size, pprint.pformat(conf)
        )
    )

    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q} not found.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref} not found.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    
    if len(pairs) == 0:
        logger.info("Skipping the matching as everything is already computed.")
        return match_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using Device:", device)
    
    # Initialize XFeat once
    xfeat = XFeat()
    
    # Use original dataset but with better batching logic
    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    
    # Simple single-item loader to avoid tensor issues
    loader = torch.utils.data.DataLoader(
        dataset, 
        num_workers=2,   # Reduced workers for stability
        batch_size=1,    # Keep at 1 to avoid collation
        shuffle=False, 
        pin_memory=True,
    )
    
    # Initialize batch processor and writer
    batch_processor = BatchProcessor(xfeat, batch_size)
    writer_queue = WorkQueue(partial(batch_writer_fn, match_path=match_path), 3)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        # Safe tensor extraction
        data = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in data.items()}
        name0, name1 = pairs[idx]
        pair_name = names_to_pair(name0, name1)
        
        # Add to batch processor
        batch_processor.add(data, pair_name, device)
        
        # Process when batch is ready or at the end
        if batch_processor.is_ready() or idx == len(pairs) - 1:
            batch_results = batch_processor.process_and_reset(device)
            if batch_results:
                writer_queue.put(batch_results)

    writer_queue.join()
    logger.info("Finished exporting matches with optimized batching.")
    return match_path

@torch.no_grad()
def match_from_paths(
    conf: Dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
    use_dicemap: bool = True,
) -> Path:
    """Original implementation maintained for compatibility"""
    logger.info("Using legacy single-pair processing...")
    return match_from_paths_batched_optimized(
        conf, pairs_path, match_path, feature_path_q, feature_path_ref,
        overwrite, use_dicemap, batch_size=1, num_workers=1
    )

def main(
    conf: Dict,
    pairs: Path,
    features: Union[Path, str],
    export_dir: Optional[Path] = None,
    matches: Optional[Path] = None,
    features_ref: Optional[Path] = None,
    overwrite: bool = False,
    use_dicemap: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    max_keypoints: int = 4096,
    use_batching: bool = True,
) -> Path:
    """Main entry point with full batching support"""
    
    # Path resolution
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError("Provide matches path when features is a file path")
    else:
        if export_dir is None:
            raise ValueError(f"Provide export_dir for features name: {features}")
        features_q = Path(export_dir, features)
        if matches is None:
            suffix = "dicemap" if use_dicemap else "faces"
            matches = Path(export_dir, f'{features}_{conf["output"]}_{suffix}_{pairs.stem}.h5')
    
    if features_ref is None:
        features_ref = features_q
    
    # Choose processing method
    if use_batching:
        print(f"Using batched processing (batch_size={batch_size})")
        return match_from_paths_batched_optimized(
            conf, pairs, matches, features_q, features_ref,
            overwrite, use_dicemap, batch_size, num_workers, max_keypoints
        )
    else:
        print("Using legacy single-pair processing")
        return match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite, use_dicemap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultra-optimized XFeat matching with batching")
    
    # Required arguments
    parser.add_argument("--pairs", type=Path, required=True, help="Path to pairs file")
    parser.add_argument("--export_dir", type=Path, help="Export directory")
    parser.add_argument("--features", type=Path, help="Features file path")
    parser.add_argument("--matches", type=Path, help="Output matches file")
    
    # Processing options
    parser.add_argument("--use_dicemap", action="store_true", help="Use dicemap features for matching")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing matches")
    
    # Batching and optimization parameters
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for processing (higher = faster but more memory)")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of data loading workers")
    parser.add_argument("--max_keypoints", type=int, default=4096,
                       help="Maximum keypoints per image (for memory management)")
    parser.add_argument("--disable_batching", action="store_true",
                       help="Disable batching and use original single-pair processing")
    
    # Advanced options
    parser.add_argument("--memory_limit_gb", type=float, default=8.0,
                       help="GPU memory limit in GB")
    
    args = parser.parse_args()
    
    # Validation
    if not CUPY_AVAILABLE:
        print("WARNING: CuPy not available. Install with: pip install cupy-cuda12x")
        print("Performance will be significantly reduced without GPU acceleration.")
    
    # Configuration
    conf = {"output": "matches"}
    
    # Adjust batch size based on available memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if args.batch_size > 32 and gpu_memory_gb < 8:
            print(f"Warning: Large batch size ({args.batch_size}) with limited GPU memory ({gpu_memory_gb:.1f}GB)")
            print("Consider reducing batch_size to avoid OOM errors")
    
    print(f"Configuration:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max keypoints: {args.max_keypoints}")
    print(f"  - Workers: {args.num_workers}")
    print(f"  - Batching enabled: {not args.disable_batching}")
    print(f"  - CuPy acceleration: {CUPY_AVAILABLE}")
    
    start_time = time.time()
    
    # Run optimized matching
    output_path = main(
        conf=conf,
        pairs=args.pairs,
        features=args.features,
        export_dir=args.export_dir,
        matches=args.matches,
        overwrite=args.overwrite,
        use_dicemap=args.use_dicemap,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_keypoints=args.max_keypoints,
        use_batching=not args.disable_batching
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f}s")
    print(f"Output saved to: {output_path}")