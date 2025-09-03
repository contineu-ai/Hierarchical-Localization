import os
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from enum import Enum
from typing import Tuple, List, Optional, Dict, Union
import random
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, deque
import time
import asyncio
import aiofiles
from functools import lru_cache
import gc
import psutil
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import CuPy for GPU operations
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cupyx_ndimage
    import cupy.cuda.runtime as cuda_runtime
    CUPY_AVAILABLE = True
    print(f"CuPy available - GPU memory: {cp.cuda.Device().mem_info[1] // 1024**3}GB")
except ImportError:
    CUPY_AVAILABLE = False
    print("ERROR: CuPy required for optimal performance. Install: pip install cupy-cuda12x")
    exit(1)

class MaskType(Enum):
    SOLID_COLOR = "solid_color"
    BLUR = "blur" 
    PIXELATE = "pixelate"

class MemoryPool:
    """GPU memory pool manager for efficient allocation reuse"""
    
    def __init__(self, initial_size_gb=2.0):
        self.pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        self.pool.set_limit(size=int(initial_size_gb * 1024**3))
        cp.cuda.set_allocator(self.pool.malloc)
        self.stats = {'allocated': 0, 'reused': 0}
    
    def get_memory_info(self):
        """Get current memory usage statistics"""
        used = self.pool.used_bytes()
        total = self.pool.total_bytes()
        return {'used_mb': used // 1024**2, 'total_mb': total // 1024**2}
    
    def cleanup(self):
        """Force cleanup of unused memory"""
        self.pool.free_all_blocks()
        cp.cuda.Device().synchronize()

class GPU_Convert:
    """GPU-accelerated cubemap converter using the provided implementation"""
    def __init__(self, image_shape):
        h, w, *_ = image_shape
        SQUARE_SIDE = h // 2
        self.cp = cp
        # Fixed import for cupyx.scipy.ndimage
        import cupyx.scipy.ndimage
        self.cupyx_ndimage = cupyx.scipy.ndimage
        self.coors_xy = self.uv2coor(self.xyz2uv(self.xyzcube(SQUARE_SIDE)), h, w)
        print(f"GPU_Convert initialized for {h}x{w} images")
    
    def xyzcube(self, face_w):
        """
        Return the xyz coordinates of the unit cube in [F R B L U D] format.
        Fixed for continuous faces.
        """
        out = self.cp.zeros((face_w, face_w * 6, 3), dtype=self.cp.float32)
        rng = self.cp.linspace(-0.5, 0.5, num=face_w, dtype=self.cp.float32)
        
        # Create base grid for each face
        x_grid, y_grid = self.cp.meshgrid(rng, rng, indexing='xy')
        
        # Front face (z = 0.5) - looking towards +z
        out[:, 0 * face_w : 1 * face_w, 0] = x_grid  # x
        out[:, 0 * face_w : 1 * face_w, 1] = -y_grid  # y (flipped for proper orientation)
        out[:, 0 * face_w : 1 * face_w, 2] = 0.5      # z
        
        # Right face (x = 0.5) - looking towards +x
        out[:, 1 * face_w : 2 * face_w, 0] = 0.5      # x
        out[:, 1 * face_w : 2 * face_w, 1] = -y_grid  # y (flipped for proper orientation)
        out[:, 1 * face_w : 2 * face_w, 2] = -x_grid  # z (flipped to maintain continuity)
        
        # Back face (z = -0.5) - looking towards -z
        out[:, 2 * face_w : 3 * face_w, 0] = -x_grid  # x (flipped for proper orientation)
        out[:, 2 * face_w : 3 * face_w, 1] = -y_grid  # y (flipped for proper orientation)
        out[:, 2 * face_w : 3 * face_w, 2] = -0.5     # z
        
        # Left face (x = -0.5) - looking towards -x
        out[:, 3 * face_w : 4 * face_w, 0] = -0.5     # x
        out[:, 3 * face_w : 4 * face_w, 1] = -y_grid  # y (flipped for proper orientation)
        out[:, 3 * face_w : 4 * face_w, 2] = x_grid   # z
        
        # Up face (y = 0.5) - looking towards +y
        out[:, 4 * face_w : 5 * face_w, 0] = x_grid   # x
        out[:, 4 * face_w : 5 * face_w, 1] = 0.5      # y
        out[:, 4 * face_w : 5 * face_w, 2] = y_grid   # z
        
        # Down face (y = -0.5) - looking towards -y
        out[:, 5 * face_w : 6 * face_w, 0] = x_grid   # x
        out[:, 5 * face_w : 6 * face_w, 1] = -0.5     # y
        out[:, 5 * face_w : 6 * face_w, 2] = -y_grid  # z (flipped for proper orientation)
        
        return out

    def xyz2uv(self, xyz):
        """
        xyz: cp.ndarray in shape of [..., 3]
        """
        x, y, z = self.cp.split(xyz, 3, axis=-1)
        
        # Normalize the vector to ensure it's on the unit sphere for arcsin
        norm = self.cp.sqrt(x**2 + y**2 + z**2)
        # Add a small epsilon to prevent division by zero for the center point
        norm = self.cp.maximum(norm, 1e-9) 
        
        # Calculate longitude (u) - remains the same
        u = self.cp.arctan2(x, z)
        
        # Calculate latitude (v) using arcsin for consistency
        v = self.cp.arcsin(y / norm)
        
        return self.cp.concatenate([u, v], axis=-1)
    
    def uv2coor(self, uv, h, w):
        """
        uv: cp.ndarray in shape of [..., 2]
        h: int, height of the equirectangular image
        w: int, width of the equirectangular image
        """
        u, v = self.cp.split(uv, 2, axis=-1)
        coor_x = (u / (2 * self.cp.pi) + 0.5) * w - 0.5
        coor_y = (-v / self.cp.pi + 0.5) * h - 0.5
        return self.cp.concatenate([coor_x, coor_y], axis=-1).astype(self.cp.float32)

    def sample_equirec(self, e_img, coor_xy, order):
        cp, nd = self.cp, self.cupyx_ndimage
        H, W = e_img.shape
        
        e_pad = self.cp.pad(e_img, ((1, 1), (0, 0)), mode="edge")

        coor_x, coor_y = self.cp.split(coor_xy, 2, axis=-1)
        coor_y = coor_y + 1.0  

        coords = self.cp.concatenate([coor_y, coor_x], axis=-1).reshape(-1, 2).T.astype(self.cp.float32)
        out = nd.map_coordinates(e_pad, coords, order=order, mode="wrap")  
        return out.reshape(coor_x.shape[:-1])
    
    def e2c(self, e_img, coor_xy):
        c = e_img.shape[2]
        e_img = self.cp.asarray(e_img)
        coor_xy = self.cp.asarray(coor_xy)
        cubemap = self.cp.stack(
            [self.sample_equirec(e_img[..., i], coor_xy, order=1) for i in range(c)],
            axis=-1,
        )
        cubemap_faces = self.cp.array_split(cubemap, 6, axis=1)
        cubemap_dict = {
            k: self.cp.asnumpy(cubemap_faces[i])
            for i, k in enumerate(["F", "R", "B", "L", "U", "D"])
        }
        return cubemap_dict
    
    def convert_to_cubemaps(self, img):
        cubemap_images_dict = self.e2c(img, self.coors_xy)
        return cubemap_images_dict

def cubemap_to_equirectangular_uv(face: str, x: int, y: int, cubemap_size: int):
    """
    Convert cubemap face coordinates to equirectangular UV coordinates.
    This matches the face orientation used in the GPU_Convert class.
    
    Args:
        face (str): Face identifier ("F", "R", "B", "L", "U", "D")
        x (int): X coordinate on the face (0 to cubemap_size-1)
        y (int): Y coordinate on the face (0 to cubemap_size-1)
        cubemap_size (int): Size of the cubemap face
        
    Returns:
        list: [u, v] coordinates in equirectangular space (0 to 1)
    """
    # Normalize to [0, 1] range
    x_norm = x / (cubemap_size - 1)
    y_norm = y / (cubemap_size - 1)
    
    # Convert to [-0.5, 0.5] range (matching the rng in xyzcube)
    x_cube = x_norm - 0.5
    y_cube = y_norm - 0.5
    
    # Convert to 3D vector based on face orientation
    # This matches the coordinate assignments in the xyzcube method
    if face == "F":  # Front face (z = 0.5)
        vec = [x_cube, -y_cube, 0.5]
        
    elif face == "R":  # Right face (x = 0.5)
        vec = [0.5, -y_cube, -x_cube]
        
    elif face == "B":  # Back face (z = -0.5)
        vec = [-x_cube, -y_cube, -0.5]
        
    elif face == "L":  # Left face (x = -0.5)
        vec = [-0.5, -y_cube, x_cube]
        
    elif face == "U":  # Up face (y = 0.5)
        vec = [x_cube, 0.5, y_cube]
        
    elif face == "D":  # Down face (y = -0.5)
        vec = [x_cube, -0.5, -y_cube]
        
    else:
        raise ValueError(f"Invalid face identifier: {face}")
    
    # Normalize vector (matching GPU_Convert's approach)
    vec = np.array(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        norm = 1e-9
    vec = vec / norm
    
    # Convert to spherical coordinates (matching GPU_Convert's xyz2uv)
    x_3d, y_3d, z_3d = vec
    
    # longitude (u) = arctan2(x, z)
    # latitude (v) = arcsin(y)
    theta = np.arctan2(x_3d, z_3d)  # longitude
    phi = np.arcsin(y_3d)           # latitude
    
    # Map to equirectangular UV coordinates [0, 1] (matching GPU_Convert's uv2coor)
    u = (theta / (2 * np.pi) + 0.5)  # longitude: [-π, π] → [0, 1]
    v = (-phi / np.pi + 0.5)          # latitude: [-π/2, π/2] → [0, 1]
    
    # Ensure values are in valid range
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    return [u, v]

class CubemapBatchConverter:
    """Wrapper to handle batch processing with GPU_Convert"""
    
    def __init__(self, max_batch_size=64, memory_pool_size=4.0):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy required")
        
        self.max_batch_size = max_batch_size
        self.memory_pool = MemoryPool(memory_pool_size)
        
        # Cache for converters per image size
        self.converter_cache = {}
        self.max_cache_size = 5
        
        print(f"CubemapBatchConverter initialized (max_batch: {max_batch_size})")
    
    def _get_converter(self, h, w):
        """Get cached converter for specific image size"""
        cache_key = (h, w)
        
        if cache_key not in self.converter_cache:
            if len(self.converter_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.converter_cache))
                del self.converter_cache[oldest_key]
            
            print(f"Creating GPU_Convert for {h}x{w}")
            # Create dummy image shape for initialization
            image_shape = (h, w, 3)
            self.converter_cache[cache_key] = GPU_Convert(image_shape)
        
        return self.converter_cache[cache_key]
    
    def convert_batch(self, image_batch):
        """Convert batch of equirectangular images to cubemaps"""
        if isinstance(image_batch, list):
            image_batch = np.stack(image_batch)
        
        if isinstance(image_batch, cp.ndarray):
            image_batch = cp.asnumpy(image_batch)
        
        if image_batch.size == 0:
            return np.array([])
        
        batch_size = len(image_batch)
        h, w = image_batch.shape[1:3]
        face_size = h // 2
        
        # Get converter for this image size
        converter = self._get_converter(h, w)
        
        # Process images and collect as continuous array
        # Output shape: (batch, face_h, face_w * 6, channels)
        cubemap_batch = np.zeros((batch_size, face_size, face_size * 6, image_batch.shape[3]), dtype=image_batch.dtype)
        
        for i, img in enumerate(image_batch):
            # Convert single image
            cubemap_dict = converter.convert_to_cubemaps(img)
            
            # Arrange faces in continuous array [F, R, B, L, U, D]
            face_order = ["F", "R", "B", "L", "U", "D"]
            for j, face_name in enumerate(face_order):
                start_idx = j * face_size
                end_idx = (j + 1) * face_size
                cubemap_batch[i, :, start_idx:end_idx, :] = cubemap_dict[face_name]
        
        return cubemap_batch
    
    def create_dicemaps_batch(self, cubemap_batch):
        """Create dicemaps from cubemap batch"""
        if cubemap_batch.size == 0:
            return np.array([])
        
        B, face_h, face_w_total, C = cubemap_batch.shape
        face_w = face_h
        
        # Dicemap layout: 3 rows x 4 columns
        #     [   U   ]
        # [ L ][ F ][ R ][ B ]
        #     [   D   ]
        dicemap_h, dicemap_w = face_h * 3, face_w * 4
        
        # Pre-allocate dicemap batch
        dicemaps = np.zeros((B, dicemap_h, dicemap_w, C), dtype=cubemap_batch.dtype)
        
        # Face mapping: (source_start, source_end, dest_row, dest_col)
        # Order in cubemap: F(0), R(1), B(2), L(3), U(4), D(5)
        face_slices = [
            (0*face_w, 1*face_w, 1, 1),  # Front -> center
            (1*face_w, 2*face_w, 1, 2),  # Right -> middle right  
            (2*face_w, 3*face_w, 1, 3),  # Back -> middle far-right
            (3*face_w, 4*face_w, 1, 0),  # Left -> middle left
            (4*face_w, 5*face_w, 0, 1),  # Up -> top center
            # Omit Down face as in original (only use first 5 faces)
        ]
        
        for src_start, src_end, row, col in face_slices:
            dest_y_start, dest_y_end = row * face_h, (row + 1) * face_h
            dest_x_start, dest_x_end = col * face_w, (col + 1) * face_w
            
            dicemaps[:, dest_y_start:dest_y_end, dest_x_start:dest_x_end, :] = \
                cubemap_batch[:, :, src_start:src_end, :]
        
        return dicemaps
    
    def cleanup(self):
        """Clean up resources"""
        self.converter_cache.clear()
        self.memory_pool.cleanup()

class OptimizedYOLOProcessor:
    """Highly optimized YOLO processor with smart batching"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, 
                 max_batch_size: int = 64, input_size: int = 640):
        self.conf_threshold = conf_threshold
        self.max_batch_size = max_batch_size
        self.input_size = input_size
        
        self._initialize_session(model_path)
        self._setup_preprocessing()
    
    def _initialize_session(self, model_path: str):
        """Initialize optimized ONNX session"""
        # Advanced session options for maximum performance
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 0  # Use all available cores
        sess_options.inter_op_num_threads = 0
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        sess_options.enable_mem_reuse = True
        
        # Optimized providers with configuration
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB limit
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # Get model info
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"YOLO model loaded: {self.input_size}x{self.input_size}, batch_size: {self.max_batch_size}")
    
    def _setup_preprocessing(self):
        """Pre-compute preprocessing constants"""
        self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.std = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    
    def _preprocess_batch_optimized(self, images):
        """Highly optimized batch preprocessing"""
        batch_size = len(images)
        
        # Pre-allocate tensors
        input_batch = np.empty((batch_size, 3, self.input_size, self.input_size), dtype=np.float32)
        scale_factors = np.empty((batch_size, 2), dtype=np.float32)
        
        # Vectorized preprocessing
        for i, img in enumerate(images):
            if img is None or img.size == 0:
                scale_factors[i] = [1.0, 1.0]
                continue
            
            h, w = img.shape[:2]
            scale_factors[i] = [w / self.input_size, h / self.input_size]
            
            # Optimized resize and normalization
            resized = cv2.resize(img, (self.input_size, self.input_size), 
                               interpolation=cv2.INTER_LINEAR)
            
            # Direct tensor assignment with normalization
            input_batch[i] = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0
        
        return input_batch, scale_factors
    
    def _process_detections_vectorized(self, outputs, scale_factors):
        """Vectorized detection processing"""
        if not outputs or len(outputs) == 0:
            return [([], [], [])] * len(scale_factors)
        
        batch_detections = outputs[0]  # Shape: (batch_size, num_detections, 7)
        results = []
        
        for b, scales in enumerate(scale_factors):
            if b >= batch_detections.shape[0]:
                results.append(([], [], []))
                continue
            
            detections = batch_detections[b]
            
            # Vectorized confidence filtering
            valid_mask = detections[:, 6] >= self.conf_threshold
            valid_detections = detections[valid_mask]
            
            if len(valid_detections) == 0:
                results.append(([], [], []))
                continue
            
            # Vectorized coordinate scaling
            boxes = valid_detections[:, 1:5].copy()  # x1, y1, x2, y2
            boxes[:, [0, 2]] *= scales[0]  # Scale x coordinates
            boxes[:, [1, 3]] *= scales[1]  # Scale y coordinates
            
            # Convert to (x, y, w, h) format
            boxes[:, 2] -= boxes[:, 0]  # width = x2 - x1
            boxes[:, 3] -= boxes[:, 1]  # height = y2 - y1
            
            # Extract results
            boxes_list = [tuple(map(int, box)) for box in boxes]
            scores_list = valid_detections[:, 6].tolist()
            classes_list = valid_detections[:, 5].astype(int).tolist()
            
            results.append((boxes_list, scores_list, classes_list))
        
        return results
    
    def detect_batch(self, images: List[np.ndarray]) -> List[Tuple]:
        """Main batch detection function"""
        if not images:
            return []
        
        results = []
        
        # Process in optimally sized chunks
        for i in range(0, len(images), self.max_batch_size):
            chunk = images[i:min(i + self.max_batch_size, len(images))]
            
            # Preprocess
            input_batch, scale_factors = self._preprocess_batch_optimized(chunk)
            
            # Inference
            try:
                outputs = self.session.run(self.output_names, {self.input_name: input_batch})
                chunk_results = self._process_detections_vectorized(outputs, scale_factors)
                results.extend(chunk_results)
            except Exception as e:
                print(f"YOLO inference error: {e}")
                results.extend([([], [], [])] * len(chunk))
        
        return results

class StreamlinedMaskingProcessor:
    """Optimized masking with GPU acceleration where possible"""
    
    def __init__(self, mask_type: MaskType, mask_color: Tuple[int, int, int]):
        self.mask_type = mask_type
        self.mask_color = mask_color
        
        # Pre-compile kernels for different masking operations
        if CUPY_AVAILABLE:
            self._setup_gpu_kernels()
    
    def _setup_gpu_kernels(self):
        """Setup custom CUDA kernels for masking operations"""
        # Solid color masking kernel
        self.solid_color_kernel = cp.ElementwiseKernel(
            'raw uint8 img, raw int32 boxes, raw uint8 color',
            'raw uint8 out',
            '''
            int y = i / img.shape[2];
            int x = i % img.shape[2];
            int c = (i / (img.shape[1] * img.shape[2])) % img.shape[0];
            
            // Check if pixel is in any bounding box
            bool in_box = false;
            for (int box_idx = 0; box_idx < boxes.size() / 4; box_idx++) {
                int bx = boxes[box_idx * 4];
                int by = boxes[box_idx * 4 + 1];
                int bw = boxes[box_idx * 4 + 2];
                int bh = boxes[box_idx * 4 + 3];
                
                if (x >= bx && x < bx + bw && y >= by && y < by + bh) {
                    in_box = true;
                    break;
                }
            }
            
            if (in_box) {
                out = color[c];
            } else {
                out = img[i];
            }
            ''',
            'solid_color_mask'
        )
    
    def apply_masks_batch(self, image_batch: np.ndarray, detection_batch: List) -> np.ndarray:
        """Apply masks to entire batch efficiently"""
        if len(image_batch) == 0:
            return image_batch
        
        # Convert to GPU if available and beneficial
        if CUPY_AVAILABLE and len(image_batch) > 4:
            return self._apply_masks_gpu_batch(image_batch, detection_batch)
        else:
            return self._apply_masks_cpu_batch(image_batch, detection_batch)
    
    def _apply_masks_gpu_batch(self, image_batch: np.ndarray, detection_batch: List) -> np.ndarray:
        """GPU-accelerated batch masking"""
        gpu_batch = cp.asarray(image_batch)
        
        for i, (boxes, scores, class_ids) in enumerate(detection_batch):
            if not boxes or i >= len(gpu_batch):
                continue
            
            if self.mask_type == MaskType.SOLID_COLOR:
                # Use GPU kernel for solid color masking
                for box in boxes:
                    x, y, w, h = box
                    x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)
                    if w > 0 and h > 0:
                        gpu_batch[i, y:y+h, x:x+w, :] = cp.array(self.mask_color, dtype=cp.uint8)
        
        return cp.asnumpy(gpu_batch)
    
    def _apply_masks_cpu_batch(self, image_batch: np.ndarray, detection_batch: List) -> np.ndarray:
        """Optimized CPU batch masking"""
        result_batch = image_batch.copy()
        
        for i, (boxes, scores, class_ids) in enumerate(detection_batch):
            if not boxes or i >= len(result_batch):
                continue
            
            image = result_batch[i]
            for box in boxes:
                x, y, w, h = box
                x = max(0, min(x, image.shape[1]))
                y = max(0, min(y, image.shape[0]))
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    continue
                
                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                
                if self.mask_type == MaskType.SOLID_COLOR:
                    roi[:] = self.mask_color
                elif self.mask_type == MaskType.BLUR:
                    kernel_size = min(15, max(3, min(w, h) // 5))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    image[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                elif self.mask_type == MaskType.PIXELATE:
                    pixel_size = max(2, min(w, h) // 10)
                    small_h = max(1, h // pixel_size)
                    small_w = max(1, w // pixel_size)
                    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                    image[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return result_batch

class UltraOptimizedFeatureExtractor:
    """Main feature extraction pipeline with all optimizations"""
    
    def __init__(self, num_features: int, batch_size: int, use_yolo: bool = False,
                 yolo_model_path: str = None, yolo_conf_threshold: float = 0.5,
                 mask_type: str = "solid_color", mask_color: Tuple[int, int, int] = (0, 0, 0)):
        
        self.num_features = num_features
        self.batch_size = batch_size
        self.use_yolo = use_yolo
        
        # Initialize GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Setup memory monitoring
        self.memory_threshold = 0.9  # Use up to 90% of GPU memory
        
        # Initialize components
        self._initialize_models(yolo_model_path, yolo_conf_threshold)
        self._initialize_processors(mask_type, mask_color)
        
        # Performance tracking
        self.processing_stats = defaultdict(list)
    
    def _initialize_models(self, yolo_model_path: str, yolo_conf_threshold: float):
        """Initialize XFeat and YOLO models"""
        print("Loading XFeat model...")
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat',
                                   pretrained=True, top_k=self.num_features,
                                   trust_repo='check').cuda()
        self.xfeat.eval()
        
        # Enable optimizations
        if hasattr(self.xfeat, 'half'):
            self.xfeat = self.xfeat.half()  # Use FP16 for speed
        
        # Compile model if supported
        if hasattr(torch, 'compile'):
            try:
                self.xfeat = torch.compile(self.xfeat, mode='max-autotune')
            except Exception as e:
                print(f"Model compilation failed: {e}")
        
        # Initialize YOLO if needed
        self.yolo_processor = None
        if self.use_yolo and yolo_model_path:
            self.yolo_processor = OptimizedYOLOProcessor(
                yolo_model_path, yolo_conf_threshold, max_batch_size=self.batch_size//2
            )
    
    def _initialize_processors(self, mask_type: str, mask_color: Tuple[int, int, int]):
        """Initialize processing components"""
        self.cubemap_converter = CubemapBatchConverter(
            max_batch_size=self.batch_size, memory_pool_size=6.0
        )
        
        if self.use_yolo:
            self.mask_processor = StreamlinedMaskingProcessor(
                MaskType(mask_type), mask_color
            )
    
    async def load_images_async(self, image_paths: List[str]) -> Dict[Tuple, List]:
        """Asynchronous image loading with size grouping"""
        async def load_single_image(path):
            try:
                # Use OpenCV for fast loading
                loop = asyncio.get_event_loop()
                img = await loop.run_in_executor(None, cv2.imread, path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return path, img_rgb
            except Exception as e:
                print(f"Error loading {path}: {e}")
            return path, None
        
        # Load images concurrently
        tasks = [load_single_image(path) for path in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Group by size for batch processing
        size_groups = defaultdict(list)
        for result in results:
            if isinstance(result, Exception):
                continue
            path, img = result
            if img is not None:
                size_groups[img.shape].append((path, img))
        
        return size_groups
    
    def _extract_features_batch(self, dicemap_batch: np.ndarray) -> List[Dict]:
        """Optimized batch feature extraction"""
        if len(dicemap_batch) == 0:
            return []
        
        # Convert to tensor with optimal memory layout
        tensor_batch = torch.from_numpy(dicemap_batch).permute(0, 3, 1, 2)
        tensor_batch = tensor_batch.float().cuda(non_blocking=True) / 255.0
        
        # Use automatic mixed precision for speed
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                try:
                    # Batch inference
                    batch_outputs = self.xfeat.detectAndCompute(tensor_batch, top_k=self.num_features * 6)
                    
                    # Process outputs
                    results = []
                    for i, output in enumerate(batch_outputs):
                        keypoints = output.get('keypoints', torch.zeros((0, 2))).cpu().numpy()
                        descriptors = output.get('descriptors', torch.zeros((0, 64))).cpu().numpy()
                        scores = output.get('scores', torch.zeros((len(keypoints),))).cpu().numpy()
                        
                        results.append({
                            'keypoints': keypoints,
                            'descriptors': descriptors,
                            'scores': scores
                        })
                    
                    return results
                    
                except Exception as e:
                    print(f"XFeat batch processing error: {e}")
                    # Return empty results for failed batch
                    return [{'keypoints': np.zeros((0, 2)), 'descriptors': np.zeros((0, 64)), 'scores': np.zeros((0,))} 
                           for _ in range(len(dicemap_batch))]
    
    def _convert_coordinates_batch(self, dicemap_keypoints_batch: List[np.ndarray], 
                                 dicemap_shapes: List[Tuple], eq_shapes: List[Tuple]) -> List[np.ndarray]:
        """Batch coordinate conversion with vectorized operations"""
        results = []
        
        for keypoints, dicemap_shape, eq_shape in zip(dicemap_keypoints_batch, dicemap_shapes, eq_shapes):
            if len(keypoints) == 0:
                results.append(np.zeros((0, 2)))
                continue
            
            eq_height, eq_width = eq_shape
            face_size = dicemap_shape[0] // 3
            
            # Vectorized coordinate conversion
            eq_coords = self._dicemap_to_equirect_vectorized(keypoints, face_size, eq_width, eq_height)
            results.append(eq_coords)
        
        return results
    
    def _dicemap_to_equirect_vectorized(self, keypoints: np.ndarray, face_size: int,
                                    eq_width: int, eq_height: int) -> np.ndarray:
        """Vectorized dicemap to equirectangular conversion using cubemap_to_equirectangular_uv"""
        if len(keypoints) == 0:
            return np.zeros((0, 2))

        # Face layout in dicemap (3 rows × 4 cols):
        #     [   U   ]
        # [ L ][ F ][ R ][ B ]
        # (D is omitted in dicemap layout)
        face_regions = [
            (0, face_size, face_size, 2*face_size, 'U'),   # Up
            (face_size, 2*face_size, 0, face_size, 'L'),   # Left
            (face_size, 2*face_size, face_size, 2*face_size, 'F'),  # Front
            (face_size, 2*face_size, 2*face_size, 3*face_size, 'R'),  # Right
            (face_size, 2*face_size, 3*face_size, 4*face_size, 'B'),  # Back
        ]

        x_dice, y_dice = keypoints[:, 0], keypoints[:, 1]
        eq_coords = np.zeros((len(keypoints), 2))

        for y_min, y_max, x_min, x_max, face_name in face_regions:
            mask = ((x_dice >= x_min) & (x_dice < x_max) &
                    (y_dice >= y_min) & (y_dice < y_max))

            if not np.any(mask):
                continue

            x_face = x_dice[mask] - x_min
            y_face = y_dice[mask] - y_min

            # Use the provided cubemap_to_equirectangular_uv function for each point
            for idx, (xf, yf) in enumerate(zip(x_face, y_face)):
                uv = cubemap_to_equirectangular_uv(face_name, int(xf), int(yf), face_size)
                mask_indices = np.where(mask)[0]
                eq_coords[mask_indices[idx], 0] = uv[0] * eq_width
                eq_coords[mask_indices[idx], 1] = uv[1] * eq_height

        return eq_coords

    async def process_image_group(self, group_data: List[Tuple[str, np.ndarray]]) -> List[Dict]:
        """Process a group of same-size images"""
        if not group_data:
            return []
        
        paths, images = zip(*group_data)
        results = []
        
        # Process in optimally-sized batches
        for i in range(0, len(images), self.batch_size):
            batch_end = min(i + self.batch_size, len(images))
            batch_images = list(images[i:batch_end])
            batch_paths = paths[i:batch_end]
            
            start_time = time.time()
            
            # Step 1: Cubemap conversion
            cubemap_batch = self.cubemap_converter.convert_batch(batch_images)
            self.processing_stats['cubemap_time'].append(time.time() - start_time)
            
            # Step 2: Dicemap creation
            start_time = time.time()
            dicemap_batch = self.cubemap_converter.create_dicemaps_batch(cubemap_batch)
            self.processing_stats['dicemap_time'].append(time.time() - start_time)
            
            if len(dicemap_batch) == 0:
                continue
            
            # Convert to numpy for further processing
            dicemap_batch_np = cp.asnumpy(dicemap_batch) if isinstance(dicemap_batch, cp.ndarray) else dicemap_batch
            
            # Step 3: YOLO detection and masking
            detection_results = []
            if self.use_yolo and self.yolo_processor:
                start_time = time.time()
                detection_results = self.yolo_processor.detect_batch(list(dicemap_batch_np))
                dicemap_batch_np = self.mask_processor.apply_masks_batch(dicemap_batch_np, detection_results)
                self.processing_stats['yolo_time'].append(time.time() - start_time)
            else:
                detection_results = [([], [], [])] * len(dicemap_batch_np)
            
            # Step 4: Feature extraction
            start_time = time.time()
            feature_results = self._extract_features_batch(dicemap_batch_np)
            self.processing_stats['xfeat_time'].append(time.time() - start_time)
            
            # Step 5: Coordinate conversion
            start_time = time.time()
            dicemap_keypoints = [r['keypoints'] for r in feature_results]
            dicemap_shapes = [img.shape for img in dicemap_batch_np]
            eq_shapes = [img.shape[:2] for img in batch_images]
            
            eq_keypoints_batch = self._convert_coordinates_batch(dicemap_keypoints, dicemap_shapes, eq_shapes)
            self.processing_stats['coord_time'].append(time.time() - start_time)
            
            # Combine results
            for j, (path, img) in enumerate(zip(batch_paths, batch_images)):
                if j >= len(feature_results):
                    continue
                
                features = feature_results[j]
                detection_stats = {}
                if j < len(detection_results):
                    boxes, scores, class_ids = detection_results[j]
                    detection_stats = {
                        'num_detections': len(boxes),
                        'boxes': boxes,
                        'scores': scores,
                        'class_ids': class_ids
                    }
                
                results.append({
                    'path': path,
                    'image_size': (img.shape[1], img.shape[0]),
                    'dicemap': dicemap_batch_np[j] if j < len(dicemap_batch_np) else None,
                    'dicemap_keypoints': features['keypoints'],
                    'dicemap_descriptors': features['descriptors'],
                    'dicemap_scores': features['scores'],
                    'eq_keypoints': eq_keypoints_batch[j] if j < len(eq_keypoints_batch) else np.zeros((0, 2)),
                    'detection_stats': detection_stats
                })
        
        return results
    
    def print_performance_stats(self):
        """Print performance statistics"""
        if not self.processing_stats:
            return
        
        print("\n=== Performance Statistics ===")
        for operation, times in self.processing_stats.items():
            if times:
                avg_time = np.mean(times)
                print(f"{operation}: {avg_time:.3f}s avg ({len(times)} batches)")
        
        # GPU memory info
        if torch.cuda.is_available():
            memory_info = torch.cuda.memory_summary()
            print(f"\nGPU Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() // 1024**2}MB")
            print(f"Cached: {torch.cuda.memory_reserved() // 1024**2}MB")
    
    def cleanup(self):
        """Clean up resources"""
        self.cubemap_converter.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

async def extract_features_ultra_optimized(image_dir: str, output_file: str, export_dir: str,
                                         num_features: int = 3072, batch_size: int = 32,
                                         use_yolo: bool = False, yolo_model_path: str = None,
                                         yolo_conf_threshold: float = 0.5, mask_type: str = "solid_color",
                                         mask_color: Tuple[int, int, int] = (0, 0, 0)):
    """
    Ultra-optimized feature extraction pipeline with full async processing
    """
    
    # Initialize extractor
    extractor = UltraOptimizedFeatureExtractor(
        num_features=num_features,
        batch_size=batch_size,
        use_yolo=use_yolo,
        yolo_model_path=yolo_model_path,
        yolo_conf_threshold=yolo_conf_threshold,
        mask_type=mask_type,
        mask_color=mask_color
    )
    
    # Setup directories
    os.makedirs(export_dir, exist_ok=True)
    dicemap_dir = os.path.join(export_dir, "dicemaps")
    os.makedirs(dicemap_dir, exist_ok=True)
    
    # Find images
    print("Scanning for images...")
    extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    image_paths = []
    
    for file in os.listdir(image_dir):
        if os.path.splitext(file.lower())[1] in extensions:
            image_paths.append(os.path.join(image_dir, file))
    
    image_paths.sort()
    print(f"Found {len(image_paths)} images")
    
    if not image_paths:
        print("No images found!")
        return
    
    # Load images asynchronously and group by size
    print("Loading images asynchronously...")
    start_time = time.time()
    size_groups = await extractor.load_images_async(image_paths)
    print(f"Loading completed in {time.time() - start_time:.2f}s")
    
    total_processed = 0
    
    # Process each size group
    with h5py.File(output_file, 'w') as f:
        for image_shape, group_data in tqdm(size_groups.items(), desc="Processing size groups"):
            print(f"\nProcessing {len(group_data)} images of size {image_shape}")
            
            try:
                # Process group asynchronously
                group_results = await extractor.process_image_group(group_data)
                
                # Save results efficiently
                for result in group_results:
                    if result is None or result.get('dicemap') is None:
                        continue
                    
                    img_name = os.path.basename(result['path'])
                    
                    # Remove existing group if present
                    if img_name in f:
                        del f[img_name]
                    
                    # Create group and save data
                    try:
                        img_grp = f.create_group(img_name)
                        
                        # Save dicemap image
                        dicemap_path = os.path.join(dicemap_dir, f"dicemap_{img_name}")
                        dicemap_bgr = cv2.cvtColor(result['dicemap'], cv2.COLOR_RGB2BGR)
                        cv2.imwrite(dicemap_path, dicemap_bgr)
                        
                        # Save feature data with compression
                        datasets = {
                            "dicemap_keypoints": result['dicemap_keypoints'],
                            "dicemap_descriptors": result['dicemap_descriptors'], 
                            "dicemap_scores": result['dicemap_scores'],
                            "keypoints": result['eq_keypoints'],
                            "descriptors": result['dicemap_descriptors'],
                            "scores": result['dicemap_scores'],
                            "image_size": result['image_size'],
                            "dicemap_size": (result['dicemap'].shape[1], result['dicemap'].shape[0])
                        }
                        
                        for name, data in datasets.items():
                            if data is not None and len(data) > 0:
                                img_grp.create_dataset(name, data=data, compression='gzip', compression_opts=9)
                        
                        # Save detection statistics
                        stats = result.get('detection_stats', {})
                        if stats:
                            img_grp.create_dataset("num_detections", data=stats.get('num_detections', 0))
                            if stats.get('boxes'):
                                img_grp.create_dataset("detection_boxes", data=np.array(stats['boxes']), compression='gzip')
                                img_grp.create_dataset("detection_scores", data=np.array(stats['scores']), compression='gzip')
                                img_grp.create_dataset("detection_class_ids", data=np.array(stats['class_ids']), compression='gzip')
                        
                        total_processed += 1
                        
                    except Exception as e:
                        print(f"Error saving {img_name}: {e}")
                        if img_name in f:
                            del f[img_name]
            
            except Exception as e:
                print(f"Error processing group {image_shape}: {e}")
                continue
    
    # Print results and cleanup
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {total_processed} images")
    print(f"Output file: {output_file}")
    print(f"Dicemaps saved to: {dicemap_dir}")
    
    extractor.print_performance_stats()
    extractor.cleanup()

# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultra-optimized panoramic feature extraction")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing images")
    parser.add_argument('--output_file', type=str, default="ultra_optimized_features.h5", help="Output HDF5 file")
    parser.add_argument('--export_dir', type=str, default="./export", help="Export directory")
    parser.add_argument('--num_features', type=int, default=3072, help="Number of features to extract")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for processing")
    
    # YOLO options
    parser.add_argument('--use_yolo', action='store_true', help="Enable YOLO detection")
    parser.add_argument('--yolo_model', type=str, help="Path to YOLO ONNX model")
    parser.add_argument('--yolo_conf_threshold', type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument('--mask_type', type=str, default="solid_color", 
                       choices=["solid_color", "blur", "pixelate"], help="Masking type")
    parser.add_argument('--mask_color', type=int, nargs=3, default=[0, 0, 0], help="Mask color (R G B)")
    
    args = parser.parse_args()
    
    # Validation
    if args.use_yolo and not args.yolo_model:
        parser.error("--yolo_model required when --use_yolo is specified")
    
    if not CUPY_AVAILABLE:
        print("ERROR: CuPy is required for optimal performance")
        print("Install with: pip install cupy-cuda12x")
        exit(1)
    
    # Run the async extraction
    print("Starting ultra-optimized feature extraction...")
    print(f"Configuration:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Features: {args.num_features}")
    print(f"  - YOLO enabled: {args.use_yolo}")
    print(f"  - GPU memory available: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    
    asyncio.run(extract_features_ultra_optimized(
        image_dir=args.image_dir,
        output_file=args.output_file,
        export_dir=args.export_dir,
        num_features=args.num_features,
        batch_size=args.batch_size,
        use_yolo=args.use_yolo,
        yolo_model_path=args.yolo_model,
        yolo_conf_threshold=args.yolo_conf_threshold,
        mask_type=args.mask_type,
        mask_color=tuple(args.mask_color)
    ))