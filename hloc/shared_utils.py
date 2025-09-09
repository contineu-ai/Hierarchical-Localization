"""
Shared utilities for panoramic image processing pipeline
"""
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
import sys
from pathlib import Path

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

class MaskType(Enum):
    SOLID_COLOR = "solid_color"
    BLUR = "blur" 
    PIXELATE = "pixelate"

class MemoryPool:
    """GPU memory pool manager for efficient allocation reuse"""
    
    def __init__(self, initial_size_gb=2.0):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy required for MemoryPool")
        
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

def cubemap_to_equirectangular_uv_with_spherical(face: str, x: int, y: int, cubemap_size: int):
    """
    Convert cubemap face coordinates to equirectangular UV coordinates AND spherical coordinates.
    This matches the face orientation used in the GPU_Convert class.
    
    Args:
        face (str): Face identifier ("F", "R", "B", "L", "U", "D")
        x (int): X coordinate on the face (0 to cubemap_size-1)
        y (int): Y coordinate on the face (0 to cubemap_size-1)
        cubemap_size (int): Size of the cubemap face
        
    Returns:
        dict: {'uv': [u, v], 'spherical': [theta, phi]} coordinates
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
    
    # longitude (theta) = arctan2(x, z)
    # latitude (phi) = arcsin(y)
    theta = np.arctan2(x_3d, z_3d)  # longitude
    phi = np.arcsin(y_3d)           # latitude
    
    # Map to equirectangular UV coordinates [0, 1] (matching GPU_Convert's uv2coor)
    u = (theta / (2 * np.pi) + 0.5)  # longitude: [-π, π] → [0, 1]
    v = (-phi / np.pi + 0.5)          # latitude: [-π/2, π/2] → [0, 1]
    
    # Ensure values are in valid range
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    
    return {
        'uv': [u, v],
        'spherical': [theta, phi]  # theta (longitude), phi (latitude)
    }

async def load_images_async(image_paths: List[str]) -> Dict[Tuple, List]:
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

def find_images(image_dir: str) -> List[str]:
    """Find all supported images in directory"""
    extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    image_paths = []
    
    for file in os.listdir(image_dir):
        if os.path.splitext(file.lower())[1] in extensions:
            image_paths.append(os.path.join(image_dir, file))
    
    return sorted(image_paths)