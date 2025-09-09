"""
Dicemap Converter Module
Handles conversion from equirectangular to cubemap to dicemap with GPU acceleration
"""
import numpy as np
import cv2
import argparse
import time
import asyncio
from typing import List, Dict, Tuple
from .shared_utils import (
    CUPY_AVAILABLE, MemoryPool, cubemap_to_equirectangular_uv_with_spherical,
    load_images_async, find_images
)

if CUPY_AVAILABLE:
    import cupy as cp
    import cupyx.scipy.ndimage

class GPU_Convert:
    """GPU-accelerated cubemap converter using CuPy"""
    def __init__(self, image_shape):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy required for GPU_Convert")
        
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

async def convert_to_dicemaps(image_dir: str, output_dir: str, batch_size: int = 32, save_cubemaps: bool = False):
    """
    Convert equirectangular images to dicemaps
    
    Args:
        image_dir: Directory containing equirectangular images
        output_dir: Directory to save dicemaps
        batch_size: Batch size for processing
        save_cubemaps: Whether to save individual cubemap faces as well
    """
    import os
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    dicemap_dir = os.path.join(output_dir, "dicemaps")
    os.makedirs(dicemap_dir, exist_ok=True)
    
    if save_cubemaps:
        cubemap_dir = os.path.join(output_dir, "cubemaps")
        os.makedirs(cubemap_dir, exist_ok=True)
    
    # Initialize converter
    converter = CubemapBatchConverter(max_batch_size=batch_size)
    
    # Find and load images
    print("Finding images...")
    image_paths = find_images(image_dir)
    print(f"Found {len(image_paths)} images")
    
    if not image_paths:
        print("No images found!")
        return
    
    # Load images asynchronously and group by size
    print("Loading images...")
    start_time = time.time()
    size_groups = await load_images_async(image_paths)
    print(f"Loading completed in {time.time() - start_time:.2f}s")
    
    total_processed = 0
    processing_stats = []
    
    # Process each size group
    for image_shape, group_data in size_groups.items():
        print(f"\nProcessing {len(group_data)} images of size {image_shape}")
        
        # Process in batches
        for i in range(0, len(group_data), batch_size):
            batch_end = min(i + batch_size, len(group_data))
            batch_data = group_data[i:batch_end]
            batch_paths, batch_images = zip(*batch_data)
            
            start_time = time.time()
            
            # Convert to cubemaps
            cubemap_batch = converter.convert_batch(list(batch_images))
            cubemap_time = time.time() - start_time
            
            # Convert to dicemaps
            start_time = time.time()
            dicemap_batch = converter.create_dicemaps_batch(cubemap_batch)
            dicemap_time = time.time() - start_time
            
            # Save results
            for j, (path, dicemap) in enumerate(zip(batch_paths, dicemap_batch)):
                if j >= len(dicemap_batch):
                    continue
                
                img_name = os.path.basename(path)
                name_only = os.path.splitext(img_name)[0]
                
                # Save dicemap
                dicemap_path = os.path.join(dicemap_dir, f"{img_name}")
                dicemap_bgr = cv2.cvtColor(dicemap, cv2.COLOR_RGB2BGR)
                cv2.imwrite(dicemap_path, dicemap_bgr)
                
                # Save cubemap faces if requested
                if save_cubemaps and j < len(cubemap_batch):
                    cubemap = cubemap_batch[j]
                    face_size = cubemap.shape[0]
                    face_names = ["F", "R", "B", "L", "U", "D"]
                    
                    for k, face_name in enumerate(face_names):
                        start_idx = k * face_size
                        end_idx = (k + 1) * face_size
                        face_img = cubemap[:, start_idx:end_idx, :]
                        
                        face_path = os.path.join(cubemap_dir, f"{name_only}_{face_name}.jpg")
                        face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(face_path, face_bgr)
                
                total_processed += 1
            
            processing_stats.append({
                'batch_size': len(batch_images),
                'cubemap_time': cubemap_time,
                'dicemap_time': dicemap_time,
                'total_time': cubemap_time + dicemap_time
            })
            
            print(f"Batch {i//batch_size + 1}: {len(batch_images)} images in {cubemap_time + dicemap_time:.2f}s")
    
    # Print summary
    print(f"\n=== Conversion Complete ===")
    print(f"Successfully processed: {total_processed} images")
    print(f"Dicemaps saved to: {dicemap_dir}")
    if save_cubemaps:
        print(f"Cubemaps saved to: {cubemap_dir}")
    
    if processing_stats:
        avg_time = np.mean([s['total_time'] for s in processing_stats])
        avg_cubemap_time = np.mean([s['cubemap_time'] for s in processing_stats])
        avg_dicemap_time = np.mean([s['dicemap_time'] for s in processing_stats])
        print(f"Average batch processing time: {avg_time:.3f}s")
        print(f"  - Cubemap conversion: {avg_cubemap_time:.3f}s")
        print(f"  - Dicemap creation: {avg_dicemap_time:.3f}s")
    
    # Cleanup
    converter.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert equirectangular images to dicemaps")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing equirectangular images")
    parser.add_argument('--output_dir', type=str, default="./dicemap_output", help="Output directory")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for processing")
    parser.add_argument('--save_cubemaps', action='store_true', help="Also save individual cubemap faces")
    
    args = parser.parse_args()
    
    if not CUPY_AVAILABLE:
        print("ERROR: CuPy is required for dicemap conversion")
        print("Install with: pip install cupy-cuda12x")
        exit(1)
    
    print("Starting dicemap conversion...")
    print(f"Configuration:")
    print(f"  - Input directory: {args.image_dir}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Save cubemaps: {args.save_cubemaps}")
    
    asyncio.run(convert_to_dicemaps(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        save_cubemaps=args.save_cubemaps
    ))