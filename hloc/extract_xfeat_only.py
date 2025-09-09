"""
Feature Extractor Module - Bug Fixes and Improvements
Key issues identified and fixed:
1. XFeat import path issues
2. Batch processing error handling
3. Coordinate conversion edge cases
4. Memory management improvements
5. Error handling in async operations
"""
import numpy as np
import cv2
import h5py
import torch
import argparse
import time
import asyncio
import os
import gc
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import sys
from .shared_utils import (
    cubemap_to_equirectangular_uv_with_spherical, 
    load_images_async, find_images
)

class XFeatProcessor:
    """Optimized XFeat feature extraction processor"""
    
    def __init__(self, num_features: int = 3072, batch_size: int = 32, use_half_precision: bool = True):
        self.num_features = num_features
        self.batch_size = batch_size
        self.use_half_precision = use_half_precision
        self.xfeat = None  # Initialize as None
        
        # Initialize XFeat model
        self._initialize_model()
        
        # Performance tracking
        self.processing_stats = {'feature_extraction': [], 'coordinate_conversion': []}
    
    def _initialize_model(self):
        """Initialize XFeat model with optimizations and proper error handling"""
        print("Loading XFeat model...")
        
        # Try multiple possible XFeat paths
        possible_paths = [
            Path(__file__).parent / "../../XFeat",
            Path(__file__).parent / "../../Xfeat",  # Note: case difference
            Path(__file__).parent / "../XFeat",
            Path(__file__).parent / "../Xfeat",
            "XFeat",
            "Xfeat"
        ]
        
        xfeat_loaded = False
        for xfeat_path in possible_paths:
            try:
                if isinstance(xfeat_path, Path):
                    sys.path.insert(0, str(xfeat_path.resolve()))
                else:
                    sys.path.insert(0, xfeat_path)
                
                from modules.xfeat import XFeat
                self.xfeat = XFeat()
                self.xfeat.eval()
                xfeat_loaded = True
                print(f"Successfully loaded XFeat from: {xfeat_path}")
                break
                
            except (ImportError, ModuleNotFoundError) as e:
                print(f"Failed to load XFeat from {xfeat_path}: {e}")
                continue
        
        if not xfeat_loaded:
            raise RuntimeError(
                "Could not import XFeat from any of the attempted paths. "
                "Please ensure XFeat is installed and the path is correct. "
                f"Tried paths: {[str(p) for p in possible_paths]}"
            )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.xfeat = self.xfeat.cuda()
        
        # Enable optimizations
        if self.use_half_precision and hasattr(self.xfeat, 'half') and torch.cuda.is_available():
            try:
                self.xfeat = self.xfeat.half()  # Use FP16 for speed
                print("Using half precision (FP16) for faster inference")
            except Exception as e:
                print(f"Half precision failed, using float32: {e}")
                self.use_half_precision = False
        
        # Compile model if supported (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.xfeat = torch.compile(self.xfeat, mode='max-autotune')
                print("Model compiled with torch.compile for faster inference")
            except Exception as e:
                print(f"Model compilation failed (continuing without): {e}")
        
        # Setup CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"CUDA optimizations enabled. GPU: {torch.cuda.get_device_name()}")
    
    def extract_features_batch(self, image_batch: np.ndarray) -> List[Dict]:
        """Extract features from batch of images and pad to num_features"""
        if len(image_batch) == 0:
            return []
        
        if self.xfeat is None:
            raise RuntimeError("XFeat model not initialized")
        
        try:
            # Convert to tensor with optimal memory layout
            tensor_batch = torch.from_numpy(image_batch).permute(0, 3, 1, 2)
            tensor_batch = tensor_batch.float()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                tensor_batch = tensor_batch.cuda(non_blocking=True)
            
            tensor_batch = tensor_batch / 255.0
            
            if self.use_half_precision and torch.cuda.is_available():
                tensor_batch = tensor_batch.half()
            
            # Use automatic mixed precision for speed
            with torch.cuda.amp.autocast(enabled=self.use_half_precision and torch.cuda.is_available()):
                with torch.no_grad():
                    # Batch inference with error handling
                    batch_outputs = self.xfeat.detectAndCompute(tensor_batch, top_k=self.num_features)
                    
                    # Process outputs
                    results = []
                    for i, output in enumerate(batch_outputs):
                        # Handle different output formats
                        if isinstance(output, dict):
                            keypoints = output.get('keypoints', torch.zeros((0, 2))).cpu().numpy()
                            descriptors = output.get('descriptors', torch.zeros((0, 64))).cpu().numpy()
                            scores = output.get('scores', torch.zeros((0,))).cpu().numpy()
                        else:
                            # If output is a tuple or other format
                            print(f"Warning: Unexpected output format for batch item {i}: {type(output)}")
                            keypoints = np.zeros((0, 2))
                            descriptors = np.zeros((0, 64))
                            scores = np.zeros((0,))

                        # --- PADDING LOGIC WITH VALIDATION ---
                        num_detected = len(keypoints)
                        if num_detected > self.num_features:
                            # Truncate if we got more features than requested
                            keypoints = keypoints[:self.num_features]
                            descriptors = descriptors[:self.num_features]
                            scores = scores[:self.num_features]
                            num_detected = self.num_features
                        
                        num_to_pad = self.num_features - num_detected
                        
                        if num_to_pad > 0:
                            # Determine descriptor dimension safely
                            descriptor_dim = descriptors.shape[1] if num_detected > 0 else 64
                            
                            # Pad keypoints with (0, 0)
                            keypoints_padding = np.zeros((num_to_pad, 2), dtype=keypoints.dtype)
                            keypoints = np.concatenate((keypoints, keypoints_padding), axis=0)
                            
                            # Pad descriptors with zeros
                            descriptors_padding = np.zeros((num_to_pad, descriptor_dim), dtype=descriptors.dtype)
                            descriptors = np.concatenate((descriptors, descriptors_padding), axis=0)

                            # Pad scores with zeros
                            scores_padding = np.zeros(num_to_pad, dtype=scores.dtype)
                            scores = np.concatenate((scores, scores_padding), axis=0)
                        
                        # Validate final shapes
                        assert keypoints.shape == (self.num_features, 2), f"Keypoints shape mismatch: {keypoints.shape}"
                        assert descriptors.shape[0] == self.num_features, f"Descriptors shape mismatch: {descriptors.shape}"
                        assert scores.shape == (self.num_features,), f"Scores shape mismatch: {scores.shape}"

                        results.append({
                            'keypoints': keypoints,
                            'descriptors': descriptors,
                            'scores': scores
                        })
                    
                    return results
                    
        except Exception as e:
            print(f"XFeat batch processing error: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Return empty padded results for failed batch
            empty_kp = np.zeros((self.num_features, 2))
            empty_desc = np.zeros((self.num_features, 64))
            empty_scores = np.zeros((self.num_features,))
            return [{'keypoints': empty_kp, 'descriptors': empty_desc, 'scores': empty_scores} 
                    for _ in range(len(image_batch))]

class CoordinateConverter:
    """Handles coordinate conversion between dicemap, equirectangular, and spherical systems"""
    
    def convert_coordinates_batch(self, dicemap_keypoints_batch: List[np.ndarray], 
                                  dicemap_shapes: List[Tuple], eq_shapes: List[Tuple]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Batch coordinate conversion - returns both equirectangular and spherical coords"""
        if len(dicemap_keypoints_batch) != len(dicemap_shapes) or len(dicemap_shapes) != len(eq_shapes):
            raise ValueError(f"Batch size mismatch: keypoints={len(dicemap_keypoints_batch)}, "
                           f"dicemap_shapes={len(dicemap_shapes)}, eq_shapes={len(eq_shapes)}")
        
        eq_results = []
        spherical_results = []
        
        for keypoints, dicemap_shape, eq_shape in zip(dicemap_keypoints_batch, dicemap_shapes, eq_shapes):
            try:
                if len(keypoints) == 0:
                    eq_results.append(np.zeros((0, 2)))
                    spherical_results.append(np.zeros((0, 2)))
                    continue
                
                eq_height, eq_width = eq_shape
                
                # Validate dicemap shape
                if len(dicemap_shape) < 2:
                    print(f"Warning: Invalid dicemap shape {dicemap_shape}, using default")
                    face_size = 256  # Default fallback
                else:
                    face_size = dicemap_shape[0] // 3
                    if face_size <= 0:
                        print(f"Warning: Invalid face size calculated from shape {dicemap_shape}")
                        face_size = 256
                
                # Vectorized coordinate conversion
                eq_coords, spherical_coords = self._dicemap_to_equirect_and_spherical_vectorized(
                    keypoints, face_size, eq_width, eq_height)
                
                eq_results.append(eq_coords)
                spherical_results.append(spherical_coords)
                
            except Exception as e:
                print(f"Error in coordinate conversion: {e}")
                # Return empty arrays for this item
                eq_results.append(np.zeros((len(keypoints), 2)))
                spherical_results.append(np.zeros((len(keypoints), 2)))
        
        return eq_results, spherical_results
    
    def _dicemap_to_equirect_and_spherical_vectorized(self, keypoints: np.ndarray, face_size: int,
                                                    eq_width: int, eq_height: int) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized dicemap to both equirectangular and spherical conversion"""
        if len(keypoints) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))

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
        spherical_coords = np.zeros((len(keypoints), 2))  # [phi, theta] format

        for y_min, y_max, x_min, x_max, face_name in face_regions:
            mask = ((x_dice >= x_min) & (x_dice < x_max) &
                    (y_dice >= y_min) & (y_dice < y_max))

            if not np.any(mask):
                continue

            x_face = x_dice[mask] - x_min
            y_face = y_dice[mask] - y_min
            
            # Ensure coordinates are within valid range
            x_face = np.clip(x_face, 0, face_size - 1)
            y_face = np.clip(y_face, 0, face_size - 1)

            # Use the enhanced conversion function for each point
            mask_indices = np.where(mask)[0]
            for idx, (xf, yf) in enumerate(zip(x_face, y_face)):
                try:
                    result = cubemap_to_equirectangular_uv_with_spherical(
                        face_name, int(xf), int(yf), face_size)
                    
                    # Store equirectangular coordinates
                    eq_coords[mask_indices[idx], 0] = result['uv'][0] * eq_width
                    eq_coords[mask_indices[idx], 1] = result['uv'][1] * eq_height
                    
                    # Store spherical coordinates [phi, theta] as specified
                    spherical_coords[mask_indices[idx], 0] = result['spherical'][0]  # phi (azimuthal)
                    spherical_coords[mask_indices[idx], 1] = result['spherical'][1]  # theta (polar)
                    
                except Exception as e:
                    print(f"Warning: Coordinate conversion failed for point ({xf}, {yf}) on face {face_name}: {e}")
                    # Keep zero values for failed conversions

        return eq_coords, spherical_coords

class FeatureExtractionPipeline:
    """Main pipeline for feature extraction from dicemap images"""
    
    def __init__(self, num_features: int = 3072, batch_size: int = 32, use_half_precision: bool = True):
        self.num_features = num_features
        self.batch_size = batch_size
        
        # Initialize components with error handling
        try:
            self.xfeat_processor = XFeatProcessor(num_features, batch_size, use_half_precision)
            self.coord_converter = CoordinateConverter()
        except Exception as e:
            print(f"Failed to initialize pipeline components: {e}")
            raise
        
        print(f"FeatureExtractionPipeline initialized:")
        print(f"  - Features per image: {num_features}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Half precision: {use_half_precision}")
    
    async def process_dicemap_directory(self, dicemap_dir: str, output_file: str, 
                                        original_shapes_file: Optional[str] = None):
        """Process a directory of dicemap images and extract features"""
        
        # Find dicemap images
        dicemap_paths = find_images(dicemap_dir)
        print(f"Found {len(dicemap_paths)} dicemap images")
        
        if not dicemap_paths:
            print("No dicemap images found!")
            return
        
        # Load original shapes if provided
        original_shapes = {}
        if original_shapes_file and os.path.exists(original_shapes_file):
            try:
                with h5py.File(original_shapes_file, 'r') as f:
                    for img_name in f.keys():
                        if 'image_size' in f[img_name]:
                            original_shapes[img_name] = tuple(f[img_name]['image_size'][:])
                print(f"Loaded {len(original_shapes)} original image shapes")
            except Exception as e:
                print(f"Warning: Failed to load original shapes file: {e}")
        
        # Load dicemap images
        print("Loading dicemap images...")
        start_time = time.time()
        try:
            size_groups = await load_images_async(dicemap_paths)
        except Exception as e:
            print(f"Error loading images: {e}")
            return
        print(f"Loading completed in {time.time() - start_time:.2f}s")
        
        total_processed = 0
        
        # Process each size group
        try:
            with h5py.File(output_file, 'w') as f:
                for dicemap_shape, group_data in size_groups.items():
                    print(f"\nProcessing {len(group_data)} dicemaps of size {dicemap_shape}")
                    
                    # Process in batches
                    for i in range(0, len(group_data), self.batch_size):
                        batch_end = min(i + self.batch_size, len(group_data))
                        batch_data = group_data[i:batch_end]
                        batch_paths, batch_dicemaps = zip(*batch_data)
                        
                        # Feature extraction
                        start_time = time.time()
                        feature_results = self.xfeat_processor.extract_features_batch(np.array(batch_dicemaps))
                        feature_time = time.time() - start_time
                        self.xfeat_processor.processing_stats['feature_extraction'].append(feature_time)
                        
                        # Coordinate conversion
                        start_time = time.time()
                        dicemap_keypoints = [r['keypoints'] for r in feature_results]
                        dicemap_shapes = [dicemap.shape for dicemap in batch_dicemaps]
                        
                        # Determine original equirectangular shapes
                        eq_shapes = []
                        for path in batch_paths:
                            img_name = os.path.basename(path)
                            # Remove dicemap_ prefix if present
                            if img_name.startswith('dicemap_'):
                                original_name = img_name[8:]  # Remove 'dicemap_' prefix
                            else:
                                original_name = img_name
                            
                            # Look up original shape or estimate from dicemap
                            if original_name in original_shapes:
                                eq_shapes.append(original_shapes[original_name][::-1])  # (w, h) -> (h, w)
                            else:
                                # Estimate equirectangular size from dicemap (face_size * 2)
                                face_size = dicemap_shape[0] // 3
                                eq_shapes.append((face_size * 2, face_size * 4))
                        
                        eq_keypoints_batch, spherical_keypoints_batch = self.coord_converter.convert_coordinates_batch(
                            dicemap_keypoints, dicemap_shapes, eq_shapes)
                        
                        coord_time = time.time() - start_time
                        self.xfeat_processor.processing_stats['coordinate_conversion'].append(coord_time)
                        
                        # Save results
                        for j, (path, dicemap) in enumerate(zip(batch_paths, batch_dicemaps)):
                            if j >= len(feature_results):
                                print(f"Warning: No feature results for image {j}")
                                continue
                            
                            img_name = os.path.basename(path)
                            if img_name.startswith('dicemap_'):
                                original_name = img_name[8:]  # Remove 'dicemap_' prefix
                            else:
                                original_name = img_name
                            name_only = original_name  # Remove file extension
                            
                            # Remove existing group if present
                            if name_only in f:
                                del f[name_only]
                            
                            try:
                                features = feature_results[j]
                                
                                img_grp = f.create_group(name_only)
                                
                                # Save feature data with compression and validation
                                datasets = {
                                    "dicemap_keypoints": features['keypoints'],
                                    "dicemap_descriptors": features['descriptors'], 
                                    "dicemap_scores": features['scores'],
                                    "keypoints": eq_keypoints_batch[j] if j < len(eq_keypoints_batch) else np.zeros((self.num_features, 2)),
                                    "spherical_keypoints": spherical_keypoints_batch[j] if j < len(spherical_keypoints_batch) else np.zeros((self.num_features, 2)),
                                    "descriptors": features['descriptors'],
                                    "scores": features['scores'],
                                    "dicemap_size": (dicemap.shape[1], dicemap.shape[0]),
                                    "estimated_eq_size": eq_shapes[j] if j < len(eq_shapes) else (0, 0)
                                }
                                
                                for name, data in datasets.items():
                                    if data is not None and hasattr(data, 'shape') and len(data.shape) > 0:
                                        img_grp.create_dataset(name, data=data, compression='gzip', compression_opts=9)
                                
                                total_processed += 1
                                
                            except Exception as e:
                                print(f"Error saving {name_only}: {e}")
                                if name_only in f:
                                    del f[name_only]
                        
                        print(f"Batch {i//self.batch_size + 1}: {len(batch_dicemaps)} dicemaps in "
                              f"{feature_time + coord_time:.2f}s (features: {feature_time:.2f}s, coords: {coord_time:.2f}s)")
                        
                        # Memory cleanup between batches
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Print results
        print(f"\n=== Feature Extraction Complete ===")
        print(f"Successfully processed: {total_processed} dicemaps")
        print(f"Output file: {output_file}")
        print(f"Spherical coordinates (phi, theta) saved as 'spherical_keypoints' dataset")
        
        self._print_performance_stats()
        self._cleanup()
    
    async def process_single_images(self, image_paths: List[str], output_file: str):
        """Process individual images (not dicemaps) - will extract features directly"""
        
        print("Loading images...")
        start_time = time.time()
        try:
            size_groups = await load_images_async(image_paths)
        except Exception as e:
            print(f"Error loading images: {e}")
            return
        print(f"Loading completed in {time.time() - start_time:.2f}s")
        
        total_processed = 0
        
        # Process each size group  
        try:
            with h5py.File(output_file, 'w') as f:
                for image_shape, group_data in size_groups.items():
                    print(f"\nProcessing {len(group_data)} images of size {image_shape}")
                    
                    # Process in batches
                    for i in range(0, len(group_data), self.batch_size):
                        batch_end = min(i + self.batch_size, len(group_data))
                        batch_data = group_data[i:batch_end]
                        batch_paths, batch_images = zip(*batch_data)
                        
                        # Feature extraction
                        start_time = time.time()
                        feature_results = self.xfeat_processor.extract_features_batch(np.array(batch_images))
                        feature_time = time.time() - start_time
                        
                        # Save results (no coordinate conversion for direct images)
                        for j, (path, image) in enumerate(zip(batch_paths, batch_images)):
                            if j >= len(feature_results):
                                continue
                            
                            img_name = os.path.basename(path)
                            name_only = os.path.splitext(img_name)[0]
                            
                            # Remove existing group if present
                            if name_only in f:
                                del f[name_only]
                            
                            try:
                                features = feature_results[j]
                                
                                img_grp = f.create_group(name_only)
                                
                                # Save feature data
                                datasets = {
                                    "keypoints": features['keypoints'],
                                    "descriptors": features['descriptors'], 
                                    "scores": features['scores'],
                                    "image_size": (image.shape[1], image.shape[0])
                                }
                                
                                for name, data in datasets.items():
                                    if data is not None and hasattr(data, 'shape') and len(data.shape) > 0:
                                        img_grp.create_dataset(name, data=data, compression='gzip', compression_opts=9)
                                
                                total_processed += 1
                                
                            except Exception as e:
                                print(f"Error saving {name_only}: {e}")
                                if name_only in f:
                                    del f[name_only]
                        
                        print(f"Batch {i//self.batch_size + 1}: {len(batch_images)} images in {feature_time:.2f}s")
                        
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Print results
        print(f"\n=== Feature Extraction Complete ===")
        print(f"Successfully processed: {total_processed} images")
        print(f"Output file: {output_file}")
        
        self._print_performance_stats()
        self._cleanup()
    
    def _print_performance_stats(self):
        """Print performance statistics"""
        stats = self.xfeat_processor.processing_stats
        
        print("\n=== Performance Statistics ===")
        for operation, times in stats.items():
            if times:
                avg_time = np.mean(times)
                print(f"{operation}: {avg_time:.3f}s avg ({len(times)} batches)")
        
        # GPU memory info
        if torch.cuda.is_available():
            print(f"\nGPU Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() // 1024**2}MB")
            print(f"Cached: {torch.cuda.memory_reserved() // 1024**2}MB")
    
    def _cleanup(self):
        """Clean up resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

async def main():
    parser = argparse.ArgumentParser(description="Feature extraction from dicemap or regular images")
    parser.add_argument('--dicemap_dir', type=str, help="Directory containing dicemap images")
    parser.add_argument('--image_dir', type=str, help="Directory containing regular images")
    parser.add_argument('--image_paths', type=str, nargs='+', help="Specific image paths")
    parser.add_argument('--output_file', type=str, default="features.h5", help="Output HDF5 file")
    parser.add_argument('--original_shapes_file', type=str, help="HDF5 file with original image shapes")
    parser.add_argument('--num_features', type=int, default=3072, help="Number of features to extract")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for processing")
    parser.add_argument('--no_half_precision', action='store_true', help="Disable half precision (FP16)")
    
    args = parser.parse_args()
    
    # Validate arguments
    input_sources = [args.dicemap_dir, args.image_dir, args.image_paths]
    if sum(bool(x) for x in input_sources) != 1:
        parser.error("Specify exactly one of: --dicemap_dir, --image_dir, or --image_paths")
    
    # Initialize pipeline
    try:
        pipeline = FeatureExtractionPipeline(
            num_features=args.num_features,
            batch_size=args.batch_size,
            use_half_precision=not args.no_half_precision
        )
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return
    
    print("Starting feature extraction...")
    print(f"Configuration:")
    print(f"  - Features per image: {args.num_features}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Half precision: {not args.no_half_precision}")
    
    # Process based on input type
    try:
        if args.dicemap_dir:
            print(f"  - Processing dicemaps from: {args.dicemap_dir}")
            await pipeline.process_dicemap_directory(
                dicemap_dir=args.dicemap_dir,
                output_file=args.output_file,
                original_shapes_file=args.original_shapes_file
            )
        
        elif args.image_dir:
            image_paths = find_images(args.image_dir)
            print(f"  - Processing {len(image_paths)} images from: {args.image_dir}")
            await pipeline.process_single_images(
                image_paths=image_paths,
                output_file=args.output_file
            )
        
        elif args.image_paths:
            print(f"  - Processing {len(args.image_paths)} specified images")
            await pipeline.process_single_images(
                image_paths=args.image_paths,
                output_file=args.output_file
            )
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())