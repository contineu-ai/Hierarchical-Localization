"""
Simplified GPU Pipeline - CUDA Only

Handles fixed ONNX batch sizes by:
1. Detecting the model's fixed batch size from input shape
2. Padding last batch with zeros if needed
3. Processing only actual images (ignoring padding)
"""

import numpy as np
import cv2
import h5py
import torch
import argparse
import time
import os
import gc
import sys
from typing import List, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config, PipelineConfig

try:
    import cupy as cp
    import cupyx.scipy.ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("ERROR: CuPy required")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: Ultralytics required")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: ONNXRuntime required")
    sys.exit(1)


class PerformanceMonitor:
    """Simple performance monitoring"""
    def __init__(self):
        self.timings = {}
    
    @contextmanager
    def timer(self, name):
        if name not in self.timings:
            self.timings[name] = []
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        self.timings[name].append(start.elapsed_time(end))
    
    def report(self):
        if not self.timings:
            return
        print("\n=== Performance Report ===")
        for name, times in sorted(self.timings.items()):
            times_arr = np.array(times)
            print(f"{name:30s}: {np.mean(times_arr):7.2f}ms ± {np.std(times_arr):6.2f}ms")


class EquirectToDicemapConverter:
    """Convert equirectangular images to dicemaps using GPU"""
    def __init__(self, image_shape, device='cuda:0'):
        h, w, *_ = image_shape
        self.face_size = h // 2
        self.h, self.w = h, w
        self.device_id = int(device.split(':')[-1])
        
        with cp.cuda.Device(self.device_id):
            # Load or compute coordinate mappings
            cache_dir = Path(".cache")
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / f"coords_{h}x{w}.npy"
            
            if cache_file.exists():
                self.coors_xy = cp.load(str(cache_file))
            else:
                self.coors_xy = self._precompute_coordinates()
                cp.save(str(cache_file), self.coors_xy)
            
            self._setup_dicemap_layout()
    
    def _precompute_coordinates(self):
        """Pre-compute coordinate mappings for cubemap sampling"""
        face_w = self.face_size
        out = cp.zeros((face_w, face_w * 6, 3), dtype=cp.float32)
        rng = cp.linspace(-0.5, 0.5, num=face_w, dtype=cp.float32)
        x_grid, y_grid = cp.meshgrid(rng, rng, indexing='xy')
        
        # Define 6 cube faces
        faces_data = [
            (0, x_grid, -y_grid, 0.5),   # Front
            (1, 0.5, -y_grid, -x_grid),  # Right
            (2, -x_grid, -y_grid, -0.5), # Back
            (3, -0.5, -y_grid, x_grid),  # Left
            (4, x_grid, 0.5, y_grid),    # Top
            (5, x_grid, -0.5, -y_grid),  # Bottom
        ]
        
        for col, x, y, z in faces_data:
            start = col * face_w
            end = (col + 1) * face_w
            out[:, start:end, 0] = x
            out[:, start:end, 1] = y
            out[:, start:end, 2] = z
        
        # Convert to UV coordinates
        uv = self._xyz_to_uv(out)
        return self._uv_to_pixel_coords(uv, self.h, self.w)
    
    def _xyz_to_uv(self, xyz):
        """Convert 3D coordinates to UV"""
        x, y, z = cp.split(xyz, 3, axis=-1)
        norm = cp.maximum(cp.sqrt(x**2 + y**2 + z**2), 1e-9)
        u = cp.arctan2(x, z)
        v = cp.arcsin(y / norm)
        return cp.concatenate([u, v], axis=-1)
    
    def _uv_to_pixel_coords(self, uv, h, w):
        """Convert UV to pixel coordinates"""
        u, v = cp.split(uv, 2, axis=-1)
        coor_x = (u / (2 * cp.pi) + 0.5) * w - 0.5
        coor_y = (-v / cp.pi + 0.5) * h - 0.5
        return cp.concatenate([coor_x, coor_y], axis=-1).astype(cp.float32)
    
    def _setup_dicemap_layout(self):
        """Setup dicemap face positions"""
        fs = self.face_size
        self.dicemap_shape = (fs * 3, fs * 4)
        self.face_slices = {
            'U': (slice(0, fs), slice(fs, 2*fs)),
            'L': (slice(fs, 2*fs), slice(0, fs)),
            'F': (slice(fs, 2*fs), slice(fs, 2*fs)),
            'R': (slice(fs, 2*fs), slice(2*fs, 3*fs)),
            'B': (slice(fs, 2*fs), slice(3*fs, 4*fs)),
        }
    
    def _sample_equirect(self, images, coords):
        """Sample equirectangular image at given coordinates"""
        batch_size, H, W, C = images.shape
        coor_x, coor_y = cp.split(coords, 2, axis=-1)
        
        # Clip and wrap coordinates
        coor_y_clipped = cp.clip(coor_y, 0, H - 1)
        coor_x_wrapped = coor_x % W
        
        coords_flat = cp.stack([coor_y_clipped, coor_x_wrapped], axis=-1)
        coords_flat = coords_flat.reshape(-1, 2).T.astype(cp.float32)
        
        results = []
        for b in range(batch_size):
            channels = []
            for c in range(C):
                sampled = cupyx.scipy.ndimage.map_coordinates(
                    images[b, :, :, c], coords_flat, order=1, mode='nearest'
                )
                channels.append(sampled)
            result = cp.stack(channels, axis=-1).reshape(coords.shape[:-1] + (C,))
            results.append(result)
        
        return cp.stack(results, axis=0)
    
    def convert_batch(self, equirect_batch):
        """Convert batch of equirectangular images to dicemaps"""
        if isinstance(equirect_batch, np.ndarray):
            equirect_batch = cp.asarray(equirect_batch)
        
        batch_size = equirect_batch.shape[0]
        cubemaps = self._sample_equirect(equirect_batch, self.coors_xy)
        
        # Assemble into dicemap layout
        dicemap_h, dicemap_w = self.dicemap_shape
        C = equirect_batch.shape[3]
        dicemaps = cp.zeros((batch_size, dicemap_h, dicemap_w, C), 
                           dtype=equirect_batch.dtype)
        
        fs = self.face_size
        faces = {
            'F': cubemaps[:, :, 0:fs, :],
            'R': cubemaps[:, :, fs:2*fs, :],
            'B': cubemaps[:, :, 2*fs:3*fs, :],
            'L': cubemaps[:, :, 3*fs:4*fs, :],
            'U': cubemaps[:, :, 4*fs:5*fs, :],
        }
        
        for face_name, (row_slice, col_slice) in self.face_slices.items():
            dicemaps[:, row_slice, col_slice, :] = faces[face_name]
        
        return dicemaps


class CoordinateConverter:
    """Convert dicemap coordinates to equirectangular coordinates"""
    def __init__(self, face_size, eq_width, eq_height, device='cuda:0'):
        self.face_size = face_size
        self.eq_width = eq_width
        self.eq_height = eq_height
        self.device_id = int(device.split(':')[-1])
        
        with cp.cuda.Device(self.device_id):
            self._setup_face_regions()
    
    def _setup_face_regions(self):
        """Define face regions in dicemap"""
        fs = self.face_size
        self.face_regions = cp.array([
            [0, fs, fs, 2*fs, 0],      # U
            [fs, 2*fs, 0, fs, 1],      # L
            [fs, 2*fs, fs, 2*fs, 2],   # F
            [fs, 2*fs, 2*fs, 3*fs, 3], # R
            [fs, 2*fs, 3*fs, 4*fs, 4], # B
        ], dtype=cp.int32)
    
    def _face_to_xyz(self, face_ids, x_face, y_face):
        """Convert face coordinates to 3D XYZ"""
        fs = self.face_size
        u = (x_face / fs) - 0.5
        v = (y_face / fs) - 0.5
        
        xyz = cp.zeros((*u.shape, 3), dtype=cp.float32)
        
        # Map each face to XYZ
        for face_id in range(5):
            mask = (face_ids == face_id)
            if not cp.any(mask):
                continue
            
            if face_id == 0:  # Top
                xyz[mask, 0] = u[mask]
                xyz[mask, 1] = 0.5
                xyz[mask, 2] = v[mask]
            elif face_id == 1:  # Left
                xyz[mask, 0] = -0.5
                xyz[mask, 1] = -v[mask]
                xyz[mask, 2] = u[mask]
            elif face_id == 2:  # Front
                xyz[mask, 0] = u[mask]
                xyz[mask, 1] = -v[mask]
                xyz[mask, 2] = 0.5
            elif face_id == 3:  # Right
                xyz[mask, 0] = 0.5
                xyz[mask, 1] = -v[mask]
                xyz[mask, 2] = -u[mask]
            elif face_id == 4:  # Back
                xyz[mask, 0] = -u[mask]
                xyz[mask, 1] = -v[mask]
                xyz[mask, 2] = -0.5
        
        return xyz
    
    def convert_batch(self, keypoints_batch):
        """Convert batch of dicemap keypoints to equirectangular"""
        batch_size, num_points, _ = keypoints_batch.shape
        
        x_dice = keypoints_batch[:, :, 0]
        y_dice = keypoints_batch[:, :, 1]
        
        # Initialize outputs
        eq_coords = cp.zeros_like(keypoints_batch)
        spherical_coords = cp.zeros_like(keypoints_batch)
        face_ids = cp.full((batch_size, num_points), -1, dtype=cp.int32)
        
        # Determine which face each point belongs to
        for y_min, y_max, x_min, x_max, face_id in self.face_regions:
            mask = ((x_dice >= x_min) & (x_dice < x_max) & 
                   (y_dice >= y_min) & (y_dice < y_max))
            face_ids[mask] = face_id
        
        valid_mask = (face_ids >= 0)
        
        if cp.any(valid_mask):
            # Get face-local coordinates
            x_faces = cp.zeros_like(x_dice)
            y_faces = cp.zeros_like(y_dice)
            
            for y_min, y_max, x_min, x_max, face_id in self.face_regions:
                mask = (face_ids == face_id)
                x_faces[mask] = cp.clip(x_dice[mask] - x_min, 0, self.face_size - 1)
                y_faces[mask] = cp.clip(y_dice[mask] - y_min, 0, self.face_size - 1)
            
            # Convert to XYZ then to spherical
            xyz = self._face_to_xyz(face_ids, x_faces, y_faces)
            x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
            
            norm = cp.sqrt(x**2 + y**2 + z**2)
            norm = cp.maximum(norm, 1e-9)
            
            lon = cp.arctan2(x, z)
            lat = cp.arcsin(cp.clip(y / norm, -1, 1))
            
            # Convert to pixel coordinates
            u = (lon / (2 * cp.pi) + 0.5) * self.eq_width
            v = (-lat / cp.pi + 0.5) * self.eq_height
            
            eq_coords[:, :, 0] = u
            eq_coords[:, :, 1] = v
            spherical_coords[:, :, 0] = lon
            spherical_coords[:, :, 1] = lat
        
        return eq_coords, spherical_coords


class ImageLoader:
    """Simple batch loader for images"""
    def __init__(self, image_paths: List[str], batch_size: int, num_workers: int):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """Load single image"""
        try:
            img = cv2.imread(path)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def iter_batches(self):
        """Iterate through batches"""
        for i in range(0, len(self.image_paths), self.batch_size):
            batch_paths = self.image_paths[i:i+self.batch_size]
            
            images = []
            valid_paths = []
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(self._load_image, p): p for p in batch_paths}
                
                for future in as_completed(futures):
                    path = futures[future]
                    img = future.result()
                    if img is not None:
                        images.append(img)
                        valid_paths.append(path)
            
            if images:
                # Sort to maintain order
                sorted_pairs = sorted(zip(valid_paths, images), 
                                     key=lambda x: batch_paths.index(x[0]))
                valid_paths, images = zip(*sorted_pairs)
                yield list(valid_paths), np.array(images)


class UnifiedGPUPipeline:
    """Simplified GPU pipeline using CUDA only"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device_id = int(config.gpu.device.split(':')[-1])
        
        # Setup GPU memory
        self._setup_memory()
        
        # Caches
        self.converter_cache = {}
        self.coord_converter_cache = {}
        
        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()
        
        # Initialize models
        self._initialize_models()
        
        print(f"Pipeline initialized (device: {config.gpu.device})")
    
    def _setup_memory(self):
        """Setup GPU memory pool"""
        pool_size = int(self.config.gpu.max_memory_gb * 1024**3)
        memory_pool = cp.get_default_memory_pool()
        memory_pool.set_limit(pool_size)
        print(f"GPU memory limit: {self.config.gpu.max_memory_gb}GB")
    
    def _initialize_models(self):
        """Initialize YOLO and XFeat models"""
        print("\nInitializing models...")
        
        # YOLO model
        yolo_path = self.config.models.yolo_model
        print(f"Loading YOLO: {yolo_path}")
        self.yolo_model = YOLO(yolo_path)
        
        # XFeat ONNX model
        onnx_path = self.config.models.xfeat_onnx_path
        print(f"Loading XFeat: {onnx_path}")
        
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        # ONNX session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        
        # CUDA execution provider
        providers = []
        if self.config.gpu.device.startswith('cuda'):
            cuda_options = {
                'device_id': self.device_id,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': int(self.config.gpu.max_memory_gb * 0.5 * 1024**3),
                'cudnn_conv_algo_search': 'DEFAULT',
            }
            providers.append(('CUDAExecutionProvider', cuda_options))
        
        providers.append('CPUExecutionProvider')
        
        # Create ONNX session
        self.xfeat_session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Check active provider
        active_providers = self.xfeat_session.get_providers()
        print(f"ONNX providers: {active_providers}")
        
        # Get model info
        self.xfeat_input_name = self.xfeat_session.get_inputs()[0].name
        self.xfeat_output_names = [o.name for o in self.xfeat_session.get_outputs()]
        
        # Get ONNX batch size (assumes first dimension is batch)
        input_shape = self.xfeat_session.get_inputs()[0].shape
        self.onnx_batch_size = input_shape[0] if isinstance(input_shape[0], int) else None
        
        if self.onnx_batch_size:
            print(f"ONNX model has fixed batch size: {self.onnx_batch_size}")
            # Ensure config batch size matches or is smaller
            if self.config.batching.feature_batch_size > self.onnx_batch_size:
                print(f"WARNING: Config batch size ({self.config.batching.feature_batch_size}) "
                      f"exceeds ONNX batch size ({self.onnx_batch_size}). "
                      f"Using ONNX batch size.")
                self.config.batching.feature_batch_size = self.onnx_batch_size
        else:
            print("ONNX model has dynamic batch size")
            self.onnx_batch_size = self.config.batching.feature_batch_size
        
        # PyTorch optimizations
        torch.backends.cudnn.benchmark = True
        
        # Warmup
        self._warmup_models()
        print("Models ready\n")
    
    def _warmup_models(self):
        """Warmup models"""
        print("Warming up models...")
        
        # YOLO warmup
        dummy_img = torch.randn(1, 3, 512, 512, device=self.config.gpu.device)
        with torch.no_grad():
            try:
                _ = self.yolo_model.predict(dummy_img, verbose=False, 
                                           device=self.config.gpu.device)
            except:
                pass
        
        # XFeat warmup - use correct batch size
        dummy_input = np.random.rand(self.onnx_batch_size, 3, 512, 512).astype(np.float32)
        inputs = {
            self.xfeat_input_name: dummy_input,
            'top_k': np.array(self.config.features.num_features, dtype=np.int64)
        }
        try:
            for _ in range(3):
                _ = self.xfeat_session.run(self.xfeat_output_names, inputs)
        except:
            pass
        
        torch.cuda.synchronize()
        del dummy_img
        print("Warmup complete")
    
    def _get_converter(self, h, w):
        """Get or create converter for image size"""
        key = (h, w)
        if key not in self.converter_cache:
            with cp.cuda.Device(self.device_id):
                self.converter_cache[key] = EquirectToDicemapConverter(
                    (h, w, 3), self.config.gpu.device
                )
        return self.converter_cache[key]
    
    def _get_coord_converter(self, face_size, eq_width, eq_height):
        """Get or create coordinate converter"""
        key = (face_size, eq_width, eq_height)
        if key not in self.coord_converter_cache:
            with cp.cuda.Device(self.device_id):
                self.coord_converter_cache[key] = CoordinateConverter(
                    face_size, eq_width, eq_height, self.config.gpu.device
                )
        return self.coord_converter_cache[key]
    
    def _detect_and_mask(self, dicemap_gpu):
        """Detect humans and mask them"""
        batch_size = dicemap_gpu.shape[0]
        
        # Convert to tensor
        dicemap_bgr = dicemap_gpu[..., ::-1].copy()
        images_tensor = torch.as_tensor(
            dicemap_bgr, device=self.config.gpu.device
        ).permute(0, 3, 1, 2).float() / 255.0
        
        # Run YOLO detection
        with torch.no_grad():
            results = self.yolo_model.predict(
                images_tensor,
                conf=self.config.human_detection.confidence_threshold,
                classes=[0],  # Person class
                device=self.config.gpu.device,
                verbose=False
            )
        
        # Extract detections
        detections = []
        for result in results:
            boxes = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                boxes = [[int(x1), int(y1), int(x2-x1), int(y2-y1)] 
                        for x1, y1, x2, y2 in xyxy]
            detections.append(boxes)
        
        # Mask detections
        if self.config.human_detection.mask_type == 'solid_color':
            mask_color = cp.array(self.config.human_detection.mask_color, 
                                 dtype=dicemap_gpu.dtype)
            for i, boxes in enumerate(detections):
                for x, y, w, h in boxes:
                    dicemap_gpu[i, y:y+h, x:x+w, :] = mask_color
        
        del images_tensor
        return dicemap_gpu, detections
    
    def _extract_features(self, images_gpu):
        """Extract features using XFeat ONNX model"""
        # Convert to numpy
        if isinstance(images_gpu, cp.ndarray):
            images_np = cp.asnumpy(images_gpu)
        else:
            images_np = images_gpu
        
        # Prepare batch: (B, H, W, C) -> (B, C, H, W) and normalize
        images_batch = images_np.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        
        actual_batch_size = len(images_batch)
        
        # Handle fixed ONNX batch size
        expected_batch_size = self.onnx_batch_size
        if actual_batch_size < expected_batch_size:
            # Pad batch to match ONNX model's fixed batch size
            pad_size = expected_batch_size - actual_batch_size
            padding = np.zeros((pad_size, *images_batch.shape[1:]), dtype=np.float32)
            images_batch_padded = np.concatenate([images_batch, padding], axis=0)
            print(f"  Padding batch: {actual_batch_size} -> {expected_batch_size}")
        else:
            images_batch_padded = images_batch
        
        # Process batch
        inputs = {
            self.xfeat_input_name: images_batch_padded,
            'top_k': np.array(self.config.features.num_features, dtype=np.int64)
        }
        
        outputs = self.xfeat_session.run(self.xfeat_output_names, inputs)
        keypoints, scores, descriptors = outputs
        
        # Process only actual images (not padding)
        all_results = []
        for i in range(actual_batch_size):
            valid_mask = scores[i] > 0
            
            result = {
                'keypoints': keypoints[i][valid_mask] if np.any(valid_mask) else np.zeros((0, 2)),
                'scores': scores[i][valid_mask] if np.any(valid_mask) else np.zeros((0,)),
                'descriptors': descriptors[i][valid_mask] if np.any(valid_mask) else np.zeros((0, 64))
            }
            
            # Pad to expected size
            n_valid = len(result['keypoints'])
            if n_valid < self.config.features.num_features:
                pad_size = self.config.features.num_features - n_valid
                result['keypoints'] = np.vstack([
                    result['keypoints'],
                    np.zeros((pad_size, 2), dtype=np.float32)
                ])
                result['scores'] = np.concatenate([
                    result['scores'],
                    np.zeros((pad_size,), dtype=np.float32)
                ])
                result['descriptors'] = np.vstack([
                    result['descriptors'],
                    np.zeros((pad_size, 64), dtype=np.float32)
                ])
            
            all_results.append(result)
        
        return all_results
    
    def process_batch(self, image_paths: List[str], image_batch: np.ndarray) -> Dict:
        """Process a batch of images"""
        batch_size = len(image_batch)
        h, w = image_batch[0].shape[0], image_batch[0].shape[1]
        face_size = h // 2
        
        with cp.cuda.Device(self.device_id):
            # Convert to dicemaps
            with self.perf_monitor.timer("dicemap_conversion"):
                converter = self._get_converter(h, w)
                dicemap_gpu = converter.convert_batch(image_batch)
            
            # Detect and mask humans
            with self.perf_monitor.timer("detection_masking"):
                masked_gpu, detections = self._detect_and_mask(dicemap_gpu)
            
            # Extract features
            with self.perf_monitor.timer("feature_extraction"):
                feature_results = self._extract_features(masked_gpu)
            
            # Convert coordinates
            with self.perf_monitor.timer("coordinate_conversion"):
                coord_converter = self._get_coord_converter(face_size, w, h)
                
                keypoints_list = [cp.asarray(f['keypoints']) for f in feature_results]
                keypoints_gpu = cp.stack(keypoints_list)
                
                eq_coords_gpu, sph_coords_gpu = coord_converter.convert_batch(keypoints_gpu)
            
            # Transfer to CPU
            with self.perf_monitor.timer("gpu_to_cpu"):
                eq_coords = cp.asnumpy(eq_coords_gpu)
                sph_coords = cp.asnumpy(sph_coords_gpu)
                keypoints_np = cp.asnumpy(keypoints_gpu)
                descriptors_np = np.stack([f['descriptors'] for f in feature_results])
                scores_np = np.stack([f['scores'] for f in feature_results])
        
        # Package results
        results = {}
        for i, path in enumerate(image_paths):
            name = os.path.basename(path)
            results[name] = {
                'dicemap_keypoints': keypoints_np[i],
                'keypoints': eq_coords[i],
                'spherical_keypoints': sph_coords[i],
                'descriptors': descriptors_np[i],
                'scores': scores_np[i],
                'image_size': (w, h),
                'humans_detected': len(detections[i])
            }
        
        # Cleanup
        del dicemap_gpu, masked_gpu, keypoints_gpu, eq_coords_gpu, sph_coords_gpu
        gc.collect()
        
        return results
    
    def _write_to_hdf5(self, output_file: str, results: Dict):
        """Write results to HDF5"""
        mode = 'w' if not os.path.exists(output_file) else 'a'
        with h5py.File(output_file, mode) as f:
            for name, data in results.items():
                if name in f:
                    continue
                
                grp = f.create_group(name)
                for key, value in data.items():
                    if key != 'humans_detected':
                        grp.create_dataset(key, data=value, compression='gzip')
                grp.attrs['humans_detected'] = data['humans_detected']
    
    def process_images(self, image_paths: List[str], output_file: str):
        """Process all images"""
        print(f"\n=== Processing {len(image_paths)} images ===")
        print(f"Batch size: {self.config.batching.feature_batch_size}")
        print(f"Features per image: {self.config.features.num_features}")
        
        start_time = time.time()
        
        loader = ImageLoader(
            image_paths,
            self.config.batching.feature_batch_size,
            self.config.batching.num_workers
        )
        
        total_processed = 0
        total_humans = 0
        
        for batch_idx, (batch_paths, batch_images) in enumerate(loader.iter_batches()):
            batch_results = self.process_batch(batch_paths, batch_images)
            self._write_to_hdf5(output_file, batch_results)
            
            total_processed += len(batch_results)
            total_humans += sum(r['humans_detected'] for r in batch_results.values())
            
            progress = (batch_idx + 1) / loader.total_batches * 100
            print(f"Progress: {batch_idx+1}/{loader.total_batches} ({progress:.1f}%) | "
                  f"{total_processed} images | {total_humans} humans")
            
            del batch_images, batch_results
            gc.collect()
        
        elapsed = time.time() - start_time
        
        print(f"\n=== Complete ===")
        print(f"Processed: {total_processed} images in {elapsed:.2f}s")
        print(f"Throughput: {total_processed/elapsed:.2f} images/sec")
        print(f"Humans masked: {total_humans}")
        print(f"Output: {output_file}")
        
        self.perf_monitor.report()
        
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()


def find_images(directory: str) -> List[str]:
    """Find all images in directory"""
    extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp']
    images = []
    path = Path(directory)
    for ext in extensions:
        images.extend(path.glob(f'*{ext}'))
        images.extend(path.glob(f'*{ext.upper()}'))
    return sorted(str(p) for p in images)


def main():
    parser = argparse.ArgumentParser(description="Simplified GPU Pipeline (CUDA Only)")
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default="features.h5")
    parser.add_argument('--config', type=str, default="config.yaml")
    
    # Optional overrides
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_features', type=int)
    parser.add_argument('--num_workers', type=int)
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config, args)
    
    # Find images
    image_paths = find_images(args.image_dir)
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Run pipeline
    pipeline = UnifiedGPUPipeline(config)
    pipeline.process_images(image_paths, args.output_file)


if __name__ == "__main__":
    main()