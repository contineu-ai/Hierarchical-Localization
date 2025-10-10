"""
Simplified GPU Pipeline - Direct TensorRT Engine Loading

Loads pre-built TensorRT engines (.trt/.plan files) directly instead of ONNX files.
Faster initialization and inference by bypassing ONNX Runtime conversion.
"""
import os
os.environ['ORT_LOGLEVEL'] = 'VERBOSE'
import numpy as np
import cv2
import h5py
import torch
import argparse
import time
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
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("ERROR: TensorRT and PyCUDA required")
    print("Install with: pip install tensorrt pycuda")
    sys.exit(1)


class TRTInferenceEngine:
    """TensorRT Engine wrapper for inference"""
    
    def __init__(self, engine_path: str, device_id: int = 0):
        self.engine_path = engine_path
        self.device_id = device_id
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.device_buffers = {}  # Initialize early for __del__
        self.host_outputs = {}
        self.stream = None  # CUDA stream for async execution
        
        # Load engine
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
        
        self.context = self.engine.create_execution_context()
        
        # Create CUDA stream for async execution
        self.stream = cuda.Stream()
        
        # Get binding information - support both old and new TensorRT APIs
        self.bindings = []
        self.binding_names = []
        self.input_specs = []
        self.output_specs = []
        
        # Try new API first (TensorRT 8.5+)
        try:
            num_io = self.engine.num_io_tensors
            use_new_api = True
        except AttributeError:
            num_io = self.engine.num_bindings
            use_new_api = False
        
        for i in range(num_io):
            if use_new_api:
                # New API (TensorRT 8.5+)
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                shape = self.engine.get_tensor_shape(name)
                mode = self.engine.get_tensor_mode(name)
                is_input = (mode == trt.TensorIOMode.INPUT)
            else:
                # Old API (TensorRT < 8.5)
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                shape = self.engine.get_binding_shape(i)
                is_input = self.engine.binding_is_input(i)
            
            self.binding_names.append(name)
            
            spec = {
                'name': name,
                'dtype': dtype,
                'shape': tuple(shape),
                'is_input': is_input,
                'index': i
            }
            
            if is_input:
                self.input_specs.append(spec)
            else:
                self.output_specs.append(spec)
        
        print(f"  Engine loaded successfully")
        print(f"  Inputs: {[s['name'] + str(s['shape']) for s in self.input_specs]}")
        print(f"  Outputs: {[s['name'] + str(s['shape']) for s in self.output_specs]}")
        
        # Determine API version for later use
        self.use_new_api = use_new_api
        
        # Allocate device memory for inputs/outputs
        # Allocate device memory for inputs/outputs (skip scalars)
        for spec in self.input_specs + self.output_specs:
            # Skip scalar constants
            if len(spec['shape']) == 0:
                print(f"  Skipping scalar '{spec['name']}' - build-time constant")
                continue
            
            # Calculate size
            size = int(np.prod(spec['shape'])) if all(s > 0 for s in spec['shape']) else 0
            if size == 0:
                # Dynamic shape - will allocate during inference
                continue
            
            # Allocate device memory
            self.device_buffers[spec['name']] = cuda.mem_alloc(
                size * np.dtype(spec['dtype']).itemsize
            )
            
            # Allocate host memory for outputs
            if not spec['is_input']:
                self.host_outputs[spec['name']] = np.empty(spec['shape'], dtype=spec['dtype'])    
    def infer(self, **inputs):
        """Run inference with given inputs"""
        # Filter out scalar inputs - they're usually build-time constants
        runtime_inputs = {}
        for name, value in inputs.items():
            # Skip scalar constants (shape is empty or ())
            if hasattr(value, 'shape') and (value.shape == () or len(value.shape) == 0):
                # print(f"[DEBUG] Skipping scalar input '{name}' - likely a build-time constant")
                continue
            runtime_inputs[name] = value
        
        # Handle dynamic shapes
        for spec in self.input_specs:
            if spec['name'] in runtime_inputs:
                input_shape = runtime_inputs[spec['name']].shape
                if self.use_new_api:
                    # New API
                    self.context.set_input_shape(spec['name'], input_shape)
                else:
                    # Old API
                    self.context.set_binding_shape(spec['index'], input_shape)
        
        # Allocate/reallocate buffers if needed (for dynamic shapes)
        for spec in self.input_specs + self.output_specs:
            # Skip scalar inputs
            if len(spec['shape']) == 0:
                continue
                
            if spec['name'] not in self.device_buffers or self.context.all_binding_shapes_specified:
                if spec['is_input']:
                    if spec['name'] not in runtime_inputs:
                        continue
                    shape = runtime_inputs[spec['name']].shape
                else:
                    if self.use_new_api:
                        shape = self.context.get_tensor_shape(spec['name'])
                    else:
                        shape = self.context.get_binding_shape(spec['index'])
                
                size = int(np.prod(shape))
                if size > 0:
                    # Reallocate if needed
                    if spec['name'] in self.device_buffers:
                        self.device_buffers[spec['name']].free()
                    
                    self.device_buffers[spec['name']] = cuda.mem_alloc(
                        size * np.dtype(spec['dtype']).itemsize
                    )
                    
                    if not spec['is_input']:
                        self.host_outputs[spec['name']] = np.empty(shape, dtype=spec['dtype'])
        
        # Copy inputs to device (skip scalars)
        for spec in self.input_specs:
            if spec['name'] in runtime_inputs:
                input_data = np.ascontiguousarray(runtime_inputs[spec['name']])
                cuda.memcpy_htod_async(self.device_buffers[spec['name']], input_data, self.stream)
        
        # Prepare binding addresses and execute
        if self.use_new_api:
            # New API - set tensor addresses (skip scalars)
            for name in self.binding_names:
                if name in self.device_buffers:
                    self.context.set_tensor_address(name, int(self.device_buffers[name]))
            
            # Execute
            success = self.context.execute_async_v3(self.stream.handle)
        else:
            # Old API - use bindings list
            bindings = [int(self.device_buffers[name]) if name in self.device_buffers else 0 
                    for name in self.binding_names]
            
            # Execute
            success = self.context.execute_async_v2(bindings, self.stream.handle)
        
        if not success:
            raise RuntimeError("TensorRT inference failed")
        
        # Copy outputs from device
        outputs = {}
        for spec in self.output_specs:
            if spec['name'] in self.device_buffers:
                cuda.memcpy_dtoh_async(self.host_outputs[spec['name']], self.device_buffers[spec['name']], self.stream)
                outputs[spec['name']] = self.host_outputs[spec['name']]
        
        # Synchronize stream to ensure all operations complete
        self.stream.synchronize()
        
        return outputs    
    def __del__(self):
        """Cleanup"""
        # Clean up device buffers
        if hasattr(self, 'device_buffers'):
            for buf in self.device_buffers.values():
                try:
                    buf.free()
                except:
                    pass
        
        # Clean up stream
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                # Synchronize before cleanup
                self.stream.synchronize()
            except:
                pass


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
    """GPU pipeline using direct TensorRT engine loading"""
    
    RTDETR_INPUT_SIZE = 640
    
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
        print(f"RT-DETR fixed input size: {self.RTDETR_INPUT_SIZE}x{self.RTDETR_INPUT_SIZE}")
    
    def _setup_memory(self):
        """Setup GPU memory pool"""
        pool_size = int(self.config.gpu.max_memory_gb * 1024**3)
        memory_pool = cp.get_default_memory_pool()
        memory_pool.set_limit(pool_size)
        print(f"GPU memory limit: {self.config.gpu.max_memory_gb}GB")
    
    def _initialize_models(self):
        """Initialize RT-DETR and XFeat TensorRT engines"""
        print("\nInitializing TensorRT engines...")
        
        # RT-DETR TensorRT engine
        if hasattr(self.config.models, 'rtdetr_trt_path') and self.config.models.rtdetr_trt_path:
            rtdetr_engine_path = self.config.models.rtdetr_trt_path
        else:
            rtdetr_engine_path = self.config.models.rtdetr_onnx_path.replace('.onnx', '.trt')
            if not os.path.exists(rtdetr_engine_path):
                rtdetr_engine_path = self.config.models.rtdetr_onnx_path.replace('.onnx', '.plan')
        
        if not os.path.exists(rtdetr_engine_path):
            raise FileNotFoundError(
                f"RT-DETR TensorRT engine not found. Tried:\n"
                f"  {rtdetr_engine_path}\n"
                f"Please set 'rtdetr_trt_path' in your config or place the engine file alongside the ONNX file."
            )
        
        self.rtdetr_engine = TRTInferenceEngine(rtdetr_engine_path, self.device_id)
        
        # Get input/output names and batch size from engine
        self.rtdetr_image_input_name = self.rtdetr_engine.input_specs[0]['name']
        self.rtdetr_size_input_name = self.rtdetr_engine.input_specs[1]['name'] if len(self.rtdetr_engine.input_specs) > 1 else 'orig_target_sizes'
        
        # Check for fixed batch size
        img_shape = self.rtdetr_engine.input_specs[0]['shape']
        self.rtdetr_onnx_batch_size = img_shape[0] if img_shape[0] > 0 else None
        
        if self.rtdetr_onnx_batch_size:
            print(f"RT-DETR has fixed batch size: {self.rtdetr_onnx_batch_size}")
        else:
            print("RT-DETR has dynamic batch size")
        
        # XFeat TensorRT engine
        if hasattr(self.config.models, 'xfeat_trt_path') and self.config.models.xfeat_trt_path:
            xfeat_engine_path = self.config.models.xfeat_trt_path
        else:
            xfeat_engine_path = self.config.models.xfeat_onnx_path.replace('.onnx', '.trt')
            if not os.path.exists(xfeat_engine_path):
                xfeat_engine_path = self.config.models.xfeat_onnx_path.replace('.onnx', '.plan')
        
        if not os.path.exists(xfeat_engine_path):
            raise FileNotFoundError(
                f"XFeat TensorRT engine not found. Tried:\n"
                f"  {xfeat_engine_path}\n"
                f"Please set 'xfeat_trt_path' in your config or place the engine file alongside the ONNX file."
            )
        
        self.xfeat_engine = TRTInferenceEngine(xfeat_engine_path, self.device_id)
        
        self.xfeat_input_name = self.xfeat_engine.input_specs[0]['name']
        self.xfeat_output_names = [spec['name'] for spec in self.xfeat_engine.output_specs]
        
        input_shape = self.xfeat_engine.input_specs[0]['shape']
        self.onnx_batch_size = input_shape[0] if input_shape[0] > 0 else None
        
        # CRITICAL FIX: Read expected H/W dimensions from engine
        self.xfeat_input_height = input_shape[2] if input_shape[2] > 0 else None
        self.xfeat_input_width = input_shape[3] if input_shape[3] > 0 else None
        
        if self.onnx_batch_size:
            print(f"XFeat has fixed batch size: {self.onnx_batch_size}")
            if self.config.batching.feature_batch_size > self.onnx_batch_size:
                print(f"WARNING: Config batch size ({self.config.batching.feature_batch_size}) "
                    f"exceeds engine batch size ({self.onnx_batch_size}). "
                    f"Using engine batch size.")
                self.config.batching.feature_batch_size = self.onnx_batch_size
        else:
            print("XFeat has dynamic batch size")
            self.onnx_batch_size = self.config.batching.feature_batch_size
        
        if self.xfeat_input_height is None or self.xfeat_input_width is None:
            print("WARNING: XFeat has dynamic H/W dimensions - will use dicemap size")
        else:
            print(f"XFeat expected input size: {self.xfeat_input_height}x{self.xfeat_input_width}")
        
        # Pre-allocate padding buffers
        print("\nPre-allocating padding buffers...")
        
        if self.rtdetr_onnx_batch_size:
            self.rtdetr_padding_images = np.zeros(
                (self.rtdetr_onnx_batch_size, 3, self.RTDETR_INPUT_SIZE, self.RTDETR_INPUT_SIZE),
                dtype=np.float32
            )
            self.rtdetr_padding_sizes = np.tile(
                np.array([[self.RTDETR_INPUT_SIZE, self.RTDETR_INPUT_SIZE]], dtype=np.int64),
                (self.rtdetr_onnx_batch_size, 1)
            )
            print(f"  RT-DETR padding buffer: {self.rtdetr_padding_images.shape}")
        
        # Only pre-allocate XFeat buffer if dimensions are fixed
        if self.xfeat_input_height and self.xfeat_input_width:
            self.xfeat_padding_images = np.zeros(
                (self.onnx_batch_size, 3, self.xfeat_input_height, self.xfeat_input_width),
                dtype=np.float32
            )
            print(f"  XFeat padding buffer: {self.xfeat_padding_images.shape}")
        
        self.xfeat_padding_cache = {}
        
        with cp.cuda.Device(self.device_id):
            max_batch = self.rtdetr_onnx_batch_size if self.rtdetr_onnx_batch_size else self.config.batching.feature_batch_size
            self.rtdetr_resized_gpu = cp.zeros(
                (max_batch, self.RTDETR_INPUT_SIZE, self.RTDETR_INPUT_SIZE, 3),
                dtype=cp.float32
            )
            print(f"  RT-DETR GPU resize buffer: {self.rtdetr_resized_gpu.shape}")
        
        print("Padding buffers ready")
        print(f"\nNOTE: Ensure your XFeat TensorRT engine input dimensions match your dicemap size")
        print(f"      Dicemap size depends on your input image dimensions (face_size*3 x face_size*4)")
        
        torch.backends.cudnn.benchmark = True
        
        self._warmup_models()
        print("Models ready\n")


    def _warmup_models(self):
        """Warmup models"""
        print("Warming up TensorRT engines...")
        
        # RT-DETR warmup
        print("  Warming up RT-DETR...")
        dummy_img_list = [np.random.rand(3, self.RTDETR_INPUT_SIZE, self.RTDETR_INPUT_SIZE).astype(np.float32)]
        dummy_size_list = [np.array([self.RTDETR_INPUT_SIZE, self.RTDETR_INPUT_SIZE], dtype=np.int64)]
        
        warmup_batch_size = 1
        if self.rtdetr_onnx_batch_size and warmup_batch_size < self.rtdetr_onnx_batch_size:
            pad_size = self.rtdetr_onnx_batch_size - warmup_batch_size
            for _ in range(pad_size):
                dummy_img_list.append(np.zeros_like(dummy_img_list[0]))
                dummy_size_list.append(np.array([self.RTDETR_INPUT_SIZE, self.RTDETR_INPUT_SIZE], dtype=np.int64))
        
        dummy_img = np.stack(dummy_img_list)
        dummy_size = np.stack(dummy_size_list)
        
        print(f"    RT-DETR input shapes: images={dummy_img.shape}, sizes={dummy_size.shape}")
        
        warmup_inputs = {
            self.rtdetr_image_input_name: dummy_img,
            self.rtdetr_size_input_name: dummy_size
        }
        
        try:
            for i in range(3):
                print(f"    RT-DETR warmup iteration {i+1}/3...", end='', flush=True)
                _ = self.rtdetr_engine.infer(**warmup_inputs)
                print(" OK")
            print("  RT-DETR warmup complete")
        except Exception as e:
            print(f"\n  WARNING: RT-DETR warmup failed: {e}")
            print("  Continuing anyway - model may still work")
        
        # XFeat warmup - use engine's expected dimensions
        print("  Warming up XFeat...")
        if self.xfeat_input_height and self.xfeat_input_width:
            warmup_h, warmup_w = self.xfeat_input_height, self.xfeat_input_width
        else:
            # Fallback to reasonable size if dynamic
            warmup_h, warmup_w = 1440, 1920
        
        dummy_input = np.random.rand(self.onnx_batch_size, 3, warmup_h, warmup_w).astype(np.float32)
        print(f"    XFeat input shape: {dummy_input.shape}")
        
        # Handle top_k parameter - check if it's in the engine inputs
        top_k_spec = None
        for spec in self.xfeat_engine.input_specs:
            if 'top_k' in spec['name'].lower():
                top_k_spec = spec
                break
        
        inputs = {
            self.xfeat_input_name: dummy_input
        }
        
        if top_k_spec:
            # Create top_k based on its expected shape
            if len(top_k_spec['shape']) == 0 or top_k_spec['shape'] == ():
                # Scalar
                top_k_value = np.array(self.config.features.num_features, dtype=np.int64)
            else:
                # Array
                top_k_value = np.array([self.config.features.num_features], dtype=np.int64)
            
            inputs[top_k_spec['name']] = top_k_value
            print(f"    XFeat top_k: {top_k_value} (shape={top_k_value.shape}, dtype={top_k_value.dtype})")
        
        try:
            for i in range(3):
                print(f"    XFeat warmup iteration {i+1}/3...", end='', flush=True)
                _ = self.xfeat_engine.infer(**inputs)
                print(" OK")
            print("  XFeat warmup complete")
        except Exception as e:
            print(f"\n  WARNING: XFeat warmup failed: {e}")
            print("  Continuing anyway - model may still work")
        
        torch.cuda.synchronize()
        print("Warmup complete\n")
    
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
    
    def _resize_for_rtdetr_gpu(self, dicemap_gpu):
        """GPU-accelerated resize using CuPy"""
        batch_size = dicemap_gpu.shape[0]
        orig_h, orig_w = dicemap_gpu.shape[1], dicemap_gpu.shape[2]
        
        scale_h = self.RTDETR_INPUT_SIZE / orig_h
        scale_w = self.RTDETR_INPUT_SIZE / orig_w
        
        resized_gpu = self.rtdetr_resized_gpu[:batch_size]
        
        for i in range(batch_size):
            for c in range(3):
                resized_gpu[i, :, :, c] = cupyx.scipy.ndimage.zoom(
                    dicemap_gpu[i, :, :, c],
                    (scale_h, scale_w),
                    order=1
                )
        
        return resized_gpu, orig_h, orig_w
    
    def _scale_boxes_to_original(self, boxes, orig_h, orig_w):
        """Scale detection boxes from 640x640 to original dicemap size"""
        scale_x = orig_w / self.RTDETR_INPUT_SIZE
        scale_y = orig_h / self.RTDETR_INPUT_SIZE
        
        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            scaled_box = [
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y
            ]
            scaled_boxes.append(scaled_box)
        
        return scaled_boxes
    
    def _detect_and_mask(self, dicemap_gpu):
        """Detect humans and mask them using TensorRT"""
        batch_size = dicemap_gpu.shape[0]
        orig_h, orig_w = dicemap_gpu.shape[1], dicemap_gpu.shape[2]
        
        with self.perf_monitor.timer("rtdetr_resize_gpu"):
            resized_gpu, _, _ = self._resize_for_rtdetr_gpu(dicemap_gpu)
        
        with self.perf_monitor.timer("rtdetr_normalize_transpose"):
            normalized_gpu = resized_gpu / 255.0
            images_tensor_gpu = normalized_gpu.transpose(0, 3, 1, 2)
            images_tensor_gpu = cp.ascontiguousarray(images_tensor_gpu)
        
        with self.perf_monitor.timer("rtdetr_padding"):
            if self.rtdetr_onnx_batch_size and batch_size < self.rtdetr_onnx_batch_size:
                self.rtdetr_padding_images[:batch_size] = cp.asnumpy(images_tensor_gpu)
                images_to_run = self.rtdetr_padding_images
                sizes_to_run = self.rtdetr_padding_sizes
                print(f"  Using pre-allocated RT-DETR padding: {batch_size} -> {self.rtdetr_onnx_batch_size}")
            else:
                images_to_run = cp.asnumpy(images_tensor_gpu)
                sizes_to_run = np.tile(
                    np.array([[self.RTDETR_INPUT_SIZE, self.RTDETR_INPUT_SIZE]], dtype=np.int64),
                    (batch_size, 1)
                )
        
        # TensorRT inference
        with self.perf_monitor.timer("rtdetr_inference"):
            outputs = self.rtdetr_engine.infer(**{
                self.rtdetr_image_input_name: images_to_run,
                self.rtdetr_size_input_name: sizes_to_run
            })
        
        # Extract outputs - adapt to your engine's output names
        output_list = list(outputs.values())
        labels, boxes, scores = output_list[0], output_list[1], output_list[2]
        
        with self.perf_monitor.timer("rtdetr_postprocess"):
            detections = []
            conf_thresh = self.config.human_detection.confidence_threshold
            
            for i in range(batch_size):
                person_mask = (labels[i] == 0) & (scores[i] > conf_thresh)
                filtered_boxes = boxes[i][person_mask]
                scaled_boxes = self._scale_boxes_to_original(filtered_boxes, orig_h, orig_w)
                
                boxes_list = []
                for box in scaled_boxes:
                    x1, y1, x2, y2 = box
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    boxes_list.append([x, y, w, h])
                
                detections.append(boxes_list)
        
        with self.perf_monitor.timer("rtdetr_masking"):
            if self.config.human_detection.mask_type == 'solid_color':
                mask_color = cp.array(self.config.human_detection.mask_color, 
                                    dtype=dicemap_gpu.dtype)
                for i, boxes_list in enumerate(detections):
                    for x, y, w, h in boxes_list:
                        x = max(0, min(x, orig_w - 1))
                        y = max(0, min(y, orig_h - 1))
                        w = min(w, orig_w - x)
                        h = min(h, orig_h - y)
                        if w > 0 and h > 0:
                            dicemap_gpu[i, y:y+h, x:x+w, :] = mask_color
        
        return dicemap_gpu, detections
    
    def _extract_features(self, images_gpu):
        """Extract features using TensorRT"""
        if isinstance(images_gpu, cp.ndarray):
            images_np = cp.asnumpy(images_gpu)
        else:
            images_np = images_gpu
        
        images_batch = images_np.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        actual_batch_size = len(images_batch)
        
        expected_batch_size = self.onnx_batch_size
        if actual_batch_size < expected_batch_size:
            input_shape = images_batch.shape[1:]
            cache_key = input_shape
            
            if cache_key not in self.xfeat_padding_cache:
                self.xfeat_padding_cache[cache_key] = np.zeros(
                    (expected_batch_size, *input_shape),
                    dtype=np.float32
                )
                print(f"  Created XFeat padding buffer for shape {input_shape}")
            
            padding_buffer = self.xfeat_padding_cache[cache_key]
            padding_buffer[:actual_batch_size] = images_batch
            images_batch_padded = padding_buffer
            print(f"  Using cached XFeat padding: {actual_batch_size} -> {expected_batch_size}")
        else:
            images_batch_padded = images_batch
        
        # TensorRT inference
        inputs = {
            self.xfeat_input_name: images_batch_padded,
            'top_k': np.array(self.config.features.num_features, dtype=np.int64)
        }
        
        outputs = self.xfeat_engine.infer(**inputs)
        
        # Extract outputs - adapt to your engine's output names
        output_list = list(outputs.values())
        keypoints, scores, descriptors = output_list[0], output_list[1], output_list[2]
        
        all_results = []
        for i in range(actual_batch_size):
            valid_mask = scores[i] > 0
            
            result = {
                'keypoints': keypoints[i][valid_mask] if np.any(valid_mask) else np.zeros((0, 2)),
                'scores': scores[i][valid_mask] if np.any(valid_mask) else np.zeros((0,)),
                'descriptors': descriptors[i][valid_mask] if np.any(valid_mask) else np.zeros((0, 64))
            }
            
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
            with self.perf_monitor.timer("dicemap_conversion"):
                converter = self._get_converter(h, w)
                dicemap_gpu = converter.convert_batch(image_batch)
            
            with self.perf_monitor.timer("detection_masking"):
                masked_gpu, detections = self._detect_and_mask(dicemap_gpu)
            
            with self.perf_monitor.timer("feature_extraction"):
                feature_results = self._extract_features(masked_gpu)
            
            with self.perf_monitor.timer("coordinate_conversion"):
                coord_converter = self._get_coord_converter(face_size, w, h)
                
                keypoints_list = [cp.asarray(f['keypoints']) for f in feature_results]
                keypoints_gpu = cp.stack(keypoints_list)
                
                eq_coords_gpu, sph_coords_gpu = coord_converter.convert_batch(keypoints_gpu)
            
            with self.perf_monitor.timer("gpu_to_cpu"):
                eq_coords = cp.asnumpy(eq_coords_gpu)
                sph_coords = cp.asnumpy(sph_coords_gpu)
                keypoints_np = cp.asnumpy(keypoints_gpu)
                descriptors_np = np.stack([f['descriptors'] for f in feature_results])
                scores_np = np.stack([f['scores'] for f in feature_results])
        
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
    parser = argparse.ArgumentParser(description="GPU Pipeline with Direct TensorRT Loading")
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default="features.h5")
    parser.add_argument('--config', type=str, default="config.yaml")
    
    # Optional overrides
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_features', type=int)
    parser.add_argument('--num_workers', type=int)
    
    args = parser.parse_args()
    
    config = load_config(args.config, args)
    
    image_paths = find_images(args.image_dir)
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    pipeline = UnifiedGPUPipeline(config)
    pipeline.process_images(image_paths, args.output_file)


if __name__ == "__main__":
    main()