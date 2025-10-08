"""
Refactored Unified GPU Pipeline - Uses Centralized Config
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
from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
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
    print("WARNING: CuPy not available")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("ERROR: Ultralytics required")


class OptimizedGPU_Convert:
    """Optimized GPU converter with direct dicemap assembly"""
    def __init__(self, image_shape, device='cuda:0'):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy required")
        
        h, w, *_ = image_shape
        self.face_size = h // 2
        self.h, self.w = h, w
        self.device = device
        
        with cp.cuda.Device(int(device.split(':')[-1])):
            self.coors_xy = self._precompute_coordinates()
            self._precompute_dicemap_indices()
        
        print(f"OptimizedGPU_Convert initialized for {h}x{w}")
    
    def _precompute_coordinates(self):
        """Pre-compute all coordinate mappings"""
        face_w = self.face_size
        out = cp.zeros((face_w, face_w * 6, 3), dtype=cp.float32)
        rng = cp.linspace(-0.5, 0.5, num=face_w, dtype=cp.float32)
        x_grid, y_grid = cp.meshgrid(rng, rng, indexing='xy')
        
        faces_data = [
            (0, x_grid, -y_grid, 0.5),
            (1, 0.5, -y_grid, -x_grid),
            (2, -x_grid, -y_grid, -0.5),
            (3, -0.5, -y_grid, x_grid),
            (4, x_grid, 0.5, y_grid),
            (5, x_grid, -0.5, -y_grid),
        ]
        
        for col, x, y, z in faces_data:
            start = col * face_w
            end = (col + 1) * face_w
            out[:, start:end, 0] = x
            out[:, start:end, 1] = y
            out[:, start:end, 2] = z
        
        uv = self._xyz2uv(out)
        return self._uv2coor(uv, self.h, self.w)
    
    def _xyz2uv(self, xyz):
        x, y, z = cp.split(xyz, 3, axis=-1)
        norm = cp.maximum(cp.sqrt(x**2 + y**2 + z**2), 1e-9)
        u = cp.arctan2(x, z)
        v = cp.arcsin(y / norm)
        return cp.concatenate([u, v], axis=-1)
    
    def _uv2coor(self, uv, h, w):
        u, v = cp.split(uv, 2, axis=-1)
        coor_x = (u / (2 * cp.pi) + 0.5) * w - 0.5
        coor_y = (-v / cp.pi + 0.5) * h - 0.5
        return cp.concatenate([coor_x, coor_y], axis=-1).astype(cp.float32)
    
    def _precompute_dicemap_indices(self):
        fs = self.face_size
        self.dicemap_shape = (fs * 3, fs * 4)
        
        self.face_slices = {
            'U': (slice(0, fs), slice(fs, 2*fs)),
            'L': (slice(fs, 2*fs), slice(0, fs)),
            'F': (slice(fs, 2*fs), slice(fs, 2*fs)),
            'R': (slice(fs, 2*fs), slice(2*fs, 3*fs)),
            'B': (slice(fs, 2*fs), slice(3*fs, 4*fs)),
        }
        
        self.cube_slices = {
            'F': slice(0, fs),
            'R': slice(fs, 2*fs),
            'B': slice(2*fs, 3*fs),
            'L': slice(3*fs, 4*fs),
            'U': slice(4*fs, 5*fs),
            'D': slice(5*fs, 6*fs),
        }
    
    def _sample_equirec_batch(self, e_imgs, coor_xy):
        batch_size, H, W, C = e_imgs.shape
        e_pad = cp.pad(e_imgs, ((0, 0), (1, 1), (0, 0), (0, 0)), mode="edge")
        
        coor_x, coor_y = cp.split(coor_xy, 2, axis=-1)
        coor_y = coor_y + 1.0
        
        coords_shape = coor_x.shape[:-1]
        coords = cp.concatenate([coor_y, coor_x], axis=-1).reshape(-1, 2).T.astype(cp.float32)
        
        results = []
        for b in range(batch_size):
            channel_results = []
            for c in range(C):
                out = cupyx.scipy.ndimage.map_coordinates(
                    e_pad[b, :, :, c], coords, order=1, mode="wrap"
                )
                channel_results.append(out)
            
            result = cp.stack(channel_results, axis=-1).reshape(coords_shape + (C,))
            results.append(result)
        
        return cp.stack(results, axis=0)
    
    def convert_batch_to_dicemaps(self, equirect_batch):
        if isinstance(equirect_batch, np.ndarray):
            equirect_batch = cp.asarray(equirect_batch)
        
        batch_size = equirect_batch.shape[0]
        cubemaps = self._sample_equirec_batch(equirect_batch, self.coors_xy)
        
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


class VectorizedCoordConverter:
    """GPU-accelerated vectorized coordinate conversion"""
    def __init__(self, face_size, eq_width, eq_height, device='cuda:0'):
        self.face_size = face_size
        self.eq_width = eq_width
        self.eq_height = eq_height
        self.device = device
        
        with cp.cuda.Device(int(device.split(':')[-1])):
            self._precompute_conversion_maps()
    
    def _precompute_conversion_maps(self):
        fs = self.face_size
        self.face_regions = cp.array([
            [0, fs, fs, 2*fs, 0],
            [fs, 2*fs, 0, fs, 1],
            [fs, 2*fs, fs, 2*fs, 2],
            [fs, 2*fs, 2*fs, 3*fs, 3],
            [fs, 2*fs, 3*fs, 4*fs, 4],
        ], dtype=cp.int32)
        
        self.face_names = ['U', 'L', 'F', 'R', 'B']
    
    def _face_to_xyz(self, face_id, x_face, y_face):
        fs = self.face_size
        u = (x_face / fs) - 0.5
        v = (y_face / fs) - 0.5
        
        xyz = cp.zeros((*u.shape, 3), dtype=cp.float32)
        
        if face_id == 0:
            xyz[:, 0] = u
            xyz[:, 1] = 0.5
            xyz[:, 2] = v
        elif face_id == 1:
            xyz[:, 0] = -0.5
            xyz[:, 1] = -v
            xyz[:, 2] = u
        elif face_id == 2:
            xyz[:, 0] = u
            xyz[:, 1] = -v
            xyz[:, 2] = 0.5
        elif face_id == 3:
            xyz[:, 0] = 0.5
            xyz[:, 1] = -v
            xyz[:, 2] = -u
        elif face_id == 4:
            xyz[:, 0] = -u
            xyz[:, 1] = -v
            xyz[:, 2] = -0.5
        
        return xyz
    
    def convert_batch(self, keypoints_batch_gpu):
        batch_size, num_points, _ = keypoints_batch_gpu.shape
        
        x_dice = keypoints_batch_gpu[:, :, 0]
        y_dice = keypoints_batch_gpu[:, :, 1]
        
        eq_coords = cp.zeros_like(keypoints_batch_gpu)
        spherical_coords = cp.zeros_like(keypoints_batch_gpu)
        
        for i, (y_min, y_max, x_min, x_max, face_id) in enumerate(self.face_regions):
            mask = ((x_dice >= x_min) & (x_dice < x_max) & 
                   (y_dice >= y_min) & (y_dice < y_max))
            
            if not cp.any(mask):
                continue
            
            x_face = cp.clip(x_dice - x_min, 0, self.face_size - 1)
            y_face = cp.clip(y_dice - y_min, 0, self.face_size - 1)
            
            xyz = self._face_to_xyz(face_id, x_face[mask], y_face[mask])
            
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            norm = cp.sqrt(x**2 + y**2 + z**2)
            norm = cp.maximum(norm, 1e-9)
            
            lon = cp.arctan2(x, z)
            lat = cp.arcsin(y / norm)
            
            u = (lon / (2 * cp.pi) + 0.5) * self.eq_width
            v = (-lat / cp.pi + 0.5) * self.eq_height
            
            eq_coords[mask, 0] = u
            eq_coords[mask, 1] = v
            spherical_coords[mask, 0] = lon
            spherical_coords[mask, 1] = lat
        
        return eq_coords, spherical_coords


class UnifiedGPUPipeline:
    """Optimized unified pipeline with config-based settings"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        if not CUPY_AVAILABLE or not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("CuPy and Ultralytics required")
        
        self.device_id = int(config.gpu.device.split(':')[-1])
        self._setup_memory_pool()
        
        self.converter_cache = {}
        self.coord_converter_cache = {}
        self.gpu_buffer_pool = {}
        
        self._initialize_models()
        
        self.prefetch_queue = Queue(maxsize=config.batching.prefetch_batches)
        self.prefetch_thread = None
        
        print(f"Pipeline initialized with {config.batching.num_workers} workers")
    
    def _setup_memory_pool(self):
        pool_size = int(self.config.gpu.max_memory_gb * 1024**3)
        self.memory_pool = cp.get_default_memory_pool()
        self.pinned_memory_pool = cp.get_default_pinned_memory_pool()
        self.memory_pool.set_limit(pool_size)
        print(f"GPU memory pool: {self.config.gpu.max_memory_gb}GB")
    
    def _initialize_models(self):
        print("\nInitializing models...")
        
        # YOLO
        self.yolo_model = YOLO(self.config.models.yolo_model)
        self.yolo_model.to(self.config.gpu.device)
        
        # XFeat
        sys.path.append(self.config.models.xfeat_module_path)
        from modules.xfeat import XFeat
        self.xfeat_model = XFeat(
            weights=self.config.models.xfeat_weights,
            top_k=self.config.features.num_features,
            detection_threshold=self.config.features.detection_threshold,
            lightglue_checkpoint=self.config.models.lightglue_checkpoint  # ADD THIS
        ).eval().cuda()
                
        if self.config.gpu.use_half_precision:
            self.xfeat_model = self.xfeat_model.half()
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        if self.config.gpu.compile_models and hasattr(torch, 'compile'):
            try:
                self.xfeat_model = torch.compile(
                    self.xfeat_model, mode='max-autotune'
                )
            except:
                pass
        
        print("Models initialized\n")
    
    def _get_converter(self, h, w):
        key = (h, w)
        if key not in self.converter_cache:
            if len(self.converter_cache) >= 5:
                self.converter_cache.pop(next(iter(self.converter_cache)))
            
            with cp.cuda.Device(self.device_id):
                self.converter_cache[key] = OptimizedGPU_Convert(
                    (h, w, 3), self.config.gpu.device
                )
        return self.converter_cache[key]
    
    def _get_coord_converter(self, face_size, eq_width, eq_height):
        key = (face_size, eq_width, eq_height)
        if key not in self.coord_converter_cache:
            with cp.cuda.Device(self.device_id):
                self.coord_converter_cache[key] = VectorizedCoordConverter(
                    face_size, eq_width, eq_height, self.config.gpu.device
                )
        return self.coord_converter_cache[key]
    
    def _detect_and_mask_batch(self, dicemap_gpu):
        B = dicemap_gpu.shape[0]
        dicemap_bgr = dicemap_gpu[..., [2, 1, 0]]
        
        with cp.cuda.Device(self.device_id):
            images_tensor = torch.as_tensor(
                dicemap_bgr, device=self.config.gpu.device
            ).permute(0, 3, 1, 2).float() / 255.0
        
        with torch.no_grad():
            results = self.yolo_model.predict(
                images_tensor,
                conf=self.config.human_detection.confidence_threshold,
                classes=[0],
                device=self.config.gpu.device,
                verbose=False,
                stream=True,
                half=self.config.gpu.use_half_precision
            )
            
            detections = []
            for result in results:
                boxes = []
                if result.boxes is not None and len(result.boxes) > 0:
                    xyxy = result.boxes.xyxy.cpu().numpy()
                    boxes = [[int(x1), int(y1), int(x2-x1), int(y2-y1)] 
                            for x1, y1, x2, y2 in xyxy]
                detections.append(boxes)
        
        if self.config.human_detection.mask_type == 'solid_color':
            mask_color = cp.array(self.config.human_detection.mask_color, 
                                 dtype=dicemap_gpu.dtype)
            for i, boxes in enumerate(detections):
                for x, y, w, h in boxes:
                    dicemap_gpu[i, y:y+h, x:x+w, :] = mask_color
        
        del images_tensor, results, dicemap_bgr
        return dicemap_gpu, detections
    
    def _extract_features_optimized(self, images_gpu):
        with cp.cuda.Device(self.device_id):
            images_tensor = torch.as_tensor(
                images_gpu, device=self.config.gpu.device
            ).permute(0, 3, 1, 2).float() / 255.0
            
            if self.config.gpu.use_half_precision:
                images_tensor = images_tensor.half()
        
        sub_batch_size = min(self.config.batching.feature_batch_size, 32)
        all_results = []
        
        with torch.cuda.amp.autocast(enabled=self.config.gpu.use_half_precision):
            with torch.no_grad():
                for i in range(0, len(images_tensor), sub_batch_size):
                    batch = images_tensor[i:i+sub_batch_size]
                    outputs = self.xfeat_model.detectAndCompute(
                        batch, top_k=self.config.features.num_features
                    )
                    
                    for output in outputs:
                        if isinstance(output, dict):
                            kpts = output.get('keypoints', torch.zeros((0, 2)))
                            desc = output.get('descriptors', torch.zeros((0, 64)))
                            scores = output.get('scores', torch.zeros((0,)))
                            
                            n = len(kpts)
                            if n < self.config.features.num_features:
                                pad = self.config.features.num_features - n
                                kpts = torch.cat([kpts, torch.zeros((pad, 2), device=kpts.device)])
                                desc = torch.cat([desc, torch.zeros((pad, 64), device=desc.device)])
                                scores = torch.cat([scores, torch.zeros(pad, device=scores.device)])
                            
                            all_results.append({
                                'keypoints': kpts,
                                'descriptors': desc,
                                'scores': scores
                            })
                        else:
                            all_results.append({
                                'keypoints': torch.zeros((self.config.features.num_features, 2), 
                                                        device=self.config.gpu.device),
                                'descriptors': torch.zeros((self.config.features.num_features, 64), 
                                                          device=self.config.gpu.device),
                                'scores': torch.zeros(self.config.features.num_features, 
                                                     device=self.config.gpu.device)
                            })
        
        del images_tensor
        return all_results
    
    def process_batch(self, image_paths: List[str], image_batch: np.ndarray) -> Dict:
        B, h, w = len(image_batch), image_batch[0].shape[0], image_batch[0].shape[1]
        face_size = h // 2
        
        print(f"  Processing batch of {B} images ({h}x{w})")
        start = time.time()
        
        with cp.cuda.Device(self.device_id):
            t1 = time.time()
            converter = self._get_converter(h, w)
            dicemap_gpu = converter.convert_batch_to_dicemaps(image_batch)
            print(f"    Dicemap: {time.time()-t1:.2f}s")
            
            t2 = time.time()
            masked_gpu, detections = self._detect_and_mask_batch(dicemap_gpu)
            print(f"    Detect+Mask: {time.time()-t2:.2f}s")
            
            t3 = time.time()
            feature_results = self._extract_features_optimized(masked_gpu)
            print(f"    Features: {time.time()-t3:.2f}s")
            
            t4 = time.time()
            coord_converter = self._get_coord_converter(face_size, w, h)
            
            keypoints_list = []
            for f in feature_results:
                if isinstance(f['keypoints'], torch.Tensor):
                    kpts_cp = cp.asarray(f['keypoints'].detach())
                else:
                    kpts_cp = cp.asarray(f['keypoints'])
                keypoints_list.append(kpts_cp)
            
            keypoints_gpu = cp.stack(keypoints_list)
            
            eq_coords_gpu, sph_coords_gpu = coord_converter.convert_batch(keypoints_gpu)
            print(f"    Coords: {time.time()-t4:.2f}s")
            
            t5 = time.time()
            eq_coords = cp.asnumpy(eq_coords_gpu)
            sph_coords = cp.asnumpy(sph_coords_gpu)
            
            keypoints_np = cp.asnumpy(keypoints_gpu)
            descriptors_np = np.stack([
                f['descriptors'].cpu().numpy() if isinstance(f['descriptors'], torch.Tensor) 
                else f['descriptors'] for f in feature_results
            ])
            scores_np = np.stack([
                f['scores'].cpu().numpy() if isinstance(f['scores'], torch.Tensor)
                else f['scores'] for f in feature_results
            ])
            print(f"    Transfer: {time.time()-t5:.2f}s")
        
        total_humans = sum(len(d) for d in detections)
        print(f"    Total: {time.time()-start:.2f}s | Humans: {total_humans}")
        
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
        del feature_results, keypoints_list, keypoints_np, descriptors_np, scores_np
        torch.cuda.empty_cache()
        if (B % 4) == 0:
            self.memory_pool.free_all_blocks()
        
        return results
    
    def _write_batch_to_hdf5(self, output_file: str, batch_results: Dict, mode='w'):
        with h5py.File(output_file, mode) as f:
            for name, data in batch_results.items():
                if name in f:
                    continue
                
                grp = f.create_group(name)
                for key, value in data.items():
                    if key != 'humans_detected':
                        grp.create_dataset(
                            key, data=value,
                            compression='gzip', 
                            compression_opts=self.config.io.compression_level,
                            shuffle=True
                        )
                grp.attrs['humans_detected'] = data['humans_detected']
    
    def _prefetch_worker(self, batches):
        for batch_paths, batch_data in batches:
            self.prefetch_queue.put((batch_paths, batch_data))
        self.prefetch_queue.put(None)
    
    def process_images(self, image_paths: List[str], output_file: str):
        print(f"\n=== Processing {len(image_paths)} images ===")
        print(f"Batch size: {self.config.batching.feature_batch_size} | "
              f"Features: {self.config.features.num_features} | "
              f"Workers: {self.config.batching.num_workers}")
        
        print("\nLoading images...")
        with ThreadPoolExecutor(max_workers=self.config.batching.num_workers) as executor:
            loaded = list(executor.map(
                lambda p: (p, cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
                if cv2.imread(p) is not None else None,
                image_paths
            ))
        
        size_groups = {}
        for item in filter(None, loaded):
            path, img = item
            shape = img.shape[:2]
            size_groups.setdefault(shape, []).append((path, img))
        
        all_batches = []
        for shape, group_data in size_groups.items():
            for i in range(0, len(group_data), self.config.batching.feature_batch_size):
                batch_data = group_data[i:i+self.config.batching.feature_batch_size]
                paths, imgs = zip(*batch_data)
                all_batches.append((list(paths), np.array(imgs)))
        
        self.prefetch_thread = Thread(
            target=self._prefetch_worker, args=(all_batches,)
        )
        self.prefetch_thread.start()
        
        total_results = 0
        total_humans = 0
        first_batch = True
        
        while True:
            batch_item = self.prefetch_queue.get()
            if batch_item is None:
                break
            
            batch_paths, batch_images = batch_item
            batch_results = self.process_batch(batch_paths, batch_images)
            
            mode = 'w' if first_batch else 'a'
            self._write_batch_to_hdf5(output_file, batch_results, mode)
            first_batch = False
            
            total_results += len(batch_results)
            total_humans += sum(r['humans_detected'] for r in batch_results.values())
        
        self.prefetch_thread.join()
        
        print(f"\n=== Complete ===")
        print(f"Processed: {total_results} images")
        print(f"Total humans masked: {total_humans}")
        print(f"Output: {output_file}")
        
        self._cleanup()
    
    def _cleanup(self):
        torch.cuda.empty_cache()
        self.memory_pool.free_all_blocks()
        self.pinned_memory_pool.free_all_blocks()
        gc.collect()


def find_images(directory: str) -> List[str]:
    exts = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp']
    images = []
    path = Path(directory)
    for ext in exts:
        images.extend(path.glob(f'*{ext}'))
        images.extend(path.glob(f'*{ext.upper()}'))
    return sorted(str(p) for p in images)


def main():
    parser = argparse.ArgumentParser(
        description="Optimized GPU pipeline with config-based hyperparameters"
    )
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default="features.h5")
    parser.add_argument('--config', type=str, default="config.yaml",
                       help="Path to configuration file")
    
    # Optional overrides
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_features', type=int)
    parser.add_argument('--yolo_model', type=str)
    parser.add_argument('--yolo_conf', type=float)
    parser.add_argument('--mask_type', type=str)
    parser.add_argument('--mask_color', type=int, nargs=3)
    parser.add_argument('--gpu_memory', type=float)
    parser.add_argument('--no_half_precision', action='store_true')
    parser.add_argument('--num_workers', type=int)
    
    args = parser.parse_args()
    
    if not CUPY_AVAILABLE or not ULTRALYTICS_AVAILABLE:
        print("ERROR: CuPy and Ultralytics required")
        return
    
    # Load config with CLI overrides
    config = load_config(args.config, args)
    
    image_paths = find_images(args.image_dir)
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return
    
    pipeline = UnifiedGPUPipeline(config)
    pipeline.process_images(image_paths, args.output_file)


if __name__ == "__main__":
    main()