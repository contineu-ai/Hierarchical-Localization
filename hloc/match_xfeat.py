"""
Refactored XFeat Matching - TensorRT Version
Uses TensorRT for maximum inference performance
"""

import torch
import torch.nn.functional as F
import numpy as np
import h5py
import argparse
import sys
from pathlib import Path
from typing import Dict, List
import gc
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from hloc.config_loader import load_config, PipelineConfig

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available - install with: pip install tensorrt pycuda")

# ONNX fallback
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNXRuntime not available - install with: pip install onnxruntime-gpu")


class OptimizedBatchDataset(Dataset):
    """Optimized dataset with config-based settings"""
    
    def __init__(self, pairs, feature_path_q, feature_path_r, config: PipelineConfig):
        self.pairs = pairs
        self.feature_paths = {'query': feature_path_q, 'ref': feature_path_r}
        self.max_keypoints = config.features.max_keypoints
        self._handles = {}
        
    def _get_handle(self, key):
        if key not in self._handles:
            self._handles[key] = h5py.File(self.feature_paths[key], 'r', swmr=True)
        return self._handles[key]
    
    def _load_features(self, name: str, source: str) -> Dict:
        handle = self._get_handle(source)
        if name not in handle:
            return self._empty_features()
        
        grp = handle[name]
        features = {}
        
        for key in ['spherical_keypoints', 'keypoints', 'descriptors', 'scores', 'image_size']:
            if key in grp:
                data = grp[key][...]
                if key in ['spherical_keypoints', 'keypoints', 'descriptors', 'scores']:
                    data = data[:self.max_keypoints]
                features[key] = torch.from_numpy(data).float()
        
        return features
    
    def _empty_features(self) -> Dict:
        return {
            'spherical_keypoints': torch.zeros((0, 2), dtype=torch.float32),
            'keypoints': torch.zeros((0, 2), dtype=torch.float32),
            'descriptors': torch.zeros((0, 64), dtype=torch.float32),
            'scores': torch.zeros((0,), dtype=torch.float32),
            'image_size': torch.tensor([7680, 3840], dtype=torch.long)
        }
    
    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        
        features0 = self._load_features(name0, 'query')
        features1 = self._load_features(name1, 'ref')
        
        return {
            'pair_names': (name0, name1),
            'spherical_kpts0': features0.get('spherical_keypoints', features0.get('keypoints', torch.empty(0, 2))),
            'spherical_kpts1': features1.get('spherical_keypoints', features1.get('keypoints', torch.empty(0, 2))),
            'keypoints0': features0.get('keypoints', torch.empty(0, 2)),
            'keypoints1': features1.get('keypoints', torch.empty(0, 2)),
            'descriptors0': features0.get('descriptors', torch.empty(0, 64)),
            'descriptors1': features1.get('descriptors', torch.empty(0, 64)),
            'scores0': features0.get('scores', torch.empty(0)),
            'scores1': features1.get('scores', torch.empty(0)),
            'image_size0': features0.get('image_size', torch.tensor([7680, 3840])),
            'image_size1': features1.get('image_size', torch.tensor([7680, 3840]))
        }
    
    def __len__(self):
        return len(self.pairs)
    
    def __del__(self):
        for handle in self._handles.values():
            try:
                handle.close()
            except:
                pass


def collate_fn(batch):
    """Efficient collate function that preserves tensor structure"""
    return {
        'pair_names': [item['pair_names'] for item in batch],
        'spherical_kpts0': [item['spherical_kpts0'] for item in batch],
        'spherical_kpts1': [item['spherical_kpts1'] for item in batch],
        'keypoints0': [item['keypoints0'] for item in batch],
        'keypoints1': [item['keypoints1'] for item in batch],
        'descriptors0': [item['descriptors0'] for item in batch],
        'descriptors1': [item['descriptors1'] for item in batch],
        'scores0': [item['scores0'] for item in batch],
        'scores1': [item['scores1'] for item in batch],
        'image_size0': [item['image_size0'] for item in batch],
        'image_size1': [item['image_size1'] for item in batch]
    }


class TensorRTLightGlueMatcher:
    """TensorRT-accelerated LightGlue matcher"""
    
    def __init__(self, engine_path: str, config: PipelineConfig):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available. Install with: pip install tensorrt pycuda")
        
        self.config = config
        self.engine_path = engine_path
        self.device = config.gpu.device
        
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Get binding information
        self.num_bindings = self.engine.num_io_tensors

        self.bindings = [None] * self.num_bindings
        self.binding_shapes = {}
        self.binding_names = {}
        
        for i in range(self.num_bindings):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            
            self.binding_names[name] = i
            self.binding_shapes[name] = shape
            
            if is_input:
                print(f"Input {i}: {name}, shape={shape}, dtype={dtype}")
            else:
                print(f"Output {i}: {name}, shape={shape}, dtype={dtype}")
        
        # Extract dimensions
        kpts0_shape = self.binding_shapes['kpts0']
        self.fixed_batch_size = kpts0_shape[0]
        self.fixed_num_keypoints = kpts0_shape[1]
        self.descriptor_dim = self.binding_shapes['desc0'][2]
        self.stream = cuda.Stream()

        # Allocate device memory for bindings
        self._allocate_buffers()
        
        print(f"TensorRT LightGlue Loaded:")
        print(f"  Engine: {engine_path}")
        print(f"  Fixed batch size: {self.fixed_batch_size}")
        print(f"  Fixed num keypoints: {self.fixed_num_keypoints}")
        print(f"  Descriptor dim: {self.descriptor_dim}")
        self._warmup()
        print("Warmup complete")

    def _warmup(self):
        """Run a dummy inference to warm up the engine"""
        dummy_kpts = np.zeros((self.fixed_batch_size, self.fixed_num_keypoints, 2), dtype=np.float32)
        dummy_desc = np.zeros((self.fixed_batch_size, self.fixed_num_keypoints, self.descriptor_dim), dtype=np.float32)
        
        cuda.memcpy_htod_async(self.device_buffers['kpts0'], dummy_kpts, self.stream)
        cuda.memcpy_htod_async(self.device_buffers['kpts1'], dummy_kpts, self.stream)
        cuda.memcpy_htod_async(self.device_buffers['desc0'], dummy_desc, self.stream)
        cuda.memcpy_htod_async(self.device_buffers['desc1'], dummy_desc, self.stream)
        
        for name in self.binding_names:
            self.context.set_tensor_address(name, int(self.device_buffers[name]))
        
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.stream.synchronize()
        
    def _allocate_buffers(self):
        """Allocate GPU memory for all bindings"""
        self.device_buffers = {}
        self.host_buffers = {}
        
        for name, idx in self.binding_names.items():
            shape = self.binding_shapes[name]
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            # Allocate host memory
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            self.host_buffers[name] = host_mem
            
            # Allocate device memory
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.device_buffers[name] = device_mem
            self.bindings[idx] = int(device_mem)
    
    def pad_to_fixed_size(self, tensor: torch.Tensor, target_size: int, dim: int = 0) -> torch.Tensor:
        """Pad tensor to fixed size along specified dimension"""
        current_size = tensor.shape[dim]
        if current_size >= target_size:
            return tensor[:target_size] if dim == 0 else tensor[:, :target_size]
        
        pad_size = target_size - current_size
        if dim == 0:
            padding = torch.zeros((pad_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=0)
        elif dim == 1:
            padding = torch.zeros((tensor.shape[0], pad_size, *tensor.shape[2:]), dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=1)
        else:
            raise ValueError(f"Unsupported padding dimension: {dim}")
        
    def batch_match(self, outputs0: List[Dict], outputs1: List[Dict], min_conf: float = 0.1) -> List[Dict]:
        """Batch match using TensorRT engine with proper synchronization"""
        if len(outputs0) == 0:
            return []
        
        valid_pairs = [(i, o0, o1) for i, (o0, o1) in enumerate(zip(outputs0, outputs1)) 
                    if o0 is not None and o1 is not None]
        
        if len(valid_pairs) == 0:
            return [{'mkpts_0': np.empty((0, 2)), 'mkpts_1': np.empty((0, 2)),
                    'idxs_0': np.array([]), 'idxs_1': np.array([])} 
                for _ in range(len(outputs0))]
        
        all_results = [None] * len(outputs0)
        
        for chunk_start in range(0, len(valid_pairs), self.fixed_batch_size):
            chunk_end = min(chunk_start + self.fixed_batch_size, len(valid_pairs))
            chunk = valid_pairs[chunk_start:chunk_end]
            
            # Prepare batch tensors
            batch_kpts0 = []
            batch_kpts1 = []
            batch_desc0 = []
            batch_desc1 = []
            original_sizes0 = []
            original_sizes1 = []
            
            for _, o0, o1 in chunk:
                n0 = o0['keypoints'].shape[0]
                n1 = o1['keypoints'].shape[0]
                original_sizes0.append(n0)
                original_sizes1.append(n1)
                
                kpts0_padded = self.pad_to_fixed_size(o0['keypoints'], self.fixed_num_keypoints, dim=0)
                kpts1_padded = self.pad_to_fixed_size(o1['keypoints'], self.fixed_num_keypoints, dim=0)
                desc0_padded = self.pad_to_fixed_size(o0['descriptors'], self.fixed_num_keypoints, dim=0)
                desc1_padded = self.pad_to_fixed_size(o1['descriptors'], self.fixed_num_keypoints, dim=0)
                
                batch_kpts0.append(kpts0_padded)
                batch_kpts1.append(kpts1_padded)
                batch_desc0.append(desc0_padded)
                batch_desc1.append(desc1_padded)
            
            # Pad batch to fixed batch size
            if len(batch_kpts0) < self.fixed_batch_size:
                device = batch_kpts0[0].device
                while len(batch_kpts0) < self.fixed_batch_size:
                    batch_kpts0.append(torch.zeros((self.fixed_num_keypoints, 2), dtype=torch.float32, device=device))
                    batch_kpts1.append(torch.zeros((self.fixed_num_keypoints, 2), dtype=torch.float32, device=device))
                    batch_desc0.append(torch.zeros((self.fixed_num_keypoints, self.descriptor_dim), dtype=torch.float32, device=device))
                    batch_desc1.append(torch.zeros((self.fixed_num_keypoints, self.descriptor_dim), dtype=torch.float32, device=device))
            
            # Stack and ensure contiguous
            kpts0_batch = np.ascontiguousarray(torch.stack(batch_kpts0, dim=0).cpu().numpy().astype(np.float32))
            kpts1_batch = np.ascontiguousarray(torch.stack(batch_kpts1, dim=0).cpu().numpy().astype(np.float32))
            desc0_batch = np.ascontiguousarray(torch.stack(batch_desc0, dim=0).cpu().numpy().astype(np.float32))
            desc1_batch = np.ascontiguousarray(torch.stack(batch_desc1, dim=0).cpu().numpy().astype(np.float32))
            
            try:
                # Copy inputs to GPU (async)
                cuda.memcpy_htod_async(self.device_buffers['kpts0'], kpts0_batch, self.stream)
                cuda.memcpy_htod_async(self.device_buffers['kpts1'], kpts1_batch, self.stream)
                cuda.memcpy_htod_async(self.device_buffers['desc0'], desc0_batch, self.stream)
                cuda.memcpy_htod_async(self.device_buffers['desc1'], desc1_batch, self.stream)
                
                # Set tensor addresses
                for name in self.binding_names:
                    self.context.set_tensor_address(name, int(self.device_buffers[name]))
                
                # Execute inference (async)
                success = self.context.execute_async_v3(stream_handle=self.stream.handle)
                if not success:
                    raise RuntimeError("TensorRT execution failed")
                
                # Copy outputs from GPU (async)
                cuda.memcpy_dtoh_async(self.host_buffers['matches0'], self.device_buffers['matches0'], self.stream)
                cuda.memcpy_dtoh_async(self.host_buffers['mscores0'], self.device_buffers['mscores0'], self.stream)
                
                # Wait for all operations to complete
                self.stream.synchronize()
                
                # Reshape outputs
                matches0 = self.host_buffers['matches0'].reshape(self.fixed_batch_size, self.fixed_num_keypoints)
                mscores0 = self.host_buffers['mscores0'].reshape(self.fixed_batch_size, self.fixed_num_keypoints)
                
            except Exception as e:
                print(f"TensorRT inference error: {e}")
                import traceback
                traceback.print_exc()
                for idx, o0, o1 in chunk:
                    all_results[idx] = {
                        'mkpts_0': np.empty((0, 2)),
                        'mkpts_1': np.empty((0, 2)),
                        'idxs_0': np.array([]),
                        'idxs_1': np.array([])
                    }
                continue
            
            # Process results for each pair
            for i, (orig_idx, o0, o1) in enumerate(chunk):
                n0 = original_sizes0[i]
                n1 = original_sizes1[i]
                
                pair_matches0 = matches0[i, :n0]
                pair_scores0 = mscores0[i, :n0]
                
                valid_mask = (pair_matches0 >= 0) & (pair_matches0 < n1) & (pair_scores0 >= min_conf)
                
                if valid_mask.sum() == 0:
                    all_results[orig_idx] = {
                        'mkpts_0': np.empty((0, 2)),
                        'mkpts_1': np.empty((0, 2)),
                        'idxs_0': np.array([]),
                        'idxs_1': np.array([])
                    }
                    continue
                
                idxs_0 = np.where(valid_mask)[0]
                idxs_1 = pair_matches0[valid_mask].astype(np.int32)
                
                mkpts_0 = o0['keypoints'][idxs_0].cpu().numpy()
                mkpts_1 = o1['keypoints'][idxs_1].cpu().numpy()
                
                all_results[orig_idx] = {
                    'mkpts_0': mkpts_0,
                    'mkpts_1': mkpts_1,
                    'idxs_0': idxs_0,
                    'idxs_1': idxs_1
                }
        
        # Fill None results
        for i in range(len(all_results)):
            if all_results[i] is None:
                all_results[i] = {
                    'mkpts_0': np.empty((0, 2)),
                    'mkpts_1': np.empty((0, 2)),
                    'idxs_0': np.array([]),
                    'idxs_1': np.array([])
                }
        
        return all_results    
    def __del__(self):
        """Cleanup GPU memory"""
        try:
            for buf in self.device_buffers.values():
                buf.free()
        except:
            pass


class ONNXLightGlueMatcher:
    """ONNX fallback matcher when TensorRT is not available"""
    
    def __init__(self, onnx_path: str, config: PipelineConfig):
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNXRuntime not available. Install with: pip install onnxruntime-gpu")
        
        self.config = config
        self.onnx_path = onnx_path
        self.device = config.gpu.device
        
        # Setup ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        
        # Configure providers
        providers = []
        if torch.cuda.is_available() and 'cuda' in self.device:
            device_id = int(self.device.split(':')[1]) if ':' in self.device else 0
            providers.append(('CUDAExecutionProvider', {
                'device_id': device_id,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': int(config.gpu.max_memory_gb * 1024 * 1024 * 1024),
                'cudnn_conv_algo_search': 'DEFAULT',
            }))
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get model metadata
        self.input_meta = {inp.name: inp for inp in self.session.get_inputs()}
        self.output_meta = {out.name: out for out in self.session.get_outputs()}
        
        # Extract fixed dimensions from model
        kpts0_shape = self.input_meta['kpts0'].shape

# Check if the batch dimension is dynamic (a string)
        if isinstance(kpts0_shape[0], str):
            print(f"ONNX model has a dynamic batch size ('{kpts0_shape[0]}'). Using batch size from config: {config.batching.matching_batch_size}")
            self.fixed_batch_size = config.batching.matching_batch_size
        else:
            self.fixed_batch_size = kpts0_shape[0]

        self.fixed_num_keypoints = kpts0_shape[1]
        self.descriptor_dim = self.input_meta['desc0'].shape[2]
        
        print(f"ONNX LightGlue Loaded:")
        print(f"  Model: {onnx_path}")
        print(f"  Providers: {self.session.get_providers()}")
        print(f"  Fixed batch size: {self.fixed_batch_size}")
        print(f"  Fixed num keypoints: {self.fixed_num_keypoints}")
        print(f"  Descriptor dim: {self.descriptor_dim}")
    
    def pad_to_fixed_size(self, tensor: torch.Tensor, target_size: int, dim: int = 0) -> torch.Tensor:
        """Pad tensor to fixed size along specified dimension"""
        current_size = tensor.shape[dim]
        if current_size >= target_size:
            return tensor[:target_size] if dim == 0 else tensor[:, :target_size]
        
        pad_size = target_size - current_size
        if dim == 0:
            padding = torch.zeros((pad_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=0)
        elif dim == 1:
            padding = torch.zeros((tensor.shape[0], pad_size, *tensor.shape[2:]), dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding], dim=1)
        else:
            raise ValueError(f"Unsupported padding dimension: {dim}")
    
    def batch_match(self, outputs0: List[Dict], outputs1: List[Dict], min_conf: float = 0.1) -> List[Dict]:
        """Batch match using ONNX model (same implementation as before)"""
        if len(outputs0) == 0:
            return []
        
        valid_pairs = [(i, o0, o1) for i, (o0, o1) in enumerate(zip(outputs0, outputs1)) 
                       if o0 is not None and o1 is not None]
        
        if len(valid_pairs) == 0:
            return [{'mkpts_0': np.empty((0, 2)), 'mkpts_1': np.empty((0, 2)),
                    'idxs_0': np.array([]), 'idxs_1': np.array([])} 
                   for _ in range(len(outputs0))]
        
        all_results = [None] * len(outputs0)
        
        for chunk_start in range(0, len(valid_pairs), self.fixed_batch_size):
            chunk_end = min(chunk_start + self.fixed_batch_size, len(valid_pairs))
            chunk = valid_pairs[chunk_start:chunk_end]
            
            batch_kpts0 = []
            batch_kpts1 = []
            batch_desc0 = []
            batch_desc1 = []
            original_sizes0 = []
            original_sizes1 = []
            
            for _, o0, o1 in chunk:
                n0 = o0['keypoints'].shape[0]
                n1 = o1['keypoints'].shape[0]
                original_sizes0.append(n0)
                original_sizes1.append(n1)
                
                kpts0_padded = self.pad_to_fixed_size(o0['keypoints'], self.fixed_num_keypoints, dim=0)
                kpts1_padded = self.pad_to_fixed_size(o1['keypoints'], self.fixed_num_keypoints, dim=0)
                desc0_padded = self.pad_to_fixed_size(o0['descriptors'], self.fixed_num_keypoints, dim=0)
                desc1_padded = self.pad_to_fixed_size(o1['descriptors'], self.fixed_num_keypoints, dim=0)
                
                batch_kpts0.append(kpts0_padded)
                batch_kpts1.append(kpts1_padded)
                batch_desc0.append(desc0_padded)
                batch_desc1.append(desc1_padded)
            
            while len(batch_kpts0) < self.fixed_batch_size:
                batch_kpts0.append(torch.zeros((self.fixed_num_keypoints, 2), dtype=torch.float32))
                batch_kpts1.append(torch.zeros((self.fixed_num_keypoints, 2), dtype=torch.float32))
                batch_desc0.append(torch.zeros((self.fixed_num_keypoints, self.descriptor_dim), dtype=torch.float32))
                batch_desc1.append(torch.zeros((self.fixed_num_keypoints, self.descriptor_dim), dtype=torch.float32))
            
            kpts0_batch = torch.stack(batch_kpts0, dim=0).cpu().numpy().astype(np.float32)
            kpts1_batch = torch.stack(batch_kpts1, dim=0).cpu().numpy().astype(np.float32)
            desc0_batch = torch.stack(batch_desc0, dim=0).cpu().numpy().astype(np.float32)
            desc1_batch = torch.stack(batch_desc1, dim=0).cpu().numpy().astype(np.float32)
            
            try:
                outputs = self.session.run(None, {
                    'kpts0': kpts0_batch,
                    'kpts1': kpts1_batch,
                    'desc0': desc0_batch,
                    'desc1': desc1_batch,
                })
                
                matches0 = outputs[0]
                mscores0 = outputs[1]
                
            except Exception as e:
                print(f"ONNX inference error: {e}")
                for idx, o0, o1 in chunk:
                    all_results[idx] = {
                        'mkpts_0': np.empty((0, 2)),
                        'mkpts_1': np.empty((0, 2)),
                        'idxs_0': np.array([]),
                        'idxs_1': np.array([])
                    }
                continue
            
            for i, (orig_idx, o0, o1) in enumerate(chunk):
                n0 = original_sizes0[i]
                n1 = original_sizes1[i]
                
                pair_matches0 = matches0[i, :n0]
                pair_scores0 = mscores0[i, :n0]
                
                valid_mask = (pair_matches0 >= 0) & (pair_matches0 < n1) & (pair_scores0 >= min_conf)
                
                if valid_mask.sum() == 0:
                    all_results[orig_idx] = {
                        'mkpts_0': np.empty((0, 2)),
                        'mkpts_1': np.empty((0, 2)),
                        'idxs_0': np.array([]),
                        'idxs_1': np.array([])
                    }
                    continue
                
                idxs_0 = np.where(valid_mask)[0]
                idxs_1 = pair_matches0[valid_mask].astype(np.int32)
                
                mkpts_0 = o0['keypoints'][idxs_0].cpu().numpy()
                mkpts_1 = o1['keypoints'][idxs_1].cpu().numpy()
                
                all_results[orig_idx] = {
                    'mkpts_0': mkpts_0,
                    'mkpts_1': mkpts_1,
                    'idxs_0': idxs_0,
                    'idxs_1': idxs_1
                }
        
        for i in range(len(all_results)):
            if all_results[i] is None:
                all_results[i] = {
                    'mkpts_0': np.empty((0, 2)),
                    'mkpts_1': np.empty((0, 2)),
                    'idxs_0': np.array([]),
                    'idxs_1': np.array([])
                }
        
        return all_results


class OptimizedXFeatMatcher:
    """Optimized XFeat matcher with TensorRT/ONNX LightGlue"""
    
    def __init__(self, xfeat_model, matcher, config: PipelineConfig):
        self.config = config
        self.device = torch.device(config.gpu.device if torch.cuda.is_available() else 'cpu')
        self.xfeat = xfeat_model.to(self.device)
        self.xfeat.eval()
        self.matcher = matcher
        
        self.use_fp16 = config.gpu.use_half_precision and self.device.type == 'cuda'
        
        if config.gpu.compile_models and hasattr(torch, 'compile'):
            try:
                self.xfeat = torch.compile(self.xfeat, mode='reduce-overhead')
                print("XFeat model compiled")
            except Exception as e:
                print(f"Model compilation failed: {e}")
        
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
    
    @torch.no_grad()
    def match_batch(self, batch_data: Dict) -> List[Dict]:
        """Process a batch of image pairs efficiently"""
        
        batch_size = len(batch_data['pair_names'])
        
        outputs0 = []
        outputs1 = []
        
        dtype = torch.float16 if self.use_fp16 else torch.float32
        
        for i in range(batch_size):
            kpts0 = batch_data['spherical_kpts0'][i]
            kpts1 = batch_data['spherical_kpts1'][i]
            
            if len(kpts0) == 0 or len(kpts1) == 0:
                outputs0.append(None)
                outputs1.append(None)
                continue
            
            output0 = {
                'keypoints': kpts0.to(self.device, dtype=dtype),
                'descriptors': batch_data['descriptors0'][i].to(self.device, dtype=dtype),
                'scores': batch_data['scores0'][i].to(self.device, dtype=dtype),
                'image_size': tuple(batch_data['image_size0'][i].cpu().numpy())
            }
            output1 = {
                'keypoints': kpts1.to(self.device, dtype=dtype),
                'descriptors': batch_data['descriptors1'][i].to(self.device, dtype=dtype),
                'scores': batch_data['scores1'][i].to(self.device, dtype=dtype),
                'image_size': tuple(batch_data['image_size1'][i].cpu().numpy())
            }
            outputs0.append(output0)
            outputs1.append(output1)
        
        # Use TensorRT or ONNX matcher
        matches = self.matcher.batch_match(
            outputs0, outputs1, 
            min_conf=self.config.matching.min_confidence
        )
        
        # Convert matches to output format
        results = []
        for i in range(batch_size):
            pair_name = '_'.join(batch_data['pair_names'][i])
            num_kpts0 = len(batch_data['keypoints0'][i])
            
            if outputs0[i] is None or i >= len(matches):
                results.append({
                    'pair_name': pair_name,
                    'matches0': torch.full((num_kpts0,), -1, dtype=torch.long),
                    'matching_scores0': torch.zeros(num_kpts0, dtype=torch.float32)
                })
                continue
            
            match_result = matches[i]
            
            if 'idxs_0' in match_result and len(match_result['idxs_0']) > 0:
                matches0 = torch.full((num_kpts0,), -1, dtype=torch.long)
                scores0 = torch.zeros(num_kpts0, dtype=torch.float32)
                
                idxs_0 = match_result['idxs_0']
                idxs_1 = match_result['idxs_1']
                
                for idx0, idx1 in zip(idxs_0, idxs_1):
                    if 0 <= idx0 < num_kpts0:
                        matches0[idx0] = idx1
                        scores0[idx0] = 1.0
                
                results.append({
                    'pair_name': pair_name,
                    'matches0': matches0,
                    'matching_scores0': scores0
                })
            else:
                results.append({
                    'pair_name': pair_name,
                    'matches0': torch.full((num_kpts0,), -1, dtype=torch.long),
                    'matching_scores0': torch.zeros(num_kpts0, dtype=torch.float32)
                })
        
        return results


class OptimizedWriter:
    """Optimized HDF5 writer with config-based compression"""
    
    def __init__(self, output_path: Path, config: PipelineConfig):
        self.output_path = output_path
        self.config = config
        with h5py.File(self.output_path, 'w') as f:
            pass
    
    def add_results(self, results: List[Dict]):
        self._write_batch(results)
    
    def _write_batch(self, batch: List[Dict]):
        if not batch:
            return
            
        try:
            with h5py.File(self.output_path, 'a') as f:
                for result in batch:
                    pair_name = result['pair_name']
                    
                    if pair_name in f:
                        del f[pair_name]
                    
                    grp = f.create_group(pair_name)
                    
                    grp.create_dataset(
                        'matches0',
                        data=result['matches0'].cpu().numpy().astype(np.int32),
                        compression='gzip',
                        compression_opts=self.config.io.compression_level
                    )
                    grp.create_dataset(
                        'matching_scores0',
                        data=result['matching_scores0'].cpu().numpy().astype(np.float32),
                        compression='gzip',
                        compression_opts=self.config.io.compression_level
                    )
        except Exception as e:
            print(f"Write error: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        pass


def optimized_match_from_paths(
    pairs_path: Path,
    feature_path_q: Path,
    feature_path_r: Path,
    output_path: Path,
    config: PipelineConfig
) -> Path:
    """Main optimized matching pipeline with TensorRT/ONNX LightGlue"""
    
    use_tensorrt = config.gpu.use_tensorrt and TRT_AVAILABLE
    
    print(f"\nOptimized XFeat Matching Pipeline")
    print(f"Backend: {'TensorRT' if use_tensorrt else 'ONNX Runtime'}")
    print(f"Configuration:")
    print(f"  Batch size: {config.batching.matching_batch_size}")
    print(f"  Workers: {config.batching.num_workers}")
    print(f"  Max keypoints: {config.features.max_keypoints}")
    print(f"  FP16: {config.gpu.use_half_precision}")
    print(f"  Min confidence: {config.matching.min_confidence}")
    
    # Load pairs
    from hloc.utils.parsers import parse_retrieval
    pairs_dict = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs_dict.items() for r in rs]
    print(f"Processing {len(pairs)} pairs")
    
    # Initialize XFeat model (without LightGlue)
    sys.path.append(config.models.xfeat_module_path)
    from modules.xfeat import XFeat
    xfeat_model = XFeat(
        weights=config.models.xfeat_weights,
        top_k=config.features.max_keypoints,
        detection_threshold=config.features.detection_threshold,
        lightglue_checkpoint=None
    )
    
    # Initialize matcher (TensorRT or ONNX)
    # if use_tensorrt:
    if not config.models.lightglue_trt_path:
        raise ValueError("TensorRT enabled but lightglue_trt_path not specified in config")
    print(f"  TensorRT Engine: {config.models.lightglue_trt_path}")
    matcher = TensorRTLightGlueMatcher(config.models.lightglue_trt_path, config)
    # else:
    #     if not config.models.lightglue_onnx_path:
    #         raise ValueError("ONNX path not specified in config")
    #     print(f"  ONNX Model: {config.models.lightglue_onnx_path}")
    #     matcher = ONNXLightGlueMatcher(config.models.lightglue_onnx_path, config)
    
    # Create dataset and dataloader
    dataset = OptimizedBatchDataset(
        pairs, feature_path_q, feature_path_r, config
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batching.matching_batch_size,
        num_workers=config.batching.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if config.batching.num_workers > 0 else False,
        prefetch_factor=config.batching.prefetch_batches if config.batching.num_workers > 0 else None,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    # Initialize matcher and writer
    xfeat_matcher = OptimizedXFeatMatcher(xfeat_model, matcher, config)
    writer = OptimizedWriter(output_path, config)
    
    # Process batches
    try:
        with tqdm(total=len(pairs), desc="Matching", 
                 disable=not config.logging.progress_bars) as pbar:
            for batch_idx, batch_data in enumerate(dataloader):
                results = xfeat_matcher.match_batch(batch_data)
                writer.add_results(results)
                pbar.update(len(results))
                
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        writer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Matching complete. Output saved to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized XFeat Matching with TensorRT/ONNX")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features_ref", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    
    # Optional overrides
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--matching_batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--max_keypoints", type=int)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--min_conf", type=float)
    parser.add_argument("--use_tensorrt", action="store_true", help="Use TensorRT instead of ONNX")
    parser.add_argument("--trt_path", type=str, help="Override TensorRT engine path")
    parser.add_argument("--onnx_path", type=str, help="Override ONNX model path")
    
    args = parser.parse_args()
    
    # Load config with CLI overrides
    config = load_config(args.config, args)
    
    # Override paths if specified
    if args.trt_path:
        config.models.lightglue_trt_path = args.trt_path
    if args.onnx_path:
        config.models.lightglue_onnx_path = args.onnx_path
    
    # Use matching batch size if specified
    if args.matching_batch_size:
        config.batching.matching_batch_size = args.matching_batch_size
    
    features_ref = args.features_ref if args.features_ref else args.features
    
    optimized_match_from_paths(
        pairs_path=args.pairs,
        feature_path_q=args.features,
        feature_path_r=features_ref,
        output_path=args.output,
        config=config
    )