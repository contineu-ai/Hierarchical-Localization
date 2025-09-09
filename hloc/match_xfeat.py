import torch
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - using CPU spherical coordinate processing")


class OptimizedSphericalMatcher:
    """Memory-efficient spherical coordinate matcher with proper batching"""
    
    def __init__(self, device='cuda', use_fp16=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Pre-allocate buffers for reuse
        self._buffer_pool = {}
        
    def spherical_distance_batch_gpu(self, coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated spherical distance calculation using PyTorch"""
        # coords: [N, 2] where columns are [phi, theta]
        phi1, theta1 = coords1[:, 0], coords1[:, 1]
        phi2, theta2 = coords2[:, 0], coords2[:, 1]
        
        dphi = phi2 - phi1
        dtheta = theta2 - theta1
        
        # Haversine formula
        a = torch.sin(dtheta/2)**2 + torch.cos(theta1) * torch.cos(theta2) * torch.sin(dphi/2)**2
        c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0, 1)))
        
        return c
    
    def find_spherical_matches(self, spherical_kpts0: torch.Tensor, spherical_kpts1: torch.Tensor,
                              matched_indices0: torch.Tensor, matched_indices1: torch.Tensor,
                              distance_threshold: float = 0.001) -> Tuple[torch.Tensor, torch.Tensor]:
        """Efficiently find corresponding spherical keypoints for matched features"""
        
        if len(matched_indices0) == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        # Move to GPU if not already
        spherical_kpts0 = spherical_kpts0.to(self.device, dtype=self.dtype)
        spherical_kpts1 = spherical_kpts1.to(self.device, dtype=self.dtype)
        
        # Use matched indices directly if they correspond to keypoint indices
        valid_mask = (matched_indices0 < len(spherical_kpts0)) & (matched_indices1 < len(spherical_kpts1))
        valid_idx0 = matched_indices0[valid_mask]
        valid_idx1 = matched_indices1[valid_mask]
        
        if len(valid_idx0) > 0:
            # Verify matches are within distance threshold
            matched_sph0 = spherical_kpts0[valid_idx0]
            matched_sph1 = spherical_kpts1[valid_idx1]
            distances = self.spherical_distance_batch_gpu(matched_sph0, matched_sph1)
            
            # Keep only matches within threshold
            good_matches = distances < distance_threshold
            return valid_idx0[good_matches], valid_idx1[good_matches]
        
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)


class OptimizedBatchDataset(Dataset):
    """Optimized dataset with better memory management and caching"""
    
    def __init__(self, pairs, feature_path_q, feature_path_r, max_keypoints=4096, cache_size=100):
        self.pairs = pairs
        self.feature_paths = {'query': feature_path_q, 'ref': feature_path_r}
        self.max_keypoints = max_keypoints
        
        # LRU cache for frequently accessed features
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.cache_size = cache_size
        
        # File handles (opened lazily per worker)
        self._handles = {}
        
    def _get_handle(self, key):
        """Get or create file handle for worker"""
        if key not in self._handles:
            self._handles[key] = h5py.File(self.feature_paths[key], 'r', swmr=True)
        return self._handles[key]
    
    def _load_features(self, name: str, source: str) -> Dict:
        """Load features with caching"""
        cache_key = f"{source}_{name}"
        
        # Check cache
        if cache_key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key].copy()
        
        handle = self._get_handle(source)
        if name not in handle:
            return self._empty_features()
        
        grp = handle[name]
        features = {}
        
        # Load all feature data at once
        for key in ['spherical_keypoints', 'keypoints', 'descriptors', 'scores', 'image_size']:
            if key in grp:
                data = np.array(grp[key])
                # Limit keypoints if needed
                if 'keypoints' in key or key in ['descriptors', 'scores']:
                    data = data[:self.max_keypoints]
                features[key] = torch.from_numpy(data).float()
        
        # Add to cache and maintain size
        self.cache[cache_key] = features
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        
        return features.copy()
    
    def _empty_features(self) -> Dict:
        """Return empty feature dictionary"""
        return {
            'spherical_keypoints': torch.zeros((0, 2), dtype=torch.float32),
            'keypoints': torch.zeros((0, 2), dtype=torch.float32),
            'descriptors': torch.zeros((0, 64), dtype=torch.float32),
            'scores': torch.zeros((0,), dtype=torch.float32),
            'image_size': torch.tensor([7680, 3840], dtype=torch.long)
        }
    
    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        
        # Load features
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


class OptimizedXFeatMatcher:
    """Optimized XFeat matcher with proper batching and memory management"""
    
    def __init__(self, xfeat_model, device='cuda', use_fp16=True, compile_model=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.xfeat = xfeat_model.to(self.device)
        self.xfeat.eval()
        
        # Enable optimizations
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.xfeat = torch.compile(self.xfeat, mode='reduce-overhead')
                print("XFeat model compiled")
            except Exception as e:
                print(f"Model compilation failed: {e}")
        
        # Initialize spherical matcher
        self.spherical_matcher = OptimizedSphericalMatcher(device, use_fp16)
        
        # Enable CUDA optimizations
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
    
    @torch.no_grad()
    def match_batch(self, batch_data: Dict) -> List[Dict]:
        """Process a batch of image pairs efficiently"""
        
        # Extract batch components
        batch_size = len(batch_data['pair_names'])
        
        # Prepare XFeat inputs - process all at once
        outputs0 = []
        outputs1 = []
        
        for i in range(batch_size):
            # Create feature dictionaries for XFeat
            output0 = {
                'keypoints': batch_data['spherical_kpts0'][i].to(self.device, self.dtype),
                'descriptors': batch_data['descriptors0'][i].to(self.device, self.dtype),
                'scores': batch_data['scores0'][i].to(self.device, self.dtype),
                'image_size': tuple(batch_data['image_size0'][i].cpu().numpy())
            }
            output1 = {
                'keypoints': batch_data['spherical_kpts1'][i].to(self.device, self.dtype),
                'descriptors': batch_data['descriptors1'][i].to(self.device, self.dtype),
                'scores': batch_data['scores1'][i].to(self.device, self.dtype),
                'image_size': tuple(batch_data['image_size1'][i].cpu().numpy())
            }
            outputs0.append(output0)
            outputs1.append(output1)
        
        # Batch matching with XFeat/LightGlue
        try:
            if hasattr(self.xfeat, 'batch_match_lighterglue'):
                # Use batched LightGlue matching
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    matches = self.xfeat.batch_match_lighterglue(outputs0, outputs1)
            else:
                # Fallback to individual matching
                matches = []
                for out0, out1 in zip(outputs0, outputs1):
                    with torch.cuda.amp.autocast(enabled=self.use_fp16):
                        match_result = self._match_single(out0, out1)
                    matches.append(match_result)
        except Exception as e:
            print(f"Matching error: {e}")
            matches = [{'mkpts_0': np.empty((0, 2)), 'mkpts_1': np.empty((0, 2))} 
                      for _ in range(batch_size)]
        
        # Process results
        results = []
        for i in range(batch_size):
            pair_name = '_'.join(batch_data['pair_names'][i])
            
            if i < len(matches) and matches[i] is not None:
                # Extract matched keypoints
                mkpts0 = matches[i].get('mkpts_0', np.empty((0, 2)))
                mkpts1 = matches[i].get('mkpts_1', np.empty((0, 2)))
                
                # Convert to match indices for the original keypoints
                num_kpts0 = len(batch_data['keypoints0'][i])
                num_kpts1 = len(batch_data['keypoints1'][i])
                
                # Create match arrays
                matches0 = torch.full((num_kpts0,), -1, dtype=torch.long)
                scores0 = torch.zeros(num_kpts0, dtype=torch.float32)
                
                if len(mkpts0) > 0:
                    # Find correspondences in original keypoint arrays
                    # This is simplified - you may need more sophisticated matching
                    # based on your specific coordinate system
                    matched_idx0 = matches[i].get('idxs_0', np.arange(len(mkpts0)))
                    matched_idx1 = matches[i].get('idxs_1', np.arange(len(mkpts1)))
                    
                    # Assign matches
                    for idx0, idx1 in zip(matched_idx0, matched_idx1):
                        if idx0 < num_kpts0 and idx1 < num_kpts1:
                            matches0[idx0] = idx1
                            scores0[idx0] = 1.0
                
                results.append({
                    'pair_name': pair_name,
                    'matches0': matches0,
                    'matching_scores0': scores0
                })
            else:
                # Empty result
                results.append({
                    'pair_name': pair_name,
                    'matches0': torch.full((len(batch_data['keypoints0'][i]),), -1, dtype=torch.long),
                    'matching_scores0': torch.zeros(len(batch_data['keypoints0'][i]), dtype=torch.float32)
                })
        
        return results
    
    def _match_single(self, output0: Dict, output1: Dict) -> Dict:
        """Fallback single pair matching"""
        # Simple nearest neighbor matching
        desc0 = output0['descriptors']
        desc1 = output1['descriptors']
        
        if len(desc0) == 0 or len(desc1) == 0:
            return {'mkpts_0': np.empty((0, 2)), 'mkpts_1': np.empty((0, 2))}
        
        # Compute similarity matrix
        sim = torch.matmul(desc0, desc1.t())
        
        # Mutual nearest neighbor
        nn01 = sim.argmax(dim=1)
        nn10 = sim.argmax(dim=0)
        
        # Find mutual matches
        mutual = nn10[nn01] == torch.arange(len(nn01), device=sim.device)
        
        # Get matched keypoints
        idx0 = torch.where(mutual)[0]
        idx1 = nn01[mutual]
        
        mkpts0 = output0['keypoints'][idx0].cpu().numpy()
        mkpts1 = output1['keypoints'][idx1].cpu().numpy()
        
        return {
            'mkpts_0': mkpts0,
            'mkpts_1': mkpts1,
            'idxs_0': idx0.cpu().numpy(),
            'idxs_1': idx1.cpu().numpy()
        }


class OptimizedWriter:
    """Optimized HDF5 writer with batching and async I/O"""
    
    def __init__(self, output_path: Path, batch_size=128, num_workers=4):
        self.output_path = output_path
        self.batch_size = batch_size
        self.buffer = []
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures = []
        
    def add_results(self, results: List[Dict]):
        """Add results to buffer"""
        self.buffer.extend(results)
        
        # Write when buffer is full
        if len(self.buffer) >= self.batch_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Write buffered results"""
        if not self.buffer:
            return
        
        # Submit write task
        write_batch = self.buffer[:self.batch_size]
        self.buffer = self.buffer[self.batch_size:]
        
        future = self.executor.submit(self._write_batch, write_batch)
        self.futures.append(future)
    
    def _write_batch(self, batch: List[Dict]):
        """Write a batch of results"""
        try:
            with h5py.File(self.output_path, 'a') as f:
                for result in batch:
                    pair_name = result['pair_name']
                    
                    # Remove existing group if present
                    if pair_name in f:
                        del f[pair_name]
                    
                    # Create new group
                    grp = f.create_group(pair_name)
                    
                    # Write data with compression
                    grp.create_dataset('matches0', 
                                      data=result['matches0'].cpu().numpy().astype(np.int32),
                                      compression='gzip', compression_opts=6)
                    grp.create_dataset('matching_scores0',
                                      data=result['matching_scores0'].cpu().numpy().astype(np.float32),
                                      compression='gzip', compression_opts=6)
        except Exception as e:
            print(f"Write error: {e}")
    
    def close(self):
        """Flush remaining data and close"""
        # Write remaining buffer
        while self.buffer:
            self._flush_buffer()
        
        # Wait for all writes to complete
        for future in self.futures:
            try:
                future.result(timeout=30)
            except Exception as e:
                print(f"Write future error: {e}")
        
        self.executor.shutdown(wait=True)


def optimized_match_from_paths(
    pairs_path: Path,
    feature_path_q: Path,
    feature_path_r: Path,
    output_path: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    max_keypoints: int = 4096,
    use_fp16: bool = True,
    compile_model: bool = True
) -> Path:
    """Main optimized matching pipeline"""
    
    print(f"Optimized XFeat Matching Pipeline")
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Max keypoints: {max_keypoints}")
    print(f"  FP16: {use_fp16}")
    print(f"  Model compilation: {compile_model}")
    
    # Load pairs
    from hloc.utils.parsers import parse_retrieval
    pairs_dict = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs_dict.items() for r in rs]
    print(f"Processing {len(pairs)} pairs")
    
    # Initialize XFeat model
    import sys
    sys.path.append(str(Path(__file__).parent / "../../Xfeat"))
    from modules.xfeat import XFeat
    xfeat_model = XFeat()
    
    # Create dataset and dataloader
    dataset = OptimizedBatchDataset(
        pairs, feature_path_q, feature_path_r,
        max_keypoints=max_keypoints, cache_size=100
    )
    
    # Optimize dataloader settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
        # Custom collate to handle variable sizes
        collate_fn=lambda batch: {
            key: [item[key] for item in batch]
            for key in batch[0].keys()
        }
    )
    
    # Initialize matcher and writer
    matcher = OptimizedXFeatMatcher(
        xfeat_model, 
        device='cuda',
        use_fp16=use_fp16,
        compile_model=compile_model
    )
    writer = OptimizedWriter(output_path, batch_size=128, num_workers=4)
    
    # Process batches
    try:
        with tqdm(total=len(pairs), desc="Matching") as pbar:
            for batch_data in dataloader:
                # Process batch
                results = matcher.match_batch(batch_data)
                
                # Write results
                writer.add_results(results)
                
                # Update progress
                pbar.update(len(results))
                
                # Memory management
                if pbar.n % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    finally:
        # Cleanup
        writer.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Matching complete. Output saved to {output_path}")
    return output_path


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized XFeat Matching")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features_ref", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_keypoints", type=int, default=4096)
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--no_compile", action="store_true")
    
    args = parser.parse_args()
    
    features_ref = args.features_ref if args.features_ref else args.features
    
    optimized_match_from_paths(
        pairs_path=args.pairs,
        feature_path_q=args.features,
        feature_path_r=features_ref,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_keypoints=args.max_keypoints,
        use_fp16=not args.no_fp16,
        compile_model=not args.no_compile
    )