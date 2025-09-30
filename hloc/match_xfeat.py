import torch
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc
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


class OptimizedBatchDataset(Dataset):
    """Optimized dataset with better memory management and minimal copying"""
    
    def __init__(self, pairs, feature_path_q, feature_path_r, max_keypoints=4096):
        self.pairs = pairs
        self.feature_paths = {'query': feature_path_q, 'ref': feature_path_r}
        self.max_keypoints = max_keypoints
        
        # File handles (opened once per worker, thread-safe with SWMR)
        self._handles = {}
        
    def _get_handle(self, key):
        """Get or create file handle for worker"""
        if key not in self._handles:
            self._handles[key] = h5py.File(self.feature_paths[key], 'r', swmr=True)
        return self._handles[key]
    
    def _load_features(self, name: str, source: str) -> Dict:
        """Load features efficiently without unnecessary copies"""
        handle = self._get_handle(source)
        if name not in handle:
            return self._empty_features()
        
        grp = handle[name]
        features = {}
        
        # Load data as tensors directly, limit keypoints
        for key in ['spherical_keypoints', 'keypoints', 'descriptors', 'scores', 'image_size']:
            if key in grp:
                data = grp[key][...]  # Load directly as numpy array
                # Limit keypoints if needed
                if key in ['spherical_keypoints', 'keypoints', 'descriptors', 'scores']:
                    data = data[:self.max_keypoints]
                features[key] = torch.from_numpy(data).float()
        
        return features
    
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
        
        # Load features without caching (let OS handle file caching)
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
    # Stack tensors where possible, keep variable-length as lists
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


class OptimizedXFeatMatcher:
    """Optimized XFeat matcher with true batch processing"""
    
    def __init__(self, xfeat_model, device='cuda', use_fp16=True, compile_model=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.xfeat = xfeat_model.to(self.device)
        self.xfeat.eval()
        
        # Enable optimizations
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        
        # Note: torch.compile often causes issues with dynamic shapes, disabled by default
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.xfeat = torch.compile(self.xfeat, mode='reduce-overhead')
                print("XFeat model compiled")
            except Exception as e:
                print(f"Model compilation failed: {e}")
        
        # Enable CUDA optimizations
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
    
    @torch.no_grad()
    def match_batch(self, batch_data: Dict) -> List[Dict]:
        """Process a batch of image pairs efficiently using true batch matching"""
        
        batch_size = len(batch_data['pair_names'])
        
        # Prepare XFeat inputs efficiently
        outputs0 = []
        outputs1 = []
        
        # Convert to proper dtype and move to GPU in one go
        dtype = torch.float16 if self.use_fp16 else torch.float32
        
        for i in range(batch_size):
            # Use spherical_keypoints as the primary keypoints for matching
            kpts0 = batch_data['spherical_kpts0'][i]
            kpts1 = batch_data['spherical_kpts1'][i]
            
            # Skip pairs with no keypoints
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
        
        # Filter out None entries
        valid_indices = [i for i in range(batch_size) if outputs0[i] is not None]
        valid_outputs0 = [outputs0[i] for i in valid_indices]
        valid_outputs1 = [outputs1[i] for i in valid_indices]
        
        # Batch matching with XFeat/LightGlue
        matches = []
        if len(valid_outputs0) > 0:
            try:
                if hasattr(self.xfeat, 'batch_match_lighterglue'):
                    # Use batched LightGlue matching - this is the key optimization
                    with torch.cuda.amp.autocast(enabled=self.use_fp16):
                        batch_matches = self.xfeat.batch_match_lighterglue(
                            valid_outputs0, valid_outputs1, min_conf=0.2
                        )
                    matches = batch_matches
                else:
                    # Fallback to individual matching
                    for out0, out1 in zip(valid_outputs0, valid_outputs1):
                        with torch.cuda.amp.autocast(enabled=self.use_fp16):
                            match_result = self._match_single(out0, out1)
                        matches.append(match_result)
            except Exception as e:
                print(f"Matching error: {e}")
                import traceback
                traceback.print_exc()
                matches = [{'mkpts_0': np.empty((0, 2)), 'mkpts_1': np.empty((0, 2)),
                           'idxs_0': np.array([]), 'idxs_1': np.array([])} 
                          for _ in range(len(valid_outputs0))]
        
        # Build results for all pairs (including those skipped)
        results = []
        valid_match_idx = 0
        
        for i in range(batch_size):
            pair_name = '_'.join(batch_data['pair_names'][i])
            num_kpts0 = len(batch_data['keypoints0'][i])
            
            if outputs0[i] is None:
                # Empty result for skipped pairs
                results.append({
                    'pair_name': pair_name,
                    'matches0': torch.full((num_kpts0,), -1, dtype=torch.long),
                    'matching_scores0': torch.zeros(num_kpts0, dtype=torch.float32)
                })
            else:
                # Process valid match
                match_result = matches[valid_match_idx] if valid_match_idx < len(matches) else None
                valid_match_idx += 1
                
                if match_result and 'idxs_0' in match_result and len(match_result['idxs_0']) > 0:
                    # Create match array
                    matches0 = torch.full((num_kpts0,), -1, dtype=torch.long)
                    scores0 = torch.zeros(num_kpts0, dtype=torch.float32)
                    
                    idxs_0 = match_result['idxs_0']
                    idxs_1 = match_result['idxs_1']
                    
                    # Assign matches (indices from batch_match_lighterglue)
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
                    # Empty result
                    results.append({
                        'pair_name': pair_name,
                        'matches0': torch.full((num_kpts0,), -1, dtype=torch.long),
                        'matching_scores0': torch.zeros(num_kpts0, dtype=torch.float32)
                    })
        
        return results
    
    def _match_single(self, output0: Dict, output1: Dict) -> Dict:
        """Fallback single pair matching using mutual nearest neighbors"""
        desc0 = output0['descriptors']
        desc1 = output1['descriptors']
        
        if len(desc0) == 0 or len(desc1) == 0:
            return {
                'mkpts_0': np.empty((0, 2)), 
                'mkpts_1': np.empty((0, 2)),
                'idxs_0': np.array([]),
                'idxs_1': np.array([])
            }
        
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
    """Optimized HDF5 writer with proper synchronization"""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.buffer = []
        # Create file initially
        with h5py.File(self.output_path, 'w') as f:
            pass  # Just create empty file
    
    def add_results(self, results: List[Dict]):
        """Add results to buffer and write immediately"""
        # Write immediately to avoid memory buildup
        self._write_batch(results)
    
    def _write_batch(self, batch: List[Dict]):
        """Write a batch of results"""
        if not batch:
            return
            
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
                    grp.create_dataset(
                        'matches0',
                        data=result['matches0'].cpu().numpy().astype(np.int32),
                        compression='gzip',
                        compression_opts=4  # Faster compression
                    )
                    grp.create_dataset(
                        'matching_scores0',
                        data=result['matching_scores0'].cpu().numpy().astype(np.float32),
                        compression='gzip',
                        compression_opts=4
                    )
        except Exception as e:
            print(f"Write error: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        """Flush remaining data and close"""
        pass  # Nothing to do since we write immediately


def optimized_match_from_paths(
    pairs_path: Path,
    feature_path_q: Path,
    feature_path_r: Path,
    output_path: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    max_keypoints: int = 4096,
    use_fp16: bool = True,
    compile_model: bool = False
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
    sys.path.append("/data/sahil/new_colmap/Xfeat")
    from modules.xfeat import XFeat
    xfeat_model = XFeat()
    
    # Create dataset and dataloader
    dataset = OptimizedBatchDataset(
        pairs, feature_path_q, feature_path_r,
        max_keypoints=max_keypoints
    )
    
    # Optimize dataloader settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    # Initialize matcher and writer
    matcher = OptimizedXFeatMatcher(
        xfeat_model,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_fp16=use_fp16,
        compile_model=compile_model
    )
    writer = OptimizedWriter(output_path)
    
    # Process batches
    try:
        with tqdm(total=len(pairs), desc="Matching") as pbar:
            for batch_idx, batch_data in enumerate(dataloader):
                # Process batch
                results = matcher.match_batch(batch_data)
                
                # Write results immediately
                writer.add_results(results)
                
                # Update progress
                pbar.update(len(results))
                
                # Memory management every 50 batches
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
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
    parser.add_argument("--batch_size", type=int, default=16)
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