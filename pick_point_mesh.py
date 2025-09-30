"""
Refactored 3D Correspondence Finder for Spherical Cameras with Mesh Support

Clean, focused implementation with simplified API for finding 3D correspondences
from 2D image points using ray-mesh intersection.
"""

import os
import math
import time
import struct
import functools
import concurrent.futures
from typing import Tuple, Optional, Dict, List, Any, Union
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import cKDTree
from numba import jit, prange, njit
import psutil
from tqdm import tqdm
from collections import namedtuple

# ═══════════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RayHit:
    """Result of ray-mesh intersection"""
    hit: bool
    point: np.ndarray
    distance: float
    triangle_id: int
    normal: np.ndarray

Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("SPHERE", 3),
    12: ("SPHERICAL", 3),
}

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,    2*x*z + 2*w*y],
        [2*x*y + 2*w*z,      1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,      2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y],
    ])

def _read_binary(fid, nbytes: int, fmt: str, endian: str = "<"):
    """Helper for reading binary data"""
    return struct.unpack(endian + fmt, fid.read(nbytes))

@jit(nopython=True, parallel=True, cache=True)
def pixels_to_rays_numba(points, cx, cy, inv_width, inv_height, two_pi, pi, R_T):
    """Numba-accelerated batch conversion of pixels to world rays"""
    n_points = points.shape[0]
    rays = np.empty((n_points, 3), dtype=np.float64)
    
    for i in prange(n_points):
        x, y = points[i, 0], points[i, 1]
        
        # ERP mapping: pixel -> (longitude, latitude)
        lon = ((x - cx) * inv_width) * two_pi
        lat = -((y - cy) * inv_height) * pi
        
        cos_lat = math.cos(lat)
        sin_lat = math.sin(lat)
        sin_lon = math.sin(lon)
        cos_lon = math.cos(lon)
        
        # Unit ray in camera frame
        ray_cam_x = cos_lat * sin_lon
        ray_cam_y = -sin_lat
        ray_cam_z = cos_lat * cos_lon
        
        # Normalize
        norm = math.sqrt(ray_cam_x**2 + ray_cam_y**2 + ray_cam_z**2)
        if norm < 1e-10:
            norm = 1.0
        ray_cam_x /= norm
        ray_cam_y /= norm
        ray_cam_z /= norm
        
        # Transform to world frame
        rays[i, 0] = R_T[0, 0]*ray_cam_x + R_T[0, 1]*ray_cam_y + R_T[0, 2]*ray_cam_z
        rays[i, 1] = R_T[1, 0]*ray_cam_x + R_T[1, 1]*ray_cam_y + R_T[1, 2]*ray_cam_z
        rays[i, 2] = R_T[2, 0]*ray_cam_x + R_T[2, 1]*ray_cam_y + R_T[2, 2]*ray_cam_z
    
    return rays

# ═══════════════════════════════════════════════════════════════════════════════
# COLMAP I/O FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def read_cameras_binary(path: str) -> Dict[int, Camera]:
    """Read cameras from COLMAP binary file"""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = _read_binary(f, 8, "Q")[0]
        for _ in range(num_cameras):
            cam_id, model_id, width, height = _read_binary(f, 24, "iiQQ")
            model_name, num_params = CAMERA_MODEL_IDS[model_id]
            params = np.array(_read_binary(f, 8 * num_params, "d" * num_params))
            cameras[cam_id] = Camera(cam_id, model_name, width, height, params)
    return cameras

def read_images_binary(path: str) -> Dict[int, Image]:
    """Read images from COLMAP binary file"""
    images = {}
    with open(path, "rb") as f:
        num_images = _read_binary(f, 8, "Q")[0]
        for _ in range(num_images):
            img_id, *vals = _read_binary(f, 64, "idddddddi")
            qvec = np.array(vals[0:4])
            tvec = np.array(vals[4:7])
            cam_id = vals[7]
            
            # Read null-terminated name
            name = b""
            while True:
                c = _read_binary(f, 1, "c")[0]
                if c == b"\x00":
                    break
                name += c
            name = name.decode()
            
            # Read feature points
            num_points = _read_binary(f, 8, "Q")[0]
            data = _read_binary(f, 24 * num_points, "ddq" * num_points)
            xys = np.column_stack([data[0::3], data[1::3]])
            point_ids = np.array(data[2::3], dtype=int)
            
            images[img_id] = Image(img_id, qvec, tvec, cam_id, name, xys, point_ids)
    return images

@functools.lru_cache(maxsize=8)
def load_colmap_model(colmap_folder: str, model_ext: str = ".bin"):
    """Load COLMAP model with caching"""
    if model_ext == ".bin":
        cameras = read_cameras_binary(os.path.join(colmap_folder, "cameras.bin"))
        images = read_images_binary(os.path.join(colmap_folder, "images.bin"))
    else:
        raise NotImplementedError("Text format not implemented in refactored version")
    return cameras, images

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class MeshCorrespondenceFinder:
    """
    3D Correspondence finder using mesh ray-casting and COLMAP camera parameters.
    
    This class handles mesh loading, COLMAP data loading, and provides simple 
    interfaces for finding 3D correspondences from 2D image points.
    """
    
    def __init__(self, 
                 mesh_path: str, 
                 colmap_folder: str,
                 model_ext: str = ".bin",
                 simplify_threshold: Optional[int] = None,
                 n_workers: int = 4):
        """
        Initialize the correspondence finder.
        
        Args:
            mesh_path: Path to mesh file (OBJ, PLY, STL, etc.)
            colmap_folder: Path to COLMAP sparse reconstruction folder
            model_ext: COLMAP model extension (".bin" or ".txt")  
            simplify_threshold: If set, simplify mesh to this many faces
            n_workers: Number of parallel workers for processing
        """
        self.mesh_path = mesh_path
        self.colmap_folder = colmap_folder
        self.model_ext = model_ext
        self.n_workers = n_workers
        
        print(f"Initializing MeshCorrespondenceFinder...")
        print(f"  Mesh: {mesh_path}")
        print(f"  COLMAP: {colmap_folder}")
        
        # Load mesh
        self._load_mesh(simplify_threshold)
        
        # Load COLMAP data
        self._load_colmap_data()
        
        print(f"Initialization complete.")
    
    def _load_mesh(self, simplify_threshold: Optional[int] = None):
        """Load and process the mesh"""
        print(f"Loading mesh from {self.mesh_path}...")
        
        # Load mesh using trimesh
        self.mesh = trimesh.load_mesh(self.mesh_path, process=True)
        
        # Ensure mesh is valid
        if not self.mesh.is_watertight:
            print("Warning: Mesh is not watertight, attempting to fix...")
            self.mesh.fill_holes()
            self.mesh.fix_normals()
            self.mesh.remove_degenerate_faces()
            self.mesh.remove_duplicate_faces()
            self.mesh.remove_unreferenced_vertices()
        
        # Optional mesh simplification
        if simplify_threshold and len(self.mesh.faces) > simplify_threshold:
            print(f"Simplifying mesh from {len(self.mesh.faces)} to {simplify_threshold} faces...")
            self.mesh = self.mesh.simplify_quadric_decimation(simplify_threshold)
        
        # Store mesh properties
        self.vertices = np.array(self.mesh.vertices, dtype=np.float64)
        self.faces = np.array(self.mesh.faces, dtype=np.int32)
        self.face_normals = np.array(self.mesh.face_normals, dtype=np.float64)
        
        # Compute mesh statistics
        self.bounds = self.mesh.bounds
        self.scale = np.max(self.mesh.extents)
        
        # Pre-build ray caster if available
        self._ray_caster = None
        try:
            if trimesh.ray.has_embree:
                from trimesh.ray.ray_pyembree import RayMeshIntersector
                self._ray_caster = RayMeshIntersector(self.mesh)
                print("Using Embree accelerated ray casting")
        except Exception as e:
            print(f"Embree not available, using default ray casting: {e}")
        
        print(f"Mesh loaded: {len(self.vertices):,} vertices, {len(self.faces):,} faces")
    
    def _load_colmap_data(self):
        """Load COLMAP camera and image data"""
        print(f"Loading COLMAP data from {self.colmap_folder}...")
        
        self.cameras, self.images = load_colmap_model(self.colmap_folder, self.model_ext)
        
        # Filter to spherical cameras only
        self.spherical_images = {}
        for img_id, img in self.images.items():
            camera = self.cameras[img.camera_id]
            if camera.model in ["SPHERE", "SPHERICAL"]:
                self.spherical_images[img_id] = img
        
        print(f"Loaded {len(self.cameras)} cameras, {len(self.images)} images")
        print(f"Found {len(self.spherical_images)} spherical images")
    
    def _get_camera_params(self, image_name: str) -> Dict[str, Any]:
        """Get camera parameters for a specific image"""
        # Find image by name
        target_image = None
        for img_id, img in self.spherical_images.items():
            if img.name == image_name:
                target_image = img
                break
        
        if target_image is None:
            available_names = [img.name for img in self.spherical_images.values()]
            raise ValueError(f"Image '{image_name}' not found. Available: {available_names[:5]}...")
        
        # Get camera
        camera = self.cameras[target_image.camera_id]
        
        # Extract camera parameters
        if len(camera.params) >= 3:
            f, cx, cy = camera.params[:3]
            if f <= 0 or f > max(camera.width, camera.height):
                estimated_f = max(cx, cy)
                print(f"Warning: Invalid focal length {f}, using {estimated_f}")
                f = estimated_f
        else:
            raise ValueError(f"Insufficient camera parameters")
        
        # Build transformation matrices
        R = qvec2rotmat(target_image.qvec)
        t = target_image.tvec
        camera_center = -R.T @ t
        
        return {
            'image_name': image_name,
            'width': camera.width,
            'height': camera.height,
            'cx': cx,
            'cy': cy,
            'R': R,
            't': t,
            'camera_center': camera_center,
            'inv_width': 1.0 / camera.width,
            'inv_height': 1.0 / camera.height,
            'R_T': R.T
        }
    
    def _cast_rays_vectorized(self, ray_origins: np.ndarray, ray_directions: np.ndarray) -> List[RayHit]:
        """Cast multiple rays against the mesh"""
        # Use cached ray caster if available
        if self._ray_caster:
            locations, index_ray, index_tri = self._ray_caster.intersects_location(
                ray_origins, ray_directions, multiple_hits=False
            )
        else:
            locations, index_ray, index_tri = self.mesh.ray.intersects_location(
                ray_origins, ray_directions, multiple_hits=False
            )
        
        # Pre-allocate results
        n_rays = len(ray_origins)
        results = []
        
        # Create hit lookup
        ray_hits = {}
        if len(index_ray) > 0:
            for loc, ray_idx, tri_idx in zip(locations, index_ray, index_tri):
                distance = np.linalg.norm(loc - ray_origins[ray_idx])
                normal = self.face_normals[tri_idx]
                
                hit = RayHit(
                    hit=True,
                    point=loc,
                    distance=distance,
                    triangle_id=tri_idx,
                    normal=normal
                )
                
                if ray_idx not in ray_hits or distance < ray_hits[ray_idx].distance:
                    ray_hits[ray_idx] = hit
        
        # Build final results
        no_hit = RayHit(
            hit=False,
            point=np.array([np.nan, np.nan, np.nan]),
            distance=np.inf,
            triangle_id=-1,
            normal=np.array([0, 0, 0])
        )
        
        for i in range(n_rays):
            results.append(ray_hits.get(i, no_hit))
        
        return results
    
    def _find_correspondences_for_points(self, image_points: np.ndarray, 
                                       camera_params: Dict[str, Any]) -> np.ndarray:
        """Find 3D correspondences for 2D image points"""
        n_points = len(image_points)
        
        # Convert pixels to rays using numba
        ray_directions = pixels_to_rays_numba(
            image_points,
            camera_params['cx'], camera_params['cy'],
            camera_params['inv_width'], camera_params['inv_height'],
            2.0 * math.pi, math.pi, camera_params['R_T']
        )
        
        # Ray origins are all at camera center
        ray_origins = np.full((n_points, 3), camera_params['camera_center'])
        
        # Cast rays
        hits = self._cast_rays_vectorized(ray_origins, ray_directions)
        
        # Extract 3D points
        correspondences = np.full((n_points, 3), np.nan)
        for i, hit in enumerate(hits):
            if hit.hit:
                correspondences[i] = hit.point
        
        return correspondences
    
    def pickpoints_single(self, image_name: str, points_2d: np.ndarray) -> np.ndarray:
        """
        Find 3D correspondences for 2D points in a single image.
        
        Args:
            image_name: Name of the image file in COLMAP reconstruction
            points_2d: Nx2 array of 2D pixel coordinates
            
        Returns:
            Nx3 array of 3D world coordinates (NaN for failed correspondences)
        """
        if len(points_2d.shape) != 2 or points_2d.shape[1] != 2:
            raise ValueError("points_2d must be Nx2 array")
        
        print(f"Finding correspondences for {len(points_2d)} points in {image_name}")
        
        # Get camera parameters for this image
        camera_params = self._get_camera_params(image_name)
        
        # Find correspondences
        correspondences = self._find_correspondences_for_points(points_2d, camera_params)
        
        # Report results
        valid_count = np.sum(~np.any(np.isnan(correspondences), axis=1))
        success_rate = valid_count / len(points_2d)
        print(f"Found {valid_count}/{len(points_2d)} correspondences ({success_rate:.1%})")
        
        return correspondences
    
    def pickpoints_multiple(self, points_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Find 3D correspondences for 2D points in multiple images using batch processing.
        
        Args:
            points_dict: Dictionary with image names as keys and Nx2 arrays of 2D points as values
            
        Returns:
            Dictionary with image names as keys and Nx3 arrays of 3D points as values
        """
        if not points_dict:
            return {}
        
        # Validate inputs
        for image_name, points_2d in points_dict.items():
            if len(points_2d.shape) != 2 or points_2d.shape[1] != 2:
                raise ValueError(f"points_2d for {image_name} must be Nx2 array")
        
        print(f"Finding correspondences for {len(points_dict)} images using batch processing")
        
        # Step 1: Collect all camera parameters and compute cumulative indices
        image_names = list(points_dict.keys())
        camera_params_list = []
        point_counts = []
        
        for image_name in image_names:
            try:
                camera_params = self._get_camera_params(image_name)
                camera_params_list.append(camera_params)
                point_counts.append(len(points_dict[image_name]))
            except Exception as e:
                print(f"Error loading camera params for {image_name}: {e}")
                camera_params_list.append(None)
                point_counts.append(len(points_dict[image_name]))
        
        # Compute cumulative indices for efficient indexing
        cumulative_indices = np.cumsum([0] + point_counts)
        total_points = cumulative_indices[-1]
        
        print(f"Total points across all images: {total_points}")
        
        # Step 2: Collect all 2D points and convert to rays in batches
        all_ray_origins = []
        all_ray_directions = []
        
        for i, image_name in enumerate(image_names):
            if camera_params_list[i] is None:
                # Create dummy rays for failed images
                n_points = point_counts[i]
                all_ray_origins.append(np.zeros((n_points, 3)))
                all_ray_directions.append(np.zeros((n_points, 3)))
                continue
            
            camera_params = camera_params_list[i]
            points_2d = points_dict[image_name]
            
            # Convert pixels to rays
            ray_directions = pixels_to_rays_numba(
                points_2d,
                camera_params['cx'], camera_params['cy'],
                camera_params['inv_width'], camera_params['inv_height'],
                2.0 * math.pi, math.pi, camera_params['R_T']
            )
            
            # Ray origins are all at camera center
            ray_origins = np.full((len(points_2d), 3), camera_params['camera_center'])
            
            all_ray_origins.append(ray_origins)
            all_ray_directions.append(ray_directions)
        
        # Concatenate all rays
        all_ray_origins = np.vstack(all_ray_origins)
        all_ray_directions = np.vstack(all_ray_directions)
        
        # Step 3: Process all rays in parallel batches
        print("Processing all rays in parallel batches...")
        
        # Determine optimal batch size
        batch_size = min(2048, max(512, total_points // max(self.n_workers, 1)))
        
        if total_points <= 2000:
            # For small datasets, process all at once
            all_hits = self._cast_rays_vectorized(all_ray_origins, all_ray_directions)
        else:
            # For large datasets, use parallel batch processing
            all_hits = self._process_rays_in_batches(all_ray_origins, all_ray_directions, batch_size)
        
        # Step 4: Extract 3D points from hits
        all_correspondences = np.full((total_points, 3), np.nan)
        for i, hit in enumerate(all_hits):
            if hit.hit:
                all_correspondences[i] = hit.point
        
        # Step 5: Distribute results back to individual images
        results = {}
        total_valid = 0
        
        for i, image_name in enumerate(image_names):
            start_idx = cumulative_indices[i]
            end_idx = cumulative_indices[i + 1]
            
            correspondences = all_correspondences[start_idx:end_idx]
            results[image_name] = correspondences
            
            # Update statistics
            if camera_params_list[i] is not None:
                valid_count = np.sum(~np.any(np.isnan(correspondences), axis=1))
                total_valid += valid_count
                success_rate = valid_count / len(correspondences)
                print(f"  {image_name}: {valid_count}/{len(correspondences)} ({success_rate:.1%})")
            else:
                print(f"  {image_name}: Failed (camera params error)")
                results[image_name] = np.full((point_counts[i], 3), np.nan)
        
        # Report overall results
        overall_success = total_valid / total_points if total_points > 0 else 0
        print(f"Overall: {total_valid}/{total_points} correspondences ({overall_success:.1%})")
        
        return results
    
    def _process_rays_in_batches(self, ray_origins: np.ndarray, ray_directions: np.ndarray, 
                               batch_size: int) -> List[RayHit]:
        """Process rays in parallel batches for large datasets"""
        n_rays = len(ray_origins)
        all_hits = [None] * n_rays
        
        # Create batches
        batches = []
        for start_idx in range(0, n_rays, batch_size):
            end_idx = min(start_idx + batch_size, n_rays)
            batches.append((start_idx, end_idx))
        
        # Process batches in parallel
        def process_batch(batch_info):
            start_idx, end_idx = batch_info
            batch_origins = ray_origins[start_idx:end_idx]
            batch_directions = ray_directions[start_idx:end_idx]
            batch_hits = self._cast_rays_vectorized(batch_origins, batch_directions)
            return start_idx, batch_hits
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            batch_results = list(tqdm(
                executor.map(process_batch, batches),
                total=len(batches),
                desc="Processing ray batches"
            ))
        
        # Combine results
        for start_idx, batch_hits in batch_results:
            for i, hit in enumerate(batch_hits):
                all_hits[start_idx + i] = hit
        
        return all_hits


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

def example_usage():
    """Example of how to use the MeshCorrespondenceFinder"""
    
    # Initialize the finder
    finder = MeshCorrespondenceFinder(
        mesh_path="/data/sahil/new_colmap/Hierarchical-Localization/mesh_optimised.ply",
        colmap_folder="/data/sahil/new_colmap/Hierarchical-Localization/dense_raghuvir/sparse",
        model_ext=".bin",
        simplify_threshold=50000,  # Optional: simplify large meshes
        n_workers=4
    )
    
    # Example 1: Single image
    points_2d = np.array([
        [100, 200],
        [300, 400],
        [500, 600]
    ])
    
    points_3d = finder.pickpoints_single("frame_0427.png", points_2d)
    print("3D correspondences:", points_3d)
    
    # Example 2: Multiple images
    multiple_points = {
        "frame_0427.png": np.array([[100, 200], [300, 400]]),
        "frame_0067.png": np.array([[150, 250], [350, 450], [550, 650]]),
        "frame_0015.png": np.array([[680, 570]])
    }
    
    results = finder.pickpoints_multiple(multiple_points)
    for image_name, points_3d in results.items():
        print(f"{image_name}: {points_3d} points processed")

if __name__ == "__main__":
    example_usage()