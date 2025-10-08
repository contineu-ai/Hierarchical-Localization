import argparse
import struct
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pycolmap
from PIL import Image
from tqdm import tqdm

from .config_loader import load_config, PipelineConfig
from .utils.database import COLMAPDatabase
from .utils.io import get_keypoints, get_matches


def create_empty_db(database_path: Path):
    """Create an empty COLMAP database."""
    if database_path.exists():
        print(f"Warning: Database already exists at {database_path}, deleting it.")
        database_path.unlink()
    
    print("Creating an empty database...")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    image_list: Optional[List[str]] = None,
):
    """Import images into the COLMAP database."""
    print("Importing images into the database...")
    
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f"No images found in {image_dir}.")
    
    with pycolmap.ostream():
        pycolmap.import_images(
            database_path,
            image_dir,
            camera_mode,
            image_names=image_list or [],
        )


def get_image_ids(database_path: Path) -> Dict[str, int]:
    """Get mapping of image names to IDs from database."""
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images


def import_features(
    image_ids: Dict[str, int], 
    database_path: Path, 
    features_path: Path
):
    """Import feature keypoints into the database."""
    print("Importing features into the database...")
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm(image_ids.items(), desc="Features"):
        keypoints = get_keypoints(features_path, image_name)
        keypoints += 0.5  # COLMAP origin offset
        db.add_keypoints(image_id, keypoints)

    db.commit()
    db.close()


def import_matches(
    image_ids: Dict[str, int],
    database_path: Path,
    pairs_path: Path,
    matches_path: Path,
    min_match_score: Optional[float] = None,
    skip_geometric_verification: bool = True,
):
    """Import feature matches into the database."""
    print("Importing matches into the database...")

    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]

    db = COLMAPDatabase.connect(database_path)
    matched = set()
    skipped = 0

    for name0, name1 in tqdm(pairs, desc="Matches"):
        id0, id1 = image_ids[name0], image_ids[name1]
        
        # Skip if already matched
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        
        try:
            matches, scores = get_matches(matches_path, name0, name1)
        except Exception as e:
            skipped += 1
            print(f"Skipped {name0}-{name1}: {e}")
            continue
        
        if min_match_score:
            matches = matches[scores > min_match_score]
        
        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)

    print(f"Matches skipped: {skipped}")
    db.commit()
    db.close()


def update_camera_parameters(
    db_path: Path, 
    new_model_id: int, 
    new_params: List[float]
):
    """Update camera model and parameters in the database."""
    new_params_binary = struct.pack(f'{len(new_params)}d', *new_params)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all camera IDs
    cursor.execute("SELECT camera_id FROM cameras;")
    cameras = cursor.fetchall()
    
    # Update each camera
    for camera_id in cameras:
        cursor.execute("""
            UPDATE cameras
            SET model = ?, params = ?
            WHERE camera_id = ?;
        """, (new_model_id, new_params_binary, camera_id[0]))
    
    conn.commit()
    conn.close()
    print(f"Updated {len(cameras)} camera(s) to model ID {new_model_id}")


def get_image_dimensions(image_dir: Path) -> tuple[int, int]:
    """Get image dimensions from the first image in directory."""
    image_files = [
        f for f in image_dir.iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    ]
    
    if not image_files:
        raise ValueError(f"No valid image files found in {image_dir}")
    
    with Image.open(image_files[0]) as img:
        return img.width, img.height


def build_database(
    pairs: Path,
    image_dir: Path,
    export_dir: Path,
    matches: Path,
    features: Path,
    config: PipelineConfig,
    camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.AUTO,
    skip_geometric_verification: bool = True,
    min_match_score: Optional[float] = None,
):
    """Build COLMAP database with images, features, and matches."""
    # Validate inputs
    assert image_dir.exists(), f"Image directory not found: {image_dir}"
    assert features.exists(), f"Features file not found: {features}"
    assert pairs.exists(), f"Pairs file not found: {pairs}"
    assert matches.exists(), f"Matches file not found: {matches}"
    
    # Create output directory and database
    export_dir.mkdir(parents=True, exist_ok=True)
    database_path = export_dir / "database.db"
    
    # Get image dimensions
    width, height = get_image_dimensions(image_dir)
    print(f"Image dimensions: {width}x{height}")
    
    # Build database
    create_empty_db(database_path)
    import_images(image_dir, database_path, camera_mode)
    image_ids = get_image_ids(database_path)
    print(f"Found {len(image_ids)} images")
    
    import_features(image_ids, database_path, features)
    import_matches(
        image_ids,
        database_path,
        pairs,
        matches,
        min_match_score,
        skip_geometric_verification,
    )
    
    # Update camera parameters
    new_model_id = config.colmap.camera.model_id
    new_params = [
        width * config.colmap.camera.focal_multiplier,
        width / 2,
        height / 2
    ]
    
    update_camera_parameters(database_path, new_model_id, new_params)
    print(f"Camera updated: model={new_model_id}, focal={new_params[0]:.1f}px")
    print(f"Database created successfully at {database_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build COLMAP database from features and matches"
    )
    parser.add_argument(
        "--pairs", type=Path, required=True, 
        help="Path to image pairs file"
    )
    parser.add_argument(
        "--image_dir", type=Path, required=True, 
        help="Path to directory containing images"
    )
    parser.add_argument(
        "--export_dir", type=Path, required=True, 
        help="Path to output directory for database"
    )
    parser.add_argument(
        "--matches", type=Path, required=True, 
        help="Path to matches file"
    )
    parser.add_argument(
        "--features", type=Path, required=True, 
        help="Path to features file"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--camera_mode", type=str, default="AUTO",
        choices=list(pycolmap.CameraMode.__members__.keys()),
        help="Camera mode for COLMAP"
    )
    parser.add_argument(
        "--min_match_score", type=float,
        help="Minimum match score threshold"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    camera_mode = getattr(pycolmap.CameraMode, args.camera_mode)
    
    # Build database
    build_database(
        pairs=args.pairs,
        image_dir=args.image_dir,
        export_dir=args.export_dir,
        matches=args.matches,
        features=args.features,
        config=config,
        camera_mode=camera_mode,
        min_match_score=args.min_match_score,
    )

if __name__ == "__main__":
    main()