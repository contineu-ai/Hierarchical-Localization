import argparse
from pathlib import Path
import os
import sys
import struct
import sqlite3
from PIL import Image

sys.path.append('/data/sahil/sfm/Hierarchical-Localization')
from hloc import reconstruction

def change_camera_model_and_parameters(db_path, new_model_id, new_params):
    new_params_binary = struct.pack(f'{len(new_params)}d', *new_params)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:", tables)

    cursor.execute("SELECT camera_id FROM cameras;")
    cameras = cursor.fetchall()
    for camera_id in cameras:
        cursor.execute("""
            UPDATE cameras
            SET model = ?, params = ?
            WHERE camera_id = ?;
        """, (new_model_id, new_params_binary, camera_id[0]))

    conn.commit()
    conn.close()
    print(f"Updated all cameras to model '{new_model_id}' with new parameters.")

def get_image_dimensions(image_dir):
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not image_files:
        raise ValueError("No valid image files found in the directory.")

    with Image.open(image_files[0]) as img:
        return img.width, img.height

def main(pairs, image_dir, export_dir, matches, features):
    # Get image dimensions from the first image in the directory
    width, height = get_image_dimensions(image_dir)

    # Run reconstruction
    model = reconstruction.main(export_dir, image_dir, pairs, features, matches)

    # Update camera model and parameters in the database
    db_path = export_dir / "database.db"
    new_camera_model = 11
    new_params = [width * 1.2, width / 2, height / 2]
    change_camera_model_and_parameters(db_path, new_camera_model, new_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update camera model and parameters in a COLMAP database.")
    parser.add_argument("--pairs", type=Path, required=True, help="Path to the image pairs file.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Path to the directory containing images.")
    parser.add_argument("--export_dir", type=Path, required=True, help="Path to the directory for export output.")
    parser.add_argument("--matches", type=Path, required=True, help="Path to the matches file.")
    parser.add_argument("--features", type=Path, required=True, help="Path to the features file.")

    args = parser.parse_args()
    
    main(args.pairs, args.image_dir, args.export_dir, args.matches, args.features)