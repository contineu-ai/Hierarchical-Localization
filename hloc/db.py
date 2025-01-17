from pathlib import Path
import os
import sys
sys.path.append('/data/sahil/sfm/Hierarchical-Localization')
from hloc import (
    # extract_features_new,
    # match_features,
    reconstruction,
    # visualization,
    # pairs_from_retrieval,
    # conv_new
)
import struct
import sqlite3
WIDTH = 5120
HEIGHT = 2560

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
    print(f"Updated all cameras to model '{new_camera_model}' with new parameters.")

pairs = Path("/data/sahil/sfm/scripts/image_pairs.txt")  # Replace with the actual path
image_dir = Path("/data/sahil/data/sahil_test_videos/6759a65ed5305e001214c976")  # Replace with the actual path
export_dir = Path(".")  # Replace with the actual path
matches = Path("/data/sahil/sfm/Hierarchical-Localization/output_features_matches_image_pairs.h5")
features = Path("/data/sahil/sfm/Hierarchical-Localization/output_features.h5")

model = reconstruction.main(export_dir, image_dir, pairs, features, matches)
db_path = export_dir / "database.db"
new_camera_model = 11
new_params = [WIDTH*1.2, WIDTH/2,HEIGHT/2]  
change_camera_model_and_parameters(db_path, new_camera_model, new_params)