from pathlib import Path
import os
from hloc import (
    extract_features,
    matcher_sphereglue,
    match_features_1,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    conv_new
)
import struct
import sqlite3
WIDTH = 3920
HEIGHT = 1960

def generate_image_pairs(image_folder, output_file, k):
    # List all files in the image folder
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))])
    
    # Ensure we have at least two images to create pairs
    if len(images) < 2:
        print("Not enough images to create pairs.")
        return

    with open(output_file, 'w') as f:
        for i in range(len(images)):
            # Create pairs with the next k images
            for j in range(1, k+1):
                if i + j < len(images):
                    f.write(f"{images[i]} {images[i+j]}\n")
    
    print(f"Pairs file saved to {output_file}")

def change_camera_model_and_parameters(db_path, new_model_id, new_params):

    new_params_binary = struct.pack(f'{len(new_params)}d', *new_params)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
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


# images_cube = Path("/home/luffy/data/data_raghuvir/VID_20240622_155518_00_007_processed_resized_cube/")
images_spherical = Path("/home/luffy/data/66e143da3284089a55e96308_frames_3920_1960")
outputs = Path("./outputs/try_sphere/")
sfm_pairs = outputs / "pairs.txt"
sfm_dir = outputs / "sfm_superpoint+superglue"
matcher_conf = {"output":"sphereglue"}
# matcher_conf = match_features_1.confs["superpoint+lightglue"]
feature_conf = extract_features.confs["superpoint_inloc"]
os.makedirs(outputs,exist_ok=True)

k = 3
generate_image_pairs(images_spherical, sfm_pairs, k)
feature_path = extract_features.main(feature_conf, images_spherical, outputs)
# new_feature_path = Path(str(feature_path)[:-3]+"_new.h5")   
# feature_conf["output"] = feature_conf["output"] + "_new"
# conv_new.process_h5py_file(feature_path,new_feature_path,WIDTH,HEIGHT,WIDTH/4)
match_path = matcher_sphereglue.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
)
# match_path = match_features_1.main(
    # matcher_conf, sfm_pairs, feature_conf["output"], outputs
# )
# mloc
model = reconstruction.main(sfm_dir, images_spherical, sfm_pairs, feature_path, match_path)
db_path = sfm_dir / "database.db"
new_camera_model = 11  
new_params = [WIDTH*1.2, WIDTH/2,HEIGHT/2]  
change_camera_model_and_parameters(db_path, new_camera_model, new_params)