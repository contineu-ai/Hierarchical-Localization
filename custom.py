from pathlib import Path
import os
from hloc import (
    extract_features_new,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    conv_new
)

WIDTH = 3840
HEIGHT = 1920

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
            for j in range(1, k+1,3):
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


images_cube = Path("/home/luffy/data/data_raghuvir/VID_20240622_155518_00_007_processed_resized_cube/")
images_spherical = Path("/home/luffy/data/data_raghuvir/VID_20240622_155518_00_007_processed_resized/")
outputs = Path("/home/luffy/data/data_raghuvir/outputs/try_resized_2/")
sfm_pairs = outputs / "pairs.txt"
sfm_dir = outputs / "sfm_superpoint+superglue"
matcher_conf = match_features.confs["superpoint+lightglue"]
feature_conf = extract_features_new.confs["superpoint_inloc"]
os.makedirs(outputs,exist_ok=True)

k = 10
generate_image_pairs(images_spherical, sfm_pairs, k*3)
feature_path = extract_features_new.main(feature_conf, images_cube, outputs)
new_feature_path = Path(str(feature_path)[:-3]+"_new.h5")   
feature_conf["output"] = feature_conf["output"] + "_new"
conv_new.process_h5py_file(feature_path,new_feature_path,WIDTH,HEIGHT,WIDTH/4)
match_path = match_features.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
)
model = reconstruction.main(sfm_dir, images_spherical, sfm_pairs, feature_path, match_path)
db_path = sfm_dir / "database.db"
new_camera_model = 11  
new_params = [WIDTH*1.2, WIDTH,HEIGHT]  
change_camera_model_and_parameters(db_path, new_camera_model, new_params)