import os
import cv2
import h5py
import numpy as np
import torch
import argparse
from tqdm import tqdm
from .cubemap_utils import GPU_Convert, cubemap_to_equirectangular_uv

def pad_array(array, target_size, axis=0, fill_value=0):
    """
    Pad a numpy array to the target size along the specified axis.
    Args:
        array: The input numpy array.
        target_size: The desired size along the specified axis.
        axis: The axis along which to pad.
        fill_value: The value used for padding.
    Returns:
        Padded numpy array.
    """
    pad_width = [(0, 0)] * len(array.shape)
    pad_width[axis] = (0, max(0, target_size - array.shape[axis]))
    return np.pad(array, pad_width=pad_width, mode='constant', constant_values=fill_value)

def extract_features(image_dir, output_file, export_dir, num_feat):
    # Load XFeat model
    xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', 
                            pretrained=True, top_k=num_feat, trust_repo='check').to("cuda")

    # Ensure export directory exists
    os.makedirs(export_dir, exist_ok=True)

    # Gather image paths
    image_paths = []
    for ext in ("jpg", "jpeg", "png", "JPG", "PNG"):
        image_paths.extend([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(ext)])
    image_paths = sorted(image_paths)

    with h5py.File(output_file, "w") as f:
        for img_path in tqdm(image_paths, desc="Processing images", unit="img"):
            # Read image (equirectangular)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get original dimensions for the equirectangular image
            eq_height, eq_width = img.shape[0], img.shape[1]

            # Convert image to cubemap
            converter = GPU_Convert(img.shape)
            cubemaps = converter.convert_to_cubemaps(img)

            # Create a group for this image in the HDF5 file
            img_name = os.path.basename(img_path)
            img_grp = f.create_group(img_name)

            # Initialize lists for overall arrays
            all_keypoints_c = []
            all_descriptors_c = []
            all_scores_c = []

            # Get cubemap size from the front face (assuming all faces are square)
            cubemap_size = cubemaps["F"].shape[0]

            # Extract and save per-face features
            for face in ["F", "R", "B", "L", "U", "D"]:
                face_data = cubemaps[face]

                output = xfeat.detectAndCompute(face_data, top_k=num_feat)[0]
                keypoints = output.get('keypoints', np.zeros((num_feat, 2))).cpu().numpy()
                descriptors = output.get('descriptors', np.zeros((num_feat, 256))).cpu().numpy()
                scores = output.get('scores', np.zeros((num_feat,))).cpu().numpy()

                keypoints = pad_array(keypoints, num_feat, axis=0, fill_value=0)
                descriptors = pad_array(descriptors, num_feat, axis=0, fill_value=0)
                scores = pad_array(scores, num_feat, axis=0, fill_value=0)

                # Save per-face features
                img_grp.create_dataset(f"keypoints_{face.lower()}", data=keypoints)
                img_grp.create_dataset(f"descriptors_{face.lower()}", data=descriptors)
                img_grp.create_dataset(f"scores_{face.lower()}", data=scores)

                # Accumulate for overall arrays
                if keypoints.shape[0] > 0:
                    # Convert each face's keypoints into equirectangular coordinates
                    eq_keypoints = []
                    for (x_c, y_c) in keypoints:
                        u_eq, v_eq = cubemap_to_equirectangular_uv(face, x_c, y_c, 
                                                                  cubemap_size=cubemap_size, 
                                                                  eq_width=eq_width, 
                                                                  eq_height=eq_height)
                        eq_keypoints.append([u_eq, v_eq])

                    eq_keypoints = np.array(eq_keypoints)

                    all_keypoints_c.append(eq_keypoints)
                    all_descriptors_c.append(descriptors)
                    all_scores_c.append(scores)

            # Combine all faces into one set for overall features
            if all_keypoints_c:
                overall_keypoints = np.concatenate(all_keypoints_c, axis=0)
                overall_descriptors = np.concatenate(all_descriptors_c, axis=0)
                overall_scores = np.concatenate(all_scores_c, axis=0)
            else:
                # If no features were detected at all
                overall_keypoints = np.empty((0, 2))
                overall_descriptors = np.empty((0, 256))  # Adjust if needed
                overall_scores = np.empty((0,))

            # Save overall features
            img_grp.create_dataset("keypoints", data=overall_keypoints)
            img_grp.create_dataset("descriptors", data=overall_descriptors)
            img_grp.create_dataset("scores", data=overall_scores)
            img_grp.create_dataset("image_size", data=(img.shape[1], img.shape[0]))

    print("Features saved to:", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from images using XFeat and save to HDF5.")
    parser.add_argument('--image_dir', type=str,
                        help="Directory containing input images.")
    parser.add_argument('--output_file', type=str, default="output_features.h5",
                        help="Output HDF5 file for saving features.")
    parser.add_argument('--export_dir', type=str, default=".",
                        help="Directory for exporting additional processed data.")
    parser.add_argument('--num_feat', type=int, default=3072,
                        help="Number of features to extract per face.")
    args = parser.parse_args()

    extract_features(args.image_dir, args.output_file, args.export_dir, args.num_feat)
