import h5py
import numpy as np
from collections import defaultdict
import multiprocessing as mp
import psutil
import os

def cubemap_to_equirectangular_vectorized(points, face, cubemap_size=960, eq_width=3840, eq_height=1920):
    x, y = points[:, 0], points[:, 1]
    x = (x / cubemap_size) * 2 - 1
    y = (y / cubemap_size) * 2 - 1

    face_mapping = {
        '2': lambda x, y: np.column_stack((np.ones_like(x), x, -y)),
        '3': lambda x, y: np.column_stack((-x, np.ones_like(x), -y)),
        '0': lambda x, y: np.column_stack((-np.ones_like(x), -x, -y)),
        '1': lambda x, y: np.column_stack((x, -np.ones_like(x), -y)),
        '4': lambda x, y: np.column_stack((y, x, np.ones_like(x))),
        '5': lambda x, y: np.column_stack((-y, x, -np.ones_like(x)))
    }
    
    vec = face_mapping[face](x, y)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)

    theta = np.arctan2(vec[:, 1], vec[:, 0])
    phi = -np.arcsin(vec[:, 2])

    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi
    
    return np.column_stack((u * eq_width, v * eq_height))

def process_frame(image_data, cubemap_size, eq_width, eq_height):
    face_num = image_data['face_num']
    keypoints = image_data['keypoints']
    descriptors = image_data['descriptors']
    scores = image_data['scores']

    eq_coords = cubemap_to_equirectangular_vectorized(keypoints, face_num, cubemap_size, eq_width, eq_height)
    
    return eq_coords, descriptors.T, scores

def process_spherical_image(spherical_image_data, cubemap_size, eq_width, eq_height):
    base_name = spherical_image_data['base_name']
    image_data_list = spherical_image_data['image_data']

    all_keypoints = []
    all_descriptors = []
    all_scores = []

    for image_data in image_data_list:
        eq_coords, descriptors, scores = process_frame(image_data, cubemap_size, eq_width, eq_height)
        all_keypoints.append(eq_coords)
        all_descriptors.append(descriptors.T)
        all_scores.append(scores)

    combined_keypoints = np.vstack(all_keypoints)
    combined_descriptors = np.hstack(all_descriptors)
    combined_scores = np.concatenate(all_scores)

    return base_name, combined_keypoints, combined_descriptors, combined_scores

def process_h5py_file(original_h5_file_path, new_h5_file_path, eq_width=3840, eq_height=1920, cubemap_size=960, chunk_size=100):
    with h5py.File(original_h5_file_path, 'r') as orig_file:
        # Group images by their spherical name (e.g., frame_0001)
        grouped_data = defaultdict(list)
        for image_name in orig_file.keys():
            base_name = "_".join(image_name.split('_')[:-1])  # e.g., frame_0001
            grouped_data[base_name].append(image_name)

        total_spherical_images = len(grouped_data)
        chunks = [list(grouped_data.items())[i:i+chunk_size] for i in range(0, total_spherical_images, chunk_size)]

        with h5py.File(new_h5_file_path, 'w') as new_file:
            for chunk_idx, chunk in enumerate(chunks):
                print(f"Processing chunk {chunk_idx + 1}/{len(chunks)}...")

                # Prepare data for parallel processing
                process_args = []
                for base_name, image_names in chunk:
                    spherical_image_data = {
                        'base_name': base_name,
                        'image_data': []
                    }
                    for image_name in image_names:
                        face_num = image_name.split('_')[-1].split('.')[0][-1]
                        image_data = {
                            'face_num': face_num,
                            'keypoints': orig_file[image_name]['keypoints'][:],
                            'descriptors': orig_file[image_name]['descriptors'][:],
                            'scores': orig_file[image_name]['scores'][:]
                        }
                        spherical_image_data['image_data'].append(image_data)
                    process_args.append((spherical_image_data, cubemap_size, eq_width, eq_height))

                # Use multiprocessing to process spherical images in parallel
                with mp.Pool(processes=mp.cpu_count()) as pool:
                    results = pool.starmap(process_spherical_image, process_args)

                # Write results to the new HDF5 file
                for base_name, combined_keypoints, combined_descriptors, combined_scores in results:
                    print(f"Saving data for spherical image: {base_name}...")
                    grp = new_file.create_group(base_name[:-5] + ".jpg")
                    grp.create_dataset('keypoints', data=combined_keypoints, compression="gzip", compression_opts=9)
                    grp.create_dataset('descriptors', data=combined_descriptors, compression="gzip", compression_opts=9)
                    grp.create_dataset('scores', data=combined_scores, compression="gzip", compression_opts=9)
                    grp.create_dataset('image_size', data=np.array([eq_width, eq_height]))
                    tmp = base_name[:-5] + ".jpg"
                    print(f"Saved combined data for spherical image {tmp} Total keypoints: {len(combined_keypoints)}")

                # Monitor memory usage
                process = psutil.Process(os.getpid())
                print(f"Current memory usage: {process.memory_info().rss / 1e9:.2f} GB")

    print("Processing completed successfully.")

if __name__=="__main__":
    original_h5_file_path = '/home/luffy/data/data_raghuvir/outputs/try_resized_1/feats-superpoint-n4096-r1600.h5'
    new_h5_file_path = '/home/luffy/data/data_raghuvir/outputs/try_resized_1/feats-superpoint-n4096-r1600_new.h5'
    process_h5py_file(original_h5_file_path, new_h5_file_path)
