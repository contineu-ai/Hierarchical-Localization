import os
import cv2
import torch
import faiss
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TOP_K = 3
BATCH_SIZE = 2
FRAME_DIFF_THRESHOLD = 10
FRAME_CLOSENESS_THRESHOLD = 10
DISTANCE_THRESHOLD = 0.5  # New constant for filtering by feature distance
IMG_WIDTH = 1568

def extract_frame_number(image_path):
    """
    Extracts the frame number from image filename in format 'frameXXXX.jpg'
    """
    try:
        basename = os.path.basename(image_path)
        match = re.search(r'frame_(\d{4})\.jpg', basename)
        if match:
            return int(match.group(1))
        logger.warning(f"Unexpected filename format: {basename}")
        return None
    except Exception as e:
        logger.error(f"Error extracting frame number from {image_path}: {str(e)}")
        return None

def generate_mapping_data(image_width):
    """Generate mapping data for image transformation."""
    in_size = [image_width, int(image_width * 3 / 4)]
    edge = int(in_size[0] / 4)

    out_pix = np.zeros((in_size[1], in_size[0], 2), dtype="f4")
    xyz = np.zeros((in_size[1] * in_size[0] // 2, 3), dtype="f4")
    vals = np.zeros((in_size[1] * in_size[0] // 2, 3), dtype="i4")

    start, end = 0, 0
    rng_1 = np.arange(0, edge * 3)
    rng_2 = np.arange(edge, edge * 2)
    for i in range(in_size[0]):
        face = i // edge
        rng = rng_1 if face == 2 else rng_2

        end += len(rng)
        vals[start:end, 0] = rng
        vals[start:end, 1] = i
        vals[start:end, 2] = face
        start = end

    j, i, face = vals.T
    face[j < edge] = 4
    face[j >= 2 * edge] = 5

    a = 2.0 * i / edge
    b = 2.0 * j / edge
    one_arr = np.ones(len(a))
    for k in range(6):
        face_idx = face == k
        one_arr_idx = one_arr[face_idx]
        a_idx = a[face_idx]
        b_idx = b[face_idx]

        if k == 0:
            vals_to_use = [-one_arr_idx, 1.0 - a_idx, 3.0 - b_idx]
        elif k == 1:
            vals_to_use = [a_idx - 3.0, -one_arr_idx, 3.0 - b_idx]
        elif k == 2:
            vals_to_use = [one_arr_idx, a_idx - 5.0, 3.0 - b_idx]
        elif k == 3:
            vals_to_use = [7.0 - a_idx, one_arr_idx, 3.0 - b_idx]
        elif k == 4:
            vals_to_use = [b_idx - 1.0, a_idx - 5.0, one_arr_idx]
        elif k == 5:
            vals_to_use = [5.0 - b_idx, a_idx - 5.0, -one_arr_idx]

        xyz[face_idx] = np.array(vals_to_use).T

    x, y, z = xyz.T
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(z, r)

    uf = (2.0 * edge * (theta + np.pi) / np.pi) % in_size[0]
    uf[uf == in_size[0]] = 0.0
    vf = (2.0 * edge * (np.pi / 2 - phi) / np.pi)

    out_pix[j, i, 0] = vf
    out_pix[j, i, 1] = uf

    map_x_32 = out_pix[:, :, 1]
    map_y_32 = out_pix[:, :, 0]
    return map_x_32, map_y_32

class ImageFolderDataset(Dataset):
    """Dataset for loading and preprocessing image frames."""
    
    def __init__(self, image_folder, transform=None):
        self.image_paths = sorted([
            os.path.join(image_folder, fname) 
            for fname in os.listdir(image_folder) 
            if fname.lower().endswith('.jpg') and fname.startswith('frame')
        ])
        if not self.image_paths:
            raise ValueError(f"No valid frame images found in {image_folder}")
        
        self.transform = transform
        self.map_x_32, self.map_y_32 = generate_mapping_data(IMG_WIDTH)
        
        # Validate frame numbers during initialization
        invalid_frames = [
            path for path in self.image_paths 
            if extract_frame_number(path) is None
        ]
        if invalid_frames:
            logger.warning(f"Found {len(invalid_frames)} invalid frame filenames")
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            height, width = img.shape[:2]
            new_height = int((IMG_WIDTH / width) * height)
            img = cv2.resize(img, (IMG_WIDTH, new_height), interpolation=cv2.INTER_LANCZOS4)
            img = cv2.remap(img, self.map_x_32, self.map_y_32, cv2.INTER_LANCZOS4)
            img = cv2.remap(img, self.map_x_32, self.map_y_32, cv2.INTER_LANCZOS4)
            img = cv2.resize(img,(img.shape[1] // 14 * 14, img.shape[0] // 14 * 14))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(img_rgb) if self.transform else img_rgb
            return img_tensor, image_path
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

def extract_features_batch(dataloader, model, device):
    """Extract features from images in batches."""
    all_features, all_image_paths = [], []
    
    for images, image_paths in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        with torch.no_grad():
            batch_features = model(images).cpu().numpy()
        all_features.append(batch_features)
        all_image_paths.extend(image_paths)
    
    return np.vstack(all_features), all_image_paths

def build_faiss_index(image_folder, model, device, fc_output_dim, batch_size):
    """Build FAISS index for fast similarity search."""
    logger.info("Building FAISS index...")
    
    dataset = ImageFolderDataset(image_folder, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    features, image_paths = extract_features_batch(dataloader, model, device)
    
    faiss_index = faiss.IndexFlatL2(fc_output_dim)
    # print (features.shape)
    faiss_index.add(features)
    
    return faiss_index, image_paths

def search_top_k_matches(query_images, query_image_paths, faiss_index, model, 
                        device, image_paths, top_k, frame_diff_threshold, 
                        frame_closeness_threshold):
    """Search for similar frames with temporal and distance constraints."""
    query_images = query_images.to(device)
    with torch.no_grad():
        query_features = model(query_images).cpu().numpy()

    all_filtered_matches = []
    for idx, query_feature in enumerate(query_features):
        query_frame_number = extract_frame_number(query_image_paths[idx])
        if query_frame_number is None:
            logger.warning(f"Invalid query frame: {query_image_paths[idx]}")
            all_filtered_matches.append([])
            continue

        # Search for top-k * 5 initially to allow for filtering
        distances, indices = faiss_index.search(
            query_feature.reshape(1, -1), 
            top_k * 5
        )
        
        filtered_matches = []
        added_frame_numbers = set()

        for rank, (i, distance) in enumerate(zip(indices[0], distances[0])):
            if len(filtered_matches) >= top_k:
                break
                
            # Skip if distance is above threshold
            if distance > DISTANCE_THRESHOLD:
                continue
                
            candidate_image_path = image_paths[i]
            candidate_frame_number = extract_frame_number(candidate_image_path)
            
            if candidate_frame_number is None:
                continue
            
            frame_diff = abs(candidate_frame_number - query_frame_number)
            is_too_close = any(
                abs(candidate_frame_number - added_frame) <= frame_closeness_threshold 
                for added_frame in added_frame_numbers
            )
            
            if frame_diff > frame_diff_threshold and not is_too_close:
                filtered_matches.append(
                    (candidate_image_path, float(distance))
                )
                added_frame_numbers.add(candidate_frame_number)
        
        all_filtered_matches.append(filtered_matches)
    
    return all_filtered_matches

def save_results_to_json(results, output_file):
    """Save matching results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {output_file}")

def create_image_pairs_file(results, output_txt_file):
    """Create text file with image pairs."""
    with open(output_txt_file, 'w') as f:
        for result in results:
            query_image = result['query_image']
            for match in result['matches']:
                matched_image = match['image_path']
                distance = match['distance']
                # Include distance in the output
                f.write(f"{os.path.basename(query_image)} "
                       f"{os.path.basename(matched_image)}\n")
    logger.info(f"Image pairs written to {output_txt_file}")

def process_images(image_folder, model, device, fc_output_dim, batch_size, output_txt=None):
    """Main processing function."""
    try:
        logger.info(f"Processing images from {image_folder}")
        
        # Build FAISS index
        faiss_index, image_paths = build_faiss_index(
            image_folder, model, device, fc_output_dim, batch_size
        )
        
        # Setup query dataset
        query_dataset = ImageFolderDataset(image_folder, transform=transform)
        query_dataloader = DataLoader(
            query_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Process queries
        results = []
        total_pairs = 0
        for query_images, query_image_paths in tqdm(query_dataloader, 
                                                  desc="Processing queries"):
            top_matches = search_top_k_matches(
                query_images, query_image_paths, faiss_index, model,
                device, image_paths, TOP_K, FRAME_DIFF_THRESHOLD,
                FRAME_CLOSENESS_THRESHOLD
            )
            
            for query_image_path, matches in zip(query_image_paths, top_matches):
                if matches:  # Only add results with valid matches
                    result = {
                        'query_image': query_image_path,
                        'matches': [
                            {'image_path': match, 'distance': dist} 
                            for match, dist in matches
                        ]
                    }
                    results.append(result)
                    total_pairs += len(matches)
        
        logger.info(f"Found {total_pairs} valid matches across {len(results)} queries")
        
        if output_txt:
            create_image_pairs_file(results, output_txt)
            
        return results
        
    except Exception as e:
        logger.error(f"Error in process_images: {str(e)}")
        raise

import argparse
import sys

def main():
    # Use a local argument parser to avoid conflicts with the library's global arguments
    parser = argparse.ArgumentParser(description="Run VPR with FAISS")
    
    # Define script-specific arguments
    parser.add_argument("--image_folder", required=True, help="Folder containing images for VPR.")
    parser.add_argument("--output_txt", required=True, help="Path to save the image pairs TXT.")
    args, _ = parser.parse_known_args()  # Use `parse_known_args` to avoid unknown argument errors
    
    # Model setup
    sys.argv = [sys.argv[0]]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load VLAD-BuFF model without inheriting global CLI arguments
    logger.info("Loading VLAD-BuFF model...")
    model = torch.hub.load(
        "Ahmedest61/VLAD-BuFF",
        "vlad_buff",
        antiburst=True,
        nv_pca=192,
        wpca=True,
        num_pcs=4096
    ).to(device)
    model.eval()
    
    # Transform
    global transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Run VPR processing
    process_images(
        image_folder=args.image_folder,
        model=model,
        device=device,
        fc_output_dim=12288,
        batch_size=BATCH_SIZE,
        output_txt=args.output_txt
    )

if __name__ == "__main__":
    main()
