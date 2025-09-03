import argparse
import os
from pathlib import Path
import subprocess

def run_script(script_path, args):
    """Utility function to run a script with given arguments."""
    cmd = ["python3", "-m", script_path] + args
    subprocess.run(cmd, check=True)

def append_vpr_pairs_to_existing(pairs_file, vpr_pairs_file):
    """Append VPR-generated pairs to the existing pairs file."""
    with open(vpr_pairs_file, "r") as vpr_file:
        vpr_pairs = vpr_file.readlines()
    
    with open(pairs_file, "a") as pairs_file:
        pairs_file.writelines(vpr_pairs)

def run_colmap_mapper(database_path, image_path, output_path):
    """Run COLMAP's mapper to generate a sparse model."""
    cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_path),
        "--output_path", str(output_path),
        "--Mapper.ba_gpu_index","0",
        "--Mapper.ba_use_gpu", "1",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.abs_pose_max_error", "8",
        "--Mapper.max_reg_trials", "6"
    ]
    subprocess.run(cmd, check=True)

def main(image_dir, export_dir, num_features, num_matches, vpr_batch_size, vpr_distance_threshold):
    image_dir = Path(image_dir)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    features_file = export_dir / "output_features.h5"
    pairs_file = export_dir / "image_pairs.txt"
    matches_file = export_dir / "matches.h5"
    db_path = export_dir / "database.db"
    vpr_pairs_file = export_dir / "image_pairs_vpr.txt"
    sparse_output_path = export_dir / "sparse"

    # Step 1: Extract Features
    # print("[1/6] Extracting features...")
    # extract_args = [
    #     "--image_dir", str(image_dir),
    #     "--output_file", str(features_file),
    #     "--export_dir", str(export_dir),
    #     "--num_feat", str(num_features),
    # ]
    # run_script("hloc.extract_xfeat", extract_args)

    # # # Step 2: Generate Image Pairs
    # print("[2/6] Generating image pairs...")
    # pair_args = [
    #     "--image_folder", str(image_dir),
    #     "--output_file", str(pairs_file),
    #     "--num_matches", str(num_matches),
    # ]
    # run_script("hloc.make_pairs", pair_args)


    # # Step 4: Match Features
    print("[4/6] Matching features...")
    match_args = [
        "--pairs", str(pairs_file),
        "--export_dir", str(export_dir),
        "--features", str(features_file),
        "--matches", str(matches_file),
        "--use_dicemap"
    ]
    run_script("hloc.match_xfeat", match_args)

    # # Step 5: Generate COLMAP Database
    print("[5/6] Generating COLMAP database...")
    db_args = [
        "--pairs", str(pairs_file),
        "--image_dir", str(image_dir),
        "--export_dir", str(export_dir),
        "--matches", str(matches_file),
        "--features", str(features_file),
    ]
    run_script("hloc.db", db_args)

    print(f"COLMAP database generated at: {db_path}")

    # # # Step 6: Run COLMAP Mapper
    print("[6/6] Running COLMAP mapper...")
    sparse_output_path.mkdir(parents=True, exist_ok=True)
    run_colmap_mapper(db_path, image_dir, sparse_output_path)

    print(f"Sparse model generated at: {sparse_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a COLMAP database and sparse model from images using HLOC and VPR.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--export_dir", type=str, required=True, help="Path to the directory for export output.")
    parser.add_argument("--num_features", type=int, default=3072, help="Number of features to extract per face.")
    parser.add_argument("--num_matches", type=int, default=6, help="Number of matches to generate per image.")
    parser.add_argument("--vpr_batch_size", type=int, default=2, help="Batch size for VPR processing.")
    parser.add_argument("--vpr_distance_threshold", type=float, default=1, help="Distance threshold for VPR matches.")

    args = parser.parse_args()
    main(args.image_dir, args.export_dir, args.num_features, args.num_matches, args.vpr_batch_size, args.vpr_distance_threshold)
