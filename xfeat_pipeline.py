import argparse
import os
from pathlib import Path
import subprocess

def run_script(script_path, args):
    """Utility function to run a script with given arguments."""
    cmd = ["python3 -m", script_path] + args
    subprocess.run(cmd, check=True)

def main(image_dir, export_dir, num_features, num_matches):
    image_dir = Path(image_dir)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    features_file = export_dir / "output_features.h5"
    pairs_file = export_dir / "image_pairs.txt"
    matches_file = export_dir / "matches.h5"
    db_path = export_dir / "database.db"

    # Step 1: Extract Features
    print("[1/4] Extracting features...")
    extract_args = [
        "--image_dir", str(image_dir),
        "--output_file", str(features_file),
        "--export_dir", str(export_dir),
        "--num_feat", str(num_features),
    ]
    run_script("hloc.extract_xfeat", extract_args)

    # Step 2: Generate Image Pairs
    print("[2/4] Generating image pairs...")
    pair_args = [
        "--image_folder", str(image_dir),
        "--output_file", str(pairs_file),
        "--num_matches", str(num_matches),
    ]
    run_script("hloc.make_pairs", pair_args)

    # Step 3: Match Features
    print("[3/4] Matching features...")
    match_args = [
        "--pairs", str(pairs_file),
        "--export_dir", str(export_dir),
        "--features", str(features_file),
        "--matches", str(matches_file),
    ]
    run_script("hloc.match_xfeat", match_args)

    # Step 4: Generate COLMAP Database
    print("[4/4] Generating COLMAP database...")
    db_args = [
        "--pairs", str(pairs_file),
        "--image_dir", str(image_dir),
        "--export_dir", str(export_dir),
        "--matches", str(matches_file),
        "--features", str(features_file),
    ]
    run_script("hloc.db", db_args)

    print(f"COLMAP database generated at: {db_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a COLMAP database from images using HLOC.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--export_dir", type=str, required=True, help="Path to the directory for export output.")
    parser.add_argument("--num_features", type=int, default=3072, help="Number of features to extract per face.")
    parser.add_argument("--num_matches", type=int, default=6, help="Number of matches to generate per image.")

    args = parser.parse_args()
    main(args.image_dir, args.export_dir, args.num_features, args.num_matches)