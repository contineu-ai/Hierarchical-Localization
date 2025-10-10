"""
Refactored COLMAP Pipeline with Centralized Configuration
All hyperparameters now controlled via config.yaml
"""

import argparse
import subprocess
from pathlib import Path
from config_loader import load_config, ConfigLoader


def run_script(script_path, args):
    """Utility function to run a script with given arguments."""
    cmd = ["python3", "-m", script_path] + args
    subprocess.run(cmd, check=True)


def run_async_script(script_path, args):
    """Utility function to run an async script with given arguments."""
    cmd = ["python3", "-m", script_path] + args
    subprocess.run(cmd, check=True)


def append_pairs_file(pairs_file, additional_pairs_file):
    """Append contents of another text file to the existing pairs file."""
    if not Path(additional_pairs_file).exists():
        print(f"Warning: Additional pairs file {additional_pairs_file} does not exist. Skipping append.")
        return
    
    try:
        with open(additional_pairs_file, 'r') as add_file:
            additional_content = add_file.read().strip()
        
        if additional_content:
            with open(pairs_file, 'a') as pairs_f:
                pairs_f.write(additional_content)
            print(f"Successfully appended {additional_pairs_file} to {pairs_file}")
        else:
            print(f"Warning: Additional pairs file {additional_pairs_file} is empty. Nothing to append.")
    
    except Exception as e:
        print(f"Error appending pairs file: {e}")


def run_colmap_mapper(database_path, image_path, output_path, config):
    """Run COLMAP's mapper to generate a sparse model."""
    mapper_cfg = config.colmap.mapper
    
    cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_path),
        "--output_path", str(output_path),
        "--Mapper.ba_gpu_index", str(mapper_cfg.ba_gpu_index),
        "--Mapper.ba_use_gpu", str(mapper_cfg.ba_use_gpu),
        "--Mapper.ba_refine_focal_length", str(mapper_cfg.ba_refine_focal_length),
        "--Mapper.ba_refine_extra_params", str(mapper_cfg.ba_refine_extra_params),
        "--Mapper.abs_pose_max_error", str(mapper_cfg.abs_pose_max_error),
        "--Mapper.max_reg_trials", str(mapper_cfg.max_reg_trials),
        "--Mapper.tri_min_angle", str(mapper_cfg.tri_min_angle),
    ]
    subprocess.run(cmd, check=True)


def main(image_dir, export_dir, config, additional_pairs_file=None):
    """Main pipeline execution"""
    
    image_dir = Path(image_dir)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Setup processing directories
    processing_dir = export_dir / "processing"
    processing_dir.mkdir(parents=True, exist_ok=True)
    
    dicemap_dir = processing_dir / "dicemaps" 
    final_image_dir = processing_dir / "human_masked"

    # Paths
    features_file = export_dir / "output_features.h5"
    pairs_file = export_dir / "image_pairs.txt"
    matches_file = export_dir / "matches.h5"
    db_path = export_dir / "database.db"
    sparse_output_path = export_dir / "sparse"

    print(f"\n{'='*70}")
    print(f"COLMAP PIPELINE: Panoramic + Human Detection")
    print(f"{'='*70}")
    print(f"Input directory: {image_dir}")
    print(f"Export directory: {export_dir}")
    
    # Print config summary
    ConfigLoader.print_config(config)
    
    if additional_pairs_file:
        print(f"Additional pairs file: {additional_pairs_file}")
    print()
    
    # Step 1-3: Feature extraction with unified pipeline
    print(f"[1/6] Extracting features with unified GPU pipeline...")
    feature_args = [
        "--image_dir", str(image_dir),
        "--output_file", str(features_file),
        "--batch_size", str(config.batching.feature_batch_size),
        "--num_features", str(config.features.num_features),
        "--num_workers", str(config.batching.num_workers),
    ]
    
    if not config.gpu.use_half_precision:
        feature_args.append("--no_half_precision")
    
    # run_async_script("hloc.unify_v2", feature_args)

    # Step 4: Generate image pairs
    print(f"\n[2/6] Generating image pairs...")
    pair_args = [
        "--image_folder", str(image_dir),
        "--output_file", str(pairs_file),
        "--num_matches", str(config.pairing.num_matches),
    ]
    # run_script("hloc.make_pairs", pair_args)

    # Step 4.5: Append additional pairs file if provided
    if additional_pairs_file:
        print(f"\n[2.5/6] Appending additional pairs from {additional_pairs_file}...")
        # append_pairs_file(pairs_file, additional_pairs_file)

    # Step 5: Match features  
    print(f"\n[3/6] Matching features...")
    match_args = [
        "--pairs", str(pairs_file),
        "--features", str(features_file),
        "--output", str(matches_file),
        "--batch_size", str(config.batching.matching_batch_size),
        "--max_keypoints", str(config.features.max_keypoints),
        "--num_workers", str(config.batching.num_workers),
    ]
    
    if not config.gpu.use_half_precision:
        match_args.append("--no_fp16")
    
    # run_script("hloc.match_xfeat", match_args)

    # Step 6: Generate COLMAP database
    print(f"\n[4/6] Generating COLMAP database...")
    db_args = [
        "--pairs", str(pairs_file),
        "--image_dir", str(image_dir),
        "--export_dir", str(export_dir),
        "--matches", str(matches_file),
        "--features", str(features_file),
        "--config", str(Path(args.config).resolve()),  # Pass config file path
    ]
    # run_script("hloc.db", db_args)

    print(f"COLMAP database generated at: {db_path}")

    # Step 7: Run COLMAP mapper
    print(f"\n[5/6] Running COLMAP mapper...")
    sparse_output_path.mkdir(parents=True, exist_ok=True)
    run_colmap_mapper(db_path, image_dir, sparse_output_path, config)

    print(f"Sparse model generated at: {sparse_output_path}")

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Sparse model: {sparse_output_path}")
    print(f"Database: {db_path}")
    print(f"Features: {features_file}")
    print(f"Humans masked with {config.human_detection.mask_type}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COLMAP pipeline for panoramic images with automatic human detection and masking"
    )
    
    # Core arguments
    parser.add_argument("--image_dir", type=str, required=True, 
                       help="Directory containing panoramic images")
    parser.add_argument("--export_dir", type=str, required=True, 
                       help="Export directory")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file (default: config.yaml)")
    
    # Optional overrides
    parser.add_argument("--num_features", type=int, 
                       help="Features per dicemap (overrides config)")
    parser.add_argument("--num_matches", type=int, 
                       help="Matches per image (overrides config)")
    parser.add_argument("--batch_size", type=int, 
                       help="Processing batch size (overrides config)")
    parser.add_argument("--yolo_model", type=str, 
                       help="YOLOv11 model (overrides config)")
    parser.add_argument("--yolo_conf_threshold", type=float, 
                       help="Human detection confidence threshold (overrides config)")
    parser.add_argument("--mask_type", type=str, 
                       choices=["solid_color", "blur", "pixelate"],
                       help="Human masking type (overrides config)")
    parser.add_argument("--mask_color", type=int, nargs=3, 
                       help="Mask color RGB (overrides config)")
    parser.add_argument("--additional_pairs_file", type=str, 
                       help="Path to additional pairs text file to append")
    parser.add_argument("--keep_intermediate", action="store_true", 
                       help="Keep dicemaps and detection results")
    parser.add_argument("--gpu_memory", type=float,
                       help="Max GPU memory in GB (overrides config)")
    parser.add_argument("--no_half_precision", action="store_true",
                       help="Disable FP16 (overrides config)")

    args = parser.parse_args()
    
    # Load configuration with CLI overrides
    config = load_config(args.config, args)
    
    print("\n" + "="*70)
    print("PANORAMIC COLMAP PIPELINE")
    print("="*70)
    print("Automatic processing:")
    print("  ✓ Panoramic → Dicemap conversion")
    print(f"  ✓ Human detection with {config.models.yolo_model}")
    print(f"  ✓ Human masking: {config.human_detection.mask_type}")
    print("  ✓ Privacy-aware feature extraction")
    if args.additional_pairs_file:
        print(f"  ✓ Additional pairs append: {args.additional_pairs_file}")
    print("="*70 + "\n")

    main(args.image_dir, args.export_dir, config, args.additional_pairs_file)