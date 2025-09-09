import argparse
import os
from pathlib import Path
import subprocess
import asyncio

def run_script(script_path, args):
    """Utility function to run a script with given arguments."""
    cmd = ["python3", "-m", script_path] + args
    subprocess.run(cmd, check=True)

def run_async_script(script_path, args):
    """Utility function to run an async script with given arguments."""
    cmd = ["python3","-m" ,script_path] + args
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
        "--Mapper.abs_pose_max_error", "1.0",
        "--Mapper.max_reg_trials", "6",
        "--Mapper.tri_min_angle", "3.5"
    ]
    subprocess.run(cmd, check=True)

def main(image_dir, export_dir, num_features, num_matches, yolo_model, yolo_conf_threshold, 
         mask_type, mask_color, batch_size, keep_intermediate, additional_pairs_file):
    
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

    print(f"=== COLMAP Pipeline: Panoramic + Human Detection ===")
    print(f"Input directory: {image_dir}")
    print(f"Export directory: {export_dir}")
    print(f"YOLO model: {yolo_model}")
    print(f"Mask type: {mask_type} {mask_color}")
    if additional_pairs_file:
        print(f"Additional pairs file: {additional_pairs_file}")
    print()

    # Step 1: Convert panoramic images to dicemaps
    print(f"[1/7] Converting panoramic images to dicemaps...")
    dicemap_args = [
        "--image_dir", str(image_dir),
        "--output_dir", str(processing_dir),
        "--batch_size", str(batch_size)
    ]
    if keep_intermediate:
        dicemap_args.append("--save_cubemaps")
    
    run_async_script("hloc.dicemap_converter", dicemap_args)

    # Step 2: Detect and mask humans in dicemaps
    print(f"[2/7] Detecting and masking humans in dicemaps...")
    
    yolo_args = [
        "--image_dir", str(dicemap_dir),
        "--model_name", yolo_model,
        "--output_dir", str(processing_dir / "yolo_results"),
        "--conf_threshold", str(yolo_conf_threshold),
        "--batch_size", str(batch_size),
        "--mask_type", mask_type,
        "--mask_color"] + [str(c) for c in mask_color]
    
    if keep_intermediate:
        yolo_args.append("--save_detections")
    
    run_async_script("hloc.yolo_processor", yolo_args)
    
    # Move masked images to expected location
    masked_source = processing_dir / "yolo_results" / "human_masked"
    masked_target = processing_dir / "human_masked"
    if masked_source.exists():
        if masked_target.exists():
            import shutil
            shutil.rmtree(masked_target)
        masked_source.rename(masked_target)

    # Step 3: Extract features from human-masked dicemaps
    print(f"[3/7] Extracting features from processed dicemaps...")
    
    # Create original shapes file for coordinate conversion
    shapes_script = f"""
import h5py
import cv2
from pathlib import Path

shapes = {{}}
for img_path in Path('{image_dir}').glob('*'):
    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                shapes[img_path.name] = (w, h)
        except Exception as e:
            print(f"Warning: Could not read {{img_path}}: {{e}}")

with h5py.File('{export_dir / "temp_shapes.h5"}', 'w') as f:
    for img_name, (w, h) in shapes.items():
        name_only = img_name.split('.')[0]
        grp = f.create_group(name_only)
        grp.create_dataset("image_size", data=[w, h])
print(f"Saved {{len(shapes)}} original image dimensions")
"""
    with open(export_dir / "create_shapes.py", "w") as f:
        f.write(shapes_script)
    subprocess.run(["python3", str(export_dir / "create_shapes.py")], check=True)
    
    feature_args = [
        "--dicemap_dir", str(final_image_dir),
        "--output_file", str(features_file),
        "--original_shapes_file", str(export_dir / "temp_shapes.h5"),
        "--num_features", str(num_features),
        "--batch_size", str(batch_size)
    ]
    
    run_async_script("hloc.extract_xfeat_only", feature_args)

    # Step 4: Generate image pairs
    print(f"[4/7] Generating image pairs...")
    pair_args = [
        "--image_folder", str(final_image_dir),
        "--output_file", str(pairs_file),
        "--num_matches", str(num_matches),
    ]
    run_script("hloc.make_pairs", pair_args)

    # Step 4.5: Append additional pairs file if provided
    if additional_pairs_file:
        print(f"[4.5/7] Appending additional pairs from {additional_pairs_file}...")
        append_pairs_file(pairs_file, additional_pairs_file)

    # Step 5: Match features  
    print(f"[5/7] Matching features...")
    match_args = [
        "--pairs", str(pairs_file),
        "--features", str(features_file),
        "--output", str(matches_file),
    ]
    
    run_script("hloc.match_xfeat", match_args)

    # Step 6: Generate COLMAP database
    print(f"[6/7] Generating COLMAP database...")
    db_args = [
        "--pairs", str(pairs_file),
        "--image_dir", str(image_dir),
        "--export_dir", str(export_dir),
        "--matches", str(matches_file),
        "--features", str(features_file),
    ]
    run_script("hloc.db", db_args)

    print(f"COLMAP database generated at: {db_path}")

    # Step 7: Run COLMAP mapper
    print(f"[7/7] Running COLMAP mapper...")
    sparse_output_path.mkdir(parents=True, exist_ok=True)
    run_colmap_mapper(db_path, image_dir, sparse_output_path)

    print(f"Sparse model generated at: {sparse_output_path}")

    # # Cleanup temporary files
    # if not keep_intermediate:
    #     print("\nCleaning up intermediate files...")
    #     temp_files = [
    #         export_dir / "temp_shapes.h5",
    #         export_dir / "create_shapes.py"
    #     ]
    #     for temp_file in temp_files:
    #         if temp_file.exists():
    #             temp_file.unlink()
        
    #     import shutil
    #     if processing_dir.exists():
    #         shutil.rmtree(processing_dir)
    #         print(f"Removed processing directory: {processing_dir}")

    print(f"\n=== Pipeline Complete ===")
    print(f"Sparse model: {sparse_output_path}")
    print(f"Database: {db_path}")
    print(f"Features: {features_file}")
    print(f"Humans masked with {mask_type} {mask_color}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COLMAP pipeline for panoramic images with automatic human detection and masking")
    
    # Core arguments
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing panoramic images")
    parser.add_argument("--export_dir", type=str, required=True, help="Export directory")
    parser.add_argument("--num_features", type=int, default=3072, help="Features per dicemap")
    parser.add_argument("--num_matches", type=int, default=6, help="Matches per image")
    parser.add_argument("--batch_size", type=int, default=128, help="Processing batch size")
    
    # Human detection - always enabled
    parser.add_argument("--yolo_model", type=str, default="yolo11m.pt", 
                       help="YOLOv11 model (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)")
    parser.add_argument("--yolo_conf_threshold", type=float, default=0.1, help="Human detection confidence threshold")
    parser.add_argument("--mask_type", type=str, default="solid_color", 
                       choices=["solid_color", "blur", "pixelate"], help="Human masking type")
    parser.add_argument("--mask_color", type=int, nargs=3, default=[0, 0, 0], help="Mask color RGB (default: black)")
    
    # Additional pairs file
    parser.add_argument("--additional_pairs_file", type=str, default=None, 
                       help="Path to additional pairs text file to append after generating image pairs")
    
    # Processing options
    parser.add_argument("--keep_intermediate", action="store_true", help="Keep dicemaps and detection results")

    args = parser.parse_args()
    
    print("=== Panoramic COLMAP Pipeline ===")
    print("Automatic processing:")
    print("  ✓ Panoramic → Dicemap conversion")
    print(f"  ✓ Human detection with {args.yolo_model}")
    print(f"  ✓ Human masking: {args.mask_type}")
    print("  ✓ Privacy-aware feature extraction")
    if args.additional_pairs_file:
        print(f"  ✓ Additional pairs append: {args.additional_pairs_file}")
    print()

    main(
        args.image_dir, args.export_dir, args.num_features, args.num_matches,
        args.yolo_model, args.yolo_conf_threshold, args.mask_type, args.mask_color,
        args.batch_size, args.keep_intermediate, args.additional_pairs_file
    )