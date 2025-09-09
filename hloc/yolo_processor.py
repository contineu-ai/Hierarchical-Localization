"""
YOLO Processor Module
Handles human detection and masking using Ultralytics YOLOv11
Updated to preserve original file formats
"""
import numpy as np
import cv2
import argparse
import time
import asyncio
import os
from typing import List, Tuple, Dict
from pathlib import Path
from .shared_utils import MaskType, CUPY_AVAILABLE, load_images_async, find_images

# Import Ultralytics YOLO
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    print("Ultralytics YOLO available")
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("ERROR: Ultralytics required. Install: pip install ultralytics")

if CUPY_AVAILABLE:
    import cupy as cp


class UltralyticsYOLOProcessor:
    """Human detection processor using Ultralytics YOLOv11"""
    
    def __init__(self, model_name: str = 'yolo11n.pt', conf_threshold: float = 0.5, 
                 max_batch_size: int = 32, device: str = 'auto'):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics YOLO required. Install: pip install ultralytics")
        
        self.conf_threshold = conf_threshold
        self.max_batch_size = max_batch_size
        self.device = device
        self._initialize_model(model_name)
    
    def _initialize_model(self, model_name: str):
        """Initialize YOLOv11 model"""
        print(f"Loading YOLOv11 model: {model_name}")
        
        try:
            self.model = YOLO(model_name)
            
            # Set device properly for Ultralytics
            if self.device == 'auto':
                import torch
                if torch.cuda.is_available():
                    self.device = 'cuda:0'  # Use first GPU
                    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                else:
                    self.device = 'cpu'
                    print("No GPU detected, using CPU")
            
            # Validate device format
            if self.device.isdigit():
                self.device = f'cuda:{self.device}'
            
            print(f"Setting device to: {self.device}")
            self.model.to(self.device)
            
            print(f"YOLOv11 model loaded successfully on device: {self.device}")
            print(f"Model will detect humans (person class) with confidence >= {self.conf_threshold}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLOv11 model: {e}")
    
    def detect_humans_batch(self, images: List[np.ndarray]) -> List[Tuple]:
        """Detect humans in batch of images"""
        if not images:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(images), self.max_batch_size):
            chunk = images[i:min(i + self.max_batch_size, len(images))]
            
            try:
                # Run inference on batch - only detect person class (class 0)
                batch_results = self.model.predict(
                    chunk, 
                    conf=self.conf_threshold,
                    classes=[0],  # Only detect person class
                    device=self.device,
                    verbose=False
                )
                
                # Process results for each image
                for result in batch_results:
                    boxes = []
                    scores = []
                    class_ids = []
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        # Extract human detections
                        for box in result.boxes:
                            # Get coordinates in xyxy format
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            # Convert to (x, y, w, h) format
                            x, y = int(x1), int(y1)
                            w, h = int(x2 - x1), int(y2 - y1)
                            
                            # Get confidence and class
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            
                            # Only add if it's a person (class 0) and meets confidence threshold
                            if cls == 0 and conf >= self.conf_threshold:
                                boxes.append((x, y, w, h))
                                scores.append(conf)
                                class_ids.append(cls)
                    
                    results.append((boxes, scores, class_ids))
            
            except Exception as e:
                print(f"YOLO inference error: {e}")
                # Return empty results for failed batch
                results.extend([([], [], [])] * len(chunk))
        
        return results


class StreamlinedMaskingProcessor:
    """Optimized masking with GPU acceleration where possible"""
    
    def __init__(self, mask_type: MaskType, mask_color: Tuple[int, int, int]):
        self.mask_type = mask_type
        self.mask_color = mask_color
        
        # Pre-compile kernels for different masking operations
        if CUPY_AVAILABLE:
            self._setup_gpu_kernels()
    
    def _setup_gpu_kernels(self):
        """Setup custom CUDA kernels for masking operations"""
        # Solid color masking kernel
        self.solid_color_kernel = cp.ElementwiseKernel(
            'raw uint8 img, raw int32 boxes, raw uint8 color',
            'raw uint8 out',
            '''
            int y = i / img.shape[2];
            int x = i % img.shape[2];
            int c = (i / (img.shape[1] * img.shape[2])) % img.shape[0];
            
            // Check if pixel is in any bounding box
            bool in_box = false;
            for (int box_idx = 0; box_idx < boxes.size() / 4; box_idx++) {
                int bx = boxes[box_idx * 4];
                int by = boxes[box_idx * 4 + 1];
                int bw = boxes[box_idx * 4 + 2];
                int bh = boxes[box_idx * 4 + 3];
                
                if (x >= bx && x < bx + bw && y >= by && y < by + bh) {
                    in_box = true;
                    break;
                }
            }
            
            if (in_box) {
                out = color[c];
            } else {
                out = img[i];
            }
            ''',
            'solid_color_mask'
        )
    
    def apply_masks_batch(self, image_batch: np.ndarray, detection_batch: List) -> np.ndarray:
        """Apply masks to entire batch efficiently"""
        if len(image_batch) == 0:
            return image_batch
        
        # Convert to GPU if available and beneficial
        if CUPY_AVAILABLE and len(image_batch) > 4:
            return self._apply_masks_gpu_batch(image_batch, detection_batch)
        else:
            return self._apply_masks_cpu_batch(image_batch, detection_batch)
    
    def _apply_masks_gpu_batch(self, image_batch: np.ndarray, detection_batch: List) -> np.ndarray:
        """GPU-accelerated batch masking"""
        gpu_batch = cp.asarray(image_batch)
        
        for i, (boxes, scores, class_ids) in enumerate(detection_batch):
            if not boxes or i >= len(gpu_batch):
                continue
            
            if self.mask_type == MaskType.SOLID_COLOR:
                # Use GPU kernel for solid color masking
                for box in boxes:
                    x, y, w, h = box
                    x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)
                    if w > 0 and h > 0:
                        gpu_batch[i, y:y+h, x:x+w, :] = cp.array(self.mask_color, dtype=cp.uint8)
        
        return cp.asnumpy(gpu_batch)
    
    def _apply_masks_cpu_batch(self, image_batch: np.ndarray, detection_batch: List) -> np.ndarray:
        """Optimized CPU batch masking"""
        result_batch = image_batch.copy()
        
        for i, (boxes, scores, class_ids) in enumerate(detection_batch):
            if not boxes or i >= len(result_batch):
                continue
            
            image = result_batch[i]
            for box in boxes:
                x, y, w, h = box
                x = max(0, min(x, image.shape[1]))
                y = max(0, min(y, image.shape[0]))
                w = min(w, image.shape[1] - x)
                h = min(h, image.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    continue
                
                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                
                if self.mask_type == MaskType.SOLID_COLOR:
                    roi[:] = self.mask_color
                elif self.mask_type == MaskType.BLUR:
                    kernel_size = min(15, max(3, min(w, h) // 5))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    image[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                elif self.mask_type == MaskType.PIXELATE:
                    pixel_size = max(2, min(w, h) // 10)
                    small_h = max(1, h // pixel_size)
                    small_w = max(1, w // pixel_size)
                    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                    image[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        return result_batch


class HumanDetectionProcessor:
    """High-level processor for human detection and masking operations"""
    
    def __init__(self, model_name: str = 'yolo11n.pt', conf_threshold: float = 0.5,
                 batch_size: int = 32, mask_type: str = "solid_color", 
                 mask_color: Tuple[int, int, int] = (0, 0, 0), device: str = 'auto'):
        
        self.yolo_processor = UltralyticsYOLOProcessor(
            model_name, conf_threshold, max_batch_size=batch_size, device=device
        )
        self.mask_processor = StreamlinedMaskingProcessor(
            MaskType(mask_type), mask_color
        )
        self.batch_size = batch_size
    
    def _get_file_extension_info(self, image_paths: List[str]) -> Tuple[str, Dict[str, List[str]]]:
        """
        Analyze file extensions in image paths and return the most common extension
        and a mapping of extensions to file paths.
        """
        extension_counts = {}
        extension_to_paths = {}
        
        for path in image_paths:
            ext = Path(path).suffix.lower()
            if ext not in extension_counts:
                extension_counts[ext] = 0
                extension_to_paths[ext] = []
            extension_counts[ext] += 1
            extension_to_paths[ext].append(path)
        
        # Find most common extension
        most_common_ext = max(extension_counts.keys(), key=extension_counts.get) if extension_counts else '.jpg'
        
        print(f"File extension analysis:")
        for ext, count in sorted(extension_counts.items()):
            print(f"  {ext}: {count} files")
        print(f"Most common extension: {most_common_ext}")
        
        return most_common_ext, extension_to_paths
    
    def _save_image_with_original_format(self, image: np.ndarray, original_path: str, 
                                       output_path: str, force_extension: str = None) -> bool:
        """
        Save image preserving original format or with forced extension.
        
        Args:
            image: Image array in RGB format
            original_path: Original image file path (to get extension)
            output_path: Output file path (may have extension changed)
            force_extension: If provided, force this extension
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Determine the extension to use
            if force_extension:
                # Use forced extension
                output_path = str(Path(output_path).with_suffix(force_extension))
            else:
                # Preserve original extension
                original_ext = Path(original_path).suffix.lower()
                if original_ext:
                    output_path = str(Path(output_path).with_suffix(original_ext))
                else:
                    # Default to .jpg if no extension found
                    output_path = str(Path(output_path).with_suffix('.jpg'))
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Set quality parameters based on format
            ext = Path(output_path).suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                # High quality JPEG
                success = cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif ext == '.png':
                # PNG with compression
                success = cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            elif ext in ['.tiff', '.tif']:
                # TIFF
                success = cv2.imwrite(output_path, image_bgr)
            elif ext == '.webp':
                # WebP with high quality
                success = cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_WEBP_QUALITY, 95])
            else:
                # Default case
                success = cv2.imwrite(output_path, image_bgr)
            
            if not success:
                print(f"Warning: Failed to save image to {output_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error saving image to {output_path}: {e}")
            return False

    async def process_images(self, image_paths: List[str], output_dir: str, 
                           save_detections: bool = True, save_masked: bool = True,
                           force_extension: str = None, preserve_original_format: bool = True):
        """
        Process images with human detection and masking
        
        Args:
            image_paths: List of image file paths
            output_dir: Output directory
            save_detections: Whether to save detection visualizations
            save_masked: Whether to save masked images
            force_extension: If provided, force all output images to use this extension (e.g., '.jpg', '.png')
            preserve_original_format: If True, preserve original file formats (ignored if force_extension is set)
        """
        
        # Analyze file extensions
        most_common_ext, extension_to_paths = self._get_file_extension_info(image_paths)
        
        # Setup output directory
        os.makedirs(output_dir, exist_ok=True)
        if save_detections:
            detection_dir = os.path.join(output_dir, "human_detections")
            os.makedirs(detection_dir, exist_ok=True)
        if save_masked:
            masked_dir = os.path.join(output_dir, "human_masked")
            os.makedirs(masked_dir, exist_ok=True)
        
        # Load images
        print("Loading images...")
        start_time = time.time()
        size_groups = await load_images_async(image_paths)
        print(f"Loading completed in {time.time() - start_time:.2f}s")
        
        total_processed = 0
        total_humans_detected = 0
        processing_stats = []
        
        # Process each size group
        for image_shape, group_data in size_groups.items():
            print(f"\nProcessing {len(group_data)} images of size {image_shape}")
            
            # Process in batches
            for i in range(0, len(group_data), self.batch_size):
                batch_end = min(i + self.batch_size, len(group_data))
                batch_data = group_data[i:batch_end]
                batch_paths, batch_images = zip(*batch_data)
                
                start_time = time.time()
                
                # Human detection
                detection_results = self.yolo_processor.detect_humans_batch(list(batch_images))
                detection_time = time.time() - start_time
                
                # Apply masks to humans
                start_time = time.time()
                masked_images = self.mask_processor.apply_masks_batch(
                    np.array(batch_images), detection_results
                )
                masking_time = time.time() - start_time
                
                # Save results
                batch_humans = 0
                for j, (path, img, masked_img, detection) in enumerate(zip(
                    batch_paths, batch_images, masked_images, detection_results)):
                    
                    img_name = os.path.basename(path)
                    name_only = os.path.splitext(img_name)[0]
                    
                    boxes, scores, class_ids = detection
                    batch_humans += len(boxes)
                    
                    # Determine extension to use
                    ext_to_use = force_extension if force_extension else (Path(path).suffix if preserve_original_format else most_common_ext)
                    
                    # Save detection visualization
                    if save_detections:
                        detection_img = img.copy()
                        for box, score, class_id in zip(boxes, scores, class_ids):
                            x, y, w, h = box
                            # Draw bounding box for humans
                            cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(detection_img, f"Human: {score:.2f}",
                                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        detection_filename = f"human_detection_{name_only}"
                        detection_path = os.path.join(detection_dir, detection_filename)
                        
                        success = self._save_image_with_original_format(
                            detection_img, path, detection_path, ext_to_use
                        )
                        if not success:
                            print(f"Failed to save detection image for {img_name}")
                    
                    # Save masked image (humans blacked out)
                    if save_masked:
                        masked_path = os.path.join(masked_dir, name_only)
                        
                        success = self._save_image_with_original_format(
                            masked_img, path, masked_path, ext_to_use
                        )
                        if not success:
                            print(f"Failed to save masked image for {img_name}")
                    
                    total_processed += 1
                
                total_humans_detected += batch_humans
                processing_stats.append({
                    'batch_size': len(batch_images),
                    'detection_time': detection_time,
                    'masking_time': masking_time,
                    'total_time': detection_time + masking_time,
                    'humans_detected': batch_humans
                })
                
                print(f"Batch {i//self.batch_size + 1}: {len(batch_images)} images, "
                      f"{batch_humans} humans detected in {detection_time + masking_time:.2f}s")
        
        # Print summary
        print(f"\n=== Human Detection & Masking Complete ===")
        print(f"Successfully processed: {total_processed} images")
        print(f"Total humans detected: {total_humans_detected}")
        if force_extension:
            print(f"All output images saved with extension: {force_extension}")
        elif preserve_original_format:
            print(f"Original file formats preserved")
        else:
            print(f"All output images saved with most common extension: {most_common_ext}")
        
        if save_detections:
            print(f"Human detection visualizations saved to: {detection_dir}")
        if save_masked:
            print(f"Human-masked images saved to: {masked_dir}")
        
        if processing_stats:
            avg_time = np.mean([s['total_time'] for s in processing_stats])
            avg_detection_time = np.mean([s['detection_time'] for s in processing_stats])
            avg_masking_time = np.mean([s['masking_time'] for s in processing_stats])
            avg_humans = np.mean([s['humans_detected'] for s in processing_stats])
            
            print(f"Average batch processing time: {avg_time:.3f}s")
            print(f"  - Human detection: {avg_detection_time:.3f}s")
            print(f"  - Masking: {avg_masking_time:.3f}s")
            print(f"Average humans detected per batch: {avg_humans:.1f}")


async def main():
    parser = argparse.ArgumentParser(description="Human detection and masking using YOLOv11")
    parser.add_argument('--image_dir', type=str, help="Directory containing images")
    parser.add_argument('--image_paths', type=str, nargs='+', help="Specific image paths")
    parser.add_argument('--model_name', type=str, default='yolo11m.pt', 
                       help="YOLOv11 model name (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)")
    parser.add_argument('--output_dir', type=str, default="./human_detection_output", help="Output directory")
    parser.add_argument('--conf_threshold', type=float, default=0.1, help="Confidence threshold")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--mask_type', type=str, default="solid_color", 
                       choices=["solid_color", "blur", "pixelate"], help="Masking type")
    parser.add_argument('--mask_color', type=int, nargs=3, default=[0, 0, 0], help="Mask color (R G B) - default: black")
    parser.add_argument('--device', type=str, default='auto', help="Device to run on (auto, cpu, 0, 1, ...)")
    parser.add_argument('--save_detections', action='store_true', help="Save human detection visualizations")
    parser.add_argument('--save_masked', action='store_true', default=True, help="Save human-masked images")
    
    # New arguments for format control
    parser.add_argument('--force_extension', type=str, default=None, 
                       help="Force all output images to use this extension (e.g., .jpg, .png)")
    parser.add_argument('--preserve_original_format', action='store_true', default=True,
                       help="Preserve original file formats (default: True)")
    
    args = parser.parse_args()
    
    if not ULTRALYTICS_AVAILABLE:
        print("ERROR: Ultralytics YOLO is required for human detection")
        print("Install with: pip install ultralytics")
        exit(1)
    
    # Get image paths
    if args.image_dir:
        image_paths = find_images(args.image_dir)
    elif args.image_paths:
        image_paths = args.image_paths
    else:
        parser.error("Either --image_dir or --image_paths must be specified")
    
    print(f"Found {len(image_paths)} images")
    if not image_paths:
        print("No images found!")
        return
    
    # Initialize processor
    processor = HumanDetectionProcessor(
        model_name=args.model_name,
        conf_threshold=args.conf_threshold,
        batch_size=args.batch_size,
        mask_type=args.mask_type,
        mask_color=tuple(args.mask_color),
        device=args.device
    )
    
    print("Starting human detection and masking...")
    print(f"Configuration:")
    print(f"  - Model: {args.model_name}")
    print(f"  - Device: {args.device}")
    print(f"  - Confidence threshold: {args.conf_threshold}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Mask type: {args.mask_type}")
    print(f"  - Mask color: {args.mask_color} (black by default)")
    print(f"  - Target: Humans only (person class)")
    print(f"  - Force extension: {args.force_extension}")
    print(f"  - Preserve original format: {args.preserve_original_format}")
    
    # Process images
    await processor.process_images(
        image_paths=image_paths,
        output_dir=args.output_dir,
        save_detections=args.save_detections,
        save_masked=args.save_masked,
        force_extension=args.force_extension,
        preserve_original_format=args.preserve_original_format
    )


if __name__ == "__main__":
    asyncio.run(main())