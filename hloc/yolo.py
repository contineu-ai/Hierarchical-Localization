import cv2
import numpy as np
import onnxruntime as ort
from enum import Enum
from typing import Tuple, List, Optional
import random

class MaskType(Enum):
    SOLID_COLOR = "solid_color"
    BLUR = "blur"
    PIXELATE = "pixelate"

class YOLOv7:
    def __init__(self, model_path: str, conf_threshold: float = 0.5, iou_threshold: float = 0.4):
        """
        Initialize YOLO model
        
        Args:
            model_path: Path to ONNX model file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Class names - update these for your model
        self.class_names = ["helmet", "person"]
        
        # Generate random colors for each class
        random.seed(42)  # Fixed seed for consistent colors
        self.colors = [(random.randint(64, 255), random.randint(64, 255), random.randint(64, 255)) 
                      for _ in range(len(self.class_names))]
        
        # Initialize model
        self.session = None
        self.input_width = 480
        self.input_height = 480
        self.input_name = None
        self.output_names = None
        self.img_width = 0
        self.img_height = 0
        
        # Detection results
        self.boxes = []
        self.scores = []
        self.class_ids = []
        
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str):
        """Initialize ONNX Runtime session"""
        try:
            # Create session options
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # Get input details
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            input_shape = input_info.shape
            
            if len(input_shape) == 4:
                self.input_height = input_shape[2] if input_shape[2] > 0 else 640
                self.input_width = input_shape[3] if input_shape[3] > 0 else 640
            
            # Get output details
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"YOLOv7 model loaded successfully")
            print(f"Input shape: {input_shape}")
            print(f"Available providers: {self.session.get_providers()}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def _prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Prepare input tensor from image"""
        self.img_height, self.img_width = image.shape[:2]
        
        # Convert BGR to RGB
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        
        # Normalize to [0, 1]
        input_img = input_img.astype(np.float32) / 255.0
        
        # Convert to NCHW format
        input_tensor = np.transpose(input_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def _run_inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Run inference on input tensor"""
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return outputs
    
    def _process_output(self, outputs: List[np.ndarray]) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[int]]:
        """Process model outputs to extract detections"""
        boxes = []
        scores = []
        class_ids = []
        
        if not outputs or len(outputs) == 0:
            return boxes, scores, class_ids
        
        output = outputs[0]
        
        # Calculate scaling factors
        scale_x = self.img_width / self.input_width
        scale_y = self.img_height / self.input_height
        
        # Process different output formats
        if len(output.shape) == 2:
            # ONNX-RT format: [N, 7] - [batch_idx, x1, y1, x2, y2, conf, class_id]
            num_detections = output.shape[0]
            
            for i in range(num_detections):
                detection = output[i]
                
                if len(detection) >= 7:
                    confidence = detection[6]
                    if confidence >= self.conf_threshold:
                        x1 = detection[1] * scale_x
                        y1 = detection[2] * scale_y
                        x2 = detection[3] * scale_x
                        y2 = detection[4] * scale_y
                        class_id = int(detection[5])
                        
                        # Clamp to image bounds
                        x1 = max(0, min(self.img_width, x1))
                        y1 = max(0, min(self.img_height, y1))
                        x2 = max(0, min(self.img_width, x2))
                        y2 = max(0, min(self.img_height, y2))
                        
                        if x2 > x1 and y2 > y1:
                            boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                            scores.append(confidence)
                            class_ids.append(class_id)
        
        elif len(output.shape) == 3:
            # Alternative format: [batch, num_detections, attributes]
            if output.shape[0] == 1:  # batch size 1
                detections = output[0]
                for detection in detections:
                    if len(detection) >= 6:
                        # Assuming format: [x_center, y_center, width, height, confidence, class_scores...]
                        x_center, y_center, width, height = detection[:4]
                        confidence = detection[4]
                        
                        if confidence >= self.conf_threshold:
                            # Convert to class probabilities and find max
                            class_scores = detection[5:]
                            class_id = np.argmax(class_scores)
                            class_confidence = class_scores[class_id] * confidence
                            
                            if class_confidence >= self.conf_threshold:
                                # Convert center format to corner format
                                x1 = (x_center - width / 2) * scale_x
                                y1 = (y_center - height / 2) * scale_y
                                w = width * scale_x
                                h = height * scale_y
                                
                                # Clamp to image bounds
                                x1 = max(0, min(self.img_width - w, x1))
                                y1 = max(0, min(self.img_height - h, y1))
                                w = min(w, self.img_width - x1)
                                h = min(h, self.img_height - y1)
                                
                                if w > 0 and h > 0:
                                    boxes.append((int(x1), int(y1), int(w), int(h)))
                                    scores.append(class_confidence)
                                    class_ids.append(class_id)
        
        return boxes, scores, class_ids
    
    def detect_objects(self, image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[float], List[int]]:
        """
        Detect objects in image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        if image is None or image.size == 0:
            return [], [], []
        
        # Prepare input
        input_tensor = self._prepare_input(image)
        
        # Run inference
        outputs = self._run_inference(input_tensor)
        
        # Process outputs
        boxes, scores, class_ids = self._process_output(outputs)
        
        # Store results
        self.boxes = boxes
        self.scores = scores
        self.class_ids = class_ids
        
        return boxes, scores, class_ids
    
    def draw_detections(self, image: np.ndarray) -> np.ndarray:
        """Draw detection boxes and labels on image"""
        result = image.copy()
        
        for i, (box, score, class_id) in enumerate(zip(self.boxes, self.scores, self.class_ids)):
            x, y, w, h = box
            
            # Get class name and color
            class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else "Unknown"
            color = self.colors[class_id] if 0 <= class_id < len(self.colors) else (255, 255, 255)
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name} {int(score * 100)}%"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(result, (x, y - text_height - baseline - 5), 
                         (x + text_width, y), color, -1)
            
            # Draw label text
            cv2.putText(result, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return result
    
    def mask_detections(self, image: np.ndarray, mask_type: MaskType = MaskType.SOLID_COLOR, 
                       mask_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Apply masking to detected regions
        
        Args:
            image: Input image
            mask_type: Type of masking to apply
            mask_color: Color for solid color masking (BGR format)
            
        Returns:
            Image with masked detections
        """
        return mask_detections(image, self.boxes, self.scores, self.class_ids, mask_type, mask_color)

def mask_detections(image: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                   scores: List[float], class_ids: List[int], 
                   mask_type: MaskType = MaskType.SOLID_COLOR, 
                   mask_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Apply masking to detected regions
    
    Args:
        image: Input image
        boxes: List of bounding boxes (x, y, width, height)
        scores: Detection confidence scores
        class_ids: Class IDs for each detection
        mask_type: Type of masking to apply
        mask_color: Color for solid color masking (BGR format)
        
    Returns:
        Image with masked detections
    """
    result = image.copy()
    
    for box in boxes:
        x, y, w, h = box
        
        # Ensure box is within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            continue
        
        # Extract ROI
        roi = result[y:y+h, x:x+w]
        
        if roi.size == 0:
            continue
        
        if mask_type == MaskType.SOLID_COLOR:
            # Fill with solid color
            roi[:] = mask_color
            
        elif mask_type == MaskType.BLUR:
            # Apply Gaussian blur
            blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
            result[y:y+h, x:x+w] = blurred_roi
            
        elif mask_type == MaskType.PIXELATE:
            # Pixelate the region
            pixel_size = max(2, min(w, h) // 10)
            
            # Downsample
            small_h = max(1, h // pixel_size)
            small_w = max(1, w // pixel_size)
            small_roi = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            
            # Upsample with nearest neighbor
            pixelated_roi = cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)
            result[y:y+h, x:x+w] = pixelated_roi
    
    return result

# Example usage
def main():
    # Initialize YOLO model
    model_path = "/data/sahil/sfm/modded/sfm/Hierarchical-Localization/best.onnx"  # Replace with your model path
    yolo = YOLOv7(model_path, conf_threshold=0.5)
    
    # Load image
    image_path = "/data/sahil/sfm/modded/sfm/Hierarchical-Localization/dicemap_00000004.jpg"  # Replace with your image path
    image = cv2.imread(image_path)
    print (image.shape)
    if image is None:
        print("Could not load image")
        return
    image = image[960:,480:960]
    # Detect objects
    boxes, scores, class_ids = yolo.detect_objects(image)
    print(f"Detected {len(boxes)} objects")
    
    # Draw detections
    result_with_boxes = yolo.draw_detections(image)
    cv2.imwrite("detections.jpg", result_with_boxes)
    
    # Apply different types of masking
    
    # Solid color masking (black)
    masked_solid = yolo.mask_detections(image, MaskType.SOLID_COLOR, (0, 0, 0))
    cv2.imwrite("masked_solid_black.jpg", masked_solid)
    
    # Solid color masking (red)
    masked_red = yolo.mask_detections(image, MaskType.SOLID_COLOR, (0, 0, 255))
    cv2.imwrite("masked_solid_red.jpg", masked_red)
    
    # Blur masking
    masked_blur = yolo.mask_detections(image, MaskType.BLUR)
    cv2.imwrite("masked_blur.jpg", masked_blur)
    
    # Pixelate masking
    masked_pixelate = yolo.mask_detections(image, MaskType.PIXELATE)
    cv2.imwrite("masked_pixelate.jpg", masked_pixelate)
    
    print("Masking complete! Check the output images.")

if __name__ == "__main__":
    main()