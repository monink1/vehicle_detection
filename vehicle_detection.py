#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from PIL import Image
from model import ShuffleNetV2
from torchvision import models
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from typing import Dict, Any, Optional, Tuple, Union

def enhance_vehicle_region(image: np.ndarray) -> np.ndarray:
    """
    Enhance vehicle region image to reduce lighting impact.
    
    Args:
        image: Input image in RGB format
        
    Returns:
        Enhanced image in RGB format
    """
    if image.size == 0:
        return image
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels and convert back to RGB
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Adjust contrast and brightness
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
    
    return enhanced

class BaseColorClassifier:
    """Base class for color classifiers with common functionality."""
    def __init__(self, weights_path: str, device: str = 'cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
        self.model = None
        self.transform = None
        self.color_map = {}
        self._load_model(weights_path)
        
    def _load_model(self, weights_path: str) -> None:
        """Load model weights and initialize model."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict_color(self, image: Union[np.ndarray, Image.Image]) -> str:
        """Predict the color of a vehicle image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            
        Returns:
            Predicted color as string
        """
        try:
            image_tensor = self._preprocess_image(image)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                return self.color_map.get(predicted.item(), 'unknown')
        except Exception as e:
            print(f"Error in color prediction: {e}")
            return 'unknown'


class ShuffleNetColorClassifier(BaseColorClassifier):
    """Color classifier using ShuffleNetV2 backbone."""
    def _load_model(self, weights_path: str) -> None:
        """Initialize ShuffleNetV2 model and load weights."""
        print(f"Loading ShuffleNet color classifier on device: {self.device}")
        
        # Load model weights
        checkpoint = torch.load(weights_path, map_location=str(self.device))
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Initialize model
        self.model = ShuffleNetV2(scale=1.0, num_classes=12)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Color mapping
        self. ShuffleNetColorClassifier = {
            0: 'white', 1: 'black', 2: 'red', 3: 'blue',
            4: 'gray', 5: 'yellow', 6: 'green', 7: 'brown',
            8: 'pink', 9: 'orange', 10: 'purple', 11: 'cyan'
        }


class MobileNetColorClassifier(BaseColorClassifier):
    """Color classifier using MobileNetV2 backbone."""
    def _load_model(self, weights_path: str) -> None:
        """Initialize MobileNetV2 model and load weights."""
        print(f"Loading MobileNet color classifier on device: {self.device}")
        
        # Initialize model with custom classifier
        self.model = models.mobilenet_v2(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 15)
        
        # Load weights with error handling for DataParallel
        try:
            state_dict = torch.load(weights_path, map_location=str(self.device))
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading weights: {e}")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Color mapping
        self.color_map = {
            0: "white0", 1: "black1", 2: "blue2", 3: "red3", 4: "yellow4",
            5: "gray5", 6: "gray6", 7: "red7", 8: "red8", 9: "red9",
            10: "red10", 11: "gray11", 12: "gray12", 13: "white13", 14: "yellow14"
        }


def create_color_classifier(network_type: str, weights_path: str, device: str = 'cuda:0') -> BaseColorClassifier:
    """Factory function to create the appropriate color classifier.
    
    Args:
        network_type: Type of network ('shufflenet' or 'mobilenet')
        weights_path: Path to model weights
        device: Device to run on ('cuda:0' or 'cpu')
        
    Returns:
        Initialized color classifier instance
    """
    if network_type == 'mobilenet':
        return MobileNetColorClassifier(weights_path, device)
    return ShuffleNetColorClassifier(weights_path, device)

class VehicleTracker:
    """Handles vehicle tracking and color classification in video streams."""
    
    # Vehicle class mapping for YOLO model
    VEHICLE_CLASSES = {
        2: 'car',
        5: 'bus',
        7: 'truck',
        3: 'motorcycle',
        1:'bicycle'
    }
    
    def __init__(self, yolo_model, color_classifier, device: str = 'cuda:0'):
        """Initialize the vehicle tracker.
        
        Args:
            yolo_model: Initialized YOLO model for object detection
            color_classifier: Initialized color classifier
            device: Device to run on ('cuda:0' or 'cpu')
        """
        self.yolo_model = yolo_model
        self.color_classifier = color_classifier
        self.device = device
        self.tracked_objects = {}
        self.frame_count = 0
        self.last_color_update_frame = 0  # Track when we last updated colors
    
    def _process_detections(self, frame: np.ndarray, results: Any, width: int, height: int) -> np.ndarray:
        """Process detections from YOLO and draw bounding boxes with labels."""
        if results[0].boxes.id is None:
            return frame
            
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        class_ids = results[0].boxes.cls.int().cpu().numpy()
        
        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width-1, x2), min(height-1, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Get class name
            class_name = self.VEHICLE_CLASSES.get(
                int(cls_id), 
                self.yolo_model.names.get(int(cls_id), f'class_{int(cls_id)}')
            )
            
            # Update or initialize track
            self._update_track(track_id, class_name, frame, (x1, y1, x2, y2))
            
            # Get object info and draw
            obj_info = self.tracked_objects[track_id]
            frame = self._draw_detection(frame, (x1, y1, x2, y2), track_id, 
                                       obj_info['class'], obj_info['color'])
        
        return frame
    
    def _update_track(self, track_id: int, class_name: str, 
                     frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Update tracking information for a detected vehicle."""
        x1, y1, x2, y2 = bbox
        is_new_id = track_id not in self.tracked_objects
        
        if is_new_id:
            # Initialize new track
            self.tracked_objects[track_id] = {
                'class': class_name,
                'color': 'unknown',
                'last_seen': self.frame_count,
                'detect_count': 1,
                'samples': [],
                'needs_color_update': True  # Flag to indicate new ID needs color update
            }
        else:
            # Update existing track
            self.tracked_objects[track_id]['last_seen'] = self.frame_count
            self.tracked_objects[track_id]['detect_count'] += 1
        
        # Check if we should update colors (every 50 frames or for new IDs)
        should_update = (self.frame_count - self.last_color_update_frame >= 25) or is_new_id
        
        # Only update color if we've seen the object a few times and it's time to update
        if (self.tracked_objects[track_id]['detect_count'] >= 5 and 
            (should_update or self.tracked_objects[track_id].get('needs_color_update', False))):
            
            vehicle_region = frame[y1:y2, x1:x2]
            if vehicle_region.size > 0:
                try:
                    enhanced_region = enhance_vehicle_region(vehicle_region)
                    color = self.color_classifier.predict_color(enhanced_region)
                    self.tracked_objects[track_id]['color'] = color
                    self.tracked_objects[track_id]['needs_color_update'] = False
                    
                    # Update the last color update frame if this is a periodic update
                    if not is_new_id and should_update:
                        self.last_color_update_frame = self.frame_count
                        
                except Exception as e:
                    print(f"Error processing vehicle region: {e}")
                    color = self.color_classifier.predict_color(vehicle_region)
                    self.tracked_objects[track_id]['color'] = color
                    self.tracked_objects[track_id]['needs_color_update'] = False
    
    @staticmethod
    def _draw_detection(frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                       track_id: int, class_name: str, color_name: str) -> np.ndarray:
        """Draw bounding box and label on the frame."""
        x1, y1, x2, y2 = bbox
        
        # Generate consistent color based on track_id
        color = tuple(map(int, colors(int(track_id), True)))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = f"id:{track_id} {class_name} {color_name}"
        
        # Calculate text size and position
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        text_y = max(15, y1 - 10)
        
        # Draw background for text
        cv2.rectangle(
            frame,
            (x1, text_y - text_height - 5),
            (x1 + text_width, text_y + 5),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            1,
            cv2.LINE_AA
        )
        
        return frame
    
    def _cleanup_old_tracks(self, max_age: int = 30) -> None:
        """Remove tracks that haven't been seen for max_age frames."""
        self.tracked_objects = {
            tid: obj for tid, obj in self.tracked_objects.items()
            if self.frame_count - obj['last_seen'] < max_age
        }
    
    def process_video(self, video_path: str, output_path: str, 
                     conf_thres: float = 0.5, show_result: bool = True,
                     selected_classes: Optional[list] = None) -> None:
        """Process video with vehicle tracking and color classification.
        
        Args:
            video_path: Path to input video or '0' for webcam
            output_path: Path to save output video
            conf_thres: Confidence threshold for detection
            show_result: Whether to display the result in a window
            selected_classes: List of class IDs to detect (None for all)
        """
        # Initialize video capture
        is_webcam = video_path == '0' or video_path == 0
        cap = cv2.VideoCapture(int(video_path) if is_webcam else video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
        
        # Print video info
        print(f"\n{'Webcam' if is_webcam else 'Video'} properties:")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames if total_frames > 0 else 'Live stream'}")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("\nStarting processing... (Press 'q' to quit)")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Webcam disconnected" if is_webcam else "End of video")
                    break
                
                self.frame_count += 1
                if self.frame_count % 10 == 0 or self.frame_count == 1:
                    print(f"\rProcessing frame {self.frame_count}" + 
                         (f" of {total_frames}" if total_frames > 0 else ""), 
                         end="")
                
                # Convert to RGB for YOLO
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run YOLO tracking
                track_args = {
                    'source': frame_rgb,
                    'persist': True,
                    'conf': conf_thres,
                    'verbose': False,
                    'device': self.device,
                    'half': True,
                    'imgsz': 640
                }
                
                if selected_classes is not None:
                    track_args['classes'] = selected_classes
                
                results = self.yolo_model.track(**track_args)
                
                # Process detections and draw on frame
                frame = self._process_detections(frame, results, width, height)
                
                # Clean up old tracks
                self._cleanup_old_tracks()
                
                # Write frame to output
                out.write(frame)
                
                # Show result if requested
                if show_result:
                    cv2.imshow('Vehicle Tracking', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nUser interrupted processing")
                        break
                        
        except KeyboardInterrupt:
            print("\nProcessing stopped by user")
        except Exception as e:
            print(f"\nError during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Release resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"\nProcessing complete. Output saved to: {os.path.abspath(output_path)}")

def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='Vehicle Detection with Color Classification')
    parser.add_argument('--video', type=str, default='0',
                      help='Path to input video or 0 for webcam (default: 0)')
    parser.add_argument('--output', type=str, default='output_detection.mp4',
                      help='Path to output video (default: output_detection.mp4)')
    parser.add_argument('--yolo-weights', type=str, default='yolov8n.pt',
                      help='Path to YOLOv8 weights (default: yolov8n.pt)')
    parser.add_argument('--color-weights', type=str, required=True,
                      help='Path to color classifier weights (required)')
    parser.add_argument('--network', type=str, default='shufflenet',
                      choices=['shufflenet', 'mobilenet'],
                      help='Backbone network for color classification (default: shufflenet)')
    parser.add_argument('--conf', type=float, default=0.5,
                      help='Confidence threshold (default: 0.5)')
    return parser.parse_args()


def print_config(args: argparse.Namespace, device: str) -> None:
    """Print configuration information."""
    print("=" * 50)
    print("Vehicle Detection with Color Classification")
    print("=" * 50)
    print(f"Input: {args.video}")
    print(f"Output: {args.output}")
    print(f"YOLO weights: {args.yolo_weights}")
    print(f"Color weights: {args.color_weights}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Device: {device}")
    print("=" * 50 + "\n")


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if color weights exist
    if not os.path.isfile(args.color_weights):
        print(f"Error: Color weights file not found: {args.color_weights}")
        return
    
    # Initialize device - default to cuda:0 if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print_config(args, device)
    
    try:
        # Initialize YOLO model on GPU
        print("Initializing YOLO model...")
        yolo_model = YOLO(args.yolo_weights, task='detect').to(device)
        
        # Vehicle classes are defined in the VehicleTracker class
        color_classifier = create_color_classifier(
            args.network, 
            args.color_weights, 
            device=device
        )
        
        # Initialize vehicle tracker
        tracker = VehicleTracker(yolo_model, color_classifier, device=device)
        
        # Process video
        tracker.process_video(
            video_path=args.video,
            output_path=args.output,
            conf_thres=args.conf
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()