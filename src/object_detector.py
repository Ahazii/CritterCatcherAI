"""
Object detection module using YOLOv8 for real-time object detection.
Provides fast and accurate detection with bounding boxes.
"""
import logging
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple
from ultralytics import YOLO
import json
from datetime import datetime
import fcntl
import time

logger = logging.getLogger(__name__)

# YOLOv8 COCO dataset classes (80 classes)
YOLO_COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class ObjectDetector:
    """Object detector using YOLOv8."""
    
    def __init__(self, labels: List[str], confidence_threshold: float = 0.25, num_frames: int = 5, 
                 detected_objects_path: str = "/data/objects/detected", 
                 model_name: str = "yolov8n.pt"):
        """
        Initialize the object detector.
        
        Args:
            labels: List of object labels to detect (must be YOLO COCO classes)
            confidence_threshold: Minimum confidence score to consider a detection valid
            num_frames: Number of frames to extract and analyze from each video
            detected_objects_path: Path to save detected object images
            model_name: YOLO model to use (yolov8n/s/m/l/x.pt)
        """
        self.labels = [label.lower() for label in labels]
        self.model_name = model_name
        
        # Validate labels against YOLO COCO classes
        yolo_classes_lower = [c.lower() for c in YOLO_COCO_CLASSES]
        invalid_labels = [label for label in self.labels if label not in yolo_classes_lower]
        
        if invalid_labels:
            logger.warning(f"Invalid YOLO labels detected and will be ignored: {invalid_labels}")
            logger.warning(f"Valid YOLO COCO classes: {', '.join(YOLO_COCO_CLASSES[:20])}...")
            # Filter out invalid labels
            self.labels = [label for label in self.labels if label in yolo_classes_lower]
        self.confidence_threshold = confidence_threshold
        self.num_frames = num_frames
        self.detected_objects_path = Path(detected_objects_path)
        self.detected_objects_path.mkdir(parents=True, exist_ok=True)
        
        self.all_labels = list(self.labels)
        
        logger.info(f"Initializing YOLOv8 model: {self.model_name}")
        
        # Check if CUDA is available and configure device
        import torch
        import os
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available - using GPU: {gpu_name}")
            logger.info(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}")
            logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        else:
            device = 'cpu'
            logger.warning("CUDA not available - using CPU (performance will be significantly slower)")
        
        self.model = YOLO(self.model_name)
        # Move model to GPU if available
        self.model.to(device)
        logger.info(f"Model loaded: {self.model_name} on {device} - {len(YOLO_COCO_CLASSES)} COCO classes available")
        
        logger.info(f"Initialized ObjectDetector with {len(self.labels)} tracked labels")
        if not self.labels:
            logger.error("No valid labels configured! Please check your config.yaml")
    
    def extract_frames(self, video_path: Path, num_frames: int = 5) -> List[np.ndarray]:
        """
        Extract representative frames from video with robust error handling.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of frame images as numpy arrays
        """
        # Try multiple backends if first fails
        backends = [
            (cv2.CAP_FFMPEG, "FFMPEG"),
            (cv2.CAP_ANY, "ANY"),
        ]
        
        cap = None
        for backend, name in backends:
            try:
                cap = cv2.VideoCapture(str(video_path), backend)
                if cap.isOpened():
                    logger.debug(f"Opened video with {name} backend")
                    break
                else:
                    cap = None
            except Exception as e:
                logger.debug(f"Failed to open with {name} backend: {e}")
                cap = None
        
        if cap is None or not cap.isOpened():
            logger.error(f"Failed to open video with any backend: {video_path}")
            return []
        
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Validate video properties
            if total_frames <= 0 or width <= 0 or height <= 0:
                logger.error(f"Invalid video properties: {width}x{height}, {total_frames} frames")
                cap.release()
                return []
            
            logger.debug(f"Video properties: {width}x{height}, {total_frames} frames")
            
            # Extract frames evenly distributed throughout the video
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        # Validate frame dimensions
                        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
                            logger.warning(f"Invalid frame dimensions at index {idx}: {frame.shape}")
                            continue
                        
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                    else:
                        logger.debug(f"Failed to read frame at index {idx}")
                except Exception as e:
                    logger.warning(f"Error extracting frame {idx}: {e}")
                    continue
            
            cap.release()
            logger.debug(f"Extracted {len(frames)}/{num_frames} frames from {video_path.name}")
            return frames
            
        except Exception as e:
            logger.error(f"Error during frame extraction: {e}", exc_info=True)
            if cap:
                cap.release()
            return []
    
    def detect_objects_in_frame(self, frame: np.ndarray, video_name: str = None, frame_idx: int = None, save_detections: bool = True, return_bboxes: bool = False) -> Dict[str, float]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Image frame as numpy array (RGB)
            video_name: Name of the video being processed
            frame_idx: Frame index in the video
            save_detections: Whether to save detected object images
            return_bboxes: Whether to return bbox coordinates
            
        Returns:
            Dictionary mapping label to confidence score (or to dict with confidence and bbox if return_bboxes=True)
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        # Process detections
        detections = {}
        
        # Collect all detections first
        all_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = result.names[class_id].lower()
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                if confidence > 0.01:
                    is_focused = label in self.labels
                    all_boxes.append({
                        'label': label,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'is_focused': is_focused
                    })
        
        # Save a separate annotated image for each detected object
        for box_info in all_boxes:
            label = box_info['label']
            confidence = box_info['confidence']
            bbox = box_info['bbox']
            is_focused = box_info['is_focused']
            
            # Create a fresh copy for this specific detection
            annotated_frame = frame.copy()
            
            # First pass: Draw all boxes faded (context)
            for other_box in all_boxes:
                ox1, oy1, ox2, oy2 = other_box['bbox']
                other_label = other_box['label']
                other_conf = other_box['confidence']
                
                # Faded gray color for context boxes
                faded_color = (180, 180, 180)
                cv2.rectangle(annotated_frame, (ox1, oy1), (ox2, oy2), faded_color, 1)
                
                # Small faded label
                label_text = f"{other_label} {other_conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                if oy1 < th + baseline + 10:
                    ly = oy1 + th + baseline + 3
                    cv2.rectangle(annotated_frame, (ox1, oy1), (ox1 + tw + 2, ly), faded_color, -1)
                    cv2.putText(annotated_frame, label_text, (ox1 + 1, oy1 + th + 1),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
                else:
                    cv2.rectangle(annotated_frame, (ox1, oy1 - th - baseline - 3), (ox1 + tw, oy1), faded_color, -1)
                    cv2.putText(annotated_frame, label_text, (ox1, oy1 - baseline - 1),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
            
            # Second pass: Highlight THIS detection (the one we're saving for)
            x1, y1, x2, y2 = bbox
            
            # Choose bright color for the relevant detection
            if is_focused:
                color = (0, 255, 0)  # Bright green
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw thick highlighted rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw bold label for highlighted detection with improved placement
            label_text = f"{label} {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            # Calculate label box dimensions with padding
            label_padding = 4
            label_height = text_height + baseline + (2 * label_padding)
            label_width = text_width + (2 * label_padding)
            
            # Try to place label above box
            if y1 - label_height >= 0:
                # Place above
                label_y1 = y1 - label_height
                label_y2 = y1
                text_y = y1 - baseline - label_padding
            else:
                # Place inside box at top
                label_y1 = y1
                label_y2 = y1 + label_height
                text_y = y1 + text_height + label_padding
            
            # Ensure label doesn't go off right edge
            label_x1 = x1
            label_x2 = min(x1 + label_width, annotated_frame.shape[1])
            
            # Draw semi-transparent background for label
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            # Draw label text
            cv2.putText(annotated_frame, label_text, (x1 + label_padding, text_y),
                       font, font_scale, (0, 0, 0), font_thickness)
            
            # Save this annotated image (only for focused labels)
            if save_detections and is_focused:
                self._save_detected_object(
                    annotated_frame, label, confidence,
                    video_name, frame_idx,
                    bbox=bbox
                )
            
            # Process focused labels - keep highest confidence
            if is_focused and confidence >= self.confidence_threshold:
                if return_bboxes:
                    if label not in detections or confidence > detections[label].get('confidence', 0):
                        detections[label] = {
                            'confidence': confidence,
                            'bbox': {'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3]}
                        }
                else:
                    if label not in detections or confidence > detections[label]:
                        detections[label] = confidence
        
        return detections
    
    def detect_objects_in_video(self, video_path: Path, num_frames: int = None, return_bboxes: bool = False) -> Dict[str, float]:
        """
        Detect objects across entire video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to analyze (defaults to instance setting)
            return_bboxes: Whether to return bbox coordinates
            
        Returns:
            Dictionary mapping label to max confidence score (or to dict with confidence and bbox if return_bboxes=True)
        """
        if num_frames is None:
            num_frames = self.num_frames
        
        logger.info(f"Analyzing video: {video_path.name} ({num_frames} frames)")
        
        # Extract frames
        frames = self.extract_frames(video_path, num_frames=num_frames)
        
        if not frames:
            logger.warning(f"No frames extracted from {video_path.name}")
            return {}
        
        # Detect objects in each frame
        all_detections = {}
        for i, frame in enumerate(frames):
            frame_detections = self.detect_objects_in_frame(
                frame,
                video_name=video_path.name,
                frame_idx=i,
                save_detections=True,
                return_bboxes=return_bboxes
            )
            logger.debug(f"Frame {i+1}/{len(frames)}: {frame_detections}")
            
            # Keep the maximum confidence for each label
            if return_bboxes:
                for label, data in frame_detections.items():
                    if label not in all_detections or data['confidence'] > all_detections[label]['confidence']:
                        all_detections[label] = data
            else:
                for label, confidence in frame_detections.items():
                    if label not in all_detections or confidence > all_detections[label]:
                        all_detections[label] = confidence
        
        logger.info(f"Detections for {video_path.name}: {all_detections}")
        return all_detections
    
    
    def get_best_detection(self, detections: Dict[str, float]) -> Tuple[str, float]:
        """
        Get the label with highest confidence.
        
        Args:
            detections: Dictionary mapping label to confidence
            
        Returns:
            Tuple of (best_label, confidence) or (None, 0.0) if no detections
        """
        if not detections:
            return None, 0.0
        
        best_label = max(detections, key=detections.get)
        return best_label, detections[best_label]
    
    def classify_video(self, video_path: Path) -> str:
        """
        Classify video and return the most likely label.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Best matching label or "unknown" if no confident detection
        """
        detections = self.detect_objects_in_video(video_path)
        best_label, confidence = self.get_best_detection(detections)
        
        if best_label:
            logger.info(f"Classified {video_path.name} as '{best_label}' (confidence: {confidence:.2f})")
            return best_label
        else:
            logger.info(f"No confident detection for {video_path.name}, classifying as 'unknown'")
            return "unknown"
    
    def _save_detected_object(self, frame: np.ndarray, label: str, confidence: float,
                             video_name: str, frame_idx: int,
                             bbox: tuple = None) -> bool:
        """
        Save a detected object image and metadata.
        Auto-confirms detections above threshold to confirmed folder.
        
        Args:
            frame: Frame image with detected object
            label: Detected object label
            confidence: Detection confidence score
                    video_name: Name of source video
            frame_idx: Frame index
            
        Returns:
            True if saved, False if skipped
        """
        try:
            # Load config to check auto-confirm threshold
            import yaml
            config_path = Path("/app/config/config.yaml")
            auto_confirm_threshold = 0.85  # Default
            max_confirmed = 200  # Default
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    auto_confirm_threshold = config.get('image_review', {}).get('auto_confirm_threshold', 0.85)
                    max_confirmed = config.get('image_review', {}).get('max_confirmed_images', 200)
            
            # Determine if auto-confirm applies
            should_auto_confirm = confidence >= auto_confirm_threshold
            
            # Create label directory
            label_dir = self.detected_objects_path / label.replace(" ", "_")
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # Deduplication: Check if we already saved an image from this video+label recently
            # This prevents saving 5+ nearly identical frames from the same video
            video_base = Path(video_name).stem if video_name else "unknown"
            
            # Check both pending and confirmed folders for existing images from this video
            existing_files = list(label_dir.glob(f"*_{video_base}_*.jpg"))
            confirmed_dir_path = label_dir / "confirmed"
            if confirmed_dir_path.exists():
                existing_files.extend(list(confirmed_dir_path.glob(f"*_{video_base}_*.jpg")))
            
            if existing_files:
                # Already saved at least one detection from this video+label, skip
                logger.debug(f"Skipping duplicate detection: {label} from {video_name} (already have {len(existing_files)} image(s))")
                return False
            
            # If auto-confirm, save to confirmed subfolder
            if should_auto_confirm:
                confirmed_dir = label_dir / "confirmed"
                confirmed_dir.mkdir(parents=True, exist_ok=True)
                target_dir = confirmed_dir
                logger.debug(f"Auto-confirming {label} detection (confidence: {confidence:.2f})")
            else:
                target_dir = label_dir
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{video_base}_f{frame_idx}_{label.replace(' ', '_')}.jpg"
            
            # Save image
            image_path = target_dir / filename
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), frame_bgr)
            
            # Save metadata
            metadata = {
                "filename": filename,
                "video_name": video_name,
                "frame_idx": frame_idx,
                "label": label,
                "confidence": confidence,
                "timestamp": timestamp,
                "auto_confirmed": should_auto_confirm,
                "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]} if bbox else None
            }
            
            metadata_path = target_dir / f"{filename}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # Cleanup old confirmed images if auto-confirmed and limit exceeded
            if should_auto_confirm:
                confirmed_images = sorted(confirmed_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
                if len(confirmed_images) > max_confirmed:
                    to_delete = confirmed_images[:len(confirmed_images) - max_confirmed]
                    for img in to_delete:
                        try:
                            img.unlink()
                            meta = target_dir / f"{img.name}.json"
                            if meta.exists():
                                meta.unlink()
                            logger.debug(f"Deleted old confirmed image: {img.name}")
                        except Exception as e:
                            logger.warning(f"Failed to delete old image {img.name}: {e}")
            
            logger.debug(f"Saved detected object: {label} (confidence: {confidence:.2f}, auto_confirmed: {should_auto_confirm})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save detected object: {e}")
            return False
    
    def track_and_annotate_video(self, video_path: Path, output_path: Path = None, 
                                  save_original: bool = False) -> Dict[str, float]:
        """
        Track objects through entire video and create annotated output with bounding boxes.
        
        Args:
            video_path: Path to input video file
            output_path: Path for annotated output video (auto-generated if None)
            save_original: Whether to keep the original video alongside annotated version
            
        Returns:
            Dictionary mapping label to max confidence score across all frames
        """
        try:
            logger.info(f"Starting video tracking for: {video_path.name}")
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return {}
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
            
            # Generate output path if not provided
            if output_path is None:
                output_dir = self.detected_objects_path / "annotated_videos"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"tracked_{video_path.name}"
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set permissions for output directory (per user rule)
            try:
                import stat
                output_path.parent.chmod(stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            except Exception as e:
                logger.warning(f"Could not set permissions on {output_path.parent}: {e}")
            
            # Create video writer with codec fallback
            # Try H.264 variants first for best browser compatibility
            CODEC_FALLBACK = [
                ('H264', 'H.264 (x264) - best browser compatibility'),
                ('X264', 'H.264 (alt) - modern browsers'),
                ('avc1', 'H.264 (avc1) - Apple/Safari'),
                ('mp4v', 'MPEG-4 - fallback compatibility'),
                ('XVID', 'Xvid - legacy fallback'),
                ('MJPG', 'Motion JPEG - always works'),
            ]
            
            out = None
            successful_codec = None
            
            for codec, desc in CODEC_FALLBACK:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                    
                    if test_out.isOpened():
                        # Verify it can actually write a frame
                        test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                        test_out.write(test_frame)
                        test_out.release()
                        
                        # Check file was created and has content
                        if output_path.exists() and output_path.stat().st_size > 500:
                            # Success! Recreate writer for actual use
                            output_path.unlink()  # Delete test file
                            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                            
                            # Verify the recreated writer also opened successfully
                            if not out.isOpened():
                                logger.error(f"Codec {codec} test passed but failed to reopen")
                                out = None
                                continue
                            
                            successful_codec = codec
                            logger.info(f"Video writer successfully initialized with codec: {codec} ({desc})")
                            break
                        else:
                            logger.warning(f"Codec {codec} opened but failed to write data (size: {output_path.stat().st_size if output_path.exists() else 0} bytes)")
                            if output_path.exists():
                                output_path.unlink()
                    else:
                        logger.warning(f"Codec {codec} failed to open")
                except Exception as e:
                    logger.warning(f"Codec {codec} error: {e}")
                    continue
            
            if out is None or not out.isOpened():
                logger.error(f"All codecs failed for video writer: {output_path}")
                logger.error(f"Tried codecs: {[c[0] for c in CODEC_FALLBACK]}")
                cap.release()
                return {}
            
            # Track detections
            all_detections = {}
            frame_count = 0
            
            logger.info("Processing frames with object tracking...")
            
            # Write a test frame to verify codec works
            test_frame_success = False
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # On first frame, write immediately to verify codec works
                if frame_count == 1:
                    try:
                        out.write(frame)
                        test_frame_success = True
                        logger.info(f"Successfully wrote test frame with codec {successful_codec}")
                    except Exception as e:
                        logger.error(f"Failed to write test frame: {e}")
                        cap.release()
                        out.release()
                        if output_path.exists():
                            output_path.unlink()
                        return {}
                
                # Validate frame before processing
                if frame is None or frame.shape[0] <= 0 or frame.shape[1] <= 0:
                    logger.warning(f"Invalid frame {frame_count}, skipping")
                    continue
                
                # Track objects in frame (YOLOv8 tracking with persistent IDs)
                # Use detection-only mode to avoid Lucas-Kanade tracking errors
                try:
                    # Use simple detection instead of tracking to avoid lkpyramid.cpp errors
                    # Tracking has issues with frame dimension mismatches
                    results = self.model(frame, verbose=False)
                except Exception as detect_error:
                    logger.error(f"Detection failed on frame {frame_count}: {detect_error}")
                    # Write the frame without annotations
                    try:
                        out.write(frame)
                    except Exception as write_error:
                        logger.error(f"Failed to write frame {frame_count}: {write_error}")
                    continue
                
                # Process tracking results
                if results and len(results) > 0:
                    result = results[0]
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes
                        
                        for box in boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            label = result.names[class_id].lower()
                            
                            # Check if this label is in our tracked labels
                            if label not in self.labels:
                                continue
                            
                            # Track max confidence for this label
                            if label not in all_detections or confidence > all_detections[label]:
                                all_detections[label] = confidence
                            
                            # Only draw boxes for labels we're tracking
                            if confidence >= self.confidence_threshold:
                                bbox = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, bbox)
                                
                                # Get track ID if available
                                track_id = int(box.id[0]) if box.id is not None else None
                                
                                # Draw bounding box
                                color = (0, 255, 0)  # Green for tracked objects
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Prepare label text
                                if track_id is not None:
                                    label_text = f"{label} #{track_id} {confidence:.2f}"
                                else:
                                    label_text = f"{label} {confidence:.2f}"
                                
                                # Calculate label dimensions
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.6
                                font_thickness = 2
                                (text_width, text_height), baseline = cv2.getTextSize(
                                    label_text, font, font_scale, font_thickness
                                )
                                
                                # Improved label placement logic
                                label_padding = 4
                                label_height = text_height + baseline + (2 * label_padding)
                                label_width = text_width + (2 * label_padding)
                                
                                # Try to place label above box
                                if y1 - label_height >= 0:
                                    # Place above
                                    label_y1 = y1 - label_height
                                    label_y2 = y1
                                    text_y = y1 - baseline - label_padding
                                else:
                                    # Place inside box at top
                                    label_y1 = y1
                                    label_y2 = y1 + label_height
                                    text_y = y1 + text_height + label_padding
                                
                                # Ensure label doesn't go off right edge
                                label_x1 = x1
                                label_x2 = min(x1 + label_width, width)
                                
                                # Draw semi-transparent background for label
                                overlay = frame.copy()
                                cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), 
                                            color, -1)
                                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                                
                                # Draw label text
                                cv2.putText(frame, label_text, (x1 + label_padding, text_y),
                                          font, font_scale, (0, 0, 0), font_thickness)
                
                # Write annotated frame
                out.write(frame)
                
                # Log progress every 10% and check file size
                if frame_count % (total_frames // 10 + 1) == 0:
                    progress = (frame_count / total_frames) * 100
                    
                    # Check if data is actually being written
                    if output_path.exists():
                        current_size = output_path.stat().st_size
                        logger.info(f"Tracking progress: {progress:.0f}% ({frame_count}/{total_frames} frames, {current_size / 1024:.1f} KB written)")
                        
                        # If we've written many frames but file is still tiny, something is wrong
                        if frame_count > 30 and current_size < 5000:
                            logger.error(f"Video writer failure detected: {frame_count} frames written but file only {current_size} bytes")
                            logger.error(f"Codec: {successful_codec}, aborting to prevent wasted processing")
                            cap.release()
                            out.release()
                            return {}
                    else:
                        logger.warning(f"Tracking progress: {progress:.0f}% but output file not yet created")
            
            # Cleanup
            cap.release()
            out.release()
            
            # Verify output file was created successfully
            if not output_path.exists():
                logger.error(f"Output video file was not created: {output_path}")
                return {}
            
            file_size = output_path.stat().st_size
            if file_size < 10000:  # Less than 10KB is suspicious for a video
                logger.error(f"Output video file is too small ({file_size} bytes), likely corrupt: {output_path}")
                logger.error(f"Codec used: {successful_codec}, Frames processed: {frame_count}, Video properties: {width}x{height} @ {fps}fps")
                # Don't return empty - let the file exist so user can see the problem
            else:
                logger.info(f"Video tracking complete: {output_path} ({file_size / 1024 / 1024:.2f} MB)")
            
            # Convert mp4v to H.264 using ffmpeg for browser compatibility
            if successful_codec == 'mp4v':
                logger.info("Converting mp4v video to H.264 for browser compatibility...")
                temp_output = output_path.with_suffix('.temp.mp4')
                
                try:
                    import subprocess
                    # Use ffmpeg to convert: copy video stream but re-encode with libx264
                    result = subprocess.run([
                        'ffmpeg', '-y',  # Overwrite output
                        '-i', str(output_path),  # Input file
                        '-c:v', 'libx264',  # H.264 codec
                        '-preset', 'fast',  # Encoding speed
                        '-crf', '23',  # Quality (lower = better, 18-28 is good range)
                        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                        str(temp_output)
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and temp_output.exists():
                        # Replace original with converted version
                        temp_output.replace(output_path)
                        new_size = output_path.stat().st_size
                        logger.info(f"Successfully converted to H.264: {output_path} ({new_size / 1024 / 1024:.2f} MB)")
                    else:
                        logger.warning(f"ffmpeg conversion failed (returncode {result.returncode}): {result.stderr}")
                        logger.warning("Keeping original mp4v video (may not play in all browsers)")
                        if temp_output.exists():
                            temp_output.unlink()
                except subprocess.TimeoutExpired:
                    logger.error("ffmpeg conversion timed out after 5 minutes")
                    if temp_output.exists():
                        temp_output.unlink()
                except Exception as e:
                    logger.error(f"Failed to convert video with ffmpeg: {e}")
                    if temp_output.exists():
                        temp_output.unlink()
            
            logger.info(f"Detected objects: {all_detections}")
            
            # Optionally move original video to a separate folder
            if save_original:
                original_dir = self.detected_objects_path / "original_videos"
                original_dir.mkdir(parents=True, exist_ok=True)
                original_path = original_dir / video_path.name
                
                try:
                    import shutil
                    if not original_path.exists():
                        shutil.copy2(str(video_path), str(original_path))
                        logger.info(f"Saved original video to: {original_path}")
                except Exception as e:
                    logger.warning(f"Failed to save original video: {e}")
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Failed to track video: {e}", exc_info=True)
            return {}
    
