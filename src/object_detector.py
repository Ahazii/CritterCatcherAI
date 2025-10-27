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
                 discovery_mode: bool = False, discovery_threshold: float = 0.30,
                 ignored_labels: List[str] = None, model_name: str = "yolov8n.pt"):
        """
        Initialize the object detector.
        
        Args:
            labels: List of object labels to detect (must be YOLO COCO classes)
            confidence_threshold: Minimum confidence score to consider a detection valid
            num_frames: Number of frames to extract and analyze from each video
            detected_objects_path: Path to save detected object images
            discovery_mode: Enable automatic discovery of new objects
            discovery_threshold: Minimum confidence for discovered objects
            ignored_labels: List of labels to ignore in discovery mode
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
        
        # Discovery mode settings
        self.discovery_mode = discovery_mode
        self.discovery_threshold = discovery_threshold
        self.ignored_labels = set([label.lower() for label in (ignored_labels or [])])
        
        # Combined labels for detection
        yolo_classes_lower = [c.lower() for c in YOLO_COCO_CLASSES]
        if self.discovery_mode:
            # Add YOLO COCO classes that aren't already tracked or ignored
            discovery_labels = [obj for obj in yolo_classes_lower
                              if obj not in self.labels and obj not in self.ignored_labels]
            self.all_labels = list(self.labels) + discovery_labels
            logger.info(f"Discovery mode enabled: tracking {len(self.labels)} focused + {len(discovery_labels)} discovery labels")
        else:
            self.all_labels = list(self.labels)
        
        logger.info(f"Initializing YOLOv8 model: {self.model_name}")
        self.model = YOLO(self.model_name)
        logger.info(f"Model loaded: {self.model_name} - {len(YOLO_COCO_CLASSES)} COCO classes available")
        
        logger.info(f"Initialized ObjectDetector with {len(self.labels)} tracked labels")
        if not self.labels:
            logger.error("No valid labels configured! Please check your config.yaml")
    
    def extract_frames(self, video_path: Path, num_frames: int = 5) -> List[np.ndarray]:
        """
        Extract representative frames from video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            List of frame images as numpy arrays
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract frames evenly distributed throughout the video
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        logger.debug(f"Extracted {len(frames)} frames from {video_path.name}")
        return frames
    
    def detect_objects_in_frame(self, frame: np.ndarray, video_name: str = None, frame_idx: int = None, save_detections: bool = True) -> Dict[str, float]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Image frame as numpy array (RGB)
            video_name: Name of the video being processed
            frame_idx: Frame index in the video
            save_detections: Whether to save detected object images
            
        Returns:
            Dictionary mapping label to confidence score
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        # Process detections
        detections = {}
        discovered = {}  # New discoveries
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = result.names[class_id].lower()
                
                # Determine if this is a focused or discovery label
                is_focused = label in self.labels
                is_discovery = label not in self.labels and label not in self.ignored_labels and self.discovery_mode
                
                # Save detections if enabled
                if save_detections and confidence > 0.01:
                    self._save_detected_object(
                        frame, label, confidence,
                        video_name, frame_idx,
                        is_discovery=is_discovery
                    )
                
                # Process focused labels - keep highest confidence
                if is_focused and confidence >= self.confidence_threshold:
                    if label not in detections or confidence > detections[label]:
                        detections[label] = confidence
                
                # Process discoveries
                elif is_discovery and confidence >= self.discovery_threshold:
                    if label not in discovered or confidence > discovered[label]:
                        discovered[label] = confidence
                        logger.info(f"🔍 Discovered new object: {label} (confidence: {confidence:.2f})")
        
        # Store discoveries for later retrieval
        if discovered:
            self._save_discoveries(discovered, video_name)
        
        return detections
    
    def detect_objects_in_video(self, video_path: Path, num_frames: int = None) -> Dict[str, float]:
        """
        Detect objects across entire video.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to analyze (defaults to instance setting)
            
        Returns:
            Dictionary mapping label to max confidence score across all frames
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
                save_detections=True
            )
            logger.debug(f"Frame {i+1}/{len(frames)}: {frame_detections}")
            
            # Keep the maximum confidence for each label
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
                             video_name: str, frame_idx: int, is_discovery: bool = False) -> bool:
        """
        Save a detected object image and metadata.
        
        Args:
            frame: Frame image with detected object
            label: Detected object label
            confidence: Detection confidence score
            video_name: Name of source video
            frame_idx: Frame index
            is_discovery: Whether this is a newly discovered object
            
        Returns:
            True if saved, False if skipped
        """
        try:
            # Create label directory
            label_dir = self.detected_objects_path / label.replace(" ", "_")
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_base = Path(video_name).stem if video_name else "unknown"
            filename = f"{timestamp}_{video_base}_f{frame_idx}_{label.replace(' ', '_')}.jpg"
            
            # Save image
            image_path = label_dir / filename
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
                "is_discovery": is_discovery
            }
            
            metadata_path = label_dir / f"{filename}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.debug(f"Saved detected object: {label} (confidence: {confidence:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save detected object: {e}")
            return False
    
    def _save_discoveries(self, discovered: Dict[str, float], video_name: str):
        """
        Save discovered objects for user confirmation.
        
        Args:
            discovered: Dictionary of discovered labels and confidence scores
            video_name: Name of source video
        """
        discoveries_file = Path("/data/objects/discoveries.json")
        discoveries_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Retry logic with file locking
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Acquire exclusive lock on the file
                with open(discoveries_file, 'a+') as lock_file:
                    try:
                        # Try to acquire lock (non-blocking on first attempt, then blocking)
                        if attempt == 0:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        else:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                        
                        # Read existing discoveries
                        lock_file.seek(0)
                        content = lock_file.read()
                        
                        if content.strip():
                            all_discoveries = json.loads(content)
                        else:
                            all_discoveries = {}
                        
                        # Add new discoveries
                        timestamp = datetime.now().isoformat()
                        for label, confidence in discovered.items():
                            if label not in all_discoveries:
                                all_discoveries[label] = {
                                    "label": label,
                                    "first_seen": timestamp,
                                    "detection_count": 0,
                                    "total_confidence": 0.0,
                                    "videos": []
                                }
                            
                            # Update stats
                            all_discoveries[label]["detection_count"] += 1
                            all_discoveries[label]["total_confidence"] += confidence
                            if video_name not in all_discoveries[label]["videos"]:
                                all_discoveries[label]["videos"].append(video_name)
                        
                        # Write back
                        lock_file.seek(0)
                        lock_file.truncate()
                        json.dump(all_discoveries, lock_file, indent=2)
                        
                        # Release lock
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        
                        logger.debug(f"Saved {len(discovered)} discoveries")
                        return  # Success!
                        
                    except BlockingIOError:
                        # File is locked, retry after short delay
                        if attempt < max_retries - 1:
                            time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                            continue
                        else:
                            raise
                            
            except json.JSONDecodeError as e:
                logger.warning(f"Corrupted discoveries JSON on attempt {attempt + 1}, reinitializing: {e}")
                # If JSON is corrupted, reinitialize with current discoveries only
                if attempt == max_retries - 1:
                    with open(discoveries_file, 'w') as f:
                        all_discoveries = {}
                        timestamp = datetime.now().isoformat()
                        for label, confidence in discovered.items():
                            all_discoveries[label] = {
                                "label": label,
                                "first_seen": timestamp,
                                "detection_count": 1,
                                "total_confidence": confidence,
                                "videos": [video_name]
                            }
                        json.dump(all_discoveries, f, indent=2)
                    logger.info("Reinitialized discoveries.json")
                    return
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to save discoveries (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
