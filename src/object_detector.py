"""
Object detection module using CLIP for open-vocabulary detection.
Allows detection of arbitrary objects without training.
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Broad detection list for discovery mode
DISCOVERY_OBJECTS = [
    # Wildlife
    "hedgehog", "fox", "badger", "deer", "rabbit", "squirrel", "mouse", "rat",
    "bird", "crow", "robin", "pigeon", "owl", "hawk", "duck", "goose",
    # Pets
    "cat", "dog", "kitten", "puppy",
    # People & Activities  
    "person", "delivery person", "mail carrier", "child", "adult",
    # Vehicles
    "car", "truck", "van", "bicycle", "motorcycle", "scooter",
    # Objects
    "package", "box", "bag", "bottle", "umbrella",
    # Nature (often false positives, consider blacklisting)
    "tree", "bush", "flower", "leaf",
]


class ObjectDetector:
    """Open-vocabulary object detector using CLIP."""
    
    def __init__(self, labels: List[str], confidence_threshold: float = 0.25, num_frames: int = 5, 
                 detected_objects_path: str = "/data/objects/detected", 
                 discovery_mode: bool = False, discovery_threshold: float = 0.30,
                 ignored_labels: List[str] = None):
        """
        Initialize the object detector.
        
        Args:
            labels: List of object labels to detect (e.g., ["hedgehog", "fox", "bird"])
            confidence_threshold: Minimum confidence score to consider a detection valid
            num_frames: Number of frames to extract and analyze from each video
            detected_objects_path: Path to save detected object images
            discovery_mode: Enable automatic discovery of new objects
            discovery_threshold: Minimum confidence for discovered objects
            ignored_labels: List of labels to ignore in discovery mode
        """
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.num_frames = num_frames
        self.detected_objects_path = Path(detected_objects_path)
        self.detected_objects_path.mkdir(parents=True, exist_ok=True)
        
        # Discovery mode settings
        self.discovery_mode = discovery_mode
        self.discovery_threshold = discovery_threshold
        self.ignored_labels = set(ignored_labels or [])
        
        # Combined labels for detection
        if self.discovery_mode:
            # Add discovery objects that aren't already tracked or ignored
            discovery_labels = [obj for obj in DISCOVERY_OBJECTS 
                              if obj not in self.labels and obj not in self.ignored_labels]
            self.all_labels = list(self.labels) + discovery_labels
            logger.info(f"Discovery mode enabled: tracking {len(self.labels)} focused + {len(discovery_labels)} discovery labels")
        else:
            self.all_labels = list(self.labels)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing CLIP model on device: {self.device}")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        
        # Prepare text prompts for each label
        self.text_prompts = [f"a photo of a {label}" for label in self.all_labels]
        logger.info(f"Initialized ObjectDetector with {len(self.all_labels)} total labels ({len(self.labels)} focused)")
    
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
        # Convert frame to PIL Image
        image = Image.fromarray(frame)
        
        # Process inputs
        inputs = self.processor(
            text=self.text_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Convert to dictionary and save detections
        detections = {}
        discovered = {}  # New discoveries
        
        for idx, label in enumerate(self.all_labels):
            confidence = probs[0][idx].item()
            
            # Determine if this is a focused or discovery label
            is_focused = label in self.labels
            is_discovery = not is_focused and label not in self.ignored_labels
            
            # Save ALL detections with confidence scores
            if save_detections and confidence > 0.01:  # Very low threshold to catch all
                self._save_detected_object(
                    frame, label, confidence,
                    video_name, frame_idx,
                    is_discovery=is_discovery
                )
            
            # Process focused labels
            if is_focused and confidence >= self.confidence_threshold:
                detections[label] = confidence
            
            # Process discoveries
            elif is_discovery and confidence >= self.discovery_threshold:
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
        discoveries_file = Path("/data/discoveries.json")
        discoveries_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load existing discoveries
            if discoveries_file.exists():
                with open(discoveries_file, 'r') as f:
                    all_discoveries = json.load(f)
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
            
            # Save
            with open(discoveries_file, 'w') as f:
                json.dump(all_discoveries, f, indent=2)
            
            logger.debug(f"Saved {len(discovered)} discoveries")
            
        except Exception as e:
            logger.error(f"Failed to save discoveries: {e}")
