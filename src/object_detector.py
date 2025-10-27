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

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Open-vocabulary object detector using CLIP."""
    
    def __init__(self, labels: List[str], confidence_threshold: float = 0.25, num_frames: int = 5):
        """
        Initialize the object detector.
        
        Args:
            labels: List of object labels to detect (e.g., ["hedgehog", "fox", "bird"])
            confidence_threshold: Minimum confidence score to consider a detection valid
            num_frames: Number of frames to extract and analyze from each video
        """
        self.labels = labels
        self.confidence_threshold = confidence_threshold
        self.num_frames = num_frames
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing CLIP model on device: {self.device}")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        
        # Prepare text prompts for each label
        self.text_prompts = [f"a photo of a {label}" for label in labels]
        logger.info(f"Initialized ObjectDetector with {len(labels)} labels: {labels}")
    
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
    
    def detect_objects_in_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Image frame as numpy array (RGB)
            
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
        
        # Convert to dictionary
        detections = {}
        for idx, label in enumerate(self.labels):
            confidence = probs[0][idx].item()
            if confidence >= self.confidence_threshold:
                detections[label] = confidence
        
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
            frame_detections = self.detect_objects_in_frame(frame)
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
