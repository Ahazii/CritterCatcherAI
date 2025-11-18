"""Two-stage processing pipeline: YOLO Stage 1 + CLIP/ViT Stage 2."""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import shutil
import cv2
import numpy as np

from animal_profile import AnimalProfile, AnimalProfileManager
from clip_vit_classifier import CLIPVitClassifier, AnimalIdentifier
from object_detector import ObjectDetector

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract frames from video files."""
    
    def __init__(self, target_fps: int = 1):
        """
        Initialize frame extractor.
        
        Args:
            target_fps: Target frames per second to extract (e.g., 1 = every 1 second)
        """
        self.target_fps = target_fps
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: str,
        max_frames: Optional[int] = None
    ) -> List[str]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            max_frames: Maximum frames to extract (None = all)
        
        Returns:
            List of extracted frame paths
        
        Raises:
            RuntimeError: If video processing fails
        """
        try:
            video_path = Path(video_path)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / self.target_fps) if fps > 0 else 1
            
            frames = []
            frame_count = 0
            extracted_count = 0
            
            logger.info(f"Extracting frames from {video_path.name} (FPS: {fps}, Interval: {frame_interval})")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract every nth frame based on target_fps
                if frame_count % frame_interval == 0:
                    if max_frames and extracted_count >= max_frames:
                        break
                    
                    # Save frame
                    frame_filename = f"frame_{extracted_count:06d}.jpg"
                    frame_path = output_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    frames.append(str(frame_path))
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {extracted_count} frames from {video_path.name}")
            return frames
        
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            raise RuntimeError(f"Frame extraction failed: {e}")


class ProcessingResult:
    """Container for processing results."""
    
    def __init__(self, profile_id: str, profile_name: str):
        self.profile_id = profile_id
        self.profile_name = profile_name
        self.sorted_count = 0
        self.review_count = 0
        self.sorted_frames = []
        self.review_frames = []
        self.errors = []
        self.timestamp = datetime.now()
    
    def to_dict(self):
        """Convert to dictionary for logging/storage."""
        return {
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "sorted_count": self.sorted_count,
            "review_count": self.review_count,
            "total_count": self.sorted_count + self.review_count,
            "sorted_frames": self.sorted_frames,
            "review_frames": self.review_frames,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat()
        }


class TwoStageProcessor:
    """Two-stage processing pipeline: YOLO + CLIP/ViT."""
    
    def __init__(
        self,
        profile_manager: AnimalProfileManager,
        clip_classifier: Optional[CLIPVitClassifier] = None,
        object_detector: Optional[ObjectDetector] = None,
        base_data_path: str = "/data"
    ):
        """
        Initialize processor.
        
        Args:
            profile_manager: AnimalProfileManager instance
            clip_classifier: CLIPVitClassifier instance (creates default if None)
            object_detector: ObjectDetector instance (YOLO detector)
            base_data_path: Base path for data directories
        """
        self.profile_manager = profile_manager
        self.clip_classifier = clip_classifier or CLIPVitClassifier()
        self.animal_identifier = AnimalIdentifier(self.clip_classifier)
        self.object_detector = object_detector
        self.base_data_path = Path(base_data_path)
        
        # Create necessary directories
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary data directories."""
        dirs = [
            self.base_data_path / "sorted",
            self.base_data_path / "review",
            self.base_data_path / "training",
            self.base_data_path / "models",
            self.base_data_path / "temp",
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process_frames_for_profile(
        self,
        profile: AnimalProfile,
        frame_paths: List[str]
    ) -> ProcessingResult:
        """
        Process frames for a single animal profile.
        
        Stage 1: Filter by YOLO categories
        Stage 2: Score with CLIP/ViT
        
        Args:
            profile: AnimalProfile to process
            frame_paths: List of frame image paths
        
        Returns:
            ProcessingResult with sorted/review frames
        """
        result = ProcessingResult(profile.id, profile.name)
        
        if not frame_paths:
            logger.warning(f"No frames to process for {profile.name}")
            return result
        
        try:
            logger.info(f"Processing {len(frame_paths)} frames for {profile.name}")
            
            # Stage 1: Filter by YOLO categories
            if self.object_detector and profile.yolo_categories:
                logger.debug(f"Stage 1: Filtering for YOLO categories {profile.yolo_categories}")
                yolo_filtered_frames = self._stage1_yolo_filter(
                    frame_paths,
                    profile.yolo_categories
                )
                logger.info(f"Stage 1 filtered: {len(yolo_filtered_frames)}/{len(frame_paths)} frames")
            else:
                logger.warning("No object detector or YOLO categories, skipping Stage 1")
                yolo_filtered_frames = frame_paths
            
            # Stage 2: Score with CLIP/ViT
            logger.debug(f"Stage 2: Scoring with CLIP ({profile.text_description})")
            processing_results = self.animal_identifier.process_frames(
                yolo_filtered_frames,
                profile.text_description,
                threshold=profile.confidence_threshold
            )
            
            # Organize results
            result = self._organize_results(
                profile,
                processing_results,
                result
            )
            
            logger.info(
                f"Completed processing {profile.name}: "
                f"{result.sorted_count} sorted, {result.review_count} review"
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing frames for {profile.name}: {e}", exc_info=True)
            result.errors.append(str(e))
            return result
    
    def _stage1_yolo_filter(self, frame_paths: List[str], categories: List[str]) -> List[str]:
        """Filter frames containing YOLO categories."""
        filtered = []
        
        for frame_path in frame_paths:
            try:
                # Run YOLO detection
                detections = self.object_detector.detect(frame_path)
                
                # Check if any detection matches requested categories
                has_match = False
                for detection in detections:
                    if detection.get('class_name') in categories:
                        has_match = True
                        break
                
                if has_match:
                    filtered.append(frame_path)
            
            except Exception as e:
                logger.warning(f"Error running YOLO on {frame_path}: {e}")
                # Include frame on error (conservative approach)
                filtered.append(frame_path)
        
        return filtered
    
    def _organize_results(
        self,
        profile: AnimalProfile,
        processing_results: dict,
        result: ProcessingResult
    ) -> ProcessingResult:
        """
        Organize frames into sorted/review directories.
        
        Args:
            profile: AnimalProfile
            processing_results: Output from AnimalIdentifier.process_frames()
            result: ProcessingResult to update
        
        Returns:
            Updated ProcessingResult
        """
        # Get profile directories
        sorted_dir = self.base_data_path / "sorted" / profile.id
        review_dir = self.base_data_path / "review" / profile.id
        
        sorted_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)
        
        # Process high confidence frames
        for frame_path, score in processing_results['high_confidence']:
            try:
                if profile.auto_approval_enabled:
                    # Move to sorted
                    dest_path = sorted_dir / Path(frame_path).name
                    shutil.copy2(frame_path, dest_path)
                    result.sorted_frames.append(str(dest_path))
                    result.sorted_count += 1
                    self._save_frame_metadata(dest_path, score, profile.text_description)
                else:
                    # Move to review even if high confidence
                    dest_path = review_dir / Path(frame_path).name
                    shutil.copy2(frame_path, dest_path)
                    result.review_frames.append(str(dest_path))
                    result.review_count += 1
                    self._save_frame_metadata(dest_path, score, profile.text_description)
            
            except Exception as e:
                logger.warning(f"Error organizing frame {frame_path}: {e}")
                result.errors.append(f"Failed to organize frame: {e}")
        
        # Process low confidence frames
        for frame_path, score in processing_results['low_confidence']:
            try:
                # Move to review
                dest_path = review_dir / Path(frame_path).name
                shutil.copy2(frame_path, dest_path)
                result.review_frames.append(str(dest_path))
                result.review_count += 1
                self._save_frame_metadata(dest_path, score, profile.text_description)
            
            except Exception as e:
                logger.warning(f"Error organizing frame {frame_path}: {e}")
                result.errors.append(f"Failed to organize frame: {e}")
        
        return result
    
    def _save_frame_metadata(self, frame_path: str, confidence: float, description: str):
        """Save frame metadata as JSON."""
        try:
            metadata = {
                "confidence": confidence,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "frame_path": str(frame_path)
            }
            
            metadata_path = Path(str(frame_path) + ".json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        except Exception as e:
            logger.warning(f"Failed to save metadata for {frame_path}: {e}")
    
    def process_videos(
        self,
        video_paths: List[str],
        profile_ids: Optional[List[str]] = None,
        max_frames_per_video: Optional[int] = None
    ) -> Dict[str, ProcessingResult]:
        """
        Process multiple videos through the pipeline.
        
        Args:
            video_paths: List of video file paths
            profile_ids: Specific profiles to process (None = all enabled)
            max_frames_per_video: Limit frames per video
        
        Returns:
            Dict mapping profile_id to ProcessingResult
        """
        results = {}
        
        # Get profiles to process
        if profile_ids:
            profiles = [
                self.profile_manager.get_profile(pid)
                for pid in profile_ids
            ]
            profiles = [p for p in profiles if p is not None]
        else:
            profiles = [p for p in self.profile_manager.list_profiles() if p.enabled]
        
        if not profiles:
            logger.warning("No profiles to process")
            return results
        
        # Extract frames from all videos
        logger.info(f"Extracting frames from {len(video_paths)} videos")
        extractor = FrameExtractor(target_fps=1)
        temp_dir = self.base_data_path / "temp" / f"processing_{datetime.now().timestamp()}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        all_frames = []
        try:
            for video_path in video_paths:
                try:
                    frames = extractor.extract_frames(
                        video_path,
                        str(temp_dir),
                        max_frames=max_frames_per_video
                    )
                    all_frames.extend(frames)
                except Exception as e:
                    logger.error(f"Failed to extract frames from {video_path}: {e}")
            
            logger.info(f"Total frames extracted: {len(all_frames)}")
            
            # Process frames for each profile
            for profile in profiles:
                logger.info(f"Processing {profile.name}")
                result = self.process_frames_for_profile(profile, all_frames)
                results[profile.id] = result
        
        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp directory: {e}")
        
        return results
    
    def process_and_log(
        self,
        video_paths: List[str],
        profile_ids: Optional[List[str]] = None,
        max_frames_per_video: Optional[int] = None
    ) -> Dict:
        """
        Process videos and return summary.
        
        Args:
            video_paths: List of video file paths
            profile_ids: Specific profiles to process
            max_frames_per_video: Limit frames per video
        
        Returns:
            Dict with processing summary
        """
        logger.info(f"Starting pipeline processing: {len(video_paths)} videos")
        start_time = datetime.now()
        
        results = self.process_videos(video_paths, profile_ids, max_frames_per_video)
        
        # Aggregate results
        total_sorted = sum(r.sorted_count for r in results.values())
        total_review = sum(r.review_count for r in results.values())
        total_errors = sum(len(r.errors) for r in results.values())
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        summary = {
            "status": "success",
            "timestamp": start_time.isoformat(),
            "duration_seconds": elapsed,
            "videos_processed": len(video_paths),
            "profiles_processed": len(results),
            "total_sorted": total_sorted,
            "total_review": total_review,
            "total_errors": total_errors,
            "results": {pid: r.to_dict() for pid, r in results.items()}
        }
        
        logger.info(
            f"Pipeline complete: {total_sorted} sorted, {total_review} review, "
            f"{total_errors} errors ({elapsed:.1f}s)"
        )
        
        return summary
