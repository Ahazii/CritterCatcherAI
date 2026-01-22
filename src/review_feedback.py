"""Review and feedback system for handling user confirmations/rejections."""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import shutil

from animal_profile import AnimalProfileManager, AnimalProfile

logger = logging.getLogger(__name__)


class ReviewFrame:
    """Represents a frame pending review."""
    
    def __init__(self, frame_path: str, profile_id: str):
        """
        Initialize review frame.
        
        Args:
            frame_path: Path to frame image
            profile_id: Associated animal profile ID
        """
        self.frame_path = Path(frame_path)
        self.profile_id = profile_id
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> dict:
        """Load metadata from JSON file."""
        metadata_path = Path(str(self.frame_path) + ".json")
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {self.frame_path}: {e}")
        
        return {
            "confidence": 0.0,
            "description": "",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_confidence(self) -> float:
        """Get frame confidence score."""
        return float(self.metadata.get("confidence", 0.0))
    
    def get_description(self) -> str:
        """Get animal description used for scoring."""
        return self.metadata.get("description", "")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "frame_path": str(self.frame_path),
            "filename": self.frame_path.name,
            "profile_id": self.profile_id,
            "confidence": self.get_confidence(),
            "description": self.get_description(),
            "timestamp": self.metadata.get("timestamp", "")
        }


class ReviewManager:
    """Manage the review process for pending frames."""
    
    def __init__(
        self,
        profile_manager: AnimalProfileManager,
        base_data_path: str = "/data"
    ):
        """
        Initialize review manager.
        
        Args:
            profile_manager: AnimalProfileManager instance
            base_data_path: Base path for data directories
        """
        self.profile_manager = profile_manager
        self.base_data_path = Path(base_data_path)
        self.clip_classifier = None

    def _load_training_config(self) -> dict:
        config_path = Path("/config/config.yaml")
        defaults = {
            "enabled": True,
            "batch_size": 10,
            "min_negatives": 10,
            "epochs": 200,
            "learning_rate": 0.1,
            "l2": 0.001,
            "batch_size_embeddings": 8
        }
        try:
            if config_path.exists():
                import yaml
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f) or {}
                return {**defaults, **(config.get("animal_training", {}) or {})}
        except Exception as e:
            logger.warning(f"Failed to load training config: {e}")
        return defaults

    def _maybe_train_profile(self, profile: AnimalProfile):
        config = self._load_training_config()
        if not config.get("enabled", True):
            return
        
        batch_size = int(config.get("batch_size", 10))
        if (profile.confirmed_count - profile.last_trained_confirmed) < batch_size:
            return
        
        # Require negatives
        min_negatives = int(config.get("min_negatives", 10))
        positives_dir = self.base_data_path / "training" / profile.id / "confirmed"
        negatives_dir = self.base_data_path / "training" / profile.id / "rejected"
        
        positive_paths = sorted(str(p) for p in positives_dir.glob("*.jpg")) if positives_dir.exists() else []
        negative_paths = sorted(str(p) for p in negatives_dir.glob("*.jpg")) if negatives_dir.exists() else []
        
        if len(positive_paths) < batch_size or len(negative_paths) < min_negatives:
            logger.info(
                f"Skipping training for {profile.id}: "
                f"{len(positive_paths)} positives, {len(negative_paths)} negatives"
            )
            return
        
        from clip_vit_classifier import CLIPVitClassifier
        if self.clip_classifier is None:
            force_cpu = False
            try:
                import yaml
                with open("/config/config.yaml", "r") as f:
                    full_config = yaml.safe_load(f) or {}
                force_cpu = bool(full_config.get("detection", {}).get("force_cpu", False))
            except Exception:
                force_cpu = False
            self.clip_classifier = CLIPVitClassifier(force_cpu=force_cpu)
        
        model_path = self.base_data_path / "models" / profile.id / "classifier.json"
        try:
            self.clip_classifier.train_classifier(
                positive_paths=positive_paths,
                negative_paths=negative_paths,
                model_path=model_path,
                epochs=int(config.get("epochs", 200)),
                learning_rate=float(config.get("learning_rate", 0.1)),
                l2=float(config.get("l2", 0.001)),
                batch_size=int(config.get("batch_size_embeddings", 8))
            )
            profile.last_trained_confirmed = profile.confirmed_count
            profile.last_trained_rejected = profile.rejected_count
            profile.classifier_model_path = str(model_path)
            profile.last_training_date = datetime.now().isoformat()
            self.profile_manager._save_profile(profile)
            logger.info(f"Trained classifier for {profile.id}")
        except Exception as e:
            logger.error(f"Failed training for {profile.id}: {e}")
    
    def list_pending_reviews(self, profile_id: str) -> List[ReviewFrame]:
        """
        List all frames pending review for a profile.
        
        Args:
            profile_id: Animal profile ID
        
        Returns:
            List of ReviewFrame objects
        """
        review_dir = self.base_data_path / "review" / profile_id
        if not review_dir.exists():
            return []
        
        frames = []
        for frame_file in sorted(review_dir.glob("*.jpg")):
            try:
                frame = ReviewFrame(str(frame_file), profile_id)
                frames.append(frame)
            except Exception as e:
                logger.warning(f"Error loading review frame {frame_file}: {e}")
        
        return frames
    
    def list_all_pending_reviews(self) -> Dict[str, List[ReviewFrame]]:
        """
        List all pending reviews across all profiles.
        
        Returns:
            Dict mapping profile_id to list of ReviewFrame objects
        """
        result = {}
        review_base = self.base_data_path / "review"
        
        if not review_base.exists():
            return result
        
        for profile_dir in review_base.iterdir():
            if profile_dir.is_dir():
                profile_id = profile_dir.name
                frames = self.list_pending_reviews(profile_id)
                if frames:
                    result[profile_id] = frames
        
        return result
    
    def get_pending_count(self, profile_id: str) -> int:
        """Get count of pending frames for a profile."""
        return len(self.list_pending_reviews(profile_id))
    
    def get_all_pending_counts(self) -> Dict[str, int]:
        """Get pending frame counts for all profiles."""
        return {
            pid: len(frames)
            for pid, frames in self.list_all_pending_reviews().items()
        }
    
    def confirm_frame(self, profile_id: str, frame_filename: str) -> bool:
        """
        Confirm a frame as correct.
        
        Moves frame from review to training/confirmed and updates accuracy.
        
        Args:
            profile_id: Animal profile ID
            frame_filename: Name of frame file (e.g., "frame_000001.jpg")
        
        Returns:
            True if successful
        """
        try:
            profile = self.profile_manager.get_profile(profile_id)
            if not profile:
                raise ValueError(f"Profile '{profile_id}' not found")
            
            # Source and destination paths
            review_path = self.base_data_path / "review" / profile_id / frame_filename
            training_path = self.base_data_path / "training" / profile_id / "confirmed" / frame_filename
            
            if not review_path.exists():
                raise FileNotFoundError(f"Frame not found: {review_path}")
            
            # Create destination directory
            training_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move frame and metadata
            shutil.move(str(review_path), str(training_path))
            
            # Move metadata if it exists
            metadata_src = Path(str(review_path) + ".json")
            if metadata_src.exists():
                metadata_dst = Path(str(training_path) + ".json")
                shutil.move(str(metadata_src), str(metadata_dst))
            
            # Update accuracy tracking
            profile.confirmed_count += 1
            self.profile_manager._save_profile(profile)
            self._maybe_train_profile(profile)
            
            logger.info(f"Confirmed frame {frame_filename} for {profile.name}")
            return True
        
        except Exception as e:
            logger.error(f"Error confirming frame {frame_filename}: {e}")
            raise
    
    def reject_frame(
        self,
        profile_id: str,
        frame_filename: str,
        save_as_negative: bool = False
    ) -> bool:
        """
        Reject a frame.
        
        Can delete the frame or save as negative training example.
        
        Args:
            profile_id: Animal profile ID
            frame_filename: Name of frame file
            save_as_negative: If True, save to training/rejected instead of deleting
        
        Returns:
            True if successful
        """
        try:
            profile = self.profile_manager.get_profile(profile_id)
            if not profile:
                raise ValueError(f"Profile '{profile_id}' not found")
            
            # Source and destination paths
            review_path = self.base_data_path / "review" / profile_id / frame_filename
            
            if not review_path.exists():
                raise FileNotFoundError(f"Frame not found: {review_path}")
            
            if save_as_negative:
                # Move to training/rejected for use as negative examples
                training_path = self.base_data_path / "training" / profile_id / "rejected" / frame_filename
                training_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(review_path), str(training_path))
                
                # Move metadata if it exists
                metadata_src = Path(str(review_path) + ".json")
                if metadata_src.exists():
                    metadata_dst = Path(str(training_path) + ".json")
                    shutil.move(str(metadata_src), str(metadata_dst))
                
                logger.info(f"Rejected frame {frame_filename} (saved as negative) for {profile.name}")
            else:
                # Delete the frame
                review_path.unlink()
                
                # Delete metadata if it exists
                metadata_path = Path(str(review_path) + ".json")
                if metadata_path.exists():
                    metadata_path.unlink()
                
                logger.info(f"Rejected frame {frame_filename} (deleted) for {profile.name}")
            
            # Update accuracy tracking
            profile.rejected_count += 1
            self.profile_manager._save_profile(profile)
            
            return True
        
        except Exception as e:
            logger.error(f"Error rejecting frame {frame_filename}: {e}")
            raise
    
    def bulk_confirm_frames(
        self,
        profile_id: str,
        frame_filenames: List[str]
    ) -> Dict:
        """
        Confirm multiple frames in bulk.
        
        Args:
            profile_id: Animal profile ID
            frame_filenames: List of frame filenames
        
        Returns:
            Dict with results and statistics
        """
        results = {
            "confirmed": [],
            "failed": [],
            "total": len(frame_filenames)
        }
        
        for filename in frame_filenames:
            try:
                self.confirm_frame(profile_id, filename)
                results["confirmed"].append(filename)
            except Exception as e:
                logger.warning(f"Failed to confirm {filename}: {e}")
                results["failed"].append({"filename": filename, "error": str(e)})
        
        # Get updated profile for accuracy info
        profile = self.profile_manager.get_profile(profile_id)
        if profile:
            results["updated_accuracy"] = profile.accuracy_percentage
            results["confirmed_count"] = profile.confirmed_count
            results["rejected_count"] = profile.rejected_count
            self._maybe_train_profile(profile)
        
        logger.info(
            f"Bulk confirm for {profile_id}: {len(results['confirmed'])} confirmed, "
            f"{len(results['failed'])} failed"
        )
        
        return results
    
    def bulk_reject_frames(
        self,
        profile_id: str,
        frame_filenames: List[str],
        save_as_negative: bool = False
    ) -> Dict:
        """
        Reject multiple frames in bulk.
        
        Args:
            profile_id: Animal profile ID
            frame_filenames: List of frame filenames
            save_as_negative: If True, save as negative training examples
        
        Returns:
            Dict with results and statistics
        """
        results = {
            "rejected": [],
            "failed": [],
            "total": len(frame_filenames),
            "saved_as_negative": save_as_negative
        }
        
        for filename in frame_filenames:
            try:
                self.reject_frame(profile_id, filename, save_as_negative)
                results["rejected"].append(filename)
            except Exception as e:
                logger.warning(f"Failed to reject {filename}: {e}")
                results["failed"].append({"filename": filename, "error": str(e)})
        
        # Get updated profile for accuracy info
        profile = self.profile_manager.get_profile(profile_id)
        if profile:
            results["updated_accuracy"] = profile.accuracy_percentage
            results["confirmed_count"] = profile.confirmed_count
            results["rejected_count"] = profile.rejected_count
        
        logger.info(
            f"Bulk reject for {profile_id}: {len(results['rejected'])} rejected, "
            f"{len(results['failed'])} failed"
        )
        
        return results
    
    def get_review_stats(self, profile_id: str) -> Dict:
        """
        Get statistics for a profile's review status.
        
        Args:
            profile_id: Animal profile ID
        
        Returns:
            Dict with review statistics
        """
        profile = self.profile_manager.get_profile(profile_id)
        pending_frames = self.list_pending_reviews(profile_id)
        
        stats = {
            "profile_id": profile_id,
            "profile_name": profile.name if profile else "Unknown",
            "pending_count": len(pending_frames),
            "confirmed_count": profile.confirmed_count if profile else 0,
            "rejected_count": profile.rejected_count if profile else 0,
            "total_feedback": (profile.confirmed_count + profile.rejected_count) if profile else 0,
            "accuracy_percentage": profile.accuracy_percentage if profile else 0.0,
            "average_confidence": self._calculate_average_confidence(pending_frames),
            "should_recommend_retraining": profile.should_recommend_retraining if profile else (False, "")
        }
        
        return stats
    
    def get_all_review_stats(self) -> Dict:
        """Get statistics for all profiles."""
        all_profiles = self.profile_manager.list_profiles()
        stats = {}
        
        for profile in all_profiles:
            stats[profile.id] = self.get_review_stats(profile.id)
        
        return stats
    
    def _calculate_average_confidence(self, frames: List[ReviewFrame]) -> float:
        """Calculate average confidence of pending frames."""
        if not frames:
            return 0.0
        
        total = sum(f.get_confidence() for f in frames)
        return total / len(frames)
    
    def move_confirmed_to_sorted(
        self,
        profile_id: str,
        frame_filenames: List[str]
    ) -> Dict:
        """
        Move confirmed frames from training/confirmed to sorted.
        
        This is for when user wants to manually move frames to sorted.
        
        Args:
            profile_id: Animal profile ID
            frame_filenames: List of frame filenames in training/confirmed
        
        Returns:
            Dict with results
        """
        results = {
            "moved": [],
            "failed": [],
            "total": len(frame_filenames)
        }
        
        for filename in frame_filenames:
            try:
                src_path = self.base_data_path / "training" / profile_id / "confirmed" / filename
                dst_path = self.base_data_path / "sorted" / profile_id / filename
                
                if not src_path.exists():
                    raise FileNotFoundError(f"Frame not found: {src_path}")
                
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(dst_path))
                
                # Move metadata if it exists
                metadata_src = Path(str(src_path) + ".json")
                if metadata_src.exists():
                    metadata_dst = Path(str(dst_path) + ".json")
                    shutil.move(str(metadata_src), str(metadata_dst))
                
                results["moved"].append(filename)
                logger.info(f"Moved {filename} to sorted for {profile_id}")
            
            except Exception as e:
                logger.warning(f"Failed to move {filename}: {e}")
                results["failed"].append({"filename": filename, "error": str(e)})
        
        return results


class ReviewSummary:
    """Generate summary reports for review activities."""
    
    def __init__(self, review_manager: ReviewManager):
        """
        Initialize review summary.
        
        Args:
            review_manager: ReviewManager instance
        """
        self.review_manager = review_manager
    
    def get_dashboard_summary(self) -> Dict:
        """Get summary for dashboard display."""
        all_stats = self.review_manager.get_all_review_stats()
        
        total_pending = sum(s["pending_count"] for s in all_stats.values())
        total_confirmed = sum(s["confirmed_count"] for s in all_stats.values())
        total_rejected = sum(s["rejected_count"] for s in all_stats.values())
        
        profiles_with_pending = [
            (pid, stats["pending_count"])
            for pid, stats in all_stats.items()
            if stats["pending_count"] > 0
        ]
        
        profiles_with_retraining_recommendation = [
            (pid, stats["should_recommend_retraining"][1])
            for pid, stats in all_stats.items()
            if stats["should_recommend_retraining"][0]
        ]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_pending": total_pending,
            "total_confirmed": total_confirmed,
            "total_rejected": total_rejected,
            "total_feedback": total_confirmed + total_rejected,
            "profiles_count": len(all_stats),
            "profiles_with_pending": len(profiles_with_pending),
            "profiles_with_pending_details": profiles_with_pending,
            "profiles_recommending_retraining": profiles_with_retraining_recommendation,
            "all_profiles_stats": all_stats
        }
