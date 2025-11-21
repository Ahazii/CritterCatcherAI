"""
Video sorter module for organizing videos by detected class.
Moves videos to class-specific directories.
Supports clip extraction mode for specialized species detection.
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, Set, List

logger = logging.getLogger(__name__)


class VideoSorter:
    """Sorts videos into directories based on detected classes."""
    
    def __init__(self, sorted_base_path: str):
        """
        Initialize the video sorter.
        
        Args:
            sorted_base_path: Base path where sorted videos will be stored
        """
        self.sorted_base_path = Path(sorted_base_path)
        self.sorted_base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized VideoSorter with base path: {sorted_base_path}")
    
    def create_class_directory(self, class_name: str) -> Path:
        """
        Create directory for a specific class if it doesn't exist.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Path to the class directory
        """
        class_dir = self.sorted_base_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        return class_dir
    
    def move_video(self, video_path: Path, class_name: str, copy: bool = False) -> Path:
        """
        Move (or copy) video to the appropriate class directory.
        
        Args:
            video_path: Path to the video file
            class_name: Classification label for the video
            copy: If True, copy instead of move
            
        Returns:
            Path to the new video location
        """
        if not video_path.exists():
            logger.error(f"Video file does not exist: {video_path}")
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Create class directory
        class_dir = self.create_class_directory(class_name)
        
        # Destination path
        dest_path = class_dir / video_path.name
        
        # Handle duplicate filenames
        if dest_path.exists():
            # Add counter to filename
            counter = 1
            stem = video_path.stem
            suffix = video_path.suffix
            while dest_path.exists():
                new_name = f"{stem}_{counter}{suffix}"
                dest_path = class_dir / new_name
                counter += 1
        
        try:
            if copy:
                shutil.copy2(video_path, dest_path)
                logger.info(f"Copied video to {class_name}: {video_path.name}")
            else:
                shutil.move(str(video_path), str(dest_path))
                logger.info(f"Moved video to {class_name}: {video_path.name}")
            
            return dest_path
            
        except Exception as e:
            logger.error(f"Failed to move/copy video {video_path.name}: {e}", exc_info=True)
            raise
    
    def sort_by_object(self, video_path: Path, detected_object: str) -> Path:
        """
        Sort video based on detected object.
        
        Args:
            video_path: Path to the video file
            detected_object: Detected object label
            
        Returns:
            Path to sorted video
        """
        return self.move_video(video_path, detected_object)
    
    def sort_by_person(self, video_path: Path, person_name: str) -> Path:
        """
        Sort video based on recognized person.
        
        Args:
            video_path: Path to the video file
            person_name: Name of recognized person
            
        Returns:
            Path to sorted video
        """
        # Store in a "people" subdirectory with person name
        class_name = f"people/{person_name}"
        return self.move_video(video_path, class_name)
    
    def sort_video(
        self, 
        video_path: Path, 
        detected_objects: Dict[str, float] = None,
        recognized_people: Set[str] = None,
        priority: str = "people"
    ) -> Path:
        """
        Sort video based on all detections with priority rules.
        
        Args:
            video_path: Path to the video file
            detected_objects: Dictionary of detected objects and their confidence scores
            recognized_people: Set of recognized person names
            priority: What to prioritize ("people" or "objects")
            
        Returns:
            Path to sorted video
        """
        # Priority 1: People (if priority is "people" and people were detected)
        if priority == "people" and recognized_people:
            # If multiple people, use the first one alphabetically
            primary_person = sorted(recognized_people)[0]
            logger.info(f"Sorting {video_path.name} by person: {primary_person}")
            return self.sort_by_person(video_path, primary_person)
        
        # Priority 2: Objects
        if detected_objects:
            # Get object with highest confidence
            best_object = max(detected_objects, key=detected_objects.get)
            logger.info(f"Sorting {video_path.name} by object: {best_object}")
            return self.sort_by_object(video_path, best_object)
        
        # Priority 3: People (if priority was "objects" but no objects detected)
        if recognized_people:
            primary_person = sorted(recognized_people)[0]
            logger.info(f"Sorting {video_path.name} by person (fallback): {primary_person}")
            return self.sort_by_person(video_path, primary_person)
        
        # No detections - move to "unknown" category
        logger.info(f"No detections for {video_path.name}, sorting to 'unknown'")
        return self.move_video(video_path, "unknown")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about sorted videos.
        
        Returns:
            Dictionary mapping class name to video count
        """
        stats = {}
        
        def count_videos_recursive(base_path: Path, prefix: str = ""):
            """Recursively count videos in nested directories."""
            for item in base_path.iterdir():
                if item.is_dir():
                    # Check for videos directly in this directory
                    video_count = len(list(item.glob("*.mp4")))
                    
                    # Build the full path name
                    full_name = f"{prefix}/{item.name}" if prefix else item.name
                    
                    # If videos exist here, add to stats
                    if video_count > 0:
                        stats[full_name] = video_count
                    
                    # Recursively check subdirectories
                    count_videos_recursive(item, full_name)
        
        count_videos_recursive(self.sorted_base_path)
        
        logger.debug(f"Video sorting statistics: {stats}")
        return stats
    
