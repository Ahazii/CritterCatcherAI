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
        
        for class_dir in self.sorted_base_path.iterdir():
            if class_dir.is_dir():
                # Count video files
                video_count = len(list(class_dir.glob("*.mp4")))
                
                # Handle nested people directories
                if class_dir.name == "people":
                    for person_dir in class_dir.iterdir():
                        if person_dir.is_dir():
                            person_count = len(list(person_dir.glob("*.mp4")))
                            stats[f"people/{person_dir.name}"] = person_count
                else:
                    stats[class_dir.name] = video_count
        
        logger.debug(f"Video sorting statistics: {stats}")
        return stats
    
    def sort_with_specialization(
        self,
        video_path: Path,
        yolo_detections: Dict[str, float],
        species_results: Dict[str, float],
        config: dict,
        detected_objects_path: Path,
        recognized_people: Set[str] = None
    ) -> Dict[str, any]:
        """
        Sort video with specialized species detection and optional clip extraction.
        
        Args:
            video_path: Path to the video file
            yolo_detections: YOLO Stage 1 detection results
            species_results: Specialized Stage 2 detection results
            config: Full configuration dictionary
            detected_objects_path: Path to detection metadata
            recognized_people: Set of recognized person names
            
        Returns:
            Dictionary with sorting results and actions taken
        """
        result = {
            "video_path": video_path,
            "action": None,  # "sorted", "clip_extracted", "deleted"
            "target": None,  # class name or species name
            "clips": []  # extracted clip paths
        }
        
        clip_config = config.get('specialized_detection', {}).get('clip_extraction', {})
        clip_mode_enabled = clip_config.get('enabled', False)
        auto_delete = clip_config.get('auto_delete_non_matches', False)
        
        # Check if clip extraction mode is enabled and we have species detections
        if clip_mode_enabled and species_results:
            logger.info(f"Clip extraction mode enabled for {video_path.name}")
            
            # Get species confidence thresholds
            species_config = config.get('specialized_detection', {}).get('species', [])
            thresholds = {s['name']: s.get('confidence_threshold', 0.75) 
                         for s in species_config}
            
            # Check which species meet the threshold
            matching_species = {name: conf for name, conf in species_results.items()
                              if conf >= thresholds.get(name, 0.75)}
            
            if matching_species:
                # Extract clips for matching species
                logger.info(f"Extracting clips for species: {list(matching_species.keys())}")
                
                try:
                    from clip_extractor import ClipExtractor
                    
                    extractor = ClipExtractor(
                        padding_seconds=clip_config.get('padding_seconds', 10),
                        output_path=clip_config.get('output_path', '/data/clips'),
                        merge_overlapping=clip_config.get('merge_overlapping', True)
                    )
                    
                    extracted_clips = extractor.extract_clips_from_detections(
                        video_path,
                        matching_species,
                        detected_objects_path,
                        thresholds
                    )
                    
                    if extracted_clips:
                        result["action"] = "clip_extracted"
                        result["target"] = list(matching_species.keys())
                        result["clips"] = [clip for clips in extracted_clips.values() 
                                          for clip in clips]
                        logger.info(f"✓ Extracted {len(result['clips'])} clip(s) from {video_path.name}")
                        
                        # Delete original video after successful clip extraction
                        try:
                            video_path.unlink()
                            logger.info(f"Deleted original video: {video_path.name}")
                        except Exception as e:
                            logger.error(f"Failed to delete original video: {e}")
                        
                        return result
                    
                except ImportError:
                    logger.warning("ClipExtractor not available")
                except Exception as e:
                    logger.error(f"Error extracting clips: {e}", exc_info=True)
            
            elif auto_delete:
                # No matching species and auto-delete is enabled
                logger.info(f"No target species detected in {video_path.name}, deleting (auto_delete=true)")
                try:
                    video_path.unlink()
                    result["action"] = "deleted"
                    result["target"] = "no_match"
                    logger.info(f"✗ Deleted non-matching video: {video_path.name}")
                    return result
                except Exception as e:
                    logger.error(f"Failed to delete video: {e}")
        
        # Fallback: Standard sorting (original behavior)
        logger.debug(f"Using standard sorting for {video_path.name}")
        
        # Prioritize specialized species results over YOLO for sorting
        if species_results:
            best_species = max(species_results, key=species_results.get)
            sorted_path = self.sort_by_object(video_path, best_species)
            result["action"] = "sorted"
            result["target"] = best_species
            return result
        
        # Use original sorting logic
        priority = config.get('detection', {}).get('priority', 'people')
        sorted_path = self.sort_video(
            video_path,
            detected_objects=yolo_detections,
            recognized_people=recognized_people,
            priority=priority
        )
        
        result["action"] = "sorted"
        result["target"] = sorted_path.parent.name
        return result
