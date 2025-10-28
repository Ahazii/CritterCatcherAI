"""
Video clip extraction module.
Extracts clips from videos with time padding and merging of overlapping segments.
"""
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ClipExtractor:
    """Extract video clips with intelligent time range merging."""
    
    def __init__(self, padding_seconds: int = 10, output_path: str = "/data/clips",
                 merge_overlapping: bool = True):
        """
        Initialize clip extractor.
        
        Args:
            padding_seconds: Seconds to add before/after detections
            output_path: Directory to save extracted clips
            merge_overlapping: Whether to merge overlapping time ranges
        """
        self.padding_seconds = padding_seconds
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.merge_overlapping = merge_overlapping
        
        logger.info(f"ClipExtractor initialized: padding={padding_seconds}s, "
                   f"merge={merge_overlapping}, output={output_path}")
    
    def get_video_duration(self, video_path: Path) -> float:
        """
        Get video duration in seconds using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Duration in seconds
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            logger.debug(f"Video duration: {duration}s")
            return duration
            
        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return 0.0
    
    def get_detection_times(self, video_path: Path, detected_objects_path: Path,
                           species_name: str) -> List[float]:
        """
        Extract detection times from metadata files.
        
        Args:
            video_path: Source video path
            detected_objects_path: Path to detection metadata
            species_name: Species to look for
            
        Returns:
            List of detection times in seconds
        """
        video_name = video_path.stem
        detection_times = []
        
        # Check both YOLO detections and specialized detections
        paths_to_check = [
            detected_objects_path / species_name.replace(" ", "_"),
            Path("/data/objects/specialized") / species_name.replace(" ", "_")
        ]
        
        for label_dir in paths_to_check:
            if not label_dir.exists():
                continue
            
            for metadata_file in label_dir.glob(f"*{video_name}*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract frame index
                    frame_idx = metadata.get('frame_idx', 0)
                    
                    # Estimate time (assuming metadata contains frame index)
                    # We'll need to calculate this from total frames
                    # For now, use a simple approximation
                    video_duration = self.get_video_duration(video_path)
                    
                    # Get total frames and calculate time
                    # This is an approximation - will be refined
                    num_frames = metadata.get('num_frames', 5)
                    detection_time = (frame_idx / max(num_frames - 1, 1)) * video_duration
                    
                    detection_times.append(detection_time)
                    logger.debug(f"Found detection at {detection_time:.2f}s in {metadata_file.name}")
                    
                except Exception as e:
                    logger.error(f"Error reading metadata {metadata_file}: {e}")
        
        return sorted(detection_times)
    
    def merge_time_ranges(self, times: List[float], video_duration: float) -> List[Tuple[float, float]]:
        """
        Convert detection times to padded ranges and merge overlapping ones.
        
        Args:
            times: List of detection times in seconds
            video_duration: Total video duration in seconds
            
        Returns:
            List of (start, end) tuples in seconds
        """
        if not times:
            return []
        
        # Create padded ranges
        ranges = []
        for time in times:
            start = max(0, time - self.padding_seconds)
            end = min(video_duration, time + self.padding_seconds)
            ranges.append((start, end))
        
        if not self.merge_overlapping or len(ranges) == 1:
            return ranges
        
        # Sort by start time
        ranges.sort()
        
        # Merge overlapping ranges
        merged = [ranges[0]]
        for current_start, current_end in ranges[1:]:
            last_start, last_end = merged[-1]
            
            if current_start <= last_end:
                # Overlapping or adjacent - merge
                merged[-1] = (last_start, max(last_end, current_end))
                logger.debug(f"Merged overlapping ranges: [{last_start:.2f}-{last_end:.2f}] "
                           f"+ [{current_start:.2f}-{current_end:.2f}] "
                           f"= [{merged[-1][0]:.2f}-{merged[-1][1]:.2f}]")
            else:
                # Non-overlapping - add as new range
                merged.append((current_start, current_end))
        
        logger.info(f"Merged {len(ranges)} ranges into {len(merged)} clips")
        return merged
    
    def extract_clip(self, video_path: Path, start_time: float, end_time: float,
                    output_path: Path) -> bool:
        """
        Extract a single clip from video using ffmpeg.
        
        Args:
            video_path: Source video path
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output clip path
            
        Returns:
            True if successful
        """
        try:
            duration = end_time - start_time
            
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',  # H.264 codec
                '-c:a', 'aac',      # AAC audio
                '-preset', 'fast',  # Encoding speed
                '-crf', '23',       # Quality (lower = better)
                '-y',               # Overwrite output
                str(output_path)
            ]
            
            logger.info(f"Extracting clip: {start_time:.2f}s - {end_time:.2f}s "
                       f"(duration: {duration:.2f}s)")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            logger.info(f"✓ Clip extracted: {output_path.name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Failed to extract clip: {e}")
            return False
    
    def extract_clips_for_species(self, video_path: Path, species_name: str,
                                  detected_objects_path: Path) -> List[Path]:
        """
        Extract all clips for a detected species from a video.
        
        Args:
            video_path: Source video path
            species_name: Species that was detected
            detected_objects_path: Path to detection metadata
            
        Returns:
            List of paths to extracted clips
        """
        logger.info(f"Extracting clips for {species_name} from {video_path.name}")
        
        # Get video duration
        video_duration = self.get_video_duration(video_path)
        if video_duration == 0:
            logger.error("Could not determine video duration")
            return []
        
        # Get detection times
        detection_times = self.get_detection_times(
            video_path, detected_objects_path, species_name
        )
        
        if not detection_times:
            logger.warning(f"No detection times found for {species_name}")
            # Fallback: extract entire video if we have a positive detection
            detection_times = [video_duration / 2]  # Middle of video
        
        logger.info(f"Found {len(detection_times)} detection(s) at: "
                   f"{[f'{t:.2f}s' for t in detection_times]}")
        
        # Merge time ranges
        time_ranges = self.merge_time_ranges(detection_times, video_duration)
        logger.info(f"Merged into {len(time_ranges)} clip(s)")
        
        # Create species output directory
        species_dir = self.output_path / species_name.replace(" ", "_")
        species_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract each clip
        extracted_clips = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, (start, end) in enumerate(time_ranges):
            # Generate output filename
            if len(time_ranges) == 1:
                clip_name = f"{timestamp}_{video_path.stem}_{species_name}.mp4"
            else:
                clip_name = f"{timestamp}_{video_path.stem}_{species_name}_part{idx+1}.mp4"
            
            output_path = species_dir / clip_name
            
            # Extract clip
            success = self.extract_clip(video_path, start, end, output_path)
            
            if success:
                extracted_clips.append(output_path)
                
                # Save clip metadata
                metadata = {
                    "source_video": video_path.name,
                    "species": species_name,
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start,
                    "clip_index": idx,
                    "total_clips": len(time_ranges),
                    "timestamp": datetime.now().isoformat()
                }
                
                metadata_path = output_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully extracted {len(extracted_clips)} clip(s) for {species_name}")
        return extracted_clips
    
    def extract_clips_from_detections(self, video_path: Path, 
                                     species_results: Dict[str, float],
                                     detected_objects_path: Path,
                                     confidence_thresholds: Dict[str, float]) -> Dict[str, List[Path]]:
        """
        Extract clips for all detected species that meet confidence thresholds.
        
        Args:
            video_path: Source video path
            species_results: Dict of {species_name: confidence}
            detected_objects_path: Path to detection metadata
            confidence_thresholds: Dict of {species_name: threshold}
            
        Returns:
            Dict of {species_name: [clip_paths]}
        """
        extracted = {}
        
        for species_name, confidence in species_results.items():
            threshold = confidence_thresholds.get(species_name, 0.75)
            
            if confidence >= threshold:
                logger.info(f"Extracting clips for {species_name} "
                           f"(confidence: {confidence:.3f} >= {threshold:.3f})")
                
                clips = self.extract_clips_for_species(
                    video_path, species_name, detected_objects_path
                )
                
                if clips:
                    extracted[species_name] = clips
            else:
                logger.debug(f"Skipping {species_name} "
                           f"(confidence: {confidence:.3f} < {threshold:.3f})")
        
        return extracted
