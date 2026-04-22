"""
Script to reprocess person videos in review/person/unknown with face recognition.
Extracts faces and sorts videos by recognized person names.
"""
import logging
from pathlib import Path
from face_recognizer import FaceRecognizer
from video_sorter import VideoSorter
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reprocess_person_videos():
    """Reprocess all videos in review/person/unknown with face recognition."""
    
    # Load config
    with open('/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    face_recognizer = FaceRecognizer(
        encodings_path=config['paths']['face_encodings'],
        tolerance=config['detection'].get('face_tolerance', 0.6),
        model=config['detection'].get('face_model', 'hog')
    )
    
    video_sorter = VideoSorter(config['paths']['sorted'])
    
    # Find all videos in review/person/unknown
    review_path = Path("/data/review/person/unknown")
    if not review_path.exists():
        logger.info("No review/person/unknown folder found")
        return
    
    videos = list(review_path.glob("*.mp4"))
    logger.info(f"Found {len(videos)} videos to reprocess")
    
    recognized_count = 0
    unknown_count = 0
    
    for video_path in videos:
        try:
            logger.info(f"Processing: {video_path.name}")
            
            # Run face recognition
            recognized_people = face_recognizer.recognize_faces_in_video(video_path)
            
            if recognized_people:
                # Get primary person (first alphabetically)
                primary_person = sorted(recognized_people)[0]
                logger.info(f"  ✓ Recognized: {primary_person}")
                
                # Move to sorted/person/{name}
                dest_path = video_sorter.move_video(
                    video_path,
                    class_name=f"person/{primary_person}"
                )
                logger.info(f"  → Moved to: {dest_path}")
                recognized_count += 1
            else:
                logger.info(f"  ? No face recognized - leaving in review")
                unknown_count += 1
                
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}", exc_info=True)
    
    logger.info(f"\nReprocessing complete:")
    logger.info(f"  Recognized and sorted: {recognized_count}")
    logger.info(f"  Unknown (in review): {unknown_count}")

if __name__ == "__main__":
    reprocess_person_videos()
