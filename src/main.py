"""
CritterCatcherAI - Main application.
Orchestrates video download, analysis, and sorting.
"""
import os
import sys
import time
import logging
from pathlib import Path
import yaml

from ring_downloader import RingDownloader
from object_detector import ObjectDetector
from face_recognizer import FaceRecognizer
from video_sorter import VideoSorter


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the application."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str = "/app/config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    logger = logging.getLogger(__name__)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {}
    except Exception as e:
        logger.error(f"Failed to load config: {e}", exc_info=True)
        return {}


def process_videos(config: dict):
    """Main video processing pipeline."""
    logger = logging.getLogger(__name__)
    
    # Get configuration values (environment variables override config file)
    download_path = config.get('paths', {}).get('downloads', '/data/downloads')
    sorted_path = config.get('paths', {}).get('sorted', '/data/sorted')
    
    ring_config = config.get('ring', {})
    detection_config = config.get('detection', {})
    
    # Allow environment variable overrides for key settings
    if os.environ.get('DOWNLOAD_HOURS'):
        ring_config['download_hours'] = int(os.environ.get('DOWNLOAD_HOURS'))
    if os.environ.get('INTERVAL_MINUTES'):
        config['interval_minutes'] = int(os.environ.get('INTERVAL_MINUTES'))
    if os.environ.get('CONFIDENCE_THRESHOLD'):
        detection_config['confidence_threshold'] = float(os.environ.get('CONFIDENCE_THRESHOLD'))
    if os.environ.get('FACE_TOLERANCE'):
        detection_config['face_tolerance'] = float(os.environ.get('FACE_TOLERANCE'))
    if os.environ.get('DETECTION_PRIORITY'):
        detection_config['priority'] = os.environ.get('DETECTION_PRIORITY')
    
    # Initialize components
    logger.info("Initializing CritterCatcherAI components")
    
    ring_downloader = RingDownloader(download_path)
    video_sorter = VideoSorter(sorted_path)
    
    # Initialize object detector with configured labels
    object_labels = detection_config.get('object_labels', ['hedgehog', 'fox', 'bird', 'cat', 'dog'])
    object_detector = ObjectDetector(
        labels=object_labels,
        confidence_threshold=detection_config.get('confidence_threshold', 0.25)
    )
    
    # Initialize face recognizer
    face_recognizer = FaceRecognizer(
        encodings_path=config.get('paths', {}).get('face_encodings', '/data/faces/encodings.pkl'),
        tolerance=detection_config.get('face_tolerance', 0.6)
    )
    
    # Check if Ring token exists, skip auth if not (force web GUI setup)
    token_file = Path("/data/tokens/ring_token.json")
    logger.info(f"Checking for Ring token at: {token_file}")
    logger.info(f"Token file exists: {token_file.exists()}")
    
    if not token_file.exists():
        logger.warning("="*80)
        logger.warning("NO RING TOKEN FOUND - AUTHENTICATION REQUIRED")
        logger.warning("="*80)
        logger.warning("Please complete Ring authentication using the web interface:")
        logger.warning("")
        logger.warning("  1. Open the web interface in your browser")
        logger.warning("  2. Navigate to the 'Ring Setup' tab")
        logger.warning("  3. Enter your Ring credentials")
        logger.warning("  4. Complete 2FA verification when prompted")
        logger.warning("")
        logger.warning("Video processing will begin automatically after authentication.")
        logger.warning("="*80)
        return
    
    # Authenticate with existing token
    logger.info("Authenticating with Ring using saved token")
    
    if not ring_downloader.authenticate():
        logger.error("Failed to authenticate with Ring (token may be expired)")
        logger.error("Please re-authenticate using the web interface (Ring Setup tab)")
        # Delete invalid token
        if token_file.exists():
            token_file.unlink()
            logger.info("Removed invalid token file")
        return
    
    # Download recent videos
    logger.info("Downloading recent Ring videos")
    video_hours = ring_config.get('download_hours', 24)
    video_limit = ring_config.get('download_limit')
    
    downloaded_videos = ring_downloader.download_recent_videos(
        hours=video_hours,
        limit=video_limit
    )
    
    if not downloaded_videos:
        logger.info("No new videos to process")
        return
    
    logger.info(f"Processing {len(downloaded_videos)} videos")
    
    # Process each video
    for video_path in downloaded_videos:
        try:
            logger.info(f"Processing video: {video_path.name}")
            
            # Run object detection
            detected_objects = object_detector.detect_objects_in_video(video_path)
            
            # Run face recognition
            recognized_people = face_recognizer.recognize_faces_in_video(video_path)
            
            # Sort video based on detections
            priority = detection_config.get('priority', 'people')
            sorted_path = video_sorter.sort_video(
                video_path,
                detected_objects=detected_objects,
                recognized_people=recognized_people,
                priority=priority
            )
            
            logger.info(f"Video processed and sorted to: {sorted_path}")
            
        except Exception as e:
            logger.error(f"Failed to process video {video_path.name}: {e}", exc_info=True)
    
    # Log statistics
    stats = video_sorter.get_stats()
    logger.info(f"Sorting statistics: {stats}")


def main():
    """Main entry point."""
    # Setup logging
    log_level = os.environ.get('LOG_LEVEL', 'INFO')
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting CritterCatcherAI")
    
    # Start web server in separate thread
    import threading
    from webapp import start_web_server
    
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    logger.info("Web interface started on http://0.0.0.0:8080")
    
    # Load configuration
    config = load_config()
    
    # Get run mode
    run_once = config.get('run_once', False) or os.environ.get('RUN_ONCE', '').lower() == 'true'
    interval_minutes = config.get('interval_minutes', 60)
    
    if run_once:
        logger.info("Running in single-run mode")
        process_videos(config)
        logger.info("Processing complete")
    else:
        logger.info(f"Running in continuous mode with {interval_minutes} minute interval")
        while True:
            try:
                process_videos(config)
                logger.info(f"Sleeping for {interval_minutes} minutes")
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                logger.info("Shutting down")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                logger.info("Waiting 5 minutes before retry")
                time.sleep(300)


if __name__ == "__main__":
    main()
