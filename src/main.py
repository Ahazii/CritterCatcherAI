"""
CritterCatcherAI - Main application.
Orchestrates video download, analysis, and sorting.
"""
import os
import sys
import time
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
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


def load_config(config_path: str = "/config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    logger = logging.getLogger(__name__)
    
    # Ensure config directory exists
    config_dir = Path(config_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy default config if it doesn't exist
    if not Path(config_path).exists():
        default_config = Path("/app/config/config.yaml")
        if default_config.exists():
            import shutil
            shutil.copy(default_config, config_path)
            logger.info(f"Copied default config to {config_path}")
        else:
            logger.warning(f"Default config not found at {default_config}")
    
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
    
    # V2: Object detection now happens in Animal Profile processing
    # Initialize object detector with configured labels (for backward compatibility)
    object_labels = detection_config.get('object_labels', ['bird', 'cat', 'dog', 'person'])
    yolo_model = detection_config.get('yolo_model', 'yolov8n')  # Default to nano model
    
    # Add .pt extension if not present
    if not yolo_model.endswith('.pt'):
        yolo_model = f"{yolo_model}.pt"
    
    object_detector = ObjectDetector(
        labels=object_labels,
        confidence_threshold=detection_config.get('confidence_threshold', 0.25),
        num_frames=detection_config.get('object_frames', 5),
        model_name=yolo_model
    )
    
    # Initialize face recognizer
    face_recognizer = FaceRecognizer(
        encodings_path=config.get('paths', {}).get('face_encodings', '/data/faces/encodings.pkl'),
        tolerance=detection_config.get('face_tolerance', 0.6),
        num_frames=detection_config.get('face_frames', 10),
        model=detection_config.get('face_model', 'hog')
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
    
    # Update progress: authentication
    try:
        from webapp import app_state
        app_state["processing_progress"]["current_step"] = "Authenticating with Ring..."
        app_state["processing_progress"]["phase"] = "authentication"
    except:
        pass
    
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
    
    # Update progress: downloading
    try:
        from webapp import app_state
        app_state["processing_progress"]["current_step"] = f"Downloading videos from Ring (last {video_hours}h)..."
        app_state["processing_progress"]["phase"] = "downloading"
    except:
        pass
    
    downloaded_videos = ring_downloader.download_recent_videos(
        hours=video_hours,
        limit=video_limit
    )
    
    if not downloaded_videos:
        logger.info("No new videos to process")
        return
    
    logger.info(f"Processing {len(downloaded_videos)} videos")
    
    # Update progress: set total count
    try:
        from webapp import app_state
        app_state["processing_progress"]["videos_total"] = len(downloaded_videos)
    except:
        pass  # webapp might not be loaded
    
    # Process each video
    for idx, video_path in enumerate(downloaded_videos, 1):
        # Check stop flag or if scheduler was disabled
        try:
            from webapp import app_state
            if app_state.get("stop_requested", False):
                logger.info("Stop requested - ending processing gracefully")
                break
            if not app_state["scheduler"]["enabled"]:
                logger.info("Scheduler disabled - stopping current processing run")
                break
        except:
            pass
        
        try:
            # Check if video still exists (may have been processed/moved already)
            if not video_path.exists():
                logger.debug(f"Skipping {video_path.name}: already processed or moved")
                continue
            
            # Update progress
            try:
                from webapp import app_state
                app_state["processing_progress"]["current_video"] = video_path.name
                app_state["processing_progress"]["videos_processed"] = idx
            except:
                pass
            
            logger.info(f"Processing video: {video_path.name}")
            
            # Update progress: object detection
            try:
                from webapp import app_state
                app_state["processing_progress"]["current_step"] = f"Analyzing {video_path.name} with YOLO..."
                app_state["processing_progress"]["phase"] = "detection"
            except:
                pass
            
            # V2: Run standard YOLO detection
            logger.debug("Running YOLO object detection")
            detected_objects = object_detector.detect_objects_in_video(video_path)
            
            # Run face recognition ONLY if enabled and conditions are met
            priority = detection_config.get('priority', 'objects')
            face_recognition_enabled = config.get('face_recognition', {}).get('enabled', False)
            recognized_people = set()
            
            if face_recognition_enabled and (priority == "people" or not detected_objects):
                # Update progress: face recognition
                try:
                    from webapp import app_state
                    app_state["processing_progress"]["current_step"] = f"Recognizing faces in {video_path.name}..."
                    app_state["processing_progress"]["phase"] = "face_recognition"
                except:
                    pass
                
                recognized_people = face_recognizer.recognize_faces_in_video(video_path)
            
            # Update progress: moving to review
            try:
                from webapp import app_state
                app_state["processing_progress"]["current_step"] = f"Moving {video_path.name} to review..."
                app_state["processing_progress"]["phase"] = "review"
            except:
                pass
            
            # V2 Review Workflow: Move videos to review folder with metadata
            # Determine primary detection for review categorization
            review_category = "unknown"
            if detected_objects:
                review_category = max(detected_objects, key=detected_objects.get)
            elif recognized_people:
                review_category = f"people_{sorted(recognized_people)[0]}"
            
            # Create review directory for this category
            review_dir = Path("/data/review") / review_category
            review_dir.mkdir(parents=True, exist_ok=True)
            
            # Move video to review
            review_path = review_dir / video_path.name
            if review_path.exists():
                # Handle duplicates
                counter = 1
                while review_path.exists():
                    review_path = review_dir / f"{video_path.stem}_{counter}{video_path.suffix}"
                    counter += 1
            
            import shutil
            shutil.move(str(video_path), str(review_path))
            
            # Save detection metadata alongside video
            metadata = {
                "detected_objects": detected_objects if detected_objects else {},
                "recognized_people": list(recognized_people) if recognized_people else [],
                "timestamp": datetime.now().isoformat(),
                "video_name": video_path.name,
                "category": review_category
            }
            metadata_path = review_path.with_suffix(review_path.suffix + ".json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Video moved to review: {review_path} (category: {review_category})")
            
        except Exception as e:
            logger.error(f"Failed to process video {video_path.name}: {e}", exc_info=True)
    
    # Log statistics
    stats = video_sorter.get_stats()
    logger.info(f"Sorting statistics: {stats}")


def main():
    """Main entry point."""
    # Ensure all required directories exist with proper permissions
    # This must happen before logging setup or any other operations
    required_dirs = [
        "/data/downloads",
        "/data/sorted",
        "/data/faces",
        "/data/faces/unknown",
        "/data/tokens",
        "/data/animal_profiles",
        "/data/review",
        "/data/training",
        "/data/models",
        "/config"
    ]
    
    for dir_path in required_dirs:
        try:
            dir_obj = Path(dir_path)
            dir_obj.mkdir(parents=True, exist_ok=True)
            # Explicitly set permissions after creation
            os.chmod(dir_path, 0o777)
        except PermissionError:
            # If we can't set permissions, that's okay - continue anyway
            pass
        except Exception as e:
            print(f"Warning: Could not create {dir_path}: {e}")
    
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
    
    # Get run mode (new scheduler block)
    scheduler = config.get('scheduler', {})
    auto_run = scheduler.get('auto_run', True)
    interval_minutes = scheduler.get('interval_minutes', config.get('interval_minutes', 60))
    run_once = config.get('run_once', False) or os.environ.get('RUN_ONCE', '').lower() == 'true'
    
    if run_once:
        logger.info("Running in single-run mode")
        process_videos(config)
        logger.info("Processing complete")
        return
    
    # Update scheduler state in webapp
    try:
        from webapp import app_state
        app_state["scheduler"]["enabled"] = auto_run
        app_state["scheduler"]["interval_minutes"] = interval_minutes
    except:
        pass
    
    logger.info("Scheduler loop started - will respect config changes in real-time")
    
    # Track if this is the first run
    first_run = True
    
    # Main scheduler loop - checks app_state dynamically
    while True:
        try:
            from webapp import app_state
            
            # Check if scheduler is enabled (respects real-time config changes)
            if not app_state["scheduler"]["enabled"]:
                # Scheduler disabled - sleep and check again
                if not hasattr(main, '_logged_disabled'):
                    logger.info("Auto Run is DISABLED. Waiting for manual trigger or config change.")
                    main._logged_disabled = True
                time.sleep(10)  # Check every 10 seconds
                first_run = True  # Reset first_run flag when disabled
                continue
            
            # Reset the disabled log flag
            if hasattr(main, '_logged_disabled'):
                delattr(main, '_logged_disabled')
                logger.info("Auto Run is ENABLED - waiting for first scheduled run")
            
            # Get current interval from app_state (may have changed)
            current_interval = app_state["scheduler"]["interval_minutes"]
            
            # Set next run time
            next_run_time = datetime.now() + timedelta(minutes=current_interval)
            app_state["scheduler"]["next_run"] = next_run_time.isoformat()
            
            # On first run, wait for the interval before processing
            if first_run:
                logger.info(f"Scheduler enabled - first run scheduled for {next_run_time.strftime('%Y-%m-%d %H:%M:%S')} ({current_interval} minutes)")
                first_run = False
            else:
                logger.info(f"Processing videos (interval: {current_interval} minutes)")
                
                # Set processing state before running
                app_state["is_processing"] = True
                app_state["stop_requested"] = False
                app_state["processing_progress"]["start_time"] = datetime.now()
                
                try:
                    process_videos(config)
                    app_state["last_run"] = datetime.now().isoformat()
                except Exception as e:
                    logger.error(f"Scheduled processing failed: {e}", exc_info=True)
                finally:
                    app_state["is_processing"] = False
                    app_state["processing_progress"]["current_step"] = "Complete" if not app_state["stop_requested"] else "Stopped"
            
            logger.info(f"Sleeping for {current_interval} minutes (next run: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')})")
            
            # Sleep in smaller intervals to allow checking for config changes
            sleep_seconds = current_interval * 60
            sleep_interval = 10  # Check every 10 seconds
            slept = 0
            
            while slept < sleep_seconds:
                # Check if scheduler was disabled during sleep
                if not app_state["scheduler"]["enabled"]:
                    logger.info("Scheduler disabled during sleep - stopping automatic processing")
                    app_state["scheduler"]["next_run"] = None
                    break
                
                time.sleep(min(sleep_interval, sleep_seconds - slept))
                slept += sleep_interval
                
        except KeyboardInterrupt:
            logger.info("Shutting down")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            logger.info("Waiting 5 minutes before retry")
            time.sleep(300)


if __name__ == "__main__":
    main()
