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
from video_sorter import VideoSorter
from face_recognizer import FaceRecognizer
from animal_profile import AnimalProfileManager
from clip_vit_classifier import CLIPVitClassifier
from face_profile import FaceProfileManager
from gpu_monitor import GPUMonitor
import cv2
import tempfile


def setup_logging(config: dict = None):
    """Configure logging for the application with file and console output.
    
    Args:
        config: Configuration dictionary. If provided, reads log level from config['logging']['level'].
               Otherwise defaults to INFO.
    """
    from logging.handlers import RotatingFileHandler
    
    # Determine log level from config or default to INFO
    log_level = "INFO"
    if config and 'logging' in config:
        log_level = config.get('logging', {}).get('level', 'INFO')
    
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Create log file directory
    log_dir = Path("/config")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "crittercatcher.log"
    
    # Create formatters
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler with rotation (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: level={log_level}, file={log_file}")


def load_config(config_path: str = "/config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    logger = logging.getLogger(__name__)
    
    # Ensure config directory exists
    config_dir = Path(config_path).parent
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy default config if it doesn't exist
    if not Path(config_path).exists():
        default_config = Path("/app/config/config.yaml")  # Template in image
        if default_config.exists():
            import shutil
            shutil.copy(default_config, config_path)  # Copy to /config volume
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


def process_videos(config: dict, manual_trigger: bool = False):
    """Main video processing pipeline.
    
    Args:
        config: Configuration dictionary
        manual_trigger: If True, bypasses scheduler enabled check (for manual "Process Now" button)
    """
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
    
    # Initialize Animal Profile Manager and CLIP Classifier for Stage 2
    profile_manager = AnimalProfileManager(Path("/data"))
    face_profile_manager = FaceProfileManager(Path("/data"))
    clip_classifier = None  # Lazy load when needed
    
    # V2: Compute active YOLO categories from enabled Animal Profiles + manual categories
    logger.info("Computing active YOLO categories from Animal Profiles and manual config")
    
    # Get manual categories from config
    manual_categories = set([c.lower() for c in config.get('yolo_manual_categories', [])])
    
    # Get categories from enabled Animal Profiles
    all_profiles = profile_manager.list_profiles()
    auto_enabled_categories = set()
    for profile in all_profiles:
        if profile.enabled:
            auto_enabled_categories.update([c.lower() for c in profile.yolo_categories])
    
    # Compute union of auto + manual
    active_categories = list(auto_enabled_categories | manual_categories)
    
    # Fallback to config object_labels if no active categories
    if not active_categories:
        logger.warning("No active YOLO categories found from profiles/manual. Using config object_labels as fallback.")
        active_categories = detection_config.get('object_labels', ['bird', 'cat', 'dog', 'person'])
    
    logger.info(f"Active YOLO categories: {active_categories}")
    logger.info(f"  Auto-enabled by profiles: {sorted(auto_enabled_categories)}")
    logger.info(f"  Manually enabled: {sorted(manual_categories)}")
    
    yolo_model = detection_config.get('yolo_model', 'yolov8n')  # Default to nano model
    
    # Add .pt extension if not present
    if not yolo_model.endswith('.pt'):
        yolo_model = f"{yolo_model}.pt"
    
    object_detector = ObjectDetector(
        labels=active_categories,
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
    
    # Check for existing unprocessed videos in downloads folder
    logger.info("Checking for unprocessed videos in downloads folder")
    download_path_obj = Path(download_path)
    existing_videos = list(download_path_obj.glob("*.mp4")) if download_path_obj.exists() else []
    
    videos_to_process = []
    
    if existing_videos:
        logger.info(f"Found {len(existing_videos)} existing videos in downloads folder")
        videos_to_process = existing_videos
    else:
        # No existing videos - download ALL available videos from Ring
        # Uses download_history.db to prevent duplicate downloads
        logger.info("No existing videos found - downloading ALL available videos from Ring")
        logger.info("Using database-tracked download to prevent duplicates")
        
        # Update progress: downloading
        try:
            from webapp import app_state
            app_state["processing_progress"]["current_step"] = "Downloading all available videos from Ring..."
            app_state["processing_progress"]["phase"] = "downloading"
        except:
            pass
        
        try:
            # Use download_all_videos() which checks database to prevent re-downloads
            # hours=None means download ALL available videos (not time-limited)
            stats = ring_downloader.download_all_videos(
                hours=None,  # Download ALL available videos
                skip_existing=True  # Skip videos already in database
            )
            
            logger.info(f"Download statistics:")
            logger.info(f"  - New downloads: {stats['new_downloads']}")
            logger.info(f"  - Already downloaded (skipped): {stats['already_downloaded']}")
            logger.info(f"  - Unavailable (404): {stats['unavailable']}")
            logger.info(f"  - Failed: {stats['failed']}")
            
            if stats['new_downloads'] == 0:
                logger.info("No new videos to process - all available videos already downloaded")
                logger.info(f"Database prevented {stats['already_downloaded']} duplicate downloads")
                return
            
            # Get the newly downloaded files from downloads folder
            downloaded_videos = list(download_path_obj.glob("*.mp4")) if download_path_obj.exists() else []
            logger.info(f"Found {len(downloaded_videos)} videos in downloads folder after download")
            videos_to_process = downloaded_videos
            
        except Exception as download_error:
            logger.error(f"Failed to download videos from Ring: {download_error}", exc_info=True)
            logger.error("Scheduled run will retry on next cycle")
            return
    
    logger.info(f"Processing {len(videos_to_process)} videos")
    
    # Update progress: set total count
    try:
        from webapp import app_state, task_tracker
        app_state["processing_progress"]["videos_total"] = len(videos_to_process)
        
        # Update task tracker with correct total count
        # Get current task_id from app_state if available
        current_tasks = task_tracker.get_active_tasks()
        if current_tasks:
            task_id = list(current_tasks.keys())[0]  # Get the first active task
            task_tracker.update_task(
                task_id,
                total=len(videos_to_process),
                current=0,
                message=f"Processing {len(videos_to_process)} videos..."
            )
    except:
        pass  # webapp might not be loaded
    
    # Process each video
    for idx, video_path in enumerate(videos_to_process, 1):
        # Check stop flag or if scheduler was disabled (skip scheduler check for manual triggers)
        try:
            from webapp import app_state
            if app_state.get("stop_requested", False):
                logger.info("Stop requested - ending processing gracefully")
                break
            # Only check scheduler state if this wasn't a manual trigger
            if not manual_trigger and not app_state["scheduler"]["enabled"]:
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
                from webapp import app_state, task_tracker
                app_state["processing_progress"]["current_video"] = video_path.name
                app_state["processing_progress"]["videos_processed"] = idx
                
                # Update task tracker progress
                current_tasks = task_tracker.get_active_tasks()
                if current_tasks:
                    task_id = list(current_tasks.keys())[0]
                    percentage = int((idx / len(videos_to_process)) * 100) if len(videos_to_process) > 0 else 0
                    task_tracker.update_task(
                        task_id,
                        current=idx,
                        message=f"Processing video {idx}/{len(videos_to_process)} ({percentage}%) - {video_path.name}"
                    )
            except Exception as e:
                logger.debug(f"Failed to update progress: {e}")
                pass
            
            logger.info(f"Processing video: {video_path.name}")
            
            # Update progress: object detection
            try:
                from webapp import app_state
                app_state["processing_progress"]["current_step"] = f"Analyzing {video_path.name} with YOLO..."
                app_state["processing_progress"]["phase"] = "detection"
            except:
                pass
            
            # V2: Run standard YOLO detection with bbox coordinates
            logger.debug("Running YOLO object detection (Stage 1)")
            detected_objects = object_detector.detect_objects_in_video(video_path, return_bboxes=True)
            
            # HYBRID WORKFLOW: Sort by YOLO category first
            yolo_sorted_path = None
            yolo_category = None
            yolo_confidence = 0.0
            
            if detected_objects:
                # Get highest confidence detection
                best_category = max(detected_objects, key=lambda k: detected_objects[k]['confidence'])
                best_confidence = detected_objects[best_category]['confidence']
                yolo_category = best_category  # Save for metadata
                yolo_confidence = best_confidence  # Save for metadata
                
                logger.info(f"HYBRID WORKFLOW - YOLO detected '{best_category}' with confidence {best_confidence:.2f}")
                
                # Sort to YOLO category folder in review
                try:
                    from webapp import app_state
                    app_state["processing_progress"]["current_step"] = f"Sorting {video_path.name} to YOLO category '{best_category}'..."
                    app_state["processing_progress"]["phase"] = "yolo_sorting"
                except:
                    pass
                
                yolo_sorted_path = video_sorter.sort_by_yolo_category(
                    video_path,
                    yolo_category=best_category,
                    confidence=best_confidence,
                    metadata={"all_detections": detected_objects}
                )
                logger.info(f"Video sorted to /data/review/{best_category}/")
                
                # Update video_path to the new location for subsequent processing
                video_path = yolo_sorted_path
                
                # Generate tracked/annotated video
                try:
                    from webapp import app_state
                    app_state["processing_progress"]["current_step"] = f"Creating tracked video for {video_path.name}..."
                    app_state["processing_progress"]["phase"] = "tracking"
                except:
                    pass
                
                logger.info(f"Generating tracked video with bounding boxes for detected objects: {list(detected_objects.keys())}")
                tracking_config = config.get('tracking', {})
                save_original = tracking_config.get('save_original_videos', False)
                
                # Create annotated video with object tracking
                tracked_detections = object_detector.track_and_annotate_video(
                    video_path,
                    output_path=None,  # Auto-generate path
                    save_original=save_original
                )
                logger.info(f"Tracked video created with detections: {tracked_detections}")
            else:
                # No objects detected - move to "unknown" category in review
                logger.info("No objects detected - sorting to 'unknown' category")
                try:
                    from webapp import app_state
                    app_state["processing_progress"]["current_step"] = f"Sorting {video_path.name} to 'unknown' category..."
                    app_state["processing_progress"]["phase"] = "yolo_sorting"
                except:
                    pass
                
                yolo_sorted_path = video_sorter.sort_by_yolo_category(
                    video_path,
                    yolo_category="unknown",
                    confidence=0.0,
                    metadata={"all_detections": {}}
                )
                logger.info("Video sorted to /data/review/unknown/")
                
                # Update video_path to the new location
                video_path = yolo_sorted_path
                yolo_category = "unknown"
                yolo_confidence = 0.0
            
            # Check if Face Recognition routing should be triggered
            # Conditions:
            # 1. YOLO detected "person"
            # 2. An Animal Profile with "person" in yolo_categories is enabled
            # 3. Face Recognition is enabled in config
            face_recognition_enabled = config.get('face_recognition', {}).get('enabled', False)
            should_run_face_recognition = False
            recognized_people = set()
            
            if face_recognition_enabled and detected_objects and 'person' in detected_objects:
                # Check if any enabled Animal Profile includes "person" category
                try:
                    all_profiles = profile_manager.list_profiles()
                    person_profile_exists = any(
                        p.enabled and 'person' in p.yolo_categories
                        for p in all_profiles
                    )
                    
                    if person_profile_exists:
                        should_run_face_recognition = True
                        logger.info("Face Recognition routing triggered (person detected + profile enabled + FR enabled)")
                except Exception as e:
                    logger.error(f"Error checking Animal Profiles for face recognition: {e}")
            
            if should_run_face_recognition:
                # Update progress: face recognition
                try:
                    from webapp import app_state
                    app_state["processing_progress"]["current_step"] = f"Recognizing faces in {video_path.name}..."
                    app_state["processing_progress"]["phase"] = "face_recognition"
                except:
                    pass
                
                logger.debug("Running face recognition")
                recognized_people = face_recognizer.recognize_faces_in_video(video_path)
            
            # CLIP Stage 2: Check for matching Animal Profiles
            clip_stage2_result = None
            final_destination = None
            should_move_to_sorted = False  # Track if CLIP approved moving to sorted
            
            if detected_objects:
                # Get YOLO categories that were detected
                detected_categories = list(detected_objects.keys())
                logger.debug(f"YOLO detected categories: {detected_categories}")
                
                # Load all enabled Animal Profiles
                try:
                    all_profiles = profile_manager.list_profiles()
                    enabled_profiles = [p for p in all_profiles if p.enabled]
                    
                    # Find profiles with matching YOLO categories
                    matching_profiles = []
                    for profile in enabled_profiles:
                        # Check if any detected category matches profile's YOLO categories
                        if any(cat in profile.yolo_categories for cat in detected_categories):
                            matching_profiles.append(profile)
                    
                    if matching_profiles:
                        logger.info(f"Found {len(matching_profiles)} matching profiles: {[p.name for p in matching_profiles]}")
                        
                        # Update progress: CLIP Stage 2
                        try:
                            from webapp import app_state
                            app_state["processing_progress"]["current_step"] = f"Running CLIP Stage 2 for {video_path.name}..."
                            app_state["processing_progress"]["phase"] = "clip_classification"
                        except:
                            pass
                        
                        # Lazy load CLIP classifier
                        if clip_classifier is None:
                            logger.info("Initializing CLIP classifier for Stage 2")
                            clip_classifier = CLIPVitClassifier()
                        
                        # Extract frames from video for CLIP analysis
                        temp_dir = None
                        try:
                            temp_dir = tempfile.mkdtemp(prefix="clip_stage2_")
                            logger.debug(f"Extracting frames to {temp_dir}")
                            
                            # Extract frames (1 fps, max 10 frames)
                            cap = cv2.VideoCapture(str(video_path))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frame_interval = int(fps) if fps > 0 else 30
                            max_frames = 10
                            
                            frame_paths = []
                            frame_count = 0
                            extracted_count = 0
                            
                            while extracted_count < max_frames:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                if frame_count % frame_interval == 0:
                                    frame_path = Path(temp_dir) / f"frame_{extracted_count:04d}.jpg"
                                    cv2.imwrite(str(frame_path), frame)
                                    frame_paths.append(str(frame_path))
                                    extracted_count += 1
                                
                                frame_count += 1
                            
                            cap.release()
                            logger.info(f"Extracted {len(frame_paths)} frames for CLIP analysis")
                            
                            # Run CLIP for each matching profile
                            profile_results = []
                            for profile in matching_profiles:
                                logger.debug(f"Running CLIP for profile: {profile.name} ({profile.text_description})")
                                
                                scores = clip_classifier.score_batch(frame_paths, profile.text_description)
                                avg_confidence = sum(scores) / len(scores) if scores else 0.0
                                
                                profile_results.append({
                                    "profile_id": profile.id,
                                    "profile_name": profile.name,
                                    "confidence": avg_confidence,
                                    "threshold": profile.confidence_threshold,
                                    "auto_approval_enabled": profile.auto_approval_enabled,
                                    "requires_manual_confirmation": profile.requires_manual_confirmation
                                })
                                
                                logger.info(f"CLIP result for {profile.name}: {avg_confidence:.3f} (threshold: {profile.confidence_threshold})")
                            
                            # Select profile with highest confidence
                            if profile_results:
                                best_result = max(profile_results, key=lambda x: x['confidence'])
                                
                                clip_stage2_result = {
                                    "matched_profiles": [p.id for p in matching_profiles],
                                    "clip_results": profile_results,
                                    "selected_profile": best_result['profile_id'],
                                    "final_confidence": best_result['confidence'],
                                    "auto_approved": False
                                }
                                
                                # HYBRID WORKFLOW: CLIP Refinement - decide if video moves to sorted
                                should_move_to_sorted = False
                                
                                if best_result['requires_manual_confirmation']:
                                    # Manual confirmation required -> stays in YOLO review folder
                                    logger.info(f"HYBRID WORKFLOW - CLIP '{best_result['profile_name']}': Staying in YOLO review (manual confirmation required)")
                                    final_destination = video_path  # Keep in current YOLO category folder
                                elif best_result['confidence'] >= best_result['threshold']:
                                    if best_result['auto_approval_enabled']:
                                        # High confidence + auto-approval -> move to sorted
                                        should_move_to_sorted = True
                                        clip_stage2_result['auto_approved'] = True
                                        logger.info(f"HYBRID WORKFLOW - CLIP '{best_result['profile_name']}': Moving to sorted (confidence: {best_result['confidence']:.2f})")
                                    else:
                                        # High confidence but no auto-approval -> stays in YOLO review
                                        logger.info(f"HYBRID WORKFLOW - CLIP '{best_result['profile_name']}': Staying in YOLO review (auto-approval disabled)")
                                        final_destination = video_path
                                else:
                                    # Below threshold -> stays in YOLO review folder
                                    logger.info(f"HYBRID WORKFLOW - CLIP '{best_result['profile_name']}': Below threshold ({best_result['confidence']:.2f} < {best_result['threshold']}), staying in YOLO review")
                                    final_destination = video_path
                                
                                # Move to sorted if approved
                                if should_move_to_sorted:
                                    try:
                                        final_destination = video_sorter.move_to_clip_sorted(
                                            video_path,
                                            clip_profile_id=best_result['profile_id'],
                                            confidence=best_result['confidence'],
                                            metadata={"clip_results": profile_results}
                                        )
                                        logger.info(f"Video moved to /data/sorted/{best_result['profile_id']}/")
                                        video_path = final_destination  # Update path
                                    except Exception as move_err:
                                        logger.error(f"Failed to move video to sorted: {move_err}", exc_info=True)
                                        final_destination = video_path  # Fall back to current location
                        
                        except Exception as clip_err:
                            logger.error(f"CLIP Stage 2 failed: {clip_err}", exc_info=True)
                            # Fall back to YOLO-only workflow
                        
                        finally:
                            # Clean up temp frames
                            if temp_dir and Path(temp_dir).exists():
                                import shutil
                                try:
                                    shutil.rmtree(temp_dir)
                                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                                except Exception as cleanup_err:
                                    logger.warning(f"Failed to cleanup temp directory: {cleanup_err}")
                    
                    else:
                        logger.debug("No matching Animal Profiles found for detected categories")
                
                except Exception as profile_err:
                    logger.error(f"Error checking Animal Profiles: {profile_err}", exc_info=True)
            
            # HYBRID WORKFLOW: Video is already in its final location
            # - Either in /data/review/{yolo_category}/ (YOLO-only or CLIP didn't match)
            # - Or in /data/sorted/{clip_profile}/ (CLIP matched and auto-approved)
            
            if final_destination is None or final_destination == video_path:
                # Video is already sorted to YOLO category, no further move needed
                dest_path = video_path
                logger.info(f"HYBRID WORKFLOW: Video remains at {video_path}")
            else:
                # CLIP moved it to sorted, already at final destination
                dest_path = final_destination
                logger.info(f"HYBRID WORKFLOW: Video at final destination {dest_path}")
            
            # Update download tracker status to 'processed'
            try:
                parts = video_path.stem.split('_')
                if len(parts) >= 3:
                    event_id = parts[-1]
                    ring_downloader.download_tracker.update_status(
                        event_id, 'processed', str(dest_path)
                    )
            except Exception as track_err:
                logger.debug(f"Could not update download tracker: {track_err}")
            
            # Save metadata alongside video
            metadata = {
                "video_name": video_path.name,
                "timestamp": datetime.now().isoformat(),
                "yolo_category": yolo_category if yolo_category else "unknown",
                "yolo_confidence": yolo_confidence,
                "detected_objects": detected_objects if detected_objects else {},
                "recognized_people": list(recognized_people) if recognized_people else [],
                "status": "clip_sorted" if should_move_to_sorted else "pending_review"
            }
            
            # Add CLIP Stage 2 results if available
            if clip_stage2_result:
                metadata["clip_results"] = clip_stage2_result
            
            metadata_path = dest_path.with_suffix(dest_path.suffix + ".json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Video moved to: {dest_path}")
            
        except Exception as e:
            logger.error(f"Failed to process video {video_path.name}: {e}", exc_info=True)
    
    # Post-processing verification: Check if downloads folder is empty
    logger.info("="*80)
    logger.info("POST-PROCESSING VERIFICATION")
    logger.info("="*80)
    
    remaining_videos = list(download_path_obj.glob("*.mp4")) if download_path_obj.exists() else []
    if remaining_videos:
        logger.warning(f"⚠ {len(remaining_videos)} videos remain in downloads folder after processing:")
        for video in remaining_videos:
            logger.warning(f"  - {video.name}")
        logger.warning("Check logs above for processing errors or skipped videos")
    else:
        logger.info("✓ All videos successfully processed and moved from downloads folder")
    
    # Log statistics
    stats = video_sorter.get_stats()
    logger.info(f"Sorting statistics: {stats}")
    logger.info("="*80)


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
    
    # Load configuration first (needed for logging setup)
    config = load_config()
    
    # Setup logging with config
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    
    # Log startup with Docker info
    logger.info("="*80)
    logger.info("Starting CritterCatcherAI")
    try:
        # Get Docker container ID from /etc/hostname
        hostname_file = Path('/etc/hostname')
        if hostname_file.exists():
            container_id = hostname_file.read_text().strip()[:12]
            logger.info(f"  Container ID: {container_id}")
        else:
            logger.info(f"  Container ID: {os.environ.get('HOSTNAME', 'unknown')}")
        
        # Get Git SHA if available
        import subprocess
        git_sha = subprocess.run(
            ['sh', '-c', 'cat /app/git_sha.txt 2>/dev/null || echo "unknown"'],
            capture_output=True,
            text=True,
            timeout=2
        ).stdout.strip()
        if git_sha and git_sha != 'unknown':
            logger.info(f"  Git Commit: {git_sha[:12]}")
    except Exception as e:
        logger.debug(f"Could not get Docker info: {e}")
    logger.info("="*80)
    
    # Initialize GPU monitoring
    gpu_config = config.get('logging', {}).get('gpu_monitoring', {})
    gpu_monitor = GPUMonitor(
        logger=logger,
        log_interval=gpu_config.get('interval_seconds', 5),
        log_on_idle=gpu_config.get('log_on_idle', False)
    )
    if gpu_config.get('enabled', True):
        gpu_monitor.start_monitoring()
    else:
        logger.info("GPU monitoring disabled in config")
    
    # Make gpu_monitor available to webapp
    import webapp
    webapp.gpu_monitor = gpu_monitor
    
    # Start web server in separate thread
    import threading
    from webapp import start_web_server
    
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    logger.info("Web interface started on http://0.0.0.0:8080")
    
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
                    app_state["processing_progress"] = {
                        "current_video": None,
                        "current_step": None,
                        "videos_processed": 0,
                        "videos_total": 0,
                        "start_time": None
                    }
            
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
