"""
CritterCatcherAI Web Interface
FastAPI-based web GUI for monitoring and configuration.
"""
import os
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import yaml
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import requests
import numpy as np

from animal_profile import AnimalProfile, AnimalProfileManager
from review_feedback import ReviewManager

logger = logging.getLogger(__name__)


def get_app_version():
    """Get application version from version.txt file or fallback"""
    version_file = Path('/app/version.txt')
    try:
        if version_file.exists():
            with open(version_file, 'r') as f:
                version = f.read().strip()
                if version:
                    return version
    except Exception as e:
        logger.debug(f"Could not read version file: {e}")
    
    # Fallback version
    return "v0.1.0-dev"


def get_build_date():
    """Get Docker image build date from build_date.txt file or fallback"""
    build_date_file = Path('/app/build_date.txt')
    try:
        if build_date_file.exists():
            with open(build_date_file, 'r') as f:
                build_date = f.read().strip()
                if build_date:
                    return build_date
    except Exception as e:
        logger.debug(f"Could not read build_date file: {e}")
    
    # Fallback
    return "Unknown"

# Initialize FastAPI app
app = FastAPI(title="CritterCatcherAI", version="1.0.0")

# Log version on startup
app_version = get_app_version()
app_build_date = get_build_date()
logger.info(f"CritterCatcherAI starting up - Version: {app_version}, Built: {app_build_date}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Global state
app_state = {
    "is_processing": False,
    "last_run": None,
    "stats": {},
    "log_buffer": [],
    "stop_requested": False,
    "processing_progress": {
        "current_video": None,
        "current_step": None,
        "videos_processed": 0,
        "videos_total": 0,
        "start_time": None
    },
    "scheduler": {
        "enabled": False,
        "interval_minutes": 60,
        "next_run": None
    }
}

CONFIG_PATH = Path("/config/config.yaml")
FACE_TRAINING_PATH = Path("/data/faces/training")
UNKNOWN_FACES_PATH = Path("/data/faces/unknown")
SORTED_PATH = Path("/data/sorted")
DOWNLOADS_PATH = Path("/data/downloads")

# Global animal profile manager
animal_profile_manager: Optional[AnimalProfileManager] = None

# Global review manager
review_manager: Optional[ReviewManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize Animal Profile Manager and Review Manager on startup."""
    global animal_profile_manager, review_manager
    
    # Ensure config directory exists (copy from defaults if needed)
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy config.yaml if it doesn't exist
        if not CONFIG_PATH.exists():
            default_config = Path("/app/config/config.yaml")
            if default_config.exists():
                import shutil
                shutil.copy(default_config, CONFIG_PATH)
                logger.info(f"Copied default config to {CONFIG_PATH}")
            else:
                logger.warning(f"Default config not found at {default_config}")
    except Exception as e:
        logger.error(f"Failed to initialize config: {e}")
    
    # Initialize animal profile manager
    try:
        animal_profile_manager = AnimalProfileManager(Path("/data"))
        logger.info("Animal profile manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize animal profile manager: {e}")
    
    # Initialize review manager
    try:
        if animal_profile_manager:
            review_manager = ReviewManager(animal_profile_manager, Path("/data"))
            logger.info("Review manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize review manager: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    index_file = static_path / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>CritterCatcherAI</h1><p>Web interface loading...</p>")
    
    # Inject version and build date into HTML
    html_content = index_file.read_text()
    html_content = html_content.replace('{{version}}', get_app_version())
    html_content = html_content.replace('{{build_date}}', get_build_date())
    return HTMLResponse(content=html_content, status_code=200)




@app.get("/api/status")
async def get_status():
    """Get current application status."""
    progress = app_state["processing_progress"]
    
    # Calculate time elapsed if processing
    time_elapsed = None
    time_remaining = None
    if app_state["is_processing"] and progress["start_time"]:
        elapsed_seconds = (datetime.now() - progress["start_time"]).total_seconds()
        time_elapsed = int(elapsed_seconds)
        
        # Estimate remaining time
        if progress["videos_processed"] > 0 and progress["videos_total"] > 0:
            avg_time_per_video = elapsed_seconds / progress["videos_processed"]
            remaining_videos = progress["videos_total"] - progress["videos_processed"]
            time_remaining = int(avg_time_per_video * remaining_videos)
    
    return {
        "is_processing": app_state["is_processing"],
        "last_run": app_state["last_run"],
        "uptime": "Running",
        "version": "1.0.0",
        "processing_progress": {
            "current_video": progress["current_video"],
            "current_step": progress["current_step"],
            "videos_processed": progress["videos_processed"],
            "videos_total": progress["videos_total"],
            "time_elapsed": time_elapsed,
            "time_remaining": time_remaining
        } if app_state["is_processing"] else None,
        "scheduler": app_state["scheduler"]
    }


@app.get("/api/stats")
async def get_stats():
    """Get video sorting statistics."""
    stats = {}
    
    if SORTED_PATH.exists():
        # Count videos in each category
        for category_dir in SORTED_PATH.iterdir():
            if category_dir.is_dir():
                if category_dir.name == "people":
                    # Count people subdirectories
                    for person_dir in category_dir.iterdir():
                        if person_dir.is_dir():
                            video_count = len(list(person_dir.glob("*.mp4")))
                            stats[f"people/{person_dir.name}"] = video_count
                else:
                    video_count = len(list(category_dir.glob("*.mp4")))
                    stats[category_dir.name] = video_count
    
    return stats


@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
            
            # Add environment variables
            env_vars = {
                "RING_USERNAME": os.environ.get("RING_USERNAME", ""),
                "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
                "RUN_ONCE": os.environ.get("RUN_ONCE", "false"),
                "TZ": os.environ.get("TZ", "UTC")
            }
            
            return {
                "config": config,
                "env_vars": env_vars
            }
        else:
            raise HTTPException(status_code=404, detail="Configuration file not found")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config")
async def update_config(config_data: dict):
    """Update configuration file."""
    try:
        # Validate config structure
        if "config" not in config_data:
            raise HTTPException(status_code=400, detail="Invalid config format")
        
        # Load existing config to preserve sections not managed by UI
        existing_config = {}
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, 'r') as f:
                existing_config = yaml.safe_load(f) or {}
        
        new_config = config_data["config"]
        
        # Preserve specialized_detection subsections (species, clip_extraction, training)
        # Only update the 'enabled' flag from UI
        if "specialized_detection" in new_config:
            if "specialized_detection" not in existing_config:
                existing_config["specialized_detection"] = {}
            
            # Preserve existing subsections
            for key in ['species', 'clip_extraction', 'training']:
                if key in existing_config.get("specialized_detection", {}):
                    new_config["specialized_detection"][key] = existing_config["specialized_detection"][key]
        
        # Write merged config to file
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
        
        # Update scheduler state if changed
        scheduler_config = new_config.get('scheduler', {})
        if scheduler_config:
            app_state["scheduler"]["enabled"] = scheduler_config.get('auto_run', False)
            app_state["scheduler"]["interval_minutes"] = scheduler_config.get('interval_minutes', 60)
            # Reset next_run when config changes - will be set by main loop
            if not app_state["scheduler"]["enabled"]:
                app_state["scheduler"]["next_run"] = None
                # If processing is active, stop it
                if app_state["is_processing"]:
                    logger.info("Scheduler disabled during processing - stopping current run")
                    app_state["stop_requested"] = True
        
        logger.info("Configuration updated successfully")
        return {"status": "success", "message": "Configuration updated"}
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos")
async def get_videos(category: Optional[str] = None, limit: int = 50):
    """Get list of sorted videos."""
    videos = []
    
    try:
        if category:
            # Get videos from specific category
            category_path = SORTED_PATH / category
            if category_path.exists():
                for video_file in sorted(category_path.glob("*.mp4"), 
                                        key=lambda x: x.stat().st_mtime, 
                                        reverse=True)[:limit]:
                    videos.append({
                        "filename": video_file.name,
                        "category": category,
                        "path": str(video_file.relative_to(SORTED_PATH)),
                        "size": video_file.stat().st_size,
                        "modified": datetime.fromtimestamp(video_file.stat().st_mtime).isoformat()
                    })
        else:
            # Get videos from all categories
            for category_dir in SORTED_PATH.iterdir():
                if category_dir.is_dir():
                    if category_dir.name == "people":
                        for person_dir in category_dir.iterdir():
                            if person_dir.is_dir():
                                for video_file in person_dir.glob("*.mp4"):
                                    videos.append({
                                        "filename": video_file.name,
                                        "category": f"people/{person_dir.name}",
                                        "path": str(video_file.relative_to(SORTED_PATH)),
                                        "size": video_file.stat().st_size,
                                        "modified": datetime.fromtimestamp(video_file.stat().st_mtime).isoformat()
                                    })
                    else:
                        for video_file in category_dir.glob("*.mp4"):
                            videos.append({
                                "filename": video_file.name,
                                "category": category_dir.name,
                                "path": str(video_file.relative_to(SORTED_PATH)),
                                "size": video_file.stat().st_size,
                                "modified": datetime.fromtimestamp(video_file.stat().st_mtime).isoformat()
                            })
            
            # Sort by modification time and limit
            videos.sort(key=lambda x: x["modified"], reverse=True)
            videos = videos[:limit]
    
    except Exception as e:
        logger.error(f"Failed to get videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return videos


@app.get("/api/faces")
async def get_trained_faces():
    """Get list of trained faces."""
    faces = []
    
    try:
        if FACE_TRAINING_PATH.exists():
            for person_dir in FACE_TRAINING_PATH.iterdir():
                if person_dir.is_dir():
                    image_count = len(list(person_dir.glob("*.jpg"))) + len(list(person_dir.glob("*.png")))
                    faces.append({
                        "name": person_dir.name,
                        "image_count": image_count
                    })
    except Exception as e:
        logger.error(f"Failed to get trained faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return faces


@app.post("/api/faces/upload")
async def upload_face_training_image(
    person_name: str,
    file: UploadFile = File(...),
):
    """Upload a face training image."""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Only JPG and PNG images are supported")
        
        # Create person directory
        person_dir = FACE_TRAINING_PATH / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = person_dir / file.filename
        content = await file.read()
        file_path.write_bytes(content)
        
        logger.info(f"Uploaded training image for {person_name}: {file.filename}")
        return {"status": "success", "message": f"Image uploaded for {person_name}"}
    
    except Exception as e:
        logger.error(f"Failed to upload face image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/faces/train")
async def train_face(person_name: str, background_tasks: BackgroundTasks):
    """Train face recognition for a person."""
    try:
        person_dir = FACE_TRAINING_PATH / person_name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail=f"No training images found for {person_name}")
        
        # Import here to avoid circular dependencies
        from face_recognizer import FaceRecognizer
        
        def train_task():
            fr = FaceRecognizer()
            images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            fr.add_person(person_name, images)
            logger.info(f"Completed face training for {person_name}")
        
        background_tasks.add_task(train_task)
        
        return {"status": "success", "message": f"Training started for {person_name}"}
    
    except Exception as e:
        logger.error(f"Failed to train face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def trigger_initial_processing():
    """Trigger initial video processing after authentication."""
    try:
        logger.info("Triggering initial video processing after authentication")
        from main import process_videos, load_config
        config = load_config()
        process_videos(config)
        logger.info("Initial processing complete")
    except Exception as e:
        logger.error(f"Initial processing failed: {e}", exc_info=True)


@app.post("/api/ring/auth")
async def ring_authenticate(credentials: dict, background_tasks: BackgroundTasks):
    """Authenticate with Ring (handles 2FA)."""
    try:
        username = credentials.get('username')
        password = credentials.get('password')
        code_2fa = credentials.get('code_2fa')  # Optional 2FA code
        
        # Use environment variables if not provided
        if not username:
            username = os.environ.get('RING_USERNAME')
        if not password:
            password = os.environ.get('RING_PASSWORD')
        
        if not username or not password:
            raise HTTPException(status_code=400, detail="Username and password required (via form or environment)")
        
        from ring_downloader import RingDownloader
        from ring_doorbell.exceptions import Requires2FAError
        
        rd = RingDownloader(
            download_path="/data/downloads",
            token_file="/data/tokens/ring_token.json"
        )
        
        # Try to authenticate
        if code_2fa:
            logger.info(f"Attempting 2FA authentication for {username}")
            # Use 2FA authentication method
            if rd.authenticate_with_2fa(username, password, code_2fa):
                logger.info("2FA authentication successful")
                # Trigger processing in background
                background_tasks.add_task(trigger_initial_processing)
                return {
                    "status": "success",
                    "message": "Ring authentication successful. Token saved. Starting video processing..."
                }
            else:
                logger.warning("2FA authentication failed")
                raise HTTPException(status_code=401, detail="2FA authentication failed. Check your code.")
        else:
            # Try regular authentication first - expect 2FA error
            logger.info(f"Attempting authentication for {username}")
            try:
                if rd.authenticate(username, password):
                    logger.info("Authentication successful without 2FA")
                    # Trigger processing in background
                    background_tasks.add_task(trigger_initial_processing)
                    return {
                        "status": "success",
                        "message": "Ring authentication successful. Token saved. Starting video processing..."
                    }
                else:
                    logger.warning("Authentication failed")
                    raise HTTPException(status_code=401, detail="Authentication failed. Check credentials.")
            except Requires2FAError as e2fa:
                # 2FA is required - return special status
                logger.info(f"2FA required for {username}")
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "needs_2fa",
                        "message": "2FA code required. Check your phone/email for the verification code."
                    }
                )
            except Exception as auth_ex:
                # Check if it's a 2FA requirement in disguise
                error_str = str(auth_ex).lower()
                if "2fa" in error_str or "verification" in error_str or "requires2faerror" in error_str:
                    logger.info(f"2FA required (detected from error: {error_str})")
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "needs_2fa",
                            "message": "2FA code required. Check your phone/email for the verification code."
                        }
                    )
                # Otherwise re-raise
                raise
            
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Ring authentication error: {error_msg}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/ring/status")
async def ring_auth_status():
    """Check if Ring token exists and return credentials from env."""
    token_file = Path("/data/tokens/ring_token.json")
    return {
        "authenticated": token_file.exists(),
        "token_path": str(token_file),
        "username": os.environ.get("RING_USERNAME", ""),
        "has_password": bool(os.environ.get("RING_PASSWORD"))
    }


@app.post("/api/process")
async def trigger_processing(background_tasks: BackgroundTasks):
    """Manually trigger video processing."""
    logger.info("=" * 80)
    logger.info("MANUAL PROCESSING TRIGGERED VIA WEB UI")
    logger.info("=" * 80)
    
    if app_state["is_processing"]:
        logger.warning("Processing already running, rejecting request")
        return {"status": "already_running", "message": "Processing is already in progress"}
    
    # Reset stop flag and progress
    app_state["stop_requested"] = False
    app_state["processing_progress"] = {
        "current_video": None,
        "current_step": "Starting...",
        "videos_processed": 0,
        "videos_total": 0,
        "start_time": datetime.now()
    }
    
    def process_task():
        app_state["is_processing"] = True
        try:
            logger.info("Starting manual processing task...")
            # Import here to avoid issues
            from main import process_videos, load_config
            config = load_config()
            logger.info("Config loaded, calling process_videos()")
            process_videos(config)
            app_state["last_run"] = datetime.now().isoformat()
            logger.info("Manual processing completed successfully")
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
        finally:
            app_state["is_processing"] = False
            app_state["processing_progress"]["current_step"] = "Complete" if not app_state["stop_requested"] else "Stopped"
    
    background_tasks.add_task(process_task)
    logger.info("Processing task added to background queue")
    return {"status": "started", "message": "Processing started"}


@app.post("/api/stop")
async def stop_processing():
    """Request to stop current processing gracefully."""
    if not app_state["is_processing"]:
        return {"status": "not_processing", "message": "No processing task is currently running"}
    
    app_state["stop_requested"] = True
    logger.info("Stop requested - will finish current video and stop")
    
    return {
        "status": "stopping",
        "message": "Stop requested. Will finish current video and stop gracefully."
    }


@app.post("/api/download-all")
async def download_all_videos(request: dict, background_tasks: BackgroundTasks):
    """Download all videos from Ring with optional time filter."""
    logger.info("=" * 80)
    logger.info("DOWNLOAD ALL TRIGGERED VIA WEB UI")
    logger.info("=" * 80)
    
    if app_state["is_processing"]:
        logger.warning("Processing already running, rejecting download request")
        return {"status": "already_running", "message": "A task is already in progress"}
    
    # Parse time range
    time_range = request.get('time_range', 'all')
    hours_map = {
        '1hour': 1,
        '6hours': 6,
        '24hours': 24,
        '7days': 24 * 7,
        '30days': 24 * 30,
        'all': None
    }
    
    hours = hours_map.get(time_range)
    if time_range not in hours_map:
        raise HTTPException(status_code=400, detail=f"Invalid time_range: {time_range}")
    
    logger.info(f"Download All: time_range={time_range}, hours={hours}")
    
    def download_task():
        app_state["is_processing"] = True
        try:
            logger.info(f"Starting download all task (time_range: {time_range})...")
            from ring_downloader import RingDownloader
            
            rd = RingDownloader(
                download_path="/data/downloads",
                token_file="/data/tokens/ring_token.json"
            )
            
            # Authenticate
            if not rd.authenticate():
                logger.error("Failed to authenticate with Ring")
                return
            
            # Download all videos
            downloaded = rd.download_all_videos(hours=hours, skip_existing=True)
            logger.info(f"Download All Complete: {len(downloaded)} new videos downloaded")
            
        except Exception as e:
            logger.error(f"Download all failed: {e}", exc_info=True)
        finally:
            app_state["is_processing"] = False
    
    background_tasks.add_task(download_task)
    logger.info("Download all task added to background queue")
    
    time_desc = f"last {time_range}" if time_range != 'all' else "all available videos"
    return {"status": "started", "message": f"Downloading {time_desc}..."}


@app.post("/api/cleanup-downloads")
async def cleanup_downloads(background_tasks: BackgroundTasks):
    """Process all videos in downloads folder regardless of age."""
    logger.info("=" * 80)
    logger.info("CLEANUP DOWNLOADS TRIGGERED VIA WEB UI")
    logger.info("=" * 80)
    
    if app_state["is_processing"]:
        logger.warning("Processing already running, rejecting cleanup request")
        return {"status": "already_running", "message": "A task is already in progress"}
    
    def cleanup_task():
        app_state["is_processing"] = True
        app_state["stop_requested"] = False
        app_state["processing_progress"] = {
            "current_video": None,
            "current_step": "Starting cleanup...",
            "videos_processed": 0,
            "videos_total": 0,
            "start_time": datetime.now()
        }
        
        try:
            logger.info("Starting cleanup task - processing ALL videos in downloads folder...")
            from main import load_config
            from object_detector import ObjectDetector, YOLO_COCO_CLASSES
            from face_recognizer import FaceRecognizer
            from video_sorter import VideoSorter
            from taxonomy_tree import TaxonomyTree
            
            config = load_config()
            detection_config = config.get('detection', {})
            
            # Load taxonomy tree for specialized detection
            taxonomy_file = Path("/app/config/taxonomy.json")
            taxonomy_tree = None
            try:
                taxonomy_tree = TaxonomyTree.load_from_file(taxonomy_file, YOLO_COCO_CLASSES)
                logger.info(f"Taxonomy tree loaded with {len(taxonomy_tree.roots)} root classes")
            except Exception as e:
                logger.warning(f"Failed to load taxonomy tree: {e}")
                taxonomy_tree = TaxonomyTree(YOLO_COCO_CLASSES)
            
            # Get all video files in downloads
            downloads_path = Path("/data/downloads")
            video_files = list(downloads_path.glob("*.mp4"))
            
            logger.info(f"Found {len(video_files)} videos in downloads folder")
            
            if not video_files:
                logger.info("No videos to process")
                return
            
            # Update progress total
            app_state["processing_progress"]["videos_total"] = len(video_files)
            
            # Initialize components
            video_sorter = VideoSorter("/data/sorted")
            
            object_labels = detection_config.get('object_labels', [])
            discovery_mode = detection_config.get('discovery_mode', False)
            discovery_threshold = detection_config.get('discovery_threshold', 0.30)
            ignored_labels = detection_config.get('ignored_labels', [])
            yolo_model = detection_config.get('yolo_model', 'yolov8n')
            
            if not yolo_model.endswith('.pt'):
                yolo_model = f"{yolo_model}.pt"
            
            object_detector = ObjectDetector(
                labels=object_labels,
                confidence_threshold=detection_config.get('confidence_threshold', 0.25),
                num_frames=detection_config.get('object_frames', 5),
                discovery_mode=discovery_mode,
                discovery_threshold=discovery_threshold,
                ignored_labels=ignored_labels,
                model_name=yolo_model
            )
            
            face_recognizer = FaceRecognizer(
                encodings_path=config.get('paths', {}).get('face_encodings', '/data/faces/encodings.pkl'),
                tolerance=detection_config.get('face_tolerance', 0.6),
                num_frames=detection_config.get('face_frames', 10),
                model=detection_config.get('face_model', 'hog')
            )
            
            # Process each video
            for idx, video_path in enumerate(video_files, 1):
                # Check stop flag
                if app_state.get("stop_requested", False):
                    logger.info("Stop requested - ending cleanup gracefully")
                    break
                
                try:
                    # Check if video still exists
                    if not video_path.exists():
                        logger.debug(f"Skipping {video_path.name}: already processed")
                        continue
                    
                    # Update progress
                    app_state["processing_progress"]["current_video"] = video_path.name
                    app_state["processing_progress"]["videos_processed"] = idx
                    app_state["processing_progress"]["current_step"] = "Running YOLO detection..."
                    
                    logger.info(f"Processing video: {video_path.name}")
                    
                    # Run object detection - use specialized detection if enabled
                    specialized_enabled = config.get('specialized_detection', {}).get('enabled', False)
                    
                    if specialized_enabled and taxonomy_tree:
                        logger.debug("Using specialized detection (Stage 1 + Stage 2)")
                        detected_objects, species_results = object_detector.detect_objects_with_specialization(
                            video_path, config, taxonomy_tree
                        )
                        
                        # Keep species results separate for specialized sorting
                        if species_results:
                            logger.info(f"Specialized detections: {list(species_results.keys())}")
                    else:
                        logger.debug("Using standard YOLO detection only (Stage 1)")
                        detected_objects = object_detector.detect_objects_in_video(video_path)
                        species_results = {}
                    
                    # Run face recognition ONLY if enabled and conditions are met
                    priority = detection_config.get('priority', 'objects')
                    face_recognition_enabled = config.get('face_recognition', {}).get('enabled', False)
                    recognized_people = set()
                    
                    if face_recognition_enabled and (priority == "people" or not detected_objects):
                        # Update progress
                        app_state["processing_progress"]["current_step"] = "Recognizing faces..."
                        recognized_people = face_recognizer.recognize_faces_in_video(video_path)
                    
                    # Update progress
                    app_state["processing_progress"]["current_step"] = "Sorting video..."
                    
                    # Sort video using specialized detection-aware sorting
                    if specialized_enabled and species_results:
                        # Use new specialized sorting with species-specific folders
                        detected_objects_path = Path("/data/objects/detected")
                        result = video_sorter.sort_with_specialization(
                            video_path,
                            yolo_detections=detected_objects,
                            species_results={name: conf for name, (conf, _) in species_results.items()},
                            config=config,
                            detected_objects_path=detected_objects_path,
                            recognized_people=recognized_people
                        )
                        sorted_path = result.get("video_path", video_path)
                    else:
                        # Use standard sorting
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
            logger.info(f"Cleanup complete. Sorting statistics: {stats}")
            app_state["last_run"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Cleanup task failed: {e}", exc_info=True)
        finally:
            app_state["is_processing"] = False
            app_state["processing_progress"]["current_step"] = "Complete" if not app_state["stop_requested"] else "Stopped"
    
    background_tasks.add_task(cleanup_task)
    logger.info("Cleanup task added to background queue")
    return {"status": "started", "message": f"Processing all videos in downloads folder..."}


@app.get("/api/logs/stream")
async def stream_logs():
    """Stream logs - Note: Real-time streaming not available from inside container."""
    async def event_generator():
        yield {"data": "=" * 80}
        yield {"data": "Log Streaming from Inside Container Not Supported"}
        yield {"data": "=" * 80}
        yield {"data": ""}
        yield {"data": "To view logs, use one of these methods:"}
        yield {"data": "  1. Unraid Dashboard: Click container icon → View Logs"}
        yield {"data": "  2. Terminal: docker logs -f crittercatcher-ai"}
        yield {"data": "  3. Unraid Terminal: docker logs --tail 100 crittercatcher-ai"}
        yield {"data": ""}
        yield {"data": "Recent activity can be seen in the Dashboard stats above."}
    
    return EventSourceResponse(event_generator())


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker."""
    return {"status": "healthy"}


@app.get("/api/version")
async def get_version():
    """Get application version information."""
    return {
        "version": get_app_version(),
        "build_date": get_build_date()
    }


@app.get("/api/unknown_faces")
async def get_unknown_faces():
    """Get list of unknown faces with grouping by similarity."""
    try:
        if not UNKNOWN_FACES_PATH.exists():
            return {"groups": []}
        
        # Load all unknown faces with metadata
        faces_data = []
        for img_file in UNKNOWN_FACES_PATH.glob("*.jpg"):
            metadata_file = UNKNOWN_FACES_PATH / f"{img_file.name}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    faces_data.append(metadata)
        
        # Group similar faces by comparing encodings
        groups = []
        used_indices = set()
        
        for i, face1 in enumerate(faces_data):
            if i in used_indices:
                continue
            
            group = [face1]
            used_indices.add(i)
            encoding1 = np.array(face1.get('encoding', []))
            
            # Find similar faces
            for j, face2 in enumerate(faces_data):
                if j <= i or j in used_indices:
                    continue
                
                encoding2 = np.array(face2.get('encoding', []))
                
                # Calculate face distance
                if len(encoding1) > 0 and len(encoding2) > 0:
                    distance = np.linalg.norm(encoding1 - encoding2)
                    if distance < 0.6:  # Similar faces
                        group.append(face2)
                        used_indices.add(j)
            
            groups.append({
                "id": f"group_{i}",
                "count": len(group),
                "faces": group
            })
        
        return {"groups": groups}
        
    except Exception as e:
        logger.error(f"Failed to get unknown faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/unknown_faces/image/{filename}")
async def get_unknown_face_image(filename: str):
    """Serve an unknown face image."""
    try:
        image_path = UNKNOWN_FACES_PATH / filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/unknown_faces/label")
async def label_unknown_faces(request: dict, background_tasks: BackgroundTasks):
    """Label unknown faces with a person name and add to training set."""
    try:
        filenames = request.get('filenames', [])
        person_name = request.get('person_name', '').strip()
        action = request.get('action', 'label')  # 'label' or 'ignore'
        
        if not person_name and action == 'label':
            raise HTTPException(status_code=400, detail="Person name required")
        
        if not filenames:
            raise HTTPException(status_code=400, detail="No files specified")
        
        # Create person directory in training folder
        if action == 'label':
            person_dir = FACE_TRAINING_PATH / person_name
            person_dir.mkdir(parents=True, exist_ok=True)
        
        moved_count = 0
        for filename in filenames:
            image_path = UNKNOWN_FACES_PATH / filename
            metadata_path = UNKNOWN_FACES_PATH / f"{filename}.json"
            
            if image_path.exists():
                if action == 'label':
                    # Move to training folder
                    dest_path = person_dir / filename
                    image_path.rename(dest_path)
                    moved_count += 1
                elif action == 'ignore':
                    # Delete the file
                    image_path.unlink()
                    moved_count += 1
                
                # Remove metadata
                if metadata_path.exists():
                    metadata_path.unlink()
        
        # Retrain face recognition if labeled
        if action == 'label' and moved_count > 0:
            def retrain_task():
                try:
                    from face_recognizer import FaceRecognizer
                    fr = FaceRecognizer()
                    images = list((FACE_TRAINING_PATH / person_name).glob("*.jpg"))
                    fr.add_person(person_name, images)
                    logger.info(f"Retrained face recognition with {len(images)} images for {person_name}")
                except Exception as e:
                    logger.error(f"Failed to retrain: {e}")
            
            background_tasks.add_task(retrain_task)
        
        message = f"{moved_count} face(s) {'labeled as ' + person_name if action == 'label' else 'ignored'}"
        return {"status": "success", "message": message}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to label faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))








@app.post("/api/reprocess")
async def return_videos_to_downloads(background_tasks: BackgroundTasks):
    """Move all sorted videos back to downloads folder (does not trigger processing)."""
    try:
        if app_state["is_processing"]:
            return {"status": "already_running", "message": "A task is already in progress"}
        
        if not SORTED_PATH.exists():
            return {"status": "no_videos", "message": "No sorted videos found"}
        
        # Count videos to move
        video_count = 0
        for video_file in SORTED_PATH.rglob("*.mp4"):
            video_count += 1
        
        if video_count == 0:
            return {"status": "no_videos", "message": "No videos found in sorted directories"}
        
        logger.info(f"Returning {video_count} videos back to downloads")
        
        def return_task():
            app_state["is_processing"] = True
            try:
                import shutil
                
                # Move all videos back
                DOWNLOADS_PATH.mkdir(parents=True, exist_ok=True)
                moved_count = 0
                
                for video_file in list(SORTED_PATH.rglob("*.mp4")):
                    dest = DOWNLOADS_PATH / video_file.name
                    # Handle name conflicts
                    counter = 1
                    while dest.exists():
                        dest = DOWNLOADS_PATH / f"{video_file.stem}_{counter}{video_file.suffix}"
                        counter += 1
                    
                    shutil.move(str(video_file), str(dest))
                    moved_count += 1
                    logger.debug(f"Moved: {video_file.name} -> {dest.name}")
                
                logger.info(f"Moved {moved_count} videos back to downloads")
                
                # Delete sorted directories
                if SORTED_PATH.exists():
                    for item in SORTED_PATH.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                    logger.info("Deleted sorted directories")
                
                logger.info("Videos returned to downloads. Use 'Cleanup Downloads' or 'Process Now' to reprocess them.")
                
            except Exception as e:
                logger.error(f"Failed to return videos to downloads: {e}", exc_info=True)
            finally:
                app_state["is_processing"] = False
        
        background_tasks.add_task(return_task)
        
        return {
            "status": "started",
            "message": f"Moving {video_count} videos back to downloads..."
        }
    
    except Exception as e:
        logger.error(f"Failed to return videos to downloads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/check_update")
async def check_update():
    """Check for updates from GitHub releases."""
    try:
        # GitHub repo details
        repo_owner = "Ahazii"
        repo_name = "CritterCatcherAI"
        
        # Get current version
        current_version = get_app_version()
        
        # Check GitHub API for latest release
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            release_data = response.json()
            latest_version = release_data.get('tag_name', '')
            release_url = release_data.get('html_url', '')
            release_notes = release_data.get('body', '')
            published_at = release_data.get('published_at', '')
            
            # Compare versions (strip 'v' prefix if present)
            current_ver = current_version.lstrip('v')
            latest_ver = latest_version.lstrip('v')
            
            # Improved version comparison:
            # - Treat 'dev' versions as pre-release (always older than proper releases)
            # - Compare semantic versions properly
            update_available = False
            if latest_ver and current_ver:
                # If current is dev version, any proper release is newer
                if 'dev' in current_ver.lower():
                    # Extract base version from dev string (e.g., "0.1.0-dev-xxx" -> "0.1.0")
                    current_base = current_ver.split('-')[0]
                    # Compare base version with latest
                    try:
                        current_parts = [int(x) for x in current_base.split('.')]
                        latest_parts = [int(x) for x in latest_ver.split('.')]
                        # Pad shorter version with zeros
                        max_len = max(len(current_parts), len(latest_parts))
                        current_parts += [0] * (max_len - len(current_parts))
                        latest_parts += [0] * (max_len - len(latest_parts))
                        update_available = latest_parts > current_parts
                    except (ValueError, AttributeError):
                        # Fallback to string comparison
                        update_available = latest_ver > current_ver
                else:
                    # Normal semantic version comparison
                    try:
                        current_parts = [int(x) for x in current_ver.split('.')]
                        latest_parts = [int(x) for x in latest_ver.split('.')]
                        max_len = max(len(current_parts), len(latest_parts))
                        current_parts += [0] * (max_len - len(current_parts))
                        latest_parts += [0] * (max_len - len(latest_parts))
                        update_available = latest_parts > current_parts
                    except (ValueError, AttributeError):
                        update_available = latest_ver > current_ver
            
            logger.debug(f"Update check: current={current_version}, latest={latest_version}, update_available={update_available}")
            
            return {
                "success": True,
                "update_available": update_available,
                "current_version": current_version,
                "latest_version": latest_version,
                "release_url": release_url,
                "release_notes": release_notes[:200] + '...' if len(release_notes) > 200 else release_notes,
                "published_at": published_at
            }
        else:
            logger.debug(f"GitHub API returned status {response.status_code}")
            return {
                "success": False,
                "message": f"GitHub API returned status {response.status_code}"
            }
            
    except Exception as e:
        logger.debug(f"Error checking for updates: {e}")
        return {
            "success": False,
            "message": f"Error checking for updates: {str(e)}"
        }


@app.get("/api/analytics/stats")
async def get_analytics_stats():
    """Get comprehensive analytics statistics."""
    try:
        stats = {
            "total_videos": 0,
            "total_detections": 0,
            "stage2_classifications": 0,
            "categories": {},
            "daily_stats": [],
            "top_detections": [],
            "face_recognition_stats": {},
            "specialized_stats": {
                "total": 0,
                "by_species": {}
            }
        }
        
        # Get YOLO classes for comparison
        from object_detector import YOLO_COCO_CLASSES
        yolo_classes_lower = [c.lower() for c in YOLO_COCO_CLASSES]
        
        # Count videos by category
        if SORTED_PATH.exists():
            for category_dir in SORTED_PATH.iterdir():
                if category_dir.is_dir():
                    if category_dir.name == "people":
                        for person_dir in category_dir.iterdir():
                            if person_dir.is_dir():
                                video_count = len(list(person_dir.glob("*.mp4")))
                                stats["categories"][f"people/{person_dir.name}"] = video_count
                                stats["total_videos"] += video_count
                                # Count as Stage 2 if not a YOLO class
                                if person_dir.name.lower() not in yolo_classes_lower:
                                    stats["stage2_classifications"] += video_count
                                    stats["specialized_stats"]["total"] += video_count
                                    stats["specialized_stats"]["by_species"][person_dir.name] = video_count
                    else:
                        video_count = len(list(category_dir.glob("*.mp4")))
                        stats["categories"][category_dir.name] = video_count
                        stats["total_videos"] += video_count
                        # Count as Stage 2 if not a YOLO class
                        if category_dir.name.lower() not in yolo_classes_lower:
                            stats["stage2_classifications"] += video_count
                            stats["specialized_stats"]["total"] += video_count
                            stats["specialized_stats"]["by_species"][category_dir.name] = video_count
        
        # V2: detections are now tracked per animal profile in review folders
        
        # Top detections
        sorted_categories = sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True)
        stats["top_detections"] = [{"label": k, "count": v} for k, v in sorted_categories[:10]]
        
        # Face recognition stats
        if FACE_TRAINING_PATH.exists():
            for person_dir in FACE_TRAINING_PATH.iterdir():
                if person_dir.is_dir():
                    image_count = len(list(person_dir.glob("*.jpg"))) + len(list(person_dir.glob("*.png")))
                    stats["face_recognition_stats"][person_dir.name] = {
                        "training_images": image_count,
                        "videos": stats["categories"].get(f"people/{person_dir.name}", 0)
                    }
        
        return stats
    except Exception as e:
        logger.error(f"Failed to get analytics stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/timeline")
async def get_analytics_timeline(days: int = 30):
    """Get timeline data for analytics charts."""
    try:
        from datetime import timedelta
        
        timeline = []
        now = datetime.now()
        
        # Generate daily stats for the last N days
        for i in range(days, -1, -1):
            date = now - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            
            day_stats = {
                "date": date_str,
                "videos": 0,
                "detections": 0,
                "categories": {}
            }
            
            # Count videos for this day
            if SORTED_PATH.exists():
                for video_file in SORTED_PATH.rglob("*.mp4"):
                    file_date = datetime.fromtimestamp(video_file.stat().st_mtime)
                    if file_date.strftime("%Y-%m-%d") == date_str:
                        day_stats["videos"] += 1
            
            timeline.append(day_stats)
        
        return {"timeline": timeline}
    except Exception as e:
        logger.error(f"Failed to get timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/storage")
async def get_storage_data():
    """Get storage information for all sorted video categories."""
    try:
        storage_data = []
        total_size = 0
        total_videos = 0
        
        if SORTED_PATH.exists():
            for category_dir in SORTED_PATH.iterdir():
                if category_dir.is_dir():
                    # Handle people subdirectories
                    if category_dir.name == "people":
                        for person_dir in category_dir.iterdir():
                            if person_dir.is_dir():
                                video_files = list(person_dir.glob("*.mp4"))
                                size = sum(f.stat().st_size for f in video_files)
                                total_size += size
                                total_videos += len(video_files)
                                
                                storage_data.append({
                                    "category": f"people/{person_dir.name}",
                                    "path": str(person_dir.relative_to(SORTED_PATH)),
                                    "video_count": len(video_files),
                                    "size_bytes": size,
                                    "size_mb": round(size / (1024 * 1024), 2)
                                })
                    else:
                        video_files = list(category_dir.glob("*.mp4"))
                        size = sum(f.stat().st_size for f in video_files)
                        total_size += size
                        total_videos += len(video_files)
                        
                        storage_data.append({
                            "category": category_dir.name,
                            "path": str(category_dir.relative_to(SORTED_PATH)),
                            "video_count": len(video_files),
                            "size_bytes": size,
                            "size_mb": round(size / (1024 * 1024), 2)
                        })
        
        # Sort by size descending
        storage_data.sort(key=lambda x: x["size_bytes"], reverse=True)
        
        return {
            "categories": storage_data,
            "total_videos": total_videos,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_size_gb": round(total_size / (1024 * 1024 * 1024), 2)
        }
    except Exception as e:
        logger.error(f"Failed to get storage data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/storage/{category_path:path}")
async def delete_category(category_path: str):
    """Delete all videos and metadata for a specific category."""
    try:
        import shutil
        
        # Sanitize path to prevent directory traversal
        category_path = category_path.strip("/")
        full_path = SORTED_PATH / category_path
        
        # Verify path is within SORTED_PATH
        if not str(full_path.resolve()).startswith(str(SORTED_PATH.resolve())):
            raise HTTPException(status_code=400, detail="Invalid category path")
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="Category not found")
        
        # Count videos before deletion
        video_count = len(list(full_path.glob("*.mp4")))
        
        # Delete the directory and all contents
        shutil.rmtree(full_path)
        logger.info(f"Deleted category '{category_path}' with {video_count} videos")
        
        return {
            "status": "success",
            "message": f"Deleted {video_count} video(s) from {category_path}",
            "deleted_videos": video_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete category: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics."""
    try:
        import psutil
        from pathlib import Path
        
        metrics = {
            "processing": {
                "is_active": app_state["is_processing"],
                "last_run": app_state["last_run"],
                "videos_processed": app_state["processing_progress"]["videos_processed"],
                "videos_total": app_state["processing_progress"]["videos_total"]
            },
            "storage": {
                "downloads": 0,
                "sorted": 0,
                "faces": 0,
                "objects": 0
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/data').percent if Path('/data').exists() else 0
            }
        }
        
        # Calculate storage sizes
        def get_dir_size(path: Path) -> int:
            total = 0
            if path.exists():
                for item in path.rglob('*'):
                    if item.is_file():
                        total += item.stat().st_size
            return total
        
        metrics["storage"]["downloads"] = get_dir_size(DOWNLOADS_PATH) // (1024 * 1024)  # MB
        metrics["storage"]["sorted"] = get_dir_size(SORTED_PATH) // (1024 * 1024)
        metrics["storage"]["faces"] = get_dir_size(FACE_TRAINING_PATH) // (1024 * 1024)
        # V2: review data tracked per profile
        review_path = Path("/data/review")
        metrics["storage"]["objects"] = get_dir_size(review_path) // (1024 * 1024) if review_path.exists() else 0
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/detections")
async def export_detections(format: str = "json"):
    """Export detection data in JSON or CSV format (V2: from animal profiles)."""
    try:
        detections = []
        
        # V2: Collect detection data from animal profile review folders
        review_path = Path("/data/review")
        if review_path.exists():
            for profile_dir in review_path.iterdir():
                if profile_dir.is_dir():
                    for img_file in profile_dir.glob("*.jpg"):
                        metadata_file = img_file.with_suffix(".json")
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                detections.append({
                                    "label": profile_dir.name,
                                    "confidence": metadata.get('confidence', 0),
                                    "filename": img_file.name,
                                    "timestamp": metadata.get('timestamp', ''),
                                    "video": metadata.get('video', '')
                                })
        
        if format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=["label", "confidence", "filename", "timestamp", "video"])
            writer.writeheader()
            writer.writerows(detections)
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=detections.csv"}
            )
        else:
            # JSON format
            return JSONResponse(content={"detections": detections})
    
    except Exception as e:
        logger.error(f"Failed to export detections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/analytics")
async def export_analytics(format: str = "json"):
    """Export analytics data."""
    try:
        # Get comprehensive stats
        stats_response = await get_analytics_stats()
        
        if format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["Category", "Count"])
            
            for category, count in stats_response["categories"].items():
                writer.writerow([category, count])
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=analytics.csv"}
            )
        else:
            return JSONResponse(content=stats_response)
    
    except Exception as e:
        logger.error(f"Failed to export analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/faces/{person_name}")
async def delete_person(person_name: str):
    """Delete a person's training data."""
    try:
        person_dir = FACE_TRAINING_PATH / person_name
        if not person_dir.exists():
            raise HTTPException(status_code=404, detail="Person not found")
        
        import shutil
        shutil.rmtree(person_dir)
        
        logger.info(f"Deleted training data for {person_name}")
        return {"status": "success", "message": f"Deleted {person_name}"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/faces/retrain_all")
async def retrain_all_faces(background_tasks: BackgroundTasks):
    """Retrain all face recognition models."""
    try:
        if not FACE_TRAINING_PATH.exists():
            raise HTTPException(status_code=404, detail="No training data found")
        
        people = [d.name for d in FACE_TRAINING_PATH.iterdir() if d.is_dir()]
        
        if not people:
            raise HTTPException(status_code=404, detail="No people to train")
        
        def retrain_task():
            try:
                from face_recognizer import FaceRecognizer
                fr = FaceRecognizer()
                
                for person in people:
                    person_dir = FACE_TRAINING_PATH / person
                    images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
                    if images:
                        fr.add_person(person, images)
                        logger.info(f"Retrained {person} with {len(images)} images")
                
                logger.info(f"Completed retraining for {len(people)} people")
            except Exception as e:
                logger.error(f"Retraining failed: {e}")
        
        background_tasks.add_task(retrain_task)
        
        return {
            "status": "success",
            "message": f"Started retraining for {len(people)} people"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================


@app.get("/api/yolo_classes")
async def get_yolo_classes():
    """Get the list of all 80 YOLO COCO classes."""
    try:
        from object_detector import YOLO_COCO_CLASSES
        return {"classes": sorted(YOLO_COCO_CLASSES)}
    except Exception as e:
        logger.error(f"Failed to get YOLO classes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config/add_object_labels")
async def add_object_labels(request: dict):
    """Add labels to the object detection list in config."""
    try:
        labels_to_add = request.get('labels', [])
        
        if not labels_to_add:
            raise HTTPException(status_code=400, detail="No labels provided")
        
        # Load config
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get current object_labels
        current_labels = config.get('detection', {}).get('object_labels', [])
        
        # Add new labels (avoiding duplicates)
        added_labels = []
        for label in labels_to_add:
            if label not in current_labels:
                current_labels.append(label)
                added_labels.append(label)
        
        # Update config
        if 'detection' not in config:
            config['detection'] = {}
        config['detection']['object_labels'] = current_labels
        
        # Save config
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Added {len(added_labels)} labels to object_labels: {added_labels}")
        
        return {
            "status": "success",
            "added_labels": added_labels,
            "current_labels": current_labels,
            "message": f"Added {len(added_labels)} label(s) to detection list"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add object labels: {e}")
        raise HTTPException(status_code=500, detail=str(e))





# ============= Animal Profile API Endpoints =============

@app.post("/api/animal-profiles")
async def create_animal_profile(request: dict):
    """Create new animal profile."""
    try:
        if animal_profile_manager is None:
            raise HTTPException(status_code=500, detail="Animal profile manager not initialized")
        
        name = request.get('name', '').strip()
        yolo_categories = request.get('yolo_categories', [])
        text_description = request.get('text_description', '')
        confidence_threshold = request.get('confidence_threshold', 0.80)
        auto_approval_enabled = request.get('auto_approval_enabled', True)
        retraining_threshold = request.get('retraining_threshold', 0.85)
        confirmation_count_recommendation = request.get('confirmation_count_recommendation', 50)
        
        if not name:
            raise HTTPException(status_code=400, detail="Animal name is required")
        if not yolo_categories:
            raise HTTPException(status_code=400, detail="At least one YOLO category is required")
        
        # Create profile
        profile = animal_profile_manager.create_profile(
            name=name,
            yolo_categories=yolo_categories,
            text_description=text_description if text_description else None
        )
        
        # Set optional parameters
        profile = animal_profile_manager.update_profile(
            profile.id,
            confidence_threshold=confidence_threshold,
            auto_approval_enabled=auto_approval_enabled,
            retraining_threshold=retraining_threshold,
            confirmation_count_recommendation=confirmation_count_recommendation
        )
        
        # Create data directories
        profile_id = profile.id
        base_path = Path("/data")
        dirs_to_create = [
            base_path / "sorted" / profile_id,
            base_path / "review" / profile_id,
            base_path / "training" / profile_id / "confirmed",
            base_path / "training" / profile_id / "rejected",
            base_path / "models" / profile_id
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            # Set permissions (rule: folders created by AI should have full permissions)
            try:
                import stat
                dir_path.chmod(stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 777 permissions
            except Exception as e:
                logger.warning(f"Could not set permissions on {dir_path}: {e}")
        
        logger.info(f"Created animal profile: {name} with categories {yolo_categories}")
        
        return {
            "status": "success",
            "profile": {
                "id": profile.id,
                "name": profile.name,
                "yolo_categories": profile.yolo_categories,
                "text_description": profile.text_description,
                "confidence_threshold": profile.confidence_threshold,
                "auto_approval_enabled": profile.auto_approval_enabled,
                "enabled": profile.enabled,
                "accuracy_percentage": profile.accuracy_percentage,
                "retraining_threshold": profile.retraining_threshold,
                "confirmation_count_recommendation": profile.confirmation_count_recommendation
            },
            "message": f"Created profile '{name}'"
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create animal profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/animal-profiles")
async def list_animal_profiles():
    """List all animal profiles."""
    try:
        if animal_profile_manager is None:
            raise HTTPException(status_code=500, detail="Animal profile manager not initialized")
        
        profiles = animal_profile_manager.list_profiles()
        
        return {
            "status": "success",
            "profiles": [
                {
                    "id": p.id,
                    "name": p.name,
                    "yolo_categories": p.yolo_categories,
                    "text_description": p.text_description,
                    "confidence_threshold": p.confidence_threshold,
                    "auto_approval_enabled": p.auto_approval_enabled,
                    "enabled": p.enabled,
                    "confirmed_count": p.confirmed_count,
                    "rejected_count": p.rejected_count,
                    "accuracy_percentage": p.accuracy_percentage,
                    "retraining_threshold": p.retraining_threshold,
                    "confirmation_count_recommendation": p.confirmation_count_recommendation,
                    "should_recommend_retraining": p.should_recommend_retraining[0],
                    "retraining_message": p.should_recommend_retraining[1]
                }
                for p in profiles
            ],
            "count": len(profiles)
        }
    except Exception as e:
        logger.error(f"Failed to list animal profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/animal-profiles/{profile_id}")
async def get_animal_profile(profile_id: str):
    """Get specific animal profile details."""
    try:
        if animal_profile_manager is None:
            raise HTTPException(status_code=500, detail="Animal profile manager not initialized")
        
        profile = animal_profile_manager.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        return {
            "status": "success",
            "profile": {
                "id": profile.id,
                "name": profile.name,
                "yolo_categories": profile.yolo_categories,
                "text_description": profile.text_description,
                "confidence_threshold": profile.confidence_threshold,
                "auto_approval_enabled": profile.auto_approval_enabled,
                "enabled": profile.enabled,
                "confirmed_count": profile.confirmed_count,
                "rejected_count": profile.rejected_count,
                "accuracy_percentage": profile.accuracy_percentage,
                "retraining_threshold": profile.retraining_threshold,
                "confirmation_count_recommendation": profile.confirmation_count_recommendation,
                "should_recommend_retraining": profile.should_recommend_retraining[0],
                "retraining_message": profile.should_recommend_retraining[1]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get animal profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/animal-profiles/{profile_id}")
async def update_animal_profile(profile_id: str, request: dict):
    """Update animal profile settings."""
    try:
        if animal_profile_manager is None:
            raise HTTPException(status_code=500, detail="Animal profile manager not initialized")
        
        profile = animal_profile_manager.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        # Prepare update data
        update_data = {}
        if 'name' in request:
            update_data['name'] = request['name']
        if 'yolo_categories' in request:
            update_data['yolo_categories'] = request['yolo_categories']
        if 'text_description' in request:
            update_data['text_description'] = request['text_description']
        if 'confidence_threshold' in request:
            update_data['confidence_threshold'] = float(request['confidence_threshold'])
        if 'auto_approval_enabled' in request:
            update_data['auto_approval_enabled'] = bool(request['auto_approval_enabled'])
        if 'retraining_threshold' in request:
            update_data['retraining_threshold'] = float(request['retraining_threshold'])
        if 'confirmation_count_recommendation' in request:
            update_data['confirmation_count_recommendation'] = int(request['confirmation_count_recommendation'])
        
        profile = animal_profile_manager.update_profile(profile_id, **update_data)
        
        logger.info(f"Updated animal profile: {profile.name}")
        
        return {
            "status": "success",
            "profile": {
                "id": profile.id,
                "name": profile.name,
                "yolo_categories": profile.yolo_categories,
                "text_description": profile.text_description,
                "confidence_threshold": profile.confidence_threshold,
                "auto_approval_enabled": profile.auto_approval_enabled,
                "enabled": profile.enabled,
                "accuracy_percentage": profile.accuracy_percentage,
                "retraining_threshold": profile.retraining_threshold,
                "confirmation_count_recommendation": profile.confirmation_count_recommendation
            },
            "message": f"Updated profile '{profile.name}'"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update animal profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/animal-profiles/{profile_id}")
async def delete_animal_profile(profile_id: str):
    """Delete animal profile and all associated data."""
    try:
        if animal_profile_manager is None:
            raise HTTPException(status_code=500, detail="Animal profile manager not initialized")
        
        profile = animal_profile_manager.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        profile_name = profile.name
        
        # Delete profile from storage
        animal_profile_manager.delete_profile(profile_id)
        
        # Optionally clean up associated directories
        # Note: We don't delete them by default to preserve review/training data
        logger.info(f"Deleted animal profile: {profile_name}")
        
        return {
            "status": "success",
            "message": f"Deleted profile '{profile_name}'. Associated data directories remain for manual cleanup if needed."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete animal profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/animal-profiles/{profile_id}/enable")
async def enable_animal_profile(profile_id: str):
    """Enable an animal profile."""
    try:
        if animal_profile_manager is None:
            raise HTTPException(status_code=500, detail="Animal profile manager not initialized")
        
        profile = animal_profile_manager.update_profile(profile_id, enabled=True)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        logger.info(f"Enabled animal profile: {profile.name}")
        
        return {
            "status": "success",
            "message": f"Enabled profile '{profile.name}'",
            "profile": {
                "id": profile.id,
                "name": profile.name,
                "enabled": profile.enabled
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to enable animal profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/animal-profiles/{profile_id}/disable")
async def disable_animal_profile(profile_id: str):
    """Disable an animal profile."""
    try:
        if animal_profile_manager is None:
            raise HTTPException(status_code=500, detail="Animal profile manager not initialized")
        
        profile = animal_profile_manager.update_profile(profile_id, enabled=False)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        logger.info(f"Disabled animal profile: {profile.name}")
        
        return {
            "status": "success",
            "message": f"Disabled profile '{profile.name}'",
            "profile": {
                "id": profile.id,
                "name": profile.name,
                "enabled": profile.enabled
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disable animal profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/animal-profiles/{profile_id}/model-stats")
async def get_model_stats(profile_id: str):
    """Get model accuracy statistics and retraining recommendations."""
    try:
        if animal_profile_manager is None:
            raise HTTPException(status_code=500, detail="Animal profile manager not initialized")
        
        profile = animal_profile_manager.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        should_retrain, retrain_message = profile.should_recommend_retraining
        
        return {
            "status": "success",
            "profile_id": profile.id,
            "profile_name": profile.name,
            "accuracy_percentage": profile.accuracy_percentage,
            "confirmed_count": profile.confirmed_count,
            "rejected_count": profile.rejected_count,
            "total_feedback": profile.confirmed_count + profile.rejected_count,
            "accuracy_threshold": profile.retraining_threshold * 100,
            "confirmation_count_recommendation": profile.confirmation_count_recommendation,
            "should_recommend_retraining": should_retrain,
            "retraining_message": retrain_message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/animal-profiles/{profile_id}/update-accuracy")
async def update_profile_accuracy(profile_id: str, request: dict):
    """Update model accuracy counters."""
    try:
        if animal_profile_manager is None:
            raise HTTPException(status_code=500, detail="Animal profile manager not initialized")
        
        confirmed = int(request.get('confirmed', 0))
        rejected = int(request.get('rejected', 0))
        
        animal_profile_manager.update_accuracy(profile_id, confirmed, rejected)
        profile = animal_profile_manager.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        logger.info(f"Updated accuracy for {profile.name}: {confirmed} confirmed, {rejected} rejected")
        
        should_retrain, retrain_message = profile.should_recommend_retraining
        
        return {
            "status": "success",
            "profile_id": profile.id,
            "accuracy_percentage": profile.accuracy_percentage,
            "confirmed_count": profile.confirmed_count,
            "rejected_count": profile.rejected_count,
            "should_recommend_retraining": should_retrain,
            "retraining_message": retrain_message,
            "message": f"Updated accuracy for '{profile.name}'"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update profile accuracy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Phase 8: Review & Retraining Endpoints =============

@app.get("/api/animal-profiles/{profile_id}/pending-reviews")
async def get_pending_reviews(profile_id: str):
    """Get list of pending frames for review."""
    try:
        if animal_profile_manager is None or review_manager is None:
            raise HTTPException(status_code=500, detail="Managers not initialized")
        
        profile = animal_profile_manager.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        pending_frames = review_manager.list_pending_reviews(profile_id)
        
        return {
            "status": "success",
            "profile_id": profile_id,
            "profile_name": profile.name,
            "pending_count": len(pending_frames),
            "frames": [
                {
                    "filename": frame.frame_path.name,
                    "confidence": frame.get_confidence(),
                    "description": frame.get_description(),
                    "timestamp": frame.metadata.get("timestamp", "")
                }
                for frame in pending_frames
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pending reviews: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/animal-profiles/{profile_id}/frame/{filename}")
async def get_frame_image(profile_id: str, filename: str):
    """Serve frame image with border."""
    try:
        from PIL import Image, ImageOps
        import io
        
        if animal_profile_manager is None:
            raise HTTPException(status_code=500, detail="Animal profile manager not initialized")
        
        profile = animal_profile_manager.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        # Construct frame path
        frame_path = Path("/data") / "review" / profile_id / filename
        
        if not frame_path.exists():
            raise HTTPException(status_code=404, detail=f"Frame '{filename}' not found")
        
        # Load image and add border
        image = Image.open(frame_path)
        
        # Add 4px border (using ImageOps)
        border_color = (100, 150, 200)  # Light blue
        bordered_image = ImageOps.expand(image, border=4, fill=border_color)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        bordered_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        logger.debug(f"Serving frame: {profile_id}/{filename}")
        
        return StreamingResponse(
            iter([img_bytes.getvalue()]),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get frame image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/animal-profiles/{profile_id}/confirm-images")
async def confirm_images(profile_id: str, request: dict):
    """Confirm multiple frames as correct."""
    try:
        if animal_profile_manager is None or review_manager is None:
            raise HTTPException(status_code=500, detail="Managers not initialized")
        
        profile = animal_profile_manager.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        filenames = request.get('filenames', [])
        if not filenames:
            raise HTTPException(status_code=400, detail="No filenames provided")
        
        # Bulk confirm frames
        results = review_manager.bulk_confirm_frames(profile_id, filenames)
        
        logger.info(f"Confirmed {len(results['confirmed'])} frames for {profile.name}")
        
        return {
            "status": "success",
            "profile_id": profile_id,
            "profile_name": profile.name,
            "confirmed_count": len(results['confirmed']),
            "failed_count": len(results['failed']),
            "results": results,
            "message": f"Confirmed {len(results['confirmed'])} frames"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to confirm images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/animal-profiles/{profile_id}/reject-images")
async def reject_images(profile_id: str, request: dict):
    """Reject multiple frames."""
    try:
        if animal_profile_manager is None or review_manager is None:
            raise HTTPException(status_code=500, detail="Managers not initialized")
        
        profile = animal_profile_manager.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        filenames = request.get('filenames', [])
        if not filenames:
            raise HTTPException(status_code=400, detail="No filenames provided")
        
        save_as_negative = request.get('save_as_negative', False)
        
        # Bulk reject frames
        results = review_manager.bulk_reject_frames(profile_id, filenames, save_as_negative)
        
        logger.info(f"Rejected {len(results['rejected'])} frames for {profile.name}")
        
        return {
            "status": "success",
            "profile_id": profile_id,
            "profile_name": profile.name,
            "rejected_count": len(results['rejected']),
            "failed_count": len(results['failed']),
            "results": results,
            "message": f"Rejected {len(results['rejected'])} frames"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/animal-profiles/{profile_id}/retrain")
async def trigger_retrain(profile_id: str, request: dict = None):
    """Trigger model retraining for a profile."""
    try:
        if animal_profile_manager is None:
            raise HTTPException(status_code=500, detail="Animal profile manager not initialized")
        
        profile = animal_profile_manager.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        logger.info(f"Retraining triggered for profile: {profile.name}")
        
        # TODO: Phase 9 - Implement actual retraining logic
        # This would involve:
        # 1. Loading training data from /data/training/{profile_id}/
        # 2. Fine-tuning the CLIP/ViT model
        # 3. Saving the updated model
        # 4. Updating model metadata
        
        return {
            "status": "success",
            "profile_id": profile_id,
            "profile_name": profile.name,
            "message": f"Retraining started for '{profile.name}'. This will run in the background.",
            "note": "Actual model retraining implementation pending for Phase 9"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger retrain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Video-Based Review Endpoints =============

@app.get("/api/review/categories")
async def list_review_categories():
    """List all YOLO categories with pending videos in review."""
    try:
        review_base = Path("/data/review")
        if not review_base.exists():
            return {"status": "success", "categories": []}
        
        categories = []
        for category_dir in review_base.iterdir():
            if category_dir.is_dir():
                # Count videos in this category
                video_count = len(list(category_dir.glob("*.mp4")))
                if video_count > 0:
                    categories.append({
                        "name": category_dir.name,
                        "video_count": video_count
                    })
        
        # Sort by name
        categories.sort(key=lambda x: x["name"])
        
        return {
            "status": "success",
            "categories": categories,
            "total_videos": sum(c["video_count"] for c in categories)
        }
    except Exception as e:
        logger.error(f"Failed to list review categories: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/review/categories/{category}/videos")
async def list_category_videos(category: str):
    """List all videos in a review category."""
    try:
        category_dir = Path("/data/review") / category
        if not category_dir.exists():
            return {"status": "success", "videos": []}
        
        videos = []
        for video_file in sorted(category_dir.glob("*.mp4")):
            # Try to load metadata
            metadata = {}
            metadata_file = video_file.with_suffix(video_file.suffix + ".json")
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {video_file.name}: {e}")
            
            videos.append({
                "filename": video_file.name,
                "category": category,
                "detected_objects": metadata.get("detected_objects", {}),
                "timestamp": metadata.get("timestamp", ""),
                "size_mb": round(video_file.stat().st_size / (1024*1024), 2)
            })
        
        return {
            "status": "success",
            "category": category,
            "video_count": len(videos),
            "videos": videos
        }
    except Exception as e:
        logger.error(f"Failed to list category videos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/review/video/{category}/{filename}")
async def serve_review_video(category: str, filename: str):
    """Serve a video file from review folder."""
    try:
        video_path = Path("/data/review") / category / filename
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {filename}")
        
        # Return video file
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=filename
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/review/assign-to-profile")
async def assign_videos_to_profile(request: dict):
    """Assign videos from review to an animal profile."""
    try:
        if animal_profile_manager is None:
            raise HTTPException(status_code=500, detail="Animal profile manager not initialized")
        
        category = request.get('category')
        filenames = request.get('filenames', [])
        profile_id = request.get('profile_id')
        
        if not category or not filenames or not profile_id:
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        profile = animal_profile_manager.get_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
        
        # Move videos from review to profile's training folder
        import shutil
        results = {"moved": [], "failed": []}
        
        for filename in filenames:
            try:
                source_path = Path("/data/review") / category / filename
                if not source_path.exists():
                    results["failed"].append({"filename": filename, "error": "File not found"})
                    continue
                
                # Create destination directory
                dest_dir = Path("/data/training") / profile_id / "confirmed"
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                dest_path = dest_dir / filename
                
                # Move video
                shutil.move(str(source_path), str(dest_path))
                
                # Move metadata if exists
                metadata_source = source_path.with_suffix(source_path.suffix + ".json")
                if metadata_source.exists():
                    metadata_dest = dest_path.with_suffix(dest_path.suffix + ".json")
                    shutil.move(str(metadata_source), str(metadata_dest))
                
                results["moved"].append(filename)
                
                # Increment confirmed count
                profile.confirmed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to move {filename}: {e}")
                results["failed"].append({"filename": filename, "error": str(e)})
        
        # Save updated profile
        animal_profile_manager._save_profile(profile)
        
        logger.info(f"Assigned {len(results['moved'])} videos to {profile.name}")
        
        return {
            "status": "success",
            "profile_id": profile_id,
            "profile_name": profile.name,
            "moved_count": len(results["moved"]),
            "failed_count": len(results["failed"]),
            "results": results,
            "message": f"Assigned {len(results['moved'])} videos to '{profile.name}'"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assign videos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/review/delete-videos")
async def delete_review_videos(request: dict):
    """Delete videos from review (reject without saving)."""
    try:
        category = request.get('category')
        filenames = request.get('filenames', [])
        
        if not category or not filenames:
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        results = {"deleted": [], "failed": []}
        
        for filename in filenames:
            try:
                video_path = Path("/data/review") / category / filename
                if video_path.exists():
                    video_path.unlink()
                    results["deleted"].append(filename)
                    
                    # Delete metadata if exists
                    metadata_path = video_path.with_suffix(video_path.suffix + ".json")
                    if metadata_path.exists():
                        metadata_path.unlink()
                else:
                    results["failed"].append({"filename": filename, "error": "File not found"})
            except Exception as e:
                logger.error(f"Failed to delete {filename}: {e}")
                results["failed"].append({"filename": filename, "error": str(e)})
        
        logger.info(f"Deleted {len(results['deleted'])} videos from review")
        
        return {
            "status": "success",
            "deleted_count": len(results["deleted"]),
            "failed_count": len(results["failed"]),
            "results": results,
            "message": f"Deleted {len(results['deleted'])} videos"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete videos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def start_web_server(host: str = "0.0.0.0", port: int = None):
    """Start the web server."""
    if port is None:
        port = int(os.environ.get('WEB_PORT', 8080))
    logger.info(f"Starting web interface on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_web_server()
