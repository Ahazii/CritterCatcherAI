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
    "log_buffer": []
}

CONFIG_PATH = Path("/app/config/config.yaml")
FACE_TRAINING_PATH = Path("/data/faces/training")
UNKNOWN_FACES_PATH = Path("/data/faces/unknown")
DETECTED_OBJECTS_PATH = Path("/data/objects/detected")
DISCOVERIES_FILE = Path("/data/discoveries.json")
SORTED_PATH = Path("/data/sorted")
DOWNLOADS_PATH = Path("/data/downloads")


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
    return {
        "is_processing": app_state["is_processing"],
        "last_run": app_state["last_run"],
        "uptime": "Running",
        "version": "1.0.0"
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
        
        # Write to config file
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(config_data["config"], f, default_flow_style=False)
        
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
    
    background_tasks.add_task(process_task)
    logger.info("Processing task added to background queue")
    return {"status": "started", "message": "Processing started"}


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


@app.get("/api/detected_objects")
async def get_detected_objects():
    """Get list of detected objects organized by label."""
    try:
        if not DETECTED_OBJECTS_PATH.exists():
            return {"objects": []}
        
        objects = []
        
        # Iterate through label directories
        for label_dir in DETECTED_OBJECTS_PATH.iterdir():
            if label_dir.is_dir():
                label_objects = []
                
                for img_file in label_dir.glob("*.jpg"):
                    metadata_file = label_dir / f"{img_file.name}.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            metadata['path'] = f"{label_dir.name}/{img_file.name}"
                            label_objects.append(metadata)
                
                if label_objects:
                    # Sort by confidence descending
                    label_objects.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                    
                    objects.append({
                        "label": label_dir.name.replace("_", " "),
                        "count": len(label_objects),
                        "detections": label_objects[:50]  # Limit to 50 per label
                    })
        
        return {"objects": objects}
        
    except Exception as e:
        logger.error(f"Failed to get detected objects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/detected_objects/image/{label}/{filename}")
async def get_detected_object_image(label: str, filename: str):
    """Serve a detected object image."""
    try:
        image_path = DETECTED_OBJECTS_PATH / label / filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detected_objects/relabel")
async def relabel_detected_object(request: dict):
    """Relabel a detected object (correct misclassification or add new label)."""
    try:
        old_label = request.get('old_label', '').replace(" ", "_")
        new_label = request.get('new_label', '').strip()
        filename = request.get('filename', '')
        action = request.get('action', 'relabel')  # 'relabel' or 'delete'
        
        if not old_label or not filename:
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        old_path = DETECTED_OBJECTS_PATH / old_label / filename
        old_metadata_path = DETECTED_OBJECTS_PATH / old_label / f"{filename}.json"
        
        if not old_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if action == 'delete':
            old_path.unlink()
            if old_metadata_path.exists():
                old_metadata_path.unlink()
            return {"status": "success", "message": "Detection deleted"}
        
        if not new_label:
            raise HTTPException(status_code=400, detail="New label required for relabeling")
        
        # Create new label directory
        new_label_dir = DETECTED_OBJECTS_PATH / new_label.replace(" ", "_")
        new_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Move files
        new_path = new_label_dir / filename
        new_metadata_path = new_label_dir / f"{filename}.json"
        
        old_path.rename(new_path)
        
        # Update metadata
        if old_metadata_path.exists():
            with open(old_metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['label'] = new_label
            with open(new_metadata_path, 'w') as f:
                json.dump(metadata, f)
            old_metadata_path.unlink()
        
        return {
            "status": "success",
            "message": f"Relabeled from '{old_label}' to '{new_label}'"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to relabel object: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/discoveries")
async def get_discoveries():
    """Get list of discovered objects pending user confirmation."""
    try:
        if not DISCOVERIES_FILE.exists():
            return {"discoveries": []}
        
        with open(DISCOVERIES_FILE, 'r') as f:
            all_discoveries = json.load(f)
        
        # Format for UI
        discoveries = []
        for label, data in all_discoveries.items():
            avg_confidence = data["total_confidence"] / data["detection_count"] if data["detection_count"] > 0 else 0
            
            # Get sample images
            label_dir = DETECTED_OBJECTS_PATH / label.replace(" ", "_")
            sample_images = []
            if label_dir.exists():
                for img_file in list(label_dir.glob("*.jpg"))[:3]:  # Get 3 samples
                    metadata_file = label_dir / f"{img_file.name}.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as mf:
                            metadata = json.load(mf)
                            if metadata.get('is_discovery', False):
                                sample_images.append({
                                    "filename": img_file.name,
                                    "path": f"{label.replace(' ', '_')}/{img_file.name}"
                                })
            
            discoveries.append({
                "label": label,
                "detection_count": data["detection_count"],
                "avg_confidence": avg_confidence,
                "first_seen": data["first_seen"],
                "videos": data["videos"],
                "sample_images": sample_images
            })
        
        # Sort by detection count
        discoveries.sort(key=lambda x: x["detection_count"], reverse=True)
        
        return {"discoveries": discoveries}
    
    except Exception as e:
        logger.error(f"Failed to get discoveries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/discoveries/approve")
async def approve_discoveries(request: dict, background_tasks: BackgroundTasks):
    """Approve discovered objects and add them to tracking list."""
    try:
        labels = request.get('labels', [])
        
        if not labels:
            raise HTTPException(status_code=400, detail="No labels specified")
        
        # Load config
        if not CONFIG_PATH.exists():
            raise HTTPException(status_code=404, detail="Config file not found")
        
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add to object_labels
        current_labels = config.get('detection', {}).get('object_labels', [])
        new_labels = [label for label in labels if label not in current_labels]
        
        if new_labels:
            config['detection']['object_labels'].extend(new_labels)
            
            # Save config
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Added {len(new_labels)} labels to tracking: {new_labels}")
        
        # Remove from discoveries
        if DISCOVERIES_FILE.exists():
            with open(DISCOVERIES_FILE, 'r') as f:
                discoveries = json.load(f)
            
            for label in labels:
                discoveries.pop(label, None)
            
            with open(DISCOVERIES_FILE, 'w') as f:
                json.dump(discoveries, f, indent=2)
        
        return {
            "status": "success",
            "message": f"Added {len(new_labels)} new object(s) to tracking list"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve discoveries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/discoveries/ignore")
async def ignore_discoveries(request: dict):
    """Ignore discovered objects (add to blacklist)."""
    try:
        labels = request.get('labels', [])
        
        if not labels:
            raise HTTPException(status_code=400, detail="No labels specified")
        
        # Load config
        if not CONFIG_PATH.exists():
            raise HTTPException(status_code=404, detail="Config file not found")
        
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add to ignored_labels
        ignored_labels = config.get('detection', {}).get('ignored_labels', [])
        new_ignored = [label for label in labels if label not in ignored_labels]
        
        if new_ignored:
            if 'ignored_labels' not in config['detection']:
                config['detection']['ignored_labels'] = []
            config['detection']['ignored_labels'].extend(new_ignored)
            
            # Save config
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Added {len(new_ignored)} labels to blacklist: {new_ignored}")
        
        # Remove from discoveries
        if DISCOVERIES_FILE.exists():
            with open(DISCOVERIES_FILE, 'r') as f:
                discoveries = json.load(f)
            
            for label in labels:
                discoveries.pop(label, None)
            
            with open(DISCOVERIES_FILE, 'w') as f:
                json.dump(discoveries, f, indent=2)
        
        # Delete detection images for ignored labels
        for label in labels:
            label_dir = DETECTED_OBJECTS_PATH / label.replace(" ", "_")
            if label_dir.exists():
                import shutil
                shutil.rmtree(label_dir)
                logger.info(f"Deleted detection images for ignored label: {label}")
        
        return {
            "status": "success",
            "message": f"Ignored {len(new_ignored)} object(s) and added to blacklist"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ignore discoveries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/blacklist/remove")
async def remove_from_blacklist(request: dict):
    """Remove labels from blacklist."""
    try:
        labels = request.get('labels', [])
        
        if not labels:
            raise HTTPException(status_code=400, detail="No labels specified")
        
        # Load config
        if not CONFIG_PATH.exists():
            raise HTTPException(status_code=404, detail="Config file not found")
        
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        
        # Remove from ignored_labels
        ignored_labels = config.get('detection', {}).get('ignored_labels', [])
        removed = [label for label in labels if label in ignored_labels]
        
        if removed:
            config['detection']['ignored_labels'] = [l for l in ignored_labels if l not in labels]
            
            # Save config
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Removed {len(removed)} labels from blacklist: {removed}")
        
        return {
            "status": "success",
            "message": f"Removed {len(removed)} object(s) from blacklist"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove from blacklist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reprocess")
async def reprocess_all_videos(background_tasks: BackgroundTasks):
    """Move all sorted videos back to downloads and trigger reprocessing."""
    try:
        if app_state["is_processing"]:
            return {"status": "already_running", "message": "Processing is already in progress"}
        
        if not SORTED_PATH.exists():
            return {"status": "no_videos", "message": "No sorted videos found"}
        
        # Count videos to move
        video_count = 0
        for video_file in SORTED_PATH.rglob("*.mp4"):
            video_count += 1
        
        if video_count == 0:
            return {"status": "no_videos", "message": "No videos found in sorted directories"}
        
        logger.info(f"Reprocessing: moving {video_count} videos back to downloads")
        
        def reprocess_task():
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
                
                # Trigger processing
                from main import process_videos, load_config
                config = load_config()
                logger.info("Starting reprocessing...")
                process_videos(config)
                logger.info("Reprocessing complete")
                
            except Exception as e:
                logger.error(f"Reprocessing failed: {e}", exc_info=True)
            finally:
                app_state["is_processing"] = False
        
        background_tasks.add_task(reprocess_task)
        
        return {
            "status": "started",
            "message": f"Reprocessing {video_count} videos..."
        }
    
    except Exception as e:
        logger.error(f"Failed to start reprocessing: {e}")
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


def start_web_server(host: str = "0.0.0.0", port: int = None):
    """Start the web server."""
    if port is None:
        port = int(os.environ.get('WEB_PORT', 8080))
    logger.info(f"Starting web interface on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_web_server()
