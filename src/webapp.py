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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="CritterCatcherAI", version="1.0.0")

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
SORTED_PATH = Path("/data/sorted")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    index_file = static_path / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>CritterCatcherAI</h1><p>Web interface loading...</p>")
    return HTMLResponse(content=index_file.read_text(), status_code=200)


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


def start_web_server(host: str = "0.0.0.0", port: int = None):
    """Start the web server."""
    if port is None:
        port = int(os.environ.get('WEB_PORT', 8080))
    logger.info(f"Starting web interface on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_web_server()
