# CritterCatcherAI - Technical Specification

**Version 2.0 - Hybrid YOLO-First Workflow**  
**Last Updated:** December 2, 2025

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Hybrid Workflow Design](#hybrid-workflow-design)
3. [Directory Structure](#directory-structure)
4. [Processing Pipeline](#processing-pipeline)
5. [Configuration System](#configuration-system)
6. [API Reference](#api-reference)
7. [Docker Deployment](#docker-deployment)
8. [Development Guidelines](#development-guidelines)

---

## System Architecture

### Overview

CritterCatcherAI is a two-stage AI detection system designed to automatically sort Ring camera videos by detected content.

**Core Components:**
- **Ring Downloader** - Fetches videos from Ring API
- **YOLO Detector** - Stage 1: Broad object detection
- **Object Tracker** - Creates annotated videos with bounding boxes
- **CLIP Classifier** - Stage 2: Precise classification (optional)
- **Face Recognizer** - Identifies specific people
- **Video Sorter** - Organizes videos by detection results
- **Web Interface** - FastAPI + HTML/JS frontend

### Technology Stack

**Backend:**
- Python 3.11
- FastAPI (async web framework)
- YOLOv8 (Ultralytics) - Object detection
- OpenAI CLIP - Zero-shot classification
- face_recognition (dlib) - Facial recognition
- OpenCV - Video processing
- ring-doorbell - Ring API client

**Frontend:**
- Vanilla HTML/CSS/JavaScript
- No external frameworks
- Server-Side Events (SSE) for real-time updates

**Deployment:**
- Docker container
- docker-compose orchestration
- Unraid-optimized configuration

---

## Hybrid Workflow Design

### Stage 1: YOLO Detection → Initial Sorting

```
Video Input
  ↓
YOLO Detection (confidence ≥ threshold)
  ↓
Detected: "dog" (confidence: 0.85)
  ↓
Create tracked video with bounding boxes
  ↓
Save to: /data/review/dog/
  ↓
Save metadata: /data/review/dog/video.mp4.json
```

**Key Points:**
- All videos are sorted by YOLO category FIRST
- Tracked videos created in parallel with annotations
- Metadata saved alongside each video
- Only enabled YOLO categories are monitored

### Stage 2: CLIP Refinement (Optional)

```
Check: Does a CLIP profile use "dog" category?
  ↓
YES: "hedgehog" profile monitors [cat, dog]
  ↓
Run CLIP: "Is this a hedgehog?"
  ↓
RESULT SCENARIOS:
  
  A) High confidence (≥0.75 + auto-approval enabled):
     → Move to /data/sorted/hedgehog/
  
  B) Low confidence (<0.75):
     → Keep in /data/review/dog/
     → User reviews manually
```

**Key Points:**
- CLIP only runs if profile exists for detected category
- Multiple profiles can monitor same YOLO category
- Highest confidence match wins
- Original video moved, tracked video deleted after confirmation

### Stage 3: Face Recognition (Optional)

```
YOLO detects "person"
  ↓
Check: Is face recognition enabled?
  ↓
YES: Extract faces from video frames
  ↓
Match against known face encodings
  ↓
Recognized: "John Doe"
  → Route to /data/sorted/person/John/
```

**Key Points:**
- Only runs if "person" detected by YOLO
- Requires trained face encodings
- Can run in parallel with CLIP

---

## Directory Structure

```
/data/
├── downloads/                          # Temporary Ring video downloads
│   └── *.mp4                          # Raw videos (deleted after processing)
│
├── review/                            # YOLO-sorted videos pending review
│   ├── car/                           # All car detections
│   │   ├── video_001.mp4              # Original video
│   │   └── video_001.mp4.json         # Metadata (YOLO + CLIP results)
│   ├── dog/
│   ├── bird/
│   ├── person/
│   └── {category}/
│
├── sorted/                            # Confirmed/auto-approved videos
│   ├── {category}/                    # YOLO category confirmed
│   ├── {clip_profile}/                # CLIP profile matched
│   └── person/
│       └── {name}/                    # Face recognition matched
│
├── objects/detected/
│   └── annotated_videos/              # Tracked videos with bounding boxes
│       └── tracked_{filename}.mp4     # Temporary, deleted after confirmation
│
├── training/faces/
│   ├── unassigned/                    # Extracted faces pending assignment
│   │   ├── face_001.jpg
│   │   └── face_001.jpg.json          # Metadata (video source, timestamp)
│   └── {profile_id}/
│       ├── confirmed/                 # Training faces for person
│       └── rejected/                  # Negative examples
│
├── tokens/
│   └── ring_token.json                # Ring OAuth2 refresh token
│
├── models/                            # CLIP models (if used)
│   └── {profile}/
│       └── model.pt
│
└── animal_profiles/                   # CLIP profile definitions
    └── {profile_id}.json

/config/
└── config.yaml                        # Main configuration file
```

---

## Processing Pipeline

### Main Processing Loop

**File:** `src/main.py`

```python
while scheduler_enabled or run_once:
    1. Authenticate with Ring (using saved token)
    2. Download recent videos → /data/downloads/
    3. For each downloaded video:
        a. Run YOLO detection (Stage 1)
        b. If object detected:
           - Sort to /data/review/{category}/
           - Create tracked video with bounding boxes
        c. Check for matching CLIP profiles
        d. If CLIP profile exists:
           - Run CLIP classification (Stage 2)
           - If high confidence: Move to /data/sorted/{profile}/
        e. If person detected + face recognition enabled:
           - Extract and match faces
           - Route to person-specific folder if matched
    4. Save metadata for all detections
    5. Update download tracker database
    6. Sleep for interval_minutes
```

### YOLO Detection

**File:** `src/object_detector.py`

```python
def detect_objects_in_video(video_path, return_bboxes=True):
    """
    Extract frames, run YOLO, return detections with bounding boxes.
    
    Returns:
    {
        "dog": {
            "confidence": 0.85,
            "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
        },
        "car": {
            "confidence": 0.72,
            "bbox": {"x1": 500, "y1": 150, "x2": 800, "y2": 350}
        }
    }
    """
```

**Key Features:**
- Extracts 5 evenly-spaced frames per video
- Returns highest confidence detection per class
- Includes bounding box coordinates
- Only returns detections ≥ confidence_threshold

### Video Tracking

**File:** `src/object_detector.py`

```python
def track_and_annotate_video(video_path, output_path=None):
    """
    Create annotated video with persistent object tracking.
    
    Features:
    - YOLOv8 track() with persistent IDs
    - Green bounding boxes with labels
    - Track ID displayed (e.g., "dog #2 0.85")
    - Codec fallback: H264 → X264 → avc1 → mp4v
    - Automatic H.264 conversion via ffmpeg if mp4v used
    """
```

**Codec Strategy:**
1. Try H264/X264/avc1 first (browser-compatible)
2. Fall back to mp4v if needed
3. Post-process with ffmpeg to convert mp4v → H.264
4. Ensures all videos play in modern browsers

### CLIP Classification

**File:** `src/clip_vit_classifier.py`

```python
def classify_video(video_path, text_description):
    """
    Zero-shot classification using CLIP.
    
    Args:
        video_path: Path to video file
        text_description: "a hedgehog" or "a Jack Russell terrier"
    
    Returns:
        confidence: float (0.0-1.0)
    """
```

---

## Configuration System

### config.yaml Structure

```yaml
# Paths (container-side)
paths:
  downloads: /data/downloads
  sorted: /data/sorted
  face_encodings: /data/faces/encodings.pkl

# Ring camera settings
ring:
  download_hours: 24          # Hours back to download videos
  download_limit: null        # Max videos (null = unlimited)

# Object detection settings
detection:
  confidence_threshold: 0.25  # YOLO minimum confidence
  object_frames: 5            # Frames to analyze per video
  face_tolerance: 0.6         # Face matching threshold (lower = stricter)
  face_frames: 10             # Frames for face detection
  face_model: hog             # 'hog' (fast) or 'cnn' (accurate)
  priority: people            # 'people' or 'objects' for routing
  yolo_model: yolov8n         # n/s/m/l/x (size/accuracy tradeoff)

# Scheduler settings
scheduler:
  auto_run: true              # Enable automatic processing
  interval_minutes: 60        # Check for new videos every N minutes

# Image review automation
image_review:
  auto_confirm_threshold: 0.85    # Auto-confirm if confidence ≥ this
  max_confirmed_images: 200       # Max images kept per label

# Video tracking
tracking:
  enabled: true                    # Create annotated videos
  save_original_videos: false      # Keep untracked versions

# Manually enabled YOLO categories
yolo_manual_categories:
  - bird
  - cat
  - dog
  - person
  - car

# Face recognition
face_recognition:
  enabled: true

# CLIP profiles (optional)
animal_profiles:
  - name: hedgehog
    enabled: true
    yolo_categories: [cat, dog]
    text_description: "a small hedgehog with spiky brown fur"
    confidence_threshold: 0.75
    auto_approval_threshold: 0.80
```

### Configuration Hierarchy

1. **Default values** in code
2. **config.yaml** file
3. **Environment variables** (override config file)
4. **Web UI updates** (writes to config.yaml)

### Environment Variables

```bash
# Ring credentials
RING_USERNAME=your@email.com
RING_PASSWORD=yourpassword

# Logging
LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
PYTHONUNBUFFERED=1          # Real-time log output

# Scheduler
RUN_ONCE=false              # true = run once and exit

# Container
PUID=99                     # User ID (Unraid default)
PGID=100                    # Group ID (Unraid default)
UMASK=0000                  # File permissions (rwxrwxrwx)
```

---

## API Reference

### Base URL
```
http://localhost:8080
```

### Compatibility Note
The API includes compatibility aliases for earlier and documented routes. The
spec endpoints remain valid, and the application may also expose additional
routes used by the current UI. When both exist, the spec route delegates to the
current implementation.

### Status & Configuration

#### GET /api/status
Get current processing status.

**Response:**
```json
{
  "is_processing": true,
  "last_run": "2025-12-02T10:30:00",
  "uptime": "Running",
  "version": "2.0",
  "processing_progress": {
    "current_video": "Front_Door_20251202_103045.mp4",
    "current_step": "Running YOLO detection...",
    "videos_processed": 15,
    "videos_total": 50,
    "start_time": "2025-12-02T10:25:00"
  }
}
```

#### GET /api/config
Get current configuration.

**Response:**
```json
{
  "status": "success",
  "config": {
    "detection": {...},
    "scheduler": {...},
    "image_review": {...}
  }
}
```

#### POST /api/config/save
Update configuration via web UI.

### YOLO Categories

#### GET /api/yolo-categories
Get all YOLO categories with usage information.

**Response:**
```json
{
  "status": "success",
  "categories": [
    {
      "name": "dog",
      "used_by_profiles": ["hedgehog"],
      "manually_enabled": true,
      "is_enabled": true
    }
  ],
  "total_categories": 80
}
```

#### POST /api/yolo-categories/manual/toggle
Enable/disable a YOLO category.

**Request:**
```json
{
  "category": "bird",
  "enabled": true
}
```

### Video Review

#### GET /api/review/categories
List all categories with pending videos.

**Response:**
```json
{
  "status": "success",
  "categories": [
    {"name": "car", "video_count": 17},
    {"name": "dog", "video_count": 3}
  ],
  "total_videos": 45
}
```

#### GET /api/review/categories/{category}/videos
List videos in a specific category.

**Response:**
```json
{
  "status": "success",
  "category": "dog",
  "video_count": 3,
  "videos": [
    {
      "filename": "Front_Door_20251202_103045.mp4",
      "category": "dog",
      "detected_objects": {
        "dog": {"confidence": 0.85, "bbox": {...}}
      },
      "yolo_category": "dog",
      "yolo_confidence": 0.85,
      "clip_results": null,
      "status": "pending_review",
      "size_mb": 5.2,
      "tracked_video_filename": "tracked_Front_Door_20251202_103045.mp4"
    }
  ]
}
```

#### GET /api/review/video/{category}/{filename}
Serve original video file.

#### GET /api/review/tracked-video/{filename}
Serve tracked video with bounding boxes.

#### POST /api/review/confirm
Confirm selected videos.

**Request:**
```json
{
  "category": "dog",
  "filenames": ["video_001.mp4", "video_002.mp4"]
}
```

#### POST /api/review/reject
Reject selected videos.

**Request:**
```json
{
  "category": "dog",
  "filenames": ["video_003.mp4"]
}
```

### Processing Control

#### POST /api/process
Start video processing immediately.

#### POST /api/stop
Stop current processing.

### Ring Authentication

#### POST /api/ring/authenticate
Authenticate with Ring.

**Request:**
```json
{
  "username": "your@email.com",
  "password": "yourpassword",
  "two_fa_code": "123456"
}
```

---

## Docker Deployment

### docker-compose.yml

```yaml
version: '3.8'

services:
  crittercatcherai:
    build: .
    container_name: CritterCatcherAI
    restart: unless-stopped
    
    ports:
      - "8080:8080"
    
    volumes:
      - ./config:/config
      - ./data:/data
    
    environment:
      - LOG_LEVEL=INFO
      - PYTHONUNBUFFERED=1
      - PUID=99
      - PGID=100
      - UMASK=0000
    
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/status"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Dockerfile

```dockerfile
FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 libopenblas-dev liblapack-dev \
    libx264-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir dlib
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Create volumes
RUN mkdir -p /data /config && \
    chmod -R 777 /data

VOLUME ["/data", "/config"]

EXPOSE 8080

CMD ["python", "-u", "src/main.py"]
```

### Building & Running

```bash
# Build image
docker-compose build

# Start container
docker-compose up -d

# View logs
docker logs -f CritterCatcherAI

# Stop container
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

---

## Development Guidelines

### Adding New YOLO Categories

YOLO detects 80 fixed COCO classes. To add support for new categories:

1. Verify category exists in COCO dataset (see list in USER_GUIDE.md)
2. Add to `yolo_manual_categories` in config.yaml
3. Restart container
4. Invalid categories are filtered out with warnings

### Adding New CLIP Profiles

1. Create profile definition in config.yaml:
```yaml
animal_profiles:
  - name: my_animal
    enabled: true
    yolo_categories: [cat, dog, bird]
    text_description: "a description of my animal"
    confidence_threshold: 0.75
```

2. Restart container
3. Profile automatically activates on next processing run

### Code Structure

```
src/
├── main.py                    # Main orchestrator
├── ring_downloader.py         # Ring API client
├── object_detector.py         # YOLO detection + tracking
├── clip_vit_classifier.py     # CLIP classification
├── face_recognizer.py         # Face recognition
├── video_sorter.py            # File organization
├── animal_profile.py          # CLIP profile management
├── face_profile.py            # Face profile management
├── review_feedback.py         # Review manager
├── download_tracker.py        # Database tracking
├── task_tracker.py            # Background task tracking
├── webapp.py                  # FastAPI application
└── static/
    ├── index.html             # Dashboard
    ├── review.html            # Video review
    ├── yolo_categories.html   # Category management
    ├── face_training.html     # Face assignment
    └── status_widget.html     # Progress widget
```

### Adding New API Endpoints

1. Edit `src/webapp.py`
2. Add route with decorator:
```python
@app.get("/api/my-endpoint")
async def my_endpoint():
    return {"status": "success", "data": [...]}
```

3. Document in this spec
4. Test with curl or browser

### Testing

No automated test framework currently. Manual testing workflow:

1. Make code changes
2. Rebuild container: `docker-compose up -d --build`
3. Run processing: Click "Process Now" in web UI
4. Check logs: `docker logs -f CritterCatcherAI`
5. Verify results in `/data/review/` and `/data/sorted/`

### Logging

All modules use Python `logging`:

```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Detailed information")
logger.info("Normal operation")
logger.warning("Something unexpected")
logger.error("Error occurred")
```

**Log levels:**
- DEBUG - Frame-by-frame details
- INFO - Pipeline progress, decisions
- WARNING - Missing data, skipped items
- ERROR - Failures, exceptions

---

## Performance Considerations

### Expected Performance (CPU-only)

- **YOLO Detection:** 30-50ms per frame
- **CLIP Scoring:** 100-150ms per frame
- **Face Recognition:** 50-100ms per frame
- **Video Tracking:** 50-100ms per frame
- **Total Throughput:** 3-5 frames/second
- **30-second video:** ~2 minutes to process

### GPU Acceleration

Uncomment GPU section in docker-compose.yml:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

**Expected speedup:** 3-5x faster processing

### Optimization Tips

1. **Reduce frames analyzed** - Lower `object_frames` to 3-5
2. **Use smaller YOLO model** - `yolov8n` instead of `yolov8m`
3. **Increase processing interval** - 120 minutes instead of 60
4. **Enable GPU** - If available
5. **Disable tracking** - If bounding boxes not needed

---

## Support & Contributing

- **Documentation:** [README.md](README.md), [USER_GUIDE.md](USER_GUIDE.md)
- **GitHub Repository:** https://github.com/Ahazii/CritterCatcherAI
- **Issues:** https://github.com/Ahazii/CritterCatcherAI/issues

---

**Last Updated:** December 2, 2025  
**Version:** 2.0 (Hybrid YOLO-First Workflow)
