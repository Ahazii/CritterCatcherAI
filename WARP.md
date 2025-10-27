# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Commands

### Docker Operations
- **Build and start**: `docker-compose up -d`
- **View logs**: `docker logs -f crittercatcher-ai`
- **Stop container**: `docker-compose down`
- **Rebuild after changes**: `docker-compose up -d --build`

### Development and Testing
- **Run once (non-continuous mode)**: `docker-compose run --rm crittercatcher python src/main.py`
- **Test specific module**: `docker exec crittercatcher-ai python -c "from <module> import <Class>; ..."`
- **Interactive 2FA setup**: `docker-compose run --rm crittercatcher python src/main.py` (for first-time Ring authentication with 2FA)

### Face Recognition Training
```powershell
docker exec -it crittercatcher-ai python -c "from face_recognizer import FaceRecognizer; from pathlib import Path; fr = FaceRecognizer(); images = list(Path('/data/faces/training/<PersonName>').glob('*.jpg')); fr.add_person('<PersonName>', images)"
```

### Configuration Management
- **Edit config**: `nano config/config.yaml` (or use any text editor on Windows)
- **View environment**: `type .env` (PowerShell) or `cat .env` (bash/WSL)

### Troubleshooting
- **Force Ring re-authentication**: Delete `/mnt/user/appdata/crittercatcher/tokens/ring_token.json` and restart
- **Check container health**: `docker ps` (look for healthy status)

## Architecture Overview

CritterCatcherAI is a pipeline-based system that downloads Ring doorbell videos, analyzes them with AI models, and automatically organizes them by detected content. The application is designed to run in Docker containers on Unraid servers.

### Pipeline Flow
```
Ring API → RingDownloader → Video Files → AI Analysis (YOLOv8 + face_recognition) → VideoSorter → Organized Directories
```

### Core Components

1. **main.py** - Orchestrator
   - Manages the overall pipeline execution
   - Handles configuration loading from `config/config.yaml`
   - Supports two run modes: continuous (default) or single-run via `RUN_ONCE` env var
   - Uses environment variables for Ring credentials (`RING_USERNAME`, `RING_PASSWORD`)
   - Configurable logging via `LOG_LEVEL` env var

2. **ring_downloader.py** - Video Acquisition
   - Uses `ring-doorbell` library (unofficial but well-maintained)
   - Supports OAuth2 refresh tokens stored in `/data/ring_token.json`
   - Handles 2FA authentication on first run
   - Downloads videos from last N hours (configurable via `ring.download_hours` in config)
   - Supports both doorbells and stickup cameras

3. **object_detector.py** - Real-Time Object Detection
   - Uses YOLOv8 with configurable model size (nano/small/medium/large/xlarge)
   - Extracts 5 frames per video (evenly distributed) for analysis
   - Detects 80 COCO dataset classes ONLY - cannot detect objects outside this set
   - Returns confidence scores and bounding boxes for detected objects
   - Discovery mode alerts to COCO classes not in your tracking list
   - Validates configured labels against YOLO COCO classes on startup

4. **face_recognizer.py** - Person Identification
   - Uses `face_recognition` library (built on dlib)
   - Stores face encodings in `/data/faces/encodings.pkl` (pickled format)
   - Requires 3-5 training photos per person for accurate recognition
   - Extracts 10 frames per video for face detection
   - Configurable tolerance (lower = stricter matching)

5. **video_sorter.py** - Organization Logic
   - Moves videos to class-specific directories under `/data/sorted/`
   - Supports priority modes: "people" (prioritize face recognition) or "objects" (prioritize object detection)
   - Falls back through detection hierarchy: priority → objects → people → unknown
   - People are organized as `people/<PersonName>/`
   - Objects are organized as `<object_label>/`
   - Handles duplicate filenames by appending counters

### Data Flow

**Input**: Ring doorbell videos from last N hours (default: 24)

**Processing**:
1. Frame extraction (5 frames for objects, 10 frames for faces)
2. Parallel AI inference (YOLOv8 for objects, face_recognition for people)
3. Decision tree based on priority and confidence scores

**Output**: Videos moved to `/data/sorted/<class>/` with original filenames preserved

### Configuration System

All configuration lives in `config/config.yaml`:
- **Paths**: Container-side paths for downloads, sorted videos, face encodings
- **Ring settings**: `download_hours`, `download_limit`
- **Detection settings**: `object_labels` (list), `confidence_threshold` (0.0-1.0), `face_tolerance` (0.0-1.0), `priority` ("people" or "objects")
- **Application settings**: `run_once` (boolean), `interval_minutes` (int)

Environment variables override config file settings:
- `RING_USERNAME`, `RING_PASSWORD` (required)
- `LOG_LEVEL` (DEBUG/INFO/WARNING/ERROR)
- `RUN_ONCE` (true/false)
- `TZ` (timezone)

### Volume Mappings (Unraid-specific)

The application expects these volumes to be mounted:
- `/app/config` → Configuration files (read-only)
- `/data/downloads` → Temporary download storage
- `/data/sorted` → Final organized video storage
- `/data/faces` → Face encodings database
- `/data` → Ring authentication token storage

### AI Models

**YOLOv8 (Object Detection)**:
- Model: Configurable via `detection.yolo_model` in config.yaml
  - `yolov8n` (nano): ~6MB, fastest, good for real-time
  - `yolov8s` (small): ~22MB, balanced speed/accuracy
  - `yolov8m` (medium): ~52MB, better accuracy
  - `yolov8l` (large): ~87MB, high accuracy
  - `yolov8x` (xlarge): ~136MB, best accuracy, slowest
- Downloads automatically on first run, cached in `~/.cache/ultralytics`
- **IMPORTANT**: Only detects 80 COCO classes - cannot detect arbitrary objects like CLIP
- Available classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
- Configure labels in config.yaml - invalid labels are filtered out with warnings

**face_recognition (Face Recognition)**:
- Uses dlib's ResNet-based face encoder
- Requires system libraries: cmake, libboost, libdlib (installed in Dockerfile)
- Face encodings are 128-dimensional vectors

### Error Handling

The application is designed to be resilient:
- Failed video downloads are logged but don't stop the pipeline
- Videos that fail AI analysis are logged with full stack traces
- Main loop catches all exceptions and retries after 5 minutes
- Ring token expiration triggers re-authentication flow

### Logging Strategy

All modules use Python's `logging` module with consistent formatting:
- Timestamp - Module - Level - Message
- DEBUG: Frame-by-frame detection details
- INFO: Pipeline progress, video processing, sorting decisions
- WARNING: Missing data, skipped videos, no detections
- ERROR: Authentication failures, file I/O errors, AI inference errors

When making changes, maintain this logging level consistency.

## Development Guidelines

### Adding New Object Labels
**IMPORTANT**: YOLOv8 only detects the 80 COCO classes listed above. You cannot add arbitrary labels.

To track new objects:
1. Check if the object matches a COCO class (see list in AI Models section)
2. Edit `config/config.yaml` and add the COCO class name to `detection.object_labels`
3. Restart the container - invalid labels will be filtered out with warnings

For objects not in COCO (e.g., hedgehog, fox):
- Use the closest available class (e.g., "cat" for small mammals, "bird" for all birds)
- Consider contributing to custom YOLO training (advanced - not covered here)

### Adding New Person Recognition
1. Create directory: `/mnt/user/appdata/crittercatcher/faces/training/<PersonName>`
2. Add 3-5 clear face photos (JPG format)
3. Run training command (see Commands section above)

### Modifying Detection Logic
The sorting priority is controlled in `video_sorter.py::sort_video()`. The decision tree is:
1. If `priority="people"` and people detected → sort by person
2. If objects detected → sort by best object
3. If people detected (fallback) → sort by person
4. Else → sort to "unknown"

### GPU Support
Uncomment the `deploy` section in `docker-compose.yml` to enable NVIDIA GPU support. Requires nvidia-docker runtime.

### Testing Changes
Since there's no test framework, manual testing workflow:
1. Make code changes
2. Rebuild container: `docker-compose up -d --build`
3. Run once: `docker-compose run --rm crittercatcher python src/main.py`
4. Check logs for errors
5. Verify video sorting in `/mnt/user/Videos/CritterCatcher/sorted/`

### Dependencies
When adding Python dependencies:
1. Add to `requirements.txt`
2. Rebuild Docker image
3. Be mindful of ARM compatibility (Unraid may run on ARM systems)

### Common Pitfalls
- **Invalid YOLO labels**: Check logs for warnings about invalid labels. Only COCO classes work.
- **Specific animals not detected**: YOLO only has "bird", "cat", "dog" for wildlife - cannot detect hedgehogs, foxes, etc. specifically
- Ring token expires: Delete token file and restart container
- Low detection confidence: Lower `confidence_threshold` in config (try 0.25-0.35 for YOLO)
- Face recognition fails: Add more training photos or increase `face_tolerance`
- Videos not downloading: Check Ring credentials and subscription status
- Path errors on Windows: Use forward slashes or raw strings for paths
- Model download fails: Check internet connection and disk space in `~/.cache/ultralytics`

### Windows Development Notes
This repository is currently on Windows (`C:\Coding\CritterCatcherAI`). When working with Docker:
- Docker Desktop must be running
- Volume paths in docker-compose.yml use Linux-style paths (inside container)
- Use PowerShell or WSL for command-line operations
- Git will convert line endings (CRLF ↔ LF) - this is normal
