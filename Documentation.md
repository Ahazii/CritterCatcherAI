Documentation

CritterCatcherAI is an AI-powered video sorting system for Ring cameras. It
downloads recent videos, detects objects with YOLO, optionally refines detections
with CLIP profiles, and organizes results into review and sorted folders.

Overview
- Stage 1: YOLO detection for broad categories (dog, person, car, etc.).
- Stage 2: CLIP profiles for finer identification (optional).
- Face recognition: matches known faces when person is detected (optional).
- Web UI: configuration, review, and status monitoring.

Quick Start
1) Clone and start container:
   - docker-compose up -d
2) Open the UI:
   - http://YOUR_SERVER_IP:8080
3) Authenticate Ring in the Ring Setup tab (2FA supported).
4) Enable YOLO categories and run processing.

Workflow
1) Download recent Ring videos to /data/downloads.
2) Run YOLO detection on a sample of frames.
3) Sort video to /data/review/{category}/ and create tracked video.
4) If a CLIP profile matches a detected category:
   - Score frames against profile text description.
   - High confidence can auto-approve into /data/sorted/{profile}/.
5) Optional face recognition matches people and routes to /data/sorted/person/{name}/.

Directory Structure
/data/
  downloads/                        Raw videos (temporary)
  review/{category}/                YOLO-sorted videos for review
  sorted/{category|profile}/        Confirmed or auto-approved videos
  objects/detected/annotated_videos Tracked videos with bounding boxes
  training/                         CLIP/face training data
  tokens/                           Ring refresh token
  animal_profiles/                  CLIP profile JSON definitions

/config/
  config.yaml                        Runtime configuration

Configuration
Primary configuration is in /config/config.yaml. Environment variables can
override selected settings.

Key settings:
- detection.confidence_threshold
- detection.object_frames
- detection.yolo_model
- ring.download_hours
- ring.download_limit
- scheduler.auto_run
- scheduler.interval_minutes

Logging
- Default: stdout/stderr and /config/crittercatcher.log
- Disable file logging: logging.file.enabled: false or LOG_TO_FILE=false

Animal Profiles (CLIP)
Profiles are stored in /data/animal_profiles/*.json and are created via the API
or UI. config.yaml profiles are not automatically loaded.

Minimum fields:
- name
- yolo_categories (must include the YOLO class that triggers the profile)
- text_description
- confidence_threshold

API Reference (selected)
GET /api/status
GET /api/status/stream (SSE)
GET /api/config
POST /api/config/save
GET /api/yolo-categories
POST /api/yolo-categories/manual/toggle
GET /api/review/categories
GET /api/review/categories/{category}/videos
POST /api/review/confirm
POST /api/review/reject
POST /api/process
POST /api/stop
POST /api/ring/authenticate

Docker
Dockerfile uses a CUDA runtime base image for GPU support. For CPU-only hosts,
ensure your environment supports the CUDA image or provide a CPU-only build.

Testing
Automated tests are available.
- docker-compose up -d --build
- docker exec -it crittercatcher-ai pytest

Troubleshooting
- No detections: lower detection.confidence_threshold and increase object_frames.
- No CLIP matches: ensure profile exists in /data/animal_profiles and includes
  the detected YOLO category, and lower confidence_threshold if needed.
- Missing videos: check Ring auth, logs, and download_hours window.
