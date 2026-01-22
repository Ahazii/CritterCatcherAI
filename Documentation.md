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
- detection.force_cpu
- ring.download_hours
- ring.download_limit
- scheduler.auto_run
- scheduler.interval_minutes
- animal_training.enabled
- animal_training.batch_size
- animal_training.min_negatives

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

Animal Training
- Training uses a lightweight classifier on CLIP embeddings.
- Triggered automatically after every 10 new confirmed positives.
- Requires negative samples from rejected images.
- Classifier stored at /data/models/{profile_id}/classifier.json.
- Falls back to text-based CLIP scoring if no classifier exists.

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
The application uses the GPU automatically when available and falls back to CPU
when not. The container can run on CPU-only hosts without changes.

Deployment essentials:
- Ports: 8080
- Volumes: /config, /data
- Build args: PUID, PGID
- Env: LOG_LEVEL, RUN_ONCE, TZ, LOG_TO_FILE

Example docker-compose:
version: "3.8"
services:
  crittercatcher:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./config:/config
      - ./data:/data
    environment:
      - LOG_LEVEL=INFO
      - RUN_ONCE=false
      - TZ=UTC

Optional GPU (NVIDIA):
- Ensure the NVIDIA container runtime is installed.
- Enable device reservations for GPU if desired. Without this, the container
  runs in CPU mode even on a GPU host.

Force CPU:
- Set detection.force_cpu: true in /config/config.yaml or toggle it in the UI.

Testing
Automated tests are available.
- docker-compose up -d --build
- docker exec -it crittercatcher-ai pytest

Troubleshooting
- No detections: lower detection.confidence_threshold and increase object_frames.
- No CLIP matches: ensure profile exists in /data/animal_profiles and includes
  the detected YOLO category, and lower confidence_threshold if needed.
- Missing videos: check Ring auth, logs, and download_hours window.

Supported YOLO Categories
- Animals: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- Vehicles: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- People: person
- Outdoor: traffic light, fire hydrant, stop sign, parking meter, bench
