# CritterCatcherAI

**Intelligent video sorting system for Ring cameras using AI object detection and facial recognition**

CritterCatcherAI automatically downloads videos from your Ring doorbell/cameras, analyzes them using a two-stage AI pipeline (YOLO + CLIP), and organizes them by detected subjects. Perfect for wildlife monitoring, security surveillance, and smart home automation.

---

## 🎯 Key Features

### Two-Stage Detection Pipeline
- **Stage 1: YOLO Detection** - Fast, broad categorization (80 object classes)
- **Stage 2: CLIP Profiles** - Precise identification of specific animals/objects
- **Face Recognition** - Identify specific people automatically
- **Pathways** - Camera-scoped media/security routing

### Hybrid Workflow
- Videos sorted by YOLO category first → `/data/review/{category}/`
- Optional CLIP refinement moves high-confidence matches → `/data/sorted/{profile}/`
- Review interface shows tracked videos with animated bounding boxes
- Manual confirmation improves accuracy over time
- Camera-scoped pathways can auto-route media/security outputs

### Smart Video Processing
- Automatic video downloads from Ring cameras
- Object tracking with persistent IDs across frames
- Browser-compatible H.264 encoding
- Configurable confidence thresholds
- Multi-select review interface

---

## 🚀 Quick Start

### Prerequisites
- Unraid server (or Docker host)
- Ring account with video subscription
- 8GB+ RAM, 4+ CPU cores recommended

### Installation

1. **Clone repository:**
   ```bash
   cd /mnt/user/appdata
   git clone https://github.com/Ahazii/CritterCatcherAI.git crittercatcher
   cd crittercatcher
   ```

2. **Deploy container:**
   ```bash
   docker-compose up -d
   ```

3. **Access web interface:**
   Open `http://YOUR_SERVER_IP:8080`

4. **Authenticate with Ring:**
   - Go to **Ring Setup** tab
   - Enter credentials
   - Complete 2FA if prompted
   - System will start downloading videos automatically

---

## 📖 Basic Usage

### 1. Enable YOLO Categories
Go to **YOLO Categories** tab and select which objects to detect:
- Birds, cats, dogs, people, cars, etc.
- Use "Select All" for maximum coverage
- Categories persist across restarts

### 2. Review Detected Videos
Videos are automatically sorted to `/data/review/{category}/`:
- Navigate to **Review** tab
- View videos by category (car, dog, bird, person, etc.)
- Tracked videos show animated bounding boxes

**📋 Queue System (Batch Processing):**
- Select videos and queue multiple actions before executing
- Queue summary shows total actions and breakdown by type
- Click "Run Queue" to batch process all queued actions at once
- Videos show "⏳ Queued" badge with opacity indicator
- Click queued video to remove from queue

**Quick Actions (Simple Batch Operations):**
- **✓ Confirm Videos** - Move to sorted folder (for person videos with profile selected, extracts face training)
- **✗ Reject Videos** - Delete videos (optionally save as negative training examples)
- **→ Assign to Profile** - Extract frames as positive OR negative (all videos treated the same)
  - For person videos: automatically uses face profiles
  - For animal videos: uses animal profiles and supports positive/negative/both

**🎯 Advanced Review (Per-Video Control):**
- Click **Advanced Review** button for granular control over each video
- Configure each video independently:
  - Choose different profiles for different videos
  - Extract positive training frames ("Yes, this IS a hedgehog")
  - Extract negative training frames ("No, this is NOT a hedgehog")
  - Both positive AND negative from the same video (edge cases)
  - Move to sorted folder or delete after processing
- **Perfect for "unknown" category** where you need to manually identify and sort videos
- Quick action buttons to apply same settings to all videos at once

**Understanding Training Data:**
- **Positive examples** - Frames containing your target animal → `/data/training/{profile}/confirmed/`
- **Negative examples** - Frames NOT containing your target → `/data/training/{profile}/rejected/`
- **CLIP requires both** - At least 10 positive AND 10 negative examples for effective training
- The more examples you provide, the better CLIP becomes at identifying your target animals

### 3. Create CLIP Profiles (Optional)
For fine-grained detection (e.g., specific bird species):
- Go to **CLIP Profiles** tab (coming soon)
- Create profile for target animal
- Upload training images
- High-confidence matches auto-sort to profile folders
 - Training runs automatically in batches of 10 confirmed positives (requires negatives)

### 4. Setup Face Recognition (Optional)
- Go to **Face Training** tab
- Confirm person videos to extract faces
- Assign faces to people names
- Future videos automatically identify people

---

## 📁 Directory Structure

```
/data/
├── downloads/              # Temporary Ring video downloads
├── review/                 # YOLO-sorted videos pending review
│   ├── car/               # All car detections
│   ├── dog/               # All dog detections
│   ├── bird/              # All bird detections
│   └── person/            # All person detections
├── sorted/                # Confirmed/auto-approved videos
│   ├── {profile}/         # CLIP profile matches
│   └── {person_name}/     # Face recognition matches
└── objects/detected/
    └── annotated_videos/  # Tracked videos with bounding boxes
```

---

## ⚙️ Configuration

### Web Interface (Recommended)
Most settings configurable via **Configuration** tab:
- YOLO categories to monitor
- Confidence thresholds
- Auto-approval settings
- Processing schedule

### Manual Config
Advanced users can edit `/config/config.yaml`:

```yaml
# Manual YOLO categories (managed via YOLO Categories page)
yolo_manual_categories:
  - person                        # Enable person detection

# Face Recognition settings
face_recognition:
  enabled: true                   # Enable face recognition for person detection

detection:
  confidence_threshold: 0.25      # YOLO detection sensitivity (0.1-0.9)
  object_frames: 5                # Frames to analyze per video (YOLO)
  face_frames: 5                  # Frames to analyze per video (face recognition)
  face_tolerance: 0.7             # Face matching tolerance (lower = stricter)
  force_cpu: false                # Force CPU even if GPU is available
  face_model: hog                 # Face detection: "hog" (fast CPU) or "cnn" (slow GPU)
  yolo_model: yolov8n             # YOLO model: yolov8n (nano, fastest) to yolov8x (extra large, most accurate)

tracking:
  generate_tracked_videos: true   # Generate videos with bounding boxes (disable for 5-10x speedup)

scheduler:
  auto_run: true                  # Enable automatic processing
  interval_minutes: 60            # Check for new videos every hour

logging:
  level: INFO                     # DEBUG, INFO, WARNING, ERROR, CRITICAL
  gpu_monitoring:
    enabled: true                 # Monitor GPU usage
    interval_seconds: 5           # Log interval
    log_on_idle: false            # Skip logging when GPU idle
    max_scale_percent: 10         # UI color scale maximum

image_review:
  auto_confirm_threshold: 0.85    # Auto-confirm high confidence (≥85%)
  max_confirmed_images: 200       # Max training images per label

pathways:
  media:
    enabled: true                 # Media pathway for CLIP profiles
    cameras: []                   # Empty = all cameras
    profiles: []                  # Empty = all profiles
  security:
    enabled: true                 # Security pathway for unknown people
    cameras: []                   # Empty = all cameras
    unknown_only: true
```

---

## 🔧 Common Tasks

### Adjust Detection Sensitivity
**More detections (may include false positives):**
- Lower confidence threshold to 0.15-0.20

**Fewer, more accurate detections:**
- Raise confidence threshold to 0.35-0.45

### Enable GPU Acceleration (Optional)
Uncomment GPU section in `docker-compose.yml` for 3-5x faster processing

### Force CPU (Optional)
Use the Configuration tab or set `detection.force_cpu: true` in config to disable GPU use.

### Manual Processing
Click **Process Now** button on Dashboard to trigger immediate video download and processing

### Camera-Scoped Pathways
Use Config → Pathways to choose which cameras feed:
- Media pathway (CLIP profile auto-saves)
- Security pathway (unknown people)

### Performance Optimization

**For Maximum Speed (5-10x faster):**
```yaml
detection:
  face_model: hog              # Use CPU-based face detection (vs cnn)
  face_frames: 5               # Reduce frames analyzed
  object_frames: 5             # Reduce YOLO frames
tracking:
  generate_tracked_videos: false  # Skip video re-encoding (no bounding box videos)
```

**GPU Acceleration:**
- YOLO detection: Automatically uses GPU if available (3-5x faster)
- Video encoding: Falls back to CPU if NVENC not available
- Face detection with CNN: Requires proper CUDA/cuDNN setup (otherwise very slow)

**Recommended Settings:**
- Use `face_model: hog` (CPU) - Fast and accurate enough for most use cases
- Keep `face_frames: 5` - Good balance of speed and accuracy  
- Enable `generate_tracked_videos: true` - Bounding box videos are useful for review

---

## 🐛 Troubleshooting

### Ring Authentication Failed
1. Check credentials in Ring Setup tab
2. Complete 2FA verification
3. Token is saved and persists across restarts

### No Videos Being Downloaded
- Verify Ring subscription is active
- Check that cameras have recorded videos in timeframe (default: 24 hours)
- Check Docker logs: `docker logs CritterCatcherAI`
- Duplicate downloads prevented by database tracking (`/data/download_history.db`)

### People Not Being Detected
- Enable face recognition: Set `face_recognition.enabled: true` in config
- Add "person" to YOLO categories via YOLO Categories page
- Unknown people saved to `/data/sorted/security/unknown/`
- Train faces in Face Training tab for recognition

### GPU Indicator Not Working
- Monitor recovers automatically from temporary errors (max 5 retries)
- Check logs for "GPU monitoring" messages
- Logs show GPU usage for each AI operation (YOLO, CLIP, Face Recognition)
- Indicator updates every 2 seconds in UI

### Videos Show Black in Browser
- Container automatically converts videos to H.264 for browser compatibility
- Check logs for "Successfully converted to H.264" messages
- Try different browser (Chrome/Firefox recommended)

### YOLO Categories Not Persisting
- Categories are saved to `/config/config.yaml`
- Ensure config volume is mounted correctly
- Check logs for save/load confirmation messages

### Training Not Happening
- Ensure confirmed/rejected JPG frames exist in `/data/training/{profile}/`
- Use “Assign to profile” and “Reject as negatives” in Review

---

## 📚 Documentation

- **[Documentation.md](Documentation.md)** - Full user and technical documentation

---

## 🎯 Workflow Overview

```
Ring Camera Videos
       ↓
[Download & Track]
       ↓
YOLO Stage 1: Broad Detection
       ↓
Sort by Category → /data/review/{category}/
       ↓
Optional CLIP Stage 2: Precise Classification
       ↓
High Confidence → /data/sorted/{profile}/
Low Confidence → Stays in review for manual confirmation
       ↓
User Reviews & Confirms
       ↓
System learns and improves accuracy
```

---

## 🔍 Supported YOLO Categories (80 COCO Classes)

**Animals:** bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe  
**Vehicles:** bicycle, car, motorcycle, airplane, bus, train, truck, boat  
**People:** person  
**Outdoor:** traffic light, fire hydrant, stop sign, parking meter, bench  
**[Full list in Documentation.md](Documentation.md#supported-yolo-categories)**

---

## 🆘 Support

- **GitHub Issues:** https://github.com/Ahazii/CritterCatcherAI/issues
- **Docker Logs:** `docker logs -f CritterCatcherAI`
- **Config File:** `/mnt/user/appdata/crittercatcher/config/config.yaml`

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

Built with:
- **YOLOv8** by Ultralytics (object detection)
- **CLIP** by OpenAI (zero-shot classification)
- **face_recognition** by Adam Geitgey (facial recognition)
- **ring-doorbell** by tchellomello (Ring API client)

---

**Happy Critter Catching!** 🦔🐦🐶🚗👤
