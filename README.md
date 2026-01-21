# CritterCatcherAI

**Intelligent video sorting system for Ring cameras using AI object detection and facial recognition**

CritterCatcherAI automatically downloads videos from your Ring doorbell/cameras, analyzes them using a two-stage AI pipeline (YOLO + CLIP), and organizes them by detected subjects. Perfect for wildlife monitoring, security surveillance, and smart home automation.

---

## 🎯 Key Features

### Two-Stage Detection Pipeline
- **Stage 1: YOLO Detection** - Fast, broad categorization (80 object classes)
- **Stage 2: CLIP Profiles** - Precise identification of specific animals/objects
- **Face Recognition** - Identify specific people automatically

### Hybrid Workflow
- Videos sorted by YOLO category first → `/data/review/{category}/`
- Optional CLIP refinement moves high-confidence matches → `/data/sorted/{profile}/`
- Review interface shows tracked videos with animated bounding boxes
- Manual confirmation improves accuracy over time

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
- Confirm or reject videos

### 3. Create CLIP Profiles (Optional)
For fine-grained detection (e.g., specific bird species):
- Go to **CLIP Profiles** tab (coming soon)
- Create profile for target animal
- Upload training images
- High-confidence matches auto-sort to profile folders

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
detection:
  confidence_threshold: 0.25      # YOLO detection sensitivity (0.1-0.9)
  object_frames: 5                # Frames to analyze per video

scheduler:
  auto_run: true                  # Enable automatic processing
  interval_minutes: 60            # Check for new videos every hour

image_review:
  auto_confirm_threshold: 0.85    # Auto-confirm high confidence (≥85%)
  max_confirmed_images: 200       # Max training images per label
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

### Manual Processing
Click **Process Now** button on Dashboard to trigger immediate video download and processing

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

### Videos Show Black in Browser
- Container automatically converts videos to H.264 for browser compatibility
- Check logs for "Successfully converted to H.264" messages
- Try different browser (Chrome/Firefox recommended)

### YOLO Categories Not Persisting
- Categories are saved to `/config/config.yaml`
- Ensure config volume is mounted correctly
- Check logs for save/load confirmation messages

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
**[Full list in USER_GUIDE.md](USER_GUIDE.md#yolo-categories)**

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
