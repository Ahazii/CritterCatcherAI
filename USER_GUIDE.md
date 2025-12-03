# CritterCatcherAI User Guide

**Version 2.0 - Hybrid YOLO-First Workflow**  
**Last Updated:** December 2, 2025

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Ring Authentication](#ring-authentication)
3. [YOLO Categories Management](#yolo-categories-management)
4. [Video Review Workflow](#video-review-workflow)
5. [CLIP Profiles (Optional)](#clip-profiles-optional)
6. [Face Recognition](#face-recognition)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [YOLO Categories Reference](#yolo-categories-reference)

---

## Getting Started

### First Time Setup

1. **Access Web Interface**
   - Open browser: `http://YOUR_SERVER_IP:8080`
   - You'll see the Dashboard

2. **Authenticate with Ring**
   - Navigate to **Ring Setup** tab
   - Enter your Ring email and password
   - If 2FA is enabled:
     - Click "Authenticate with Ring"
     - You'll see: "✓ Ring has sent a verification code!"
     - Check your phone/email for 6-digit code
     - Enter code and click "Submit 2FA Code"
   - Token is saved automatically

3. **Enable YOLO Categories**
   - Go to **YOLO Categories** tab
   - Select categories to monitor (bird, cat, dog, person, car, etc.)
   - Click "Select All" for maximum coverage
   - Categories are saved automatically

4. **Start Processing**
   - Click **Process Now** on Dashboard
   - System downloads videos from Ring
   - Videos are analyzed and sorted by category
   - Check **Review** tab to see results

---

## Ring Authentication

### Web Interface Method (Recommended)

The web UI makes Ring authentication simple:

**Initial Setup:**
1. Go to **Ring Setup** tab
2. Enter your Ring credentials
3. Click "Authenticate with Ring"

**If 2FA is Enabled:**
- After clicking authenticate, you'll see a green success message
- A 2FA input field appears
- Enter the 6-digit code from your phone/email
- Click "Submit 2FA Code"

**Token Storage:**
- Token saved to `/data/tokens/ring_token.json`
- Persists across container restarts
- Only authenticate once unless token expires

### Token Expiration

If your Ring token expires:
- You'll see authentication errors in logs
- Old token is automatically deleted
- Follow the "Initial Setup" steps again

For more details, see [RING_2FA_SETUP.md](RING_2FA_SETUP.md)

---

## YOLO Categories Management

### Understanding YOLO Categories

YOLO Stage 1 detects 80 object classes from the COCO dataset. You choose which categories to monitor.

**How it works:**
1. System downloads videos from Ring cameras
2. YOLO analyzes videos and detects objects
3. Videos are sorted to `/data/review/{category}/` folders
4. Only enabled categories are monitored

### Enabling Categories

**Via Web Interface:**
1. Go to **YOLO Categories** tab
2. You'll see organized groups:
   - **Animals** (bird, cat, dog, horse, etc.)
   - **Vehicles** (car, motorcycle, bus, etc.)
   - **People** (person)
   - **Outdoor Objects** (bench, fire hydrant, etc.)
   - And more...

3. **Category States:**
   - 🔒 **Locked (purple)** - Used by CLIP profile, cannot disable
   - ✅ **Enabled (green toggle)** - Manually enabled
   - ⬜ **Disabled (gray toggle)** - Not monitored

4. **Batch Actions:**
   - Click "✅ Select All" to enable all categories
   - Click "❌ Deselect All" to disable all (except locked)

### YOLO Detection Settings

**Confidence Threshold** (in Configuration tab):
- **0.15-0.20** - Very sensitive (many detections, more false positives)
- **0.25-0.35** - ✅ Recommended (balanced)
- **0.40-0.50** - Conservative (fewer detections, high accuracy)

**Object Detection Frames:**
- Number of video frames to analyze
- **5 frames** - ✅ Recommended (fast, sufficient)
- **10-15 frames** - More thorough, slower

---

## Video Review Workflow

### Review Interface

After processing, videos are organized in `/data/review/{category}/` folders.

**Accessing Review:**
1. Go to **Review** tab
2. You'll see category tabs: `car (17)`, `dog (3)`, `bird (5)`, `person (26)`
3. Click a tab to view videos in that category

### Tracked Videos with Bounding Boxes

All videos in review include **animated tracking**:
- Green bounding boxes around detected objects
- Persistent track IDs (e.g., "dog #1", "person #2")
- Confidence scores displayed
- Boxes follow objects through frames

**Video Files:**
- **Original:** `/data/review/{category}/video.mp4`
- **Tracked:** `/data/objects/detected/annotated_videos/tracked_video.mp4`
- Review tab automatically shows tracked version with bounding boxes

### Video Actions

**Bulk Selection:**
- **Single click** - Select one video
- **Shift+click** - Select range
- **"Select All"** button - Select all videos in category

**Actions:**
- **✓ Confirm Videos** 
  - Moves original to `/data/sorted/{category}/`
  - Deletes tracked version (no longer needed)
  - Video remains in sorted folder permanently

- **✗ Reject Videos**
  - Deletes both original and tracked versions
  - Marks as 'rejected' in download tracker
  - Won't be downloaded again

- **Or assign to:** (Dropdown)
  - Assign to specific CLIP profile
  - Moves to `/data/sorted/{profile}/`

### Video Metadata

Each video shows:
- **YOLO Category** - What YOLO detected (e.g., "car 81%")
- **All Detections** - All objects found (e.g., "potted plant: 82%, person: 86%, car: 91%")
- **CLIP Match** - If CLIP profile ran (e.g., "Profile ID hedgehog (78%)")
- **Status** - "Pending Review" or "CLIP Sorted"
- **File Size** - Video file size

---

## CLIP Profiles (Optional)

### What are CLIP Profiles?

CLIP Stage 2 provides **fine-grained classification** beyond YOLO's 80 categories.

**Example Use Cases:**
- YOLO detects "bird" → CLIP identifies specific species (finch, robin, hawk)
- YOLO detects "dog" → CLIP identifies breeds (Jack Russell, Doberman)
- YOLO detects "cat" → CLIP identifies specific animals (hedgehog, fox)

### Creating a CLIP Profile

**Coming Soon** - CLIP Profiles tab is under development.

**Current Workaround:**
Edit `/config/config.yaml` manually:

```yaml
animal_profiles:
  - name: hedgehog
    yolo_categories: [cat, dog]  # Which YOLO categories to check
    text_description: "a small hedgehog with spiky brown fur"
    confidence_threshold: 0.75    # 75% confidence required
    auto_approval: true           # Auto-move to sorted if ≥threshold
```

### How CLIP Profiles Work

1. YOLO detects video with "dog" (Stage 1)
2. Video sorted to `/data/review/dog/`
3. System checks: Is there a CLIP profile for "dog"?
4. YES → Run CLIP classification
5. **If CLIP confidence ≥ 0.75:**
   - Move to `/data/sorted/{profile}/`
   - Mark as auto-approved
6. **If CLIP confidence < 0.75:**
   - Stay in `/data/review/dog/`
   - User reviews manually

---

## Face Recognition

### Setup Face Recognition

Face recognition allows identifying specific people automatically.

**Workflow:**
1. **Enable person detection** in YOLO Categories
2. **Process person videos** - they go to `/data/review/person/`
3. **Extract faces** from confirmed videos
4. **Assign faces** to person names
5. **Future videos** automatically recognize people

### Face Training Tab

1. **Confirm Person Videos**
   - In Review tab, select person videos
   - Click "Confirm as Person" button
   - System extracts faces from video
   - Faces saved to `/data/training/faces/unassigned/`

2. **Assign Faces to Names**
   - Go to **Face Training** tab
   - View unassigned face images
   - Select faces of same person
   - Enter person name (e.g., "John", "Claire")
   - Click "Assign to Person"
   - System retrains face recognition automatically

3. **Future Processing**
   - New person videos are analyzed
   - Recognized faces are labeled
   - Videos routed to appropriate destinations

### Face Recognition Settings

**Face Tolerance** (in Configuration tab):
- **0.3-0.4** - Strict matching (fewer false positives)
- **0.5-0.7** - ✅ Recommended (balanced)
- **0.8-0.9** - Lenient (more false positives)

**Face Recognition Frames:**
- Number of frames to analyze for faces
- **10-15** - ✅ Recommended
- **20-30** - More thorough, slower

---

## Configuration

### Configuration Tab

Most settings are accessible via **Configuration** tab in web UI:

**Object Detection:**
- Confidence Threshold (slider: 0.1-0.9)
- Object Detection Frames (slider: 3-20)

**Image Review:**
- Auto-Confirm Threshold (slider: 0.50-1.00)
  - Detections ≥ this threshold are auto-confirmed
  - Bypass manual review
  - ✅ Recommended: 0.80-0.90
  
- Max Confirmed Images per Label (slider: 50-1000)
  - Maximum confirmed images kept per label
  - Older images auto-deleted when limit exceeded
  - ✅ Recommended: 100-300

**Scheduler:**
- Enable Auto Run (checkbox)
- Run Interval (dropdown: 15 min to 24 hours)

**Face Recognition:**
- Enable Face Recognition (checkbox)
- Face Tolerance (slider: 0.3-0.9)
- Face Recognition Frames (slider: 5-30)

### Manual Configuration

Advanced users can edit `/config/config.yaml` directly:

```yaml
paths:
  downloads: /data/downloads
  sorted: /data/sorted
  face_encodings: /data/faces/encodings.pkl

ring:
  download_hours: 24        # Hours back to download
  download_limit: null      # Max videos (null = unlimited)

detection:
  confidence_threshold: 0.25
  object_frames: 5
  face_tolerance: 0.6
  face_frames: 10
  face_model: hog           # or 'cnn' (slower, more accurate)
  priority: people          # or 'objects'
  yolo_model: yolov8n       # n/s/m/l/x (size/speed tradeoff)

scheduler:
  auto_run: true
  interval_minutes: 60

image_review:
  auto_confirm_threshold: 0.85
  max_confirmed_images: 200

tracking:
  enabled: true
  save_original_videos: false

yolo_manual_categories:    # Manually enabled categories
  - bird
  - cat
  - dog
  - person
  - car

face_recognition:
  enabled: true

animal_profiles: []         # CLIP profiles (optional)
```

**After editing:** Restart container to apply changes

---

## Troubleshooting

### Ring Issues

**Authentication Failed:**
- Double-check credentials in Ring Setup tab
- Ensure Ring subscription is active
- Delete token file: `/data/tokens/ring_token.json` and re-authenticate

**No Videos Downloaded:**
- Check Ring app - are there videos in the timeframe?
- Default: last 24 hours
- Increase `ring.download_hours` in config
- Check Docker logs: `docker logs CritterCatcherAI`

**2FA Code Rejected:**
- Codes expire quickly - request a new one
- Copy/paste code directly from email/SMS
- Ensure no spaces in code

### Detection Issues

**No Detections in Videos:**
- Lower confidence threshold to 0.20-0.25
- Enable more YOLO categories
- Check logs for processing errors
- Verify videos are in `/data/downloads/`

**Too Many False Positives:**
- Raise confidence threshold to 0.35-0.45
- Increase auto-confirm threshold
- Reject false positives to improve tracking

**YOLO Categories Not Persisting:**
- Check if `/config/` volume is mounted correctly
- Verify file permissions on config directory
- Check logs for "Saved X categories" messages

### Video Playback Issues

**Videos Show Black in Browser:**
- System automatically converts mp4v to H.264
- Check logs for "Successfully converted to H.264"
- Try different browser (Chrome/Firefox recommended)
- Check browser console (F12) for errors

**Tracked Videos Missing:**
- Verify video tracking is enabled in config
- Check `/data/objects/detected/annotated_videos/` folder
- Check logs for tracking errors
- Ensure `lap` module is installed (should be automatic)

**Videos Have No Bounding Boxes:**
- Check that correct video is being served
- Tracked videos should be named `tracked_{filename}.mp4`
- Original videos don't have boxes

### Performance Issues

**High CPU Usage:**
- Normal during video processing
- 60-100% CPU expected
- Enable GPU support for faster processing
- Reduce `object_frames` to 3-5

**Out of Memory:**
- Reduce Docker memory limit
- Process fewer videos at once
- Close other applications

**Slow Processing:**
- Enable GPU support (3-5x faster)
- Use smaller YOLO model (`yolov8n` instead of `yolov8m`)
- Reduce frames analyzed per video
- Increase processing interval

---

## YOLO Categories Reference

### All 80 COCO Classes

**Animals (11):**
bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles (8):**
bicycle, car, motorcycle, airplane, bus, train, truck, boat

**People (1):**
person

**Outdoor Objects (5):**
traffic light, fire hydrant, stop sign, parking meter, bench

**Sports & Recreation (10):**
frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Indoor Objects (24):**
bottle, wine glass, cup, fork, knife, spoon, bowl, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, book, clock, vase, scissors, teddy bear

**Electronics & Appliances (6):**
microwave, oven, toaster, sink, refrigerator, hair drier, toothbrush

**Personal Items (5):**
backpack, umbrella, handbag, tie, suitcase

**Food (10):**
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

---

## Tips & Best Practices

### General Usage
- Start with broad YOLO categories, narrow down with CLIP profiles later
- Use "Select All" initially to see what your cameras capture
- Review and confirm/reject regularly to keep system accurate
- Check logs periodically for errors or issues

### Detection Accuracy
- Lower confidence = more detections but more false positives
- Higher confidence = fewer detections but more accurate
- Adjust based on your specific cameras and environment
- Different cameras may need different thresholds

### Storage Management
- Confirmed videos go to `/data/sorted/` - monitor disk space
- Rejected videos are deleted - won't be downloaded again
- Tracked videos deleted after confirmation (originals kept)
- Set `max_confirmed_images` to prevent unlimited growth

### Performance
- Process during off-peak hours for better performance
- Enable GPU if available for 3-5x speedup
- Reduce `object_frames` if processing is too slow
- Use `yolov8n` model for fastest processing

---

## Quick Reference

### Common Commands

```bash
# View logs
docker logs -f CritterCatcherAI

# Restart container
docker restart CritterCatcherAI

# Check disk space
docker exec CritterCatcherAI df -h /data

# Access container shell
docker exec -it CritterCatcherAI /bin/bash
```

### File Locations

| Path | Description |
|------|-------------|
| `/config/config.yaml` | Main configuration |
| `/data/downloads/` | Temporary video downloads |
| `/data/review/{category}/` | YOLO-sorted videos pending review |
| `/data/sorted/{category}/` | Confirmed videos |
| `/data/objects/detected/annotated_videos/` | Tracked videos with bounding boxes |
| `/data/tokens/ring_token.json` | Ring authentication token |
| `/data/training/faces/` | Face recognition training data |

### Default Settings

| Setting | Default | Recommended |
|---------|---------|-------------|
| Confidence Threshold | 0.25 | 0.25-0.35 |
| Object Frames | 5 | 5-10 |
| Auto-Confirm Threshold | 0.85 | 0.80-0.90 |
| Max Confirmed Images | 200 | 100-300 |
| Process Interval | 60 min | 30-120 min |
| Face Tolerance | 0.6 | 0.5-0.7 |

---

## Support

- **Documentation:** [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md)
- **GitHub Issues:** https://github.com/Ahazii/CritterCatcherAI/issues
- **Docker Logs:** `docker logs -f CritterCatcherAI`
- **Config File:** `/config/config.yaml`

---

**Happy Monitoring!** 🎥🦔🐦🚗👤
