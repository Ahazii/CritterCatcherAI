# CritterCatcherAI - Unraid Deployment Guide

## Overview
This guide covers deploying CritterCatcherAI v2.0 to Unraid using Docker. The system detects animals in Ring camera videos using two-stage detection (YOLO + CLIP/ViT) and organizes them for review and retraining.

## Prerequisites

### Hardware Requirements
- **CPU**: 4+ cores recommended (2+ minimum)
- **RAM**: 8GB+ recommended (4GB minimum)
- **Storage**: 
  - 20GB+ for Docker image
  - 100GB+ for sorted videos (highly dependent on your setup)
  - 50GB+ for review/training frames buffer
- **GPU** (optional): NVIDIA GPU significantly speeds up inference
  - Currently set up for CPU-only; GPU support requires additional Docker configuration

### Unraid Setup
- Unraid 6.9 or later
- Docker enabled and working
- At least one user share or cache drive with available space

## Installation Steps

### 1. Clone or Download CritterCatcherAI

```bash
cd /mnt/user/appdata
git clone https://github.com/Ahazii/CritterCatcherAI.git crittercatcher
cd crittercatcher
```

Or download as ZIP and extract.

### 2. Configure docker-compose.yml

Edit `docker-compose.yml` and adjust the volume mounts for your setup:

```yaml
volumes:
  - ./config:/app/config
  - /mnt/user/appdata/crittercatcher:/data  # Main data directory
```

**Important**: The `/data` directory must be writable by the container. Unraid automatically handles permissions (PUID=99, PGID=100).

### 3. Build the Docker Image

```bash
cd /mnt/user/appdata/crittercatcher
docker-compose build
```

**Note**: First build takes 5-10 minutes. Subsequent builds are faster due to Docker layer caching.

```bash
# Expected output:
# ... (many dependencies installing)
# Build version: v0.1.0
# Build date: 2025-11-20 01:30:00 UTC
# Successfully tagged crittercatcher-ai:latest
```

### 4. Start the Container

```bash
docker-compose up -d
```

Verify it's running:

```bash
docker ps | grep crittercatcher
docker logs crittercatcher-ai
```

**Expected logs:**
```
2025-11-20 01:31:45 INFO: CritterCatcherAI starting up - Version: v0.1.0, Built: 2025-11-20 01:30:00 UTC
2025-11-20 01:31:46 INFO: Taxonomy tree initialized with 80 root classes
2025-11-20 01:31:47 INFO: Animal profile manager initialized
2025-11-20 01:31:47 INFO: Review manager initialized
2025-11-20 01:31:48 INFO: Starting web interface on http://0.0.0.0:8080
```

### 5. Access the Web Interface

Open your browser and navigate to:
- **Dashboard**: `http://<your-unraid-ip>:8080/`
- **Animal Profiles**: `http://<your-unraid-ip>:8080/static/profiles.html`
- **Review Images**: `http://<your-unraid-ip>:8080/static/review.html`
- **Model Management**: `http://<your-unraid-ip>:8080/static/models.html`

## Configuration

### Environment Variables

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
  - WEB_PORT=8080               # Web interface port
  - PUID=99                      # Unraid default user
  - PGID=100                     # Unraid default group
  - UMASK=0000                   # File permissions (rwxrwxrwx)
  - PYTHONUNBUFFERED=1           # Real-time log output
```

### Resource Limits

Adjust CPU and memory limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'        # CPU cores limit
      memory: 8G       # Memory limit
    reservations:
      cpus: '2'        # Guaranteed CPU cores
      memory: 4G       # Guaranteed memory
```

## Usage Workflow

### 1. Create Animal Profiles

Go to **Animal Profiles** and create a profile for each animal you want to detect:
- **Name**: e.g., "Hedgehog", "Golden Eagle"
- **YOLO Categories**: Select which COCO classes might contain your animal
- **Text Description**: e.g., "A small brown hedgehog with spiky fur"
- **Confidence Threshold**: 0.80 (80%) - frames above this auto-approve
- **Retraining Settings**: Accuracy threshold 85%, retrain after 50 confirmations

### 2. Process Videos

Place Ring camera videos in `/data/downloads/` or use the Ring API to download them. Videos are processed by:
1. **Stage 1 (YOLO)**: Detects objects in YOLO categories you selected
2. **Stage 2 (CLIP/ViT)**: Scores frames against your animal profile description
3. **Organization**: Moves frames to review directory if confidence is below threshold

### 3. Review Frames

Go to **Review Images** to:
- See pending frames for each profile
- Use Ctrl+click for multi-select
- Use Shift+click for range selection
- **Confirm**: Frame contains the animal (moves to training/confirmed)
- **Reject**: False positive (moves to training/rejected)

Your feedback automatically updates accuracy statistics!

### 4. Monitor Model Performance

Go to **Model Management** to:
- See accuracy percentage for each profile
- Monitor confirmed/rejected counts
- View retraining recommendations
- Check training data distribution

### 5. Retrain Models

When the system recommends retraining (or manually trigger it):
- Uses your confirmed/rejected frames as training data
- Fine-tunes the CLIP/ViT model
- Improves accuracy over time

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs crittercatcher-ai

# Check if port 8080 is in use
netstat -an | grep 8080

# Try running with verbose output
docker-compose up (without -d flag)
```

### Web Interface Not Accessible

```bash
# Verify container is running
docker ps

# Check network
docker network inspect crittercatcher_default

# Try accessing from container
docker exec crittercatcher-ai curl http://localhost:8080/
```

### Permission Denied on Data Directory

```bash
# Fix ownership (run from Unraid terminal)
sudo chown -R 99:100 /mnt/user/appdata/crittercatcher
sudo chmod -R 777 /mnt/user/appdata/crittercatcher
```

### Out of Memory

Reduce limits in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 4G    # Reduce from 8G
```

Or reduce model batch sizes (requires code changes).

### High CPU Usage

- This is normal during video processing
- Monitor with `docker stats crittercatcher-ai`
- Expected: 60-100% CPU during processing, <5% at idle

## API Reference

### Core Endpoints

```bash
# Get all animal profiles
curl http://localhost:8080/api/animal-profiles

# Create a profile
curl -X POST http://localhost:8080/api/animal-profiles \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Hedgehog",
    "yolo_categories": ["cat", "dog"],
    "text_description": "small brown hedgehog"
  }'

# Get pending reviews
curl http://localhost:8080/api/animal-profiles/{id}/pending-reviews

# Get a frame image
curl http://localhost:8080/api/animal-profiles/{id}/frame/frame_000001.jpg -o frame.jpg

# Confirm frames
curl -X POST http://localhost:8080/api/animal-profiles/{id}/confirm-images \
  -H "Content-Type: application/json" \
  -d '{"filenames": ["frame_000001.jpg", "frame_000002.jpg"]}'

# Reject frames
curl -X POST http://localhost:8080/api/animal-profiles/{id}/reject-images \
  -H "Content-Type: application/json" \
  -d '{"filenames": ["frame_000003.jpg"], "save_as_negative": true}'

# Get model statistics
curl http://localhost:8080/api/animal-profiles/{id}/model-stats

# Trigger retraining
curl -X POST http://localhost:8080/api/animal-profiles/{id}/retrain
```

## Performance Notes

### Expected Performance

- **YOLO Detection**: ~30-50ms per frame (CPU)
- **CLIP Scoring**: ~100-150ms per frame (CPU)
- **Total Throughput**: 3-5 frames/second on 4-core CPU
- **30-second video**: ~2 minutes to process (both stages)

### With GPU
- YOLO: ~5-10ms per frame
- CLIP: ~20-30ms per frame
- **Total**: 15-20 frames/second

### Optimization Tips
1. **Enable GPU** if available (NVIDIA GPUs recommended)
2. **Adjust confidence thresholds** - higher = fewer reviews
3. **Use specific YOLO categories** - reduces false positives
4. **Process in batches** - wait for off-peak hours

## Data Persistence

### Directory Structure
```
/data/
├── animal_profiles/        # Profile definitions (.json files)
├── sorted/                 # Final approved videos
├── review/                 # Pending review frames
├── training/
│   ├── confirmed/          # Training data (positives)
│   └── rejected/           # Training data (negatives)
└── models/                 # Fine-tuned model weights
```

### Backup Strategy
```bash
# Backup profiles and training data monthly
tar -czf crittercatcher-backup-$(date +%Y%m%d).tar.gz \
  /mnt/user/appdata/crittercatcher/

# Store backup in safe location
cp crittercatcher-backup-*.tar.gz /mnt/user/backups/
```

## Updating

### Pull Latest Changes
```bash
cd /mnt/user/appdata/crittercatcher
git pull origin main
```

### Rebuild Image
```bash
docker-compose build --no-cache
docker-compose down
docker-compose up -d
```

## Support & Issues

For issues or questions:
1. Check Docker logs: `docker logs crittercatcher-ai`
2. Verify volumes are mounted: `docker exec crittercatcher-ai ls -la /data/`
3. Test API: `curl http://localhost:8080/api/animal-profiles`
4. Report on GitHub: https://github.com/Ahazii/CritterCatcherAI/issues

## Known Limitations

1. **Ring API Integration**: Currently placeholder only (needs authentication setup)
2. **Retraining**: Endpoint accepts requests but implementation is Phase 9+ only
3. **Video Download**: Must be placed in `/data/downloads/` manually or via external script
4. **GPU Support**: Requires nvidia-docker and NVIDIA GPU (not yet configured)

---

**Last Updated**: 2025-11-20  
**CritterCatcherAI Version**: v0.1.0  
**Status**: Production Ready for Testing
