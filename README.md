# CritterCatcherAI

A Docker container for Unraid that automatically downloads Ring doorbell videos, analyzes them with AI, and organizes them by detected subjects.

## Overview

CritterCatcherAI monitors your Ring doorbell, downloads videos to a shared volume, analyzes them using open-source AI tools, and automatically sorts them into separate folders based on what (or who) was detected.

## Features

- **Ring Video Download**: Uses a well-maintained unofficial Ring client supporting refresh tokens and 2FA
- **Open-Vocabulary Detection**: Detect arbitrary objects like "hedgehog", "wren", "fox", etc.
- **Face Recognition**: Identify specific people like family members or frequent visitors
- **Automatic Organization**: Videos are automatically moved to class-specific shared volumes
- **Image Review Automation**: Auto-confirm high-confidence detections and manage training data size
- **Discovery Mode**: Automatically find new objects in your videos
- **Specialized Species Training**: Train custom AI models for specific wildlife
- **Web Interface**: Modern UI for monitoring, configuration, and training
- **Unraid Optimized**: Designed to run seamlessly on Unraid servers

## Architecture

```
Ring Doorbell → Download Videos → AI Analysis → Sort by Detection
                     ↓                  ↓              ↓
                Shared Volume    Object/Face     Class Volumes
                                 Detection       (Hedgehogs, Birds, 
                                                  People, etc.)
```

## Detection Classes

Configure your own detection classes in `config/config.yaml`:
- Animals: hedgehog, fox, cat, dog, squirrel
- Birds: wren, finch, robin, crow
- People: Claire, John, Delivery Person
- Vehicles: car, bicycle, truck

## Quick Start

See [Installation Guide](#installation) below for detailed setup instructions.

## Requirements

- Unraid server (or any Docker host)
- Ring account with video subscription
- Shared storage volumes for video organization

## Installation

### Prerequisites

- Unraid server (or any Docker host)
- Ring account with video subscription
- Docker and Docker Compose installed

### Quick Start

1. **Clone or download this repository to your Unraid server:**
   ```bash
   cd /mnt/user/appdata
   git clone <repository-url> crittercatcher
   cd crittercatcher
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   nano .env
   ```
   Update with your Ring credentials:
   - `RING_USERNAME`: Your Ring account email
   - `RING_PASSWORD`: Your Ring account password

3. **Edit configuration (optional):**
   ```bash
   nano config/config.yaml
   ```
   Customize detection labels, confidence thresholds, and other settings.

4. **Update volume paths in docker-compose.yml:**
   Edit the volume mappings to match your Unraid share structure.

5. **Build and start the container:**
   ```bash
   docker-compose up -d
   ```

6. **Check logs:**
   ```bash
   docker logs -f crittercatcher-ai
   ```

### First-Time Setup

#### Ring Authentication
On first run, the container will authenticate with Ring using your credentials. If you have 2FA enabled, you may need to:
1. Run the container interactively: `docker-compose run --rm crittercatcher python src/main.py`
2. Enter your 2FA code when prompted
3. The refresh token will be saved for future automatic authentication

#### Face Recognition Setup
To enable face recognition for specific people:

1. Create a directory with photos of the person:
   ```bash
   mkdir -p /mnt/user/appdata/crittercatcher/faces/training/Claire
   ```

2. Add 3-5 clear photos of the person's face to this directory

3. Run the training script:
   ```bash
   docker exec -it crittercatcher-ai python -c "
   from face_recognizer import FaceRecognizer
   from pathlib import Path
   fr = FaceRecognizer()
   images = list(Path('/data/faces/training/Claire').glob('*.jpg'))
   fr.add_person('Claire', images)
   "
   ```

4. Repeat for additional people

## Usage

### Monitoring

View real-time logs:
```bash
docker logs -f crittercatcher-ai
```

### Accessing Sorted Videos

Videos are automatically sorted into directories based on detection:
```
/mnt/user/Videos/CritterCatcher/sorted/
├── hedgehog/
│   └── FrontDoor_20240126_143022_12345.mp4
├── fox/
├── bird/
├── people/
│   ├── Claire/
│   └── John/
└── unknown/
```

### Manual Processing

To process videos on-demand:
```bash
docker exec crittercatcher-ai python src/main.py
```

## Configuration

### Main Configuration (config/config.yaml)

**Detection Labels:** Add any object you want to detect
```yaml
detection:
  object_labels:
    - hedgehog
    - your_custom_animal
    - your_custom_object
```

**Confidence Threshold:** Adjust detection sensitivity
```yaml
detection:
  confidence_threshold: 0.25  # Lower = more detections, Higher = more accurate
```

**Processing Schedule:** 
```yaml
run_once: false           # false = continuous, true = run once and exit
interval_minutes: 60      # Check for new videos every 60 minutes
```

**Priority Mode:**
```yaml
detection:
  priority: people  # or "objects" - determines what gets priority in sorting
```

**Image Review (New in v0.1.0):**
```yaml
image_review:
  auto_confirm_threshold: 0.85  # Auto-confirm detections >= 85% confidence
  max_confirmed_images: 200      # Keep max 200 confirmed images per label
```

### Advanced Configuration

**GPU Acceleration:** Uncomment the GPU section in docker-compose.yml if you have an NVIDIA GPU

**Custom Volume Paths:** Edit docker-compose.yml volume mappings for your storage setup

**Logging Level:** Set `LOG_LEVEL` environment variable (DEBUG, INFO, WARNING, ERROR)

## Troubleshooting

### Ring Authentication Issues
- Check credentials in `.env` file
- If 2FA is enabled, run container interactively for initial setup
- Delete `/mnt/user/appdata/crittercatcher/tokens/ring_token.json` to force re-authentication

### No Detections
- Lower `confidence_threshold` in config.yaml
- Check logs for errors during video processing
- Verify videos are being downloaded to `/data/downloads`

### Face Recognition Not Working
- Ensure face encodings are created (see Face Recognition Setup)
- Add more training photos for better accuracy
- Adjust `face_tolerance` in config.yaml

### Performance Issues
- Enable GPU support if available
- Reduce `download_hours` to process fewer videos
- Increase `interval_minutes` to reduce processing frequency

## License

MIT License - See LICENSE file for details
