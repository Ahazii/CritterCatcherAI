# CritterCatcherAI - Unraid Installation Guide

## Quick Start (Recommended)

### Method 1: Using Community Applications (Easiest)

*Note: This template will be available in Community Applications once submitted*

1. Open Unraid web interface
2. Go to **Apps** tab
3. Search for "CritterCatcherAI"
4. Click **Install**
5. Configure settings (see Configuration section below)
6. Click **Apply**

### Method 2: Manual Template Installation

1. Download the template:
   ```bash
   wget -O /boot/config/plugins/dockerMan/templates-user/my-CritterCatcherAI.xml https://raw.githubusercontent.com/Ahazii/CritterCatcherAI/master/my-CritterCatcherAI.xml
   ```

2. Go to **Docker** tab in Unraid
3. Click **Add Container**
4. Select **CritterCatcherAI** from template dropdown
5. Configure settings
6. Click **Apply**

### Method 3: Manual Configuration

1. Go to **Docker** tab
2. Click **Add Container**
3. Fill in the following:

**Basic Settings:**
- **Name:** CritterCatcherAI
- **Repository:** `ghcr.io/ahazii/crittercatcherai:latest`
- **Network Type:** Bridge

**Environment Variables:**
| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RING_USERNAME` | Yes | - | Your Ring account email |
| `RING_PASSWORD` | Yes | - | Your Ring account password |
| `LOG_LEVEL` | No | INFO | DEBUG, INFO, WARNING, ERROR |
| `RUN_ONCE` | No | false | true = run once, false = continuous |
| `TZ` | No | America/New_York | Your timezone |
| `DOWNLOAD_HOURS` | No | 24 | Hours back to download videos |
| `INTERVAL_MINUTES` | No | 60 | Minutes between runs |
| `CONFIDENCE_THRESHOLD` | No | 0.25 | Object detection threshold (0.0-1.0) |
| `FACE_TOLERANCE` | No | 0.6 | Face recognition tolerance (0.0-1.0) |
| `DETECTION_PRIORITY` | No | people | "people" or "objects" |
| `WEB_PORT` | No | 8080 | Web interface port (change if conflicts) |

**Volume Mappings:**
| Container Path | Host Path | Mode | Description |
|----------------|-----------|------|-------------|
| `/app/config` | `/mnt/user/appdata/crittercatcher/config` | RW | Configuration files |
| `/data/downloads` | `/mnt/user/appdata/crittercatcher/downloads` | RW | Temp video storage |
| `/data/sorted` | `/mnt/user/Videos/CritterCatcher/sorted` | RW | Organized videos |
| `/data/faces` | `/mnt/user/appdata/crittercatcher/faces` | RW | Face database & unknown faces |
| `/data/objects` | `/mnt/user/appdata/crittercatcher/objects` | RW | Detected objects & discoveries |
| `/data/tokens` | `/mnt/user/appdata/crittercatcher/tokens` | RW | Ring auth tokens |

## Initial Setup

### 1. Create Required Directories

SSH into your Unraid server and run:
```bash
mkdir -p /mnt/user/appdata/crittercatcher/{config,downloads,faces/training,objects,tokens}
mkdir -p /mnt/user/Videos/CritterCatcher/sorted
chmod -R 777 /mnt/user/appdata/crittercatcher
```

### 2. Create Configuration File

Create `/mnt/user/appdata/crittercatcher/config/config.yaml`:

```yaml
# CritterCatcherAI Configuration

paths:
  downloads: /data/downloads
  sorted: /data/sorted
  face_encodings: /data/faces/encodings.pkl

ring:
  download_hours: 24
  download_limit: null

detection:
  object_labels:
    - hedgehog
    - fox
    - cat
    - dog
    - squirrel
    - bird
    - person
    - delivery person
    - car
  
  confidence_threshold: 0.25
  face_tolerance: 0.6
  priority: people

run_once: false
interval_minutes: 60
```

### 3. Handle Ring 2FA Authentication (If Enabled)

**If you have 2FA enabled on your Ring account**, you must authenticate interactively BEFORE starting the container normally:

```bash
# SSH into your Unraid server, then run:
docker run -it --rm \
  -v /mnt/user/appdata/crittercatcher/tokens:/data \
  -e RING_USERNAME="your@email.com" \
  -e RING_PASSWORD="yourpassword" \
  ghcr.io/ahazii/crittercatcherai:latest \
  python src/ring_2fa_setup.py
```

**You will be prompted to:**
1. Enter your 2FA code (check your email/SMS/authenticator app)
2. The script will save the authentication token
3. Exit once you see "SUCCESS!"

**After 2FA setup is complete**, the token is saved and you can start the container normally.

### 4. Start the Container

1. Start the container from Unraid Docker tab
2. The container will use the saved token (no more 2FA prompts needed)

### 5. Verify Operation

Check the logs:
```bash
docker logs -f crittercatcher-ai
```

You should see:
- "Starting CritterCatcherAI"
- "Authenticating with Ring"
- "Downloading recent Ring videos"

## Face Recognition Setup

To recognize specific people (e.g., family members):

### 1. Prepare Training Photos

1. Create a folder for each person:
   ```bash
   mkdir -p /mnt/user/appdata/crittercatcher/faces/training/Claire
   mkdir -p /mnt/user/appdata/crittercatcher/faces/training/John
   ```

2. Add 3-5 clear photos of each person's face:
   - Use JPG format
   - Face should be clearly visible
   - Variety of angles helps
   - Good lighting is important

### 2. Train the Model

For each person, run:
```bash
docker exec -it crittercatcher-ai python -c "
from face_recognizer import FaceRecognizer
from pathlib import Path
fr = FaceRecognizer()
images = list(Path('/data/faces/training/Claire').glob('*.jpg'))
fr.add_person('Claire', images)
"
```

Replace "Claire" with the person's name.

### 3. Verify Training

Check the logs for confirmation:
```bash
docker logs crittercatcher-ai | grep "Added.*face encoding"
```

## Configuration Tips

### Adjust Detection Sensitivity

**More Detections (may have false positives):**
- Lower `CONFIDENCE_THRESHOLD` to 0.15-0.20
- Raise `FACE_TOLERANCE` to 0.7-0.8

**Fewer but More Accurate Detections:**
- Raise `CONFIDENCE_THRESHOLD` to 0.30-0.40
- Lower `FACE_TOLERANCE` to 0.5

### Add Custom Object Labels

Edit `/mnt/user/appdata/crittercatcher/config/config.yaml` and add any object to `detection.object_labels`:
```yaml
detection:
  object_labels:
    - hedgehog
    - your_custom_animal
    - package
    - mailman
```

Restart the container after changes:
```bash
docker restart crittercatcher-ai
```

### Change Processing Schedule

**Check every 30 minutes:**
Set `INTERVAL_MINUTES=30` in container settings

**Run once per day via Unraid User Scripts:**
1. Set `RUN_ONCE=true`
2. Install User Scripts plugin
3. Create daily script:
   ```bash
   docker start crittercatcher-ai
   ```

## GPU Acceleration (Optional)

If you have an NVIDIA GPU:

1. Install **Nvidia-Driver** plugin from Community Applications
2. Edit container settings
3. Add GPU device: `/dev/dri` or specific GPU device
4. Add environment variable: `NVIDIA_VISIBLE_DEVICES=all`

Processing will be 3-5x faster with GPU.

## Troubleshooting

### Ring Authentication Fails
```bash
# Delete token and force re-auth
rm /mnt/user/appdata/crittercatcher/tokens/ring_token.json
docker restart crittercatcher-ai
```

### 2FA Code Not Working
- Ensure you're using the LATEST code (they expire quickly)
- Check if you're looking at the right 2FA method (SMS vs email vs authenticator)
- Try running the 2FA setup script again:
  ```bash
  docker run -it --rm \
    -v /mnt/user/appdata/crittercatcher/tokens:/data \
    -e RING_USERNAME="your@email.com" \
    -e RING_PASSWORD="yourpassword" \
    ghcr.io/ahazii/crittercatcherai:latest \
    python src/ring_2fa_setup.py
  ```

### No Videos Being Downloaded
- Verify Ring credentials are correct
- Check that Ring subscription is active
- Ensure videos exist in the time range (default: last 24 hours)

### No Detections in Videos
- Lower confidence threshold
- Check logs for errors during video processing
- Verify videos are being downloaded to `/data/downloads`

### Face Recognition Not Working
- Ensure training photos were processed (check logs)
- Add more training photos for better accuracy
- Increase `FACE_TOLERANCE` if faces aren't being matched

### High CPU/Memory Usage
- Enable GPU support if available
- Reduce `DOWNLOAD_HOURS` to process fewer videos
- Increase `INTERVAL_MINUTES` to reduce frequency

### Permission Errors
Ensure proper ownership:
```bash
chown -R nobody:users /mnt/user/appdata/crittercatcher
chown -R nobody:users /mnt/user/Videos/CritterCatcher
```

## Accessing Sorted Videos

Videos are organized in:
```
/mnt/user/Videos/CritterCatcher/sorted/
├── hedgehog/
├── fox/
├── bird/
├── people/
│   ├── Claire/
│   └── John/
└── unknown/
```

You can:
- Access via SMB share: `\\tower\Videos\CritterCatcher\sorted`
- Use Krusader or other Unraid file managers
- Mount as Plex/Emby/Jellyfin library

## Monitoring

### View Logs
```bash
docker logs -f crittercatcher-ai
```

### Check Statistics
Container logs will show sorting statistics after each run:
```
Sorting statistics: {'hedgehog': 15, 'people/Claire': 8, 'bird': 23}
```

### Container Health
The container includes a health check. View status:
```bash
docker ps | grep crittercatcher-ai
```

## Updating

### Update Container
1. Go to Docker tab
2. Click **Check for Updates**
3. If update available, click **Update**

Or via CLI:
```bash
docker pull ghcr.io/ahazii/crittercatcherai:latest
docker stop crittercatcher-ai
docker rm crittercatcher-ai
# Then recreate from template or Unraid UI
```

**Note:** Your configuration and data are preserved in the mapped volumes.

## Support

- GitHub Issues: https://github.com/Ahazii/CritterCatcherAI/issues
- Documentation: https://github.com/Ahazii/CritterCatcherAI
- Unraid Forums: [Search for CritterCatcherAI]
