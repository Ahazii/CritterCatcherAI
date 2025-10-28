# CritterCatcherAI Quick Reference

**Version**: 2.0  
**Last Updated**: 2025-10-28

---

## 🚀 Getting Started (5 Minutes)

### First Time Setup
1. Access web UI: `http://YOUR_UNRAID_IP:8080`
2. Navigate to **Ring Setup** tab
3. Enter Ring credentials + 2FA code
4. Go to **Configuration** tab
5. Select objects to detect (use dropdown)
6. Click **▶ Process Now**

### Daily Use
- Videos auto-download every hour
- Check **Dashboard** for processing status
- Review detections in **Videos** tab
- Train species in **Species Training** tab

---

## 🎯 Core Features

### Standard Detection (Built-in)
- **What**: Detects 80 YOLO object classes
- **When**: Automatically on all videos
- **Output**: Sorted folders: `/data/sorted/CATEGORY/`
- **Speed**: ~30 seconds per video

### Species Detection (Optional)
- **What**: Fine-grained species ID (hedgehog, finch, etc.)
- **When**: After training custom models
- **Output**: Clips extracted: `/data/clips/SPECIES/`
- **Speed**: +10 seconds per video

### Face Recognition (Optional)
- **What**: Identifies specific people
- **When**: After uploading training photos
- **Output**: Sorted by person name
- **Speed**: +5 seconds per video

---

## 📋 Common Tasks

### Add Objects to Detect
1. **Configuration** tab → **Objects to Detect**
2. Click dropdown → search classes
3. Select checkboxes (e.g., bird, cat, dog)
4. **💾 Save Configuration**

### Train a Species Classifier
1. **Species Training** → **+ Add Species**
2. Name: `hedgehog`
3. Parent Classes: Select `cat, dog` from dropdown
4. If warning appears → Click **Add Missing Classes**
5. Upload 50+ training images
6. Click **🎓 Train** (wait 30-60 min)
7. Enable in config: `specialized_detection.enabled: true`

### Extract Clips of Target Species
Edit `/app/config/config.yaml`:
```yaml
specialized_detection:
  enabled: true
  clip_extraction:
    enabled: true
    padding_seconds: 10
    auto_delete_non_matches: false  # Start safe!
```

### Add Face Recognition
1. **Face Training** tab
2. Enter person name
3. Upload 3-5 clear photos
4. Click **🔄 Retrain All**
5. Wait for training to complete

---

## 🎨 Web Interface Navigation

### Tabs Overview
| Tab | Purpose | Key Actions |
|-----|---------|-------------|
| 📊 **Dashboard** | Status, stats, videos | Process Now, View Stats |
| 📈 **Analytics** | Charts, trends | View detection history |
| 🎥 **Videos** | Browse sorted videos | Download, review |
| 🏷️ **Review & Label** | Manage discoveries, unknown faces | Add/ignore labels |
| 🔔 **Ring Setup** | Authentication | Login, 2FA, status |
| 👤 **Face Training** | Face recognition | Upload photos, train |
| 🦔 **Species Training** | Species classifiers | Add species, train models |
| ⚙️ **Configuration** | System settings | Detection, thresholds |
| 📋 **Logs** | Real-time logs | Monitor processing |

### Dashboard Quick Actions
- **▶ Process Now**: Start processing immediately
- **⏹ Stop**: Cancel current processing
- **⬇ Download All**: Bulk download videos
- **🔄 Reprocess All**: Re-analyze all videos

---

## 🔧 Configuration Cheat Sheet

### Detection Settings (⚙️ Config Tab)

#### Confidence Threshold
- **0.1-0.2**: Very sensitive (many false positives)
- **0.25-0.4**: ✅ **Recommended** (balanced)
- **0.5-0.9**: Conservative (may miss detections)

#### Object Detection Frames
- **3-5**: ✅ **Recommended** (fast)
- **10-15**: More accurate, slower
- **20+**: Very thorough, very slow

#### Face Tolerance
- **0.3-0.4**: Strict matching (fewer false positives)
- **0.5-0.7**: ✅ **Recommended** (balanced)
- **0.8-0.9**: Lenient (more false positives)

### Species Training Settings

#### Confidence Threshold (per species)
- **0.65-0.70**: Sensitive (more detections)
- **0.75-0.80**: ✅ **Recommended** (balanced)
- **0.85-0.95**: Conservative (high precision)

#### Training Data Requirements
- **Minimum**: 10 images (poor results)
- **Good**: 50-100 images
- **Excellent**: ✅ 200+ images
- **Variety**: Different angles, lighting, backgrounds

---

## 🎯 YOLO Classes Reference

### Most Common for Wildlife
```
bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe,
person, bicycle, car, motorcycle, bench
```

### All 80 COCO Classes (Alphabetical)
```
airplane, apple, backpack, banana, baseball bat, baseball glove, 
bear, bed, bench, bicycle, bird, boat, book, bottle, bowl, broccoli,
bus, cake, car, carrot, cat, cell phone, chair, clock, couch, cow,
cup, dining table, dog, donut, elephant, fire hydrant, fork, frisbee,
giraffe, hair drier, handbag, horse, hot dog, keyboard, kite, knife,
laptop, microwave, motorcycle, mouse, orange, oven, parking meter,
person, pizza, potted plant, refrigerator, remote, sandwich, scissors,
sheep, sink, skateboard, skis, snowboard, spoon, sports ball, 
stop sign, suitcase, surfboard, teddy bear, tennis racket, tie,
toaster, toilet, toothbrush, traffic light, train, truck, tv, 
umbrella, vase, wine glass, zebra
```

---

## 📁 File Locations

### Configuration
- **Main config**: `/app/config/config.yaml`
- **Ring token**: `/app/config/ring_token.json`

### Data Directories
- **Downloads**: `/data/downloads/` (raw videos from Ring)
- **Sorted**: `/data/sorted/CATEGORY/` (organized videos)
- **Clips**: `/data/clips/SPECIES/` (extracted clips)
- **Training data**: `/data/training_data/SPECIES/train/`
- **Models**: `/data/models/SPECIES_classifier.pt`
- **Face encodings**: `/data/faces/encodings.pkl`
- **Detected objects**: `/data/objects/detected/`
- **Specialized detections**: `/data/objects/specialized/`

### Unraid Paths (Host)
- **Config**: `/mnt/user/appdata/crittercatcher/config/`
- **Data**: `/mnt/user/appdata/crittercatcher/data/`
- **Videos**: `/mnt/user/Videos/Ring/` (sorted output)

---

## 🔨 Docker Commands

### Container Management
```bash
# View logs (real-time)
docker logs -f crittercatcher-ai

# Restart container
docker restart crittercatcher-ai

# Stop container
docker stop crittercatcher-ai

# Start container
docker start crittercatcher-ai

# Access shell
docker exec -it crittercatcher-ai /bin/bash
```

### Troubleshooting Commands
```bash
# Check disk space
docker exec crittercatcher-ai df -h /data

# List training data
docker exec crittercatcher-ai ls -lh /data/training_data/

# List trained models
docker exec crittercatcher-ai ls -lh /data/models/

# Check GPU availability
docker exec crittercatcher-ai python -c "import torch; print(torch.cuda.is_available())"

# View config
docker exec crittercatcher-ai cat /app/config/config.yaml
```

---

## 🚨 Troubleshooting Quick Fixes

### Ring Authentication Failed
```bash
# Delete token and re-authenticate
docker exec crittercatcher-ai rm /app/config/ring_token.json
docker restart crittercatcher-ai
# Then re-auth via web UI
```

### Training Fails
```bash
# Check logs
docker logs crittercatcher-ai | grep -i error

# Verify training images
docker exec crittercatcher-ai ls -lh /data/training_data/SPECIES/train/

# Check GPU
docker exec crittercatcher-ai nvidia-smi
```

### Videos Not Processing
1. Check Ring authentication: **Ring Setup** tab
2. Check object labels: **Configuration** → Objects to Detect
3. Check processing status: **Dashboard**
4. Check logs: `docker logs -f crittercatcher-ai`

### Clip Extraction Not Working
```yaml
# Verify config.yaml settings
specialized_detection:
  enabled: true  # Must be true!
  species:
    - name: hedgehog
      parent_yolo_class: [cat, dog]  # Must match detection list!
      model_path: /data/models/hedgehog_classifier.pt
  clip_extraction:
    enabled: true  # Must be true!
```

### Low Detection Accuracy
1. **Not enough training data**: Upload 100+ images
2. **Poor quality images**: Use clear, varied images
3. **Threshold too high**: Lower to 0.65-0.70
4. **Wrong parent classes**: Verify YOLO detects them
5. **Need more epochs**: Increase to 100 in config

---

## 📊 API Quick Reference

### Base URL
```
http://YOUR_UNRAID_IP:8080
```

### Common Endpoints

#### Status & Stats
```http
GET /api/status          # Processing status
GET /api/stats           # Video statistics
GET /api/config          # Current configuration
```

#### Species Management
```http
GET  /api/species/list                     # List all species
POST /api/species/add                      # Add new species
POST /api/species/upload_training_data     # Upload images
POST /api/species/train                    # Start training
GET  /api/species/training_status          # Training progress
GET  /api/species/models                   # List trained models
```

#### YOLO Classes
```http
GET  /api/yolo_classes                     # Get all 80 YOLO classes
POST /api/config/add_object_labels         # Add labels to config
```

#### Ring Integration
```http
GET  /api/ring/status                      # Auth status
POST /api/ring/authenticate                # Login with credentials
```

#### Face Recognition
```http
GET  /api/faces                            # List trained faces
POST /api/faces/upload                     # Upload training photo
POST /api/faces/train                      # Train face model
POST /api/faces/retrain                    # Retrain all faces
```

#### Processing Control
```http
POST /api/process                          # Start processing
POST /api/stop                             # Stop processing
POST /api/reprocess                        # Reprocess all videos
POST /api/download-all                     # Download videos from Ring
```

---

## 💡 Pro Tips

### Performance Optimization
- ✅ Use GPU for training (10-20x faster)
- ✅ Reduce `object_frames` to 5 for faster processing
- ✅ Enable only needed object classes
- ✅ Process during off-peak hours
- ✅ Use smaller models (ResNet18) for inference

### Training Best Practices
- 🎯 **Quality > Quantity**: 50 good images > 200 poor ones
- 📸 **Variety**: Different angles, lighting, distances
- ⚖️ **Balance**: System auto-collects negative samples
- 🧪 **Test first**: Verify accuracy before clip extraction
- 📊 **Monitor**: Check training logs for issues

### Clip Extraction Safety
- 🔒 **Start safe**: `auto_delete_non_matches: false`
- 🧪 **Test thoroughly**: Review extracted clips first
- 💾 **Backup**: Keep originals until confident
- 🎯 **Conservative threshold**: Start at 0.80+
- 📊 **Monitor**: Check disk space regularly

### Multi-Select Dropdown Tips
- 🔍 **Use search**: Type to filter 80 classes instantly
- ✅ **Visual feedback**: Tags show selected classes
- ⚠️ **Validation warnings**: Auto-detects misconfigurations
- 🔧 **One-click fixes**: Add missing classes automatically
- 🎨 **Consistent UX**: Same dropdown in Config and Species pages

---

## 📱 Workflow Examples

### Daily Monitoring
```
1. Open Dashboard → Check status
2. Review Videos tab → See new detections
3. Check Species Training → Monitor model accuracy
4. Review extracted clips in Plex
```

### Adding New Species
```
1. Species Training → + Add Species
2. Name: "fox", Parent: [cat, dog]
3. Upload 100+ fox images
4. Train model (30-60 min)
5. Enable: specialized_detection.enabled: true
6. Test on a few videos
7. Enable clip extraction when confident
```

### Improving Accuracy
```
1. Check species accuracy in Training Status
2. If < 85%:
   - Upload more training images
   - Ensure image variety
   - Increase epochs to 100
   - Retrain model
3. Test on sample videos
4. Adjust confidence threshold
```

---

## 🆘 Emergency Procedures

### Videos Being Deleted Accidentally
```bash
# IMMEDIATE ACTION
1. Edit config: auto_delete_non_matches: false
2. Restart: docker restart crittercatcher-ai
3. Review remaining videos
4. Retrain models with more data
5. Test thoroughly before re-enabling
```

### Container Won't Start
```bash
# Check logs
docker logs crittercatcher-ai

# Check disk space
df -h /mnt/user/appdata/crittercatcher/

# Reset config (backup first!)
mv /mnt/user/appdata/crittercatcher/config/config.yaml /mnt/user/appdata/crittercatcher/config/config.yaml.bak
docker restart crittercatcher-ai
```

### Out of Disk Space
```bash
# Check usage
docker exec crittercatcher-ai du -sh /data/*

# Clean up options
docker exec crittercatcher-ai rm -rf /data/downloads/*  # Raw videos
docker exec crittercatcher-ai rm -rf /data/objects/detected/*  # Detection images
# Be careful with sorted videos and clips!
```

---

## 📚 Resources

### Documentation
- **User Guide**: `USER_GUIDE.md` (detailed)
- **This file**: `QUICK_REFERENCE.md` (quick lookup)
- **Installation**: `UNRAID_INSTALL.md`
- **Ring Setup**: `RING_2FA_SETUP.md`
- **Implementation**: `IMPLEMENTATION_PLAN.md`

### Support
- **GitHub**: https://github.com/Ahazii/CritterCatcherAI
- **Issues**: https://github.com/Ahazii/CritterCatcherAI/issues
- **Logs**: `docker logs -f crittercatcher-ai`

### Training Resources
- **iNaturalist**: https://www.inaturalist.org/
- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **PyTorch Docs**: https://pytorch.org/vision/stable/models.html

---

## ✅ Checklists

### New Installation Checklist
- [ ] Container running: `docker ps | grep crittercatcher`
- [ ] Web UI accessible: `http://YOUR_IP:8080`
- [ ] Ring authentication completed
- [ ] Objects to detect configured
- [ ] First video processed successfully
- [ ] Sorted folders created
- [ ] Logs show no errors

### Species Training Checklist
- [ ] Species added via web UI
- [ ] Parent YOLO classes validated
- [ ] Missing classes added to detection list
- [ ] 50+ training images uploaded
- [ ] Training completed (>85% accuracy)
- [ ] Model file exists: `/data/models/SPECIES_classifier.pt`
- [ ] Config updated: `specialized_detection.enabled: true`
- [ ] Tested on sample videos
- [ ] Clip extraction configured (optional)

### Production Deployment Checklist
- [ ] All models trained and tested
- [ ] Accuracy >85% for all species
- [ ] Clip extraction tested with safety on
- [ ] Backup strategy in place
- [ ] Disk space monitored
- [ ] Plex folders mounted correctly
- [ ] Auto-delete disabled initially
- [ ] Log monitoring setup

---

**Happy Critter Catching!** 🦔🐦🦊

*For detailed information, see `USER_GUIDE.md`*
