# CritterCatcherAI User Guide
## Specialized Species Detection & Training

**Version**: 2.0  
**Last Updated**: 2025-10-28

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Training Species Classifiers](#training-species-classifiers)
4. [Clip Extraction Mode](#clip-extraction-mode)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)

---

## Overview

CritterCatcherAI v2.0 introduces a two-stage AI detection pipeline for identifying specific wildlife species from your Ring camera videos:

### **Stage 1: YOLO Detection** (Existing)
- Detects broad categories (bird, cat, dog, etc.)
- Fast, real-time capable
- 80 COCO object classes

### **Stage 2: Specialized Classification** (New)
- Fine-grained species identification
- Custom-trained models for specific animals
- High accuracy for target species
- Example: hedgehog, finch, goldfinch

### **Clip Extraction** (New)
- Automatically extracts video clips containing target species
- Configurable padding (±10 seconds)
- Saves to Plex server
- Optional auto-delete non-matching videos

---

## Getting Started

### Prerequisites

1. **Unraid server** with CritterCatcherAI installed
2. **Ring camera** with video subscription
3. **GPU (recommended)** or powerful CPU for training
4. **Training images** (minimum 10 per species)

### Access the Web Interface

1. Open your browser and navigate to: `http://YOUR_UNRAID_IP:8080`
2. You'll see the main dashboard
3. Click **"Species Training"** in the navigation menu

---

## Training Species Classifiers

Training a custom species classifier involves 4 steps:

### Step 1: Add a New Species

1. Navigate to the **Species Training** page
2. Click **"+ Add Species"** button
3. Fill in the form:
   - **Species Name**: e.g., "hedgehog"
   - **Parent YOLO Classes**: Comma-separated list (e.g., "cat, dog")
   - **Confidence Threshold**: 0.75 (recommended)
4. Click **"Add Species"**

#### Choosing Parent YOLO Classes

The parent classes tell the system which YOLO detections to check with your specialized classifier:

| Your Species | Parent YOLO Classes | Reason |
|--------------|---------------------|---------|
| Hedgehog | cat, dog | Hedgehogs are often misclassified as cats or dogs |
| Finch | bird | Finches are birds |
| Fox | dog, cat | Foxes look like dogs/cats |
| Squirrel | cat, dog | Small mammals often detected as cats |

**Available YOLO Classes:**
```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
```

### Step 2: Collect Training Images

You need **minimum 10 images**, but **50-200+ is recommended** for good accuracy.

#### Option A: Download from Public Datasets

1. Search for datasets on:
   - **iNaturalist**: https://www.inaturalist.org/
   - **Kaggle**: https://www.kaggle.com/datasets
   - **Google Images**: Use browser extensions to batch download

2. Download 50-200 images of your target species

#### Option B: Use Your Ring Videos

1. Let YOLO run for a few days
2. Check `/data/objects/detected/` for potential matches
3. Manually select images of your target species
4. Move them to `/data/training_data/SPECIES_NAME/train/`

### Step 3: Upload Training Images

1. On the Species Training page, find your species card
2. Click **"📤 Upload"** button
3. Select multiple images (Ctrl+Click or Cmd+Click)
4. Click **"Upload Images"**
5. Wait for upload to complete

**Image Requirements:**
- Format: JPG or PNG
- Size: Any (will be resized to 224x224)
- Quality: Clear, well-lit images work best
- Variety: Different angles, lighting, backgrounds

### Step 4: Train the Model

1. Once you have 10+ training images, the **"🎓 Train"** button will activate
2. Click **"🎓 Train"**
3. Confirm the dialog (training takes 30-60 minutes)
4. Monitor progress in the **"Training Status"** section

**Training Process:**
- Uses ResNet18 with transfer learning
- Automatically collects negative samples from YOLO detections
- Trains for 50 epochs (configurable)
- Saves best model based on validation accuracy
- GPU training: ~30 minutes
- CPU training: ~2 hours

**Training Output:**
```
/data/models/hedgehog_classifier.pt (model file)
/data/models/hedgehog_metadata.json (training metrics)
```

---

## Clip Extraction Mode

Once you have trained models, you can enable clip extraction to automatically save only videos containing your target species.

### Enable Clip Extraction

1. Edit `/app/config/config.yaml` (via Unraid terminal or file editor):

```yaml
specialized_detection:
  enabled: true  # Enable specialized classifiers
  
  species:
    - name: hedgehog
      parent_yolo_class: [cat, dog]
      model_path: /data/models/hedgehog_classifier.pt
      confidence_threshold: 0.75
  
  clip_extraction:
    enabled: true  # Enable clip extraction
    padding_seconds: 10  # ±10 seconds around detection
    merge_overlapping: true  # Merge nearby detections
    auto_delete_non_matches: false  # CAUTION: Deletes non-matching videos!
    output_path: /data/clips  # Extracted clips for Plex
```

2. Restart the container:
```bash
docker restart crittercatcher-ai
```

### How Clip Extraction Works

1. **YOLO detects** animals in video (Stage 1)
2. **Specialized classifier** checks if it's your target species (Stage 2)
3. **If match found**:
   - Extracts clip with ±10s padding
   - Saves to `/data/clips/SPECIES_NAME/`
   - Deletes original video (saves space)
4. **If no match** and `auto_delete_non_matches: true`:
   - Deletes original video
   - ⚠️ **Use with caution!** Deleted files cannot be recovered

### Safety Toggle

The `auto_delete_non_matches` setting is **disabled by default** for safety:

```yaml
auto_delete_non_matches: false  # Safe: keeps all videos
```

**Only enable this when:**
- ✅ Your models are well-trained (>85% accuracy)
- ✅ You've tested with a few videos first
- ✅ You have backups of important footage
- ✅ You understand files are permanently deleted

### Accessing Extracted Clips

Clips are saved to `/data/clips/SPECIES_NAME/` which you can mount to your Plex server:

```yaml
# docker-compose.yml
volumes:
  - /mnt/user/Videos/Wildlife:/data/clips
```

**Clip Filename Format:**
```
20251028_143022_FrontDoor_20240126_143022_12345_hedgehog.mp4
```

---

## Configuration

### Main Configuration File

Location: `/app/config/config.yaml`

### Specialized Detection Section

```yaml
specialized_detection:
  # Enable/disable Stage 2 classifiers
  enabled: false  # Set to true after training models
  
  # Species definitions
  species:
    - name: hedgehog
      parent_yolo_class: [cat, dog]
      model_path: /data/models/hedgehog_classifier.pt
      confidence_threshold: 0.75  # 75% confidence required
      
    - name: finch
      parent_yolo_class: [bird]
      model_path: /data/models/finch_classifier.pt
      confidence_threshold: 0.70
      # Optional: hierarchical sub-classifiers
      sub_classifiers:
        - name: goldfinch
          model_path: /data/models/goldfinch_classifier.pt
          confidence_threshold: 0.65
  
  # Clip extraction settings
  clip_extraction:
    enabled: false  # Enable clip mode
    padding_seconds: 10  # Seconds before/after detection
    merge_overlapping: true  # Merge nearby detections
    auto_delete_non_matches: false  # Safety toggle
    output_path: /data/clips
  
  # Training hyperparameters
  training:
    data_augmentation: true
    validation_split: 0.2  # 20% for validation
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
```

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 50 | Training iterations |
| `batch_size` | 32 | Images per training batch |
| `learning_rate` | 0.001 | Learning step size |
| `validation_split` | 0.2 | Validation data percentage |
| `data_augmentation` | true | Random transforms during training |

**Adjust for better results:**
- Low accuracy? → Increase `epochs` to 100
- Training too slow? → Decrease `batch_size` to 16
- Overfitting? → Add more training images

---

## Troubleshooting

### Training Issues

#### "Insufficient training images"
- **Problem**: Less than 10 images uploaded
- **Solution**: Upload at least 10 images, preferably 50+

#### Training fails immediately
- **Problem**: Corrupt images or wrong format
- **Solution**: 
  1. Check logs: `docker logs crittercatcher-ai`
  2. Verify all images are JPG/PNG
  3. Remove any corrupt files

#### Low accuracy (<70%)
- **Problem**: Not enough training data or poor quality images
- **Solutions**:
  - Add more training images (aim for 100+)
  - Ensure image variety (different angles, lighting)
  - Add more negative samples
  - Increase epochs to 100

#### Training takes forever (>4 hours)
- **Problem**: CPU training is slow
- **Solutions**:
  - Enable GPU support (see UNRAID_INSTALL.md)
  - Reduce `batch_size` to 16
  - Reduce `epochs` to 25

### Detection Issues

#### Model not detecting anything
1. Check if specialized_detection is enabled in config
2. Verify model file exists: `/data/models/SPECIES_classifier.pt`
3. Check parent YOLO classes are correct
4. Lower confidence_threshold to 0.5 for testing

#### Too many false positives
- **Solution**: Increase `confidence_threshold` to 0.85
- Add more negative training samples

#### Too many false negatives (missing detections)
- **Solution**: Lower `confidence_threshold` to 0.65
- Add more training images with variety

### Clip Extraction Issues

#### Clips are too short/long
- **Problem**: Padding setting
- **Solution**: Adjust `padding_seconds` in config (try 15 or 20)

#### Clips have multiple detections
- **Expected behavior** when `merge_overlapping: true`
- Disable merging if you want separate clips per detection

#### Videos being deleted accidentally
- **IMMEDIATE ACTION**: Set `auto_delete_non_matches: false`
- Test thoroughly before re-enabling
- Consider keeping original videos in separate folder

---

## API Reference

### Species Management

#### List Species
```http
GET /api/species/list
```

Response:
```json
{
  "species": [
    {
      "name": "hedgehog",
      "parent_yolo_class": ["cat", "dog"],
      "confidence_threshold": 0.75,
      "has_model": true,
      "training_images": 150
    }
  ]
}
```

#### Add Species
```http
POST /api/species/add
Content-Type: application/json

{
  "name": "hedgehog",
  "parent_yolo_class": ["cat", "dog"],
  "confidence_threshold": 0.75
}
```

#### Upload Training Data
```http
POST /api/species/upload_training_data?species_name=hedgehog
Content-Type: multipart/form-data

files: [image1.jpg, image2.jpg, ...]
```

#### Train Species
```http
POST /api/species/train
Content-Type: application/json

{
  "species_name": "hedgehog"
}
```

#### Training Status
```http
GET /api/species/training_status
```

Response:
```json
{
  "is_training": true,
  "current_species": "hedgehog",
  "device": "cuda",
  "available_models": [...]
}
```

#### List Trained Models
```http
GET /api/species/models
```

Response:
```json
{
  "models": [
    {
      "species": "hedgehog",
      "accuracy": 89.5,
      "size_mb": 42.3,
      "created": "2025-10-28T12:00:00",
      "training_time": 1834
    }
  ]
}
```

---

## Advanced Topics

### Hierarchical Classification

For fine-grained species identification (e.g., bird → finch → goldfinch):

```yaml
species:
  - name: bird
    parent_yolo_class: [bird]
    model_path: /data/models/bird_classifier.pt
    confidence_threshold: 0.70
    sub_classifiers:
      - name: finch
        model_path: /data/models/finch_classifier.pt
        confidence_threshold: 0.65
```

### Custom Model Architecture

Edit `src/training_manager.py` line 224 to use different architectures:

```python
# Current: ResNet18
model = models.resnet18(pretrained=True)

# Alternatives:
# model = models.resnet50(pretrained=True)  # More accurate, slower
# model = models.efficientnet_b0(pretrained=True)  # Balanced
# model = models.mobilenet_v3_small(pretrained=True)  # Faster, less accurate
```

### Data Augmentation

Controlled in config:

```yaml
training:
  data_augmentation: true  # Enables random transforms
```

Augmentations applied:
- Random crop
- Horizontal flip
- Rotation (±15°)
- Color jitter (brightness, contrast, saturation)

---

## Best Practices

### Training
1. **Start with quality over quantity**: 50 good images > 200 poor images
2. **Vary your dataset**: Different times of day, weather, angles
3. **Balance positive/negative samples**: System auto-collects negatives from YOLO
4. **Test before production**: Train, test accuracy, then enable clip extraction
5. **Monitor logs**: `docker logs -f crittercatcher-ai`

### Clip Extraction
1. **Test first**: Keep `auto_delete_non_matches: false` initially
2. **Review extracted clips**: Verify accuracy before enabling auto-delete
3. **Backup important videos**: Before enabling deletion
4. **Start conservative**: High confidence threshold (0.80+)
5. **Monitor disk space**: Extracted clips accumulate quickly

### Performance
1. **GPU training**: 10-20x faster than CPU
2. **Batch processing**: Process videos in batches during low-usage periods
3. **Model size**: Smaller models (ResNet18) are faster for inference
4. **Frame sampling**: Default 5 frames per video is usually sufficient

---

## Support & Resources

- **Project Repository**: https://github.com/Ahazii/CritterCatcherAI
- **Docker Logs**: `docker logs -f crittercatcher-ai`
- **Configuration**: `/app/config/config.yaml`
- **Training Data**: `/data/training_data/`
- **Models**: `/data/models/`
- **Extracted Clips**: `/data/clips/`

---

## Quick Reference

### File Locations

| Path | Description |
|------|-------------|
| `/app/config/config.yaml` | Main configuration |
| `/data/training_data/SPECIES/train/` | Training images |
| `/data/models/SPECIES_classifier.pt` | Trained model |
| `/data/clips/SPECIES/` | Extracted clips |
| `/data/objects/detected/` | YOLO detection images |
| `/data/objects/specialized/` | Stage 2 detection metadata |

### Common Commands

```bash
# View logs
docker logs -f crittercatcher-ai

# Restart container
docker restart crittercatcher-ai

# Access container shell
docker exec -it crittercatcher-ai /bin/bash

# Check disk space
docker exec crittercatcher-ai df -h /data

# List trained models
ls -lh /mnt/user/appdata/crittercatcher/models/
```

### Training Checklist

- [ ] Species added to config
- [ ] 50+ training images uploaded
- [ ] Training completed successfully
- [ ] Model accuracy >85%
- [ ] Tested on sample videos
- [ ] `specialized_detection.enabled: true`
- [ ] Clip extraction configured
- [ ] Safety toggle set appropriately
- [ ] Plex folder mounted

---

**Happy Critter Catching!** 🦔🐦🦊
