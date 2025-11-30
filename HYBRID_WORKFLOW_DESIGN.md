# Hybrid YOLO-First Workflow Design

## Date: 2025-11-29
## Status: Implementation In Progress

## Problem Statement

Current V2 system only sorts videos through CLIP profiles, making it impossible to see what YOLO detects before committing to CLIP training. Users want to:
1. See all detected objects (e.g., all dogs) even without CLIP profiles
2. Optionally refine detections with CLIP (e.g., separate Jack Russells from other dogs)
3. Review unmatched detections to decide on new CLIP profiles

## Current Issues

1. **Video tracking broken**: 258-byte files - codec 'mp4v' failing
2. **No YOLO-only sorting**: Videos only sorted if CLIP profile exists
3. **Dogs in wrong categories**: Hedgehog CLIP profile claims dog videos
4. **Old data confusion**: Mixed old and new data in review folders
5. **False positives**: Low confidence detections (0.2) creating noise

## Proposed Hybrid Workflow

### Stage 1: YOLO Detection → Initial Sorting
```
Video Input
  ↓
YOLO Detection (confidence ≥ 0.2)
  ↓
Detected: "dog" (confidence: 0.85)
  ↓
Save to: /data/review/dog/
  ↓
Create tracked video: /data/objects/detected/annotated_videos/tracked_video.mp4
```

### Stage 2: CLIP Refinement (Optional)
```
Check: Does a CLIP profile use "dog" category?
  ↓
YES: Jack Russell profile uses "dog"
  ↓
Run CLIP: "Is this a Jack Russell?"
  ↓
RESULT SCENARIOS:
  
  A) High confidence (≥0.80):
     → Move to /data/sorted/jack_russell/
  
  B) Maybe match (0.50-0.79):
     → Move to /data/review/jack_russell_maybe/
  
  C) No match (<0.50):
     → Keep in /data/review/dog/
```

### Stage 3: User Review
```
User browses /data/review/dog/
  ↓
Sees: Doberman videos (unmatched by Jack Russell CLIP)
  ↓
Decision: Create "Doberman" CLIP profile
  ↓
Future videos: Dobermans auto-sorted
```

## Directory Structure

```
/data/
├── review/
│   ├── dog/                        # All YOLO dog detections
│   │   ├── unmatched/              # No CLIP profile matched
│   │   ├── jack_russell_maybe/     # CLIP near-miss (0.5-0.79)
│   │   └── needs_review/           # Low YOLO confidence (0.2-0.4)
│   ├── cat/
│   ├── bird/
│   └── person/
├── sorted/
│   ├── jack_russell/               # CLIP confirmed (≥0.8)
│   ├── hedgehog/
│   └── golden_eagle/
└── objects/
    └── detected/
        └── annotated_videos/       # Tracked videos with bounding boxes
            ├── tracked_dog_FrontDoor_20251129.mp4
            └── tracked_bird_Garden_20251129.mp4
```

## Implementation Plan

### Priority 1: Fix Video Tracking (CRITICAL)

**Problem**: 258-byte files indicate codec failure

**Solution**: Implement smart codec fallback
```python
CODEC_FALLBACK = [
    ('avc1', 'H.264 - best quality, modern'),
    ('mp4v', 'MPEG-4 - most compatible'),
    ('XVID', 'Xvid - legacy fallback'),
    ('MJPG', 'Motion JPEG - always works'),
]

def create_video_writer(path, fps, width, height):
    for codec, desc in CODEC_FALLBACK:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if out.isOpened():
            logger.info(f"Video writer using codec: {codec} ({desc})")
            return out
        logger.warning(f"Codec {codec} failed, trying next...")
    
    raise RuntimeError("All codecs failed!")
```

**Files to modify**:
- `src/object_detector.py`: Update `track_and_annotate_video()`

### Priority 2: Clear Old Data

**Action**: Remove all old sorted/review data
```bash
ssh root@192.168.1.55
cd /mnt/user/data/CritterCatcher/
rm -rf sorted/*
rm -rf review/*
rm -rf objects/detected/annotated_videos/*
```

**Why**: Old data from V1 system is confusing current behavior

### Priority 3: Implement Hybrid Workflow

**A. Update Video Sorter**
- `src/video_sorter.py`: Add `sort_by_yolo_category()` method
- Create `/data/review/{yolo_category}/` folders
- Save metadata JSON with YOLO confidence

**B. Update Processing Pipeline**
- `src/main.py`: After YOLO detection, immediately sort by category
- Then run CLIP profiles for categories they monitor
- Move videos based on CLIP results

**C. Update CLIP Profile Matching**
- Check all CLIP profiles that use detected category
- Run CLIP classification for each
- Use highest confidence match
- Implement three-tier threshold system

**Files to modify**:
- `src/main.py` - Processing pipeline
- `src/video_sorter.py` - Add YOLO-based sorting
- `src/processing_pipeline.py` - Update routing logic

### Priority 4: Update Review Interface

**A. Add YOLO Category Tabs**
- Show tabs for all YOLO categories with videos
- Example: "dog (23) | cat (12) | bird (5)"

**B. Show CLIP Results (if any)**
- Display which CLIP profiles ran
- Show confidence scores
- Indicate why video is in this folder

**C. Batch Actions**
- Select multiple videos
- "Create CLIP Profile from Selection"
- "Move to different category"
- "Delete false positives"

**Files to modify**:
- `src/static/review.html` - UI updates
- `src/webapp.py` - New API endpoints

## Configuration Changes

Add to `config.yaml`:
```yaml
tracking:
  enabled: true
  save_original_videos: false  # Space saving
  codec_preference:
    - avc1
    - mp4v
    - XVID
    - MJPG

detection:
  confidence_threshold: 0.25    # YOLO minimum
  yolo_sort_threshold: 0.40     # Below this goes to needs_review/
  clip_high_confidence: 0.80    # Auto-sort
  clip_maybe_confidence: 0.50   # Goes to maybe/ folder
  
sorting:
  use_yolo_folders: true        # Enable YOLO-first sorting
  use_clip_refinement: true     # Enable CLIP on top
  create_maybe_folders: true    # Enable near-miss folders
```

## Success Criteria

### Video Tracking Fixed
- ✅ Videos > 1MB (not 258 bytes)
- ✅ Playable in browser/VLC
- ✅ Bounding boxes visible and tracking objects
- ✅ Labels readable at all positions

### Hybrid Workflow Working
- ✅ YOLO detects dog → goes to `/data/review/dog/`
- ✅ CLIP processes and moves high-confidence matches
- ✅ Unmatched detections stay in YOLO category folder
- ✅ User can review by YOLO category

### Clean Data
- ✅ No old V1 data remaining
- ✅ All videos have proper metadata
- ✅ Folder structure matches design

### User Experience
- ✅ Can see all dogs before creating Jack Russell profile
- ✅ Can identify new animals to train CLIP on
- ✅ Clear separation of YOLO vs CLIP results
- ✅ Fast review with video previews

## Testing Plan

1. **Clear all old data**
2. **Download 5 videos** with known content (dog, cat, bird, person, car)
3. **Run processing** without CLIP profiles
4. **Verify** videos sorted to correct YOLO categories
5. **Check tracked videos** are playable with correct bounding boxes
6. **Create Jack Russell CLIP profile**
7. **Re-process** dog videos
8. **Verify** Jack Russells moved to sorted, others stay in review
9. **Test review UI** - can browse by category

## Rollback Plan

If issues occur:
1. Keep old code in git branch
2. Can revert to pure CLIP-only system
3. Docker container can be rebuilt from previous commit

## Future Enhancements (Post-Implementation)

1. **Video preview in review page** - See first 2 seconds
2. **Statistics dashboard** - Show detection counts
3. **CLIP training wizard** - One-click profile creation
4. **Batch metadata editing** - Fix miscategorized videos
5. **Export reports** - CSV of detections for analysis

## Notes

- YOLO categories must be enabled in YOLO Categories tab
- CLIP profiles auto-enable their YOLO categories
- System is backward compatible - CLIP profiles still work
- Users can disable YOLO-first sorting via config if needed
