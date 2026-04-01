# Review Workflow Changes
**Date:** March 31, 2026  
**Changes:** Two-tier profile selection + Keep & Train workflow

---

## Problem Solved

### Issue 1: Limited Profile Selection
**Before:** When reviewing videos in a category (e.g., "dog"), you could only assign them to profiles that monitor that specific YOLO category.

**Problem:** If YOLO misclassified a hedgehog as "dog", you couldn't assign it to the "hedgehog" profile.

### Issue 2: Videos Being "Deleted"
**Before:** When assigning videos to a profile, they were moved to `/data/training/{profile}/confirmed/` for AI training only. The original videos were not kept for viewing.

**Problem:** User manually confirmed hedgehog videos were not being saved permanently - only training frames were kept.

---

## Solutions Implemented

### Change 1: Two-Tier Profile Selection
**File:** `src/static/review.html`

The profile dropdown now shows ALL profiles, organized into two groups:

```
┌─────────────────────────────────────┐
│ Assign to Profile:                  │
│ ┌─────────────────────────────────┐ │
│ │ ✓ Monitors "dog"                │ │
│ │  - Hedgehog                     │ │
│ │  - Fox                          │ │
│ │ ─────────────────────────────── │ │
│ │ 📂 Other Profiles               │ │
│ │  - Blue Jay (monitors: bird)    │ │
│ │  - Cardinal (monitors: bird)    │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

**Benefits:**
- ✅ Can assign videos to ANY profile (corrects YOLO misclassifications)
- ✅ Visual guidance - matching profiles shown first
- ✅ Clear indication of which categories each profile monitors

---

### Change 2: Keep AND Train Workflow
**File:** `src/webapp.py` - `/api/review/assign-to-profile` endpoint

When assigning videos to a profile, the system now does BOTH:

#### Before (Training Only):
```
/data/review/dog/video_001.mp4
↓
Extract frames only
↓
/data/training/hedgehog/confirmed/video_001_frame_000000.jpg
/data/training/hedgehog/confirmed/video_001_frame_000001.jpg
...
[Original video deleted]
```

#### After (Keep + Train):
```
/data/review/dog/video_001.mp4
↓
1. Extract frames for AI training
↓
/data/training/hedgehog/confirmed/video_001_frame_000000.jpg
/data/training/hedgehog/confirmed/video_001_frame_000001.jpg
...
↓
2. COPY original video to sorted folder
↓
/data/sorted/hedgehog/video_001.mp4  ← YOU KEEP THIS!
```

**What happens:**
1. **Extract frames** → `/data/training/{profile}/confirmed/*.jpg` (for CLIP training)
2. **Copy video** → `/data/sorted/{profile}/*.mp4` (for permanent storage)
3. **Remove from review** → Cleans up `/data/review/{category}/`

**Benefits:**
- ✅ Videos permanently saved in `/data/sorted/{profile}/`
- ✅ AI gets training data from extracted frames
- ✅ Review folder stays clean
- ✅ Videos NOT deleted anymore

---

## Technical Details

### Files Modified
1. **`src/static/review.html`**
   - Added `updateProfileDropdown()` function
   - Groups profiles by category match
   - Updates dropdown when switching categories

2. **`src/webapp.py`**
   - Modified `/api/review/assign-to-profile` endpoint
   - Added video copy to `/data/sorted/{profile_id}/`
   - Sets proper permissions (rwxrwxrwx per user rules)
   - Handles duplicate filenames
   - Improved logging and success messages

### Directory Structure After Assignment
```
/data/
├── review/
│   └── dog/                    # Empty after assignment
│
├── training/
│   └── hedgehog/
│       └── confirmed/
│           ├── video_001_frame_000000.jpg  # For AI training
│           ├── video_001_frame_000001.jpg
│           └── video_001.mp4               # Original in training
│
└── sorted/
    └── hedgehog/
        └── video_001.mp4                   # YOUR PERMANENT COPY
```

---

## User Impact

### For Manual Review:
- **More flexibility:** Assign videos to any profile, not just matching categories
- **Better organization:** Visual grouping shows relevant profiles first
- **Corrects mistakes:** Can fix YOLO misclassifications easily

### For Video Collection:
- **Videos are kept:** All confirmed videos saved to `/data/sorted/{profile}/`
- **Easy access:** Videos organized by profile name
- **No data loss:** Videos no longer disappear after assignment

### For AI Training:
- **Still works:** Frame extraction continues as before
- **Automatic training:** CLIP trains when batch size reached (10+ confirmations + 10+ negatives)
- **Better results:** Can now include corrected videos in training data

---

## Deployment

To deploy these changes to your Unraid server:

```powershell
# From Windows (local development)
git add -A
git commit -m "Fix review workflow: two-tier profile selection + keep videos

- Add two-tier profile dropdown (matching + other profiles)
- Copy videos to /data/sorted/ when assigning to profiles
- Videos now kept permanently instead of being deleted
- Maintain training data extraction for CLIP

Co-Authored-By: Warp <agent@warp.dev>"
git push
```

Then on Unraid server:
```bash
ssh root@192.168.1.55
cd /mnt/user/appdata/crittercatcher
git pull
docker-compose down
docker-compose up -d --build
```

Or use Unraid Docker UI to "Force Update" the container.

---

## Testing Checklist

After deployment, verify:
- [ ] Review page loads without errors
- [ ] Profile dropdown shows two groups when applicable
- [ ] Can assign videos from any category to any profile
- [ ] Videos appear in `/data/sorted/{profile}/` after assignment
- [ ] Training frames still extracted to `/data/training/{profile}/confirmed/`
- [ ] Videos removed from `/data/review/{category}/` after assignment
- [ ] Success message shows correct path
- [ ] Logs show "Copied video to sorted/{profile_id}"

---

## Configuration

No configuration changes needed. The system will:
- Extract up to 10 frames per video (configurable via `animal_training.batch_size`)
- Set folder permissions to 777 (rwxrwxrwx per user rules)
- Handle duplicate filenames automatically
- Continue automatic CLIP training when conditions met

---

---

## Additional Improvements (April 1, 2026)

### Change 3: Negative Training Examples
**Files:** `src/static/review.html`, `src/webapp.py`

Added ability to save rejected videos as negative training examples:

**New "Save rejects as negatives" checkbox:**
- When enabled, rejected videos extract frames to `/data/training/{profile}/rejected/`
- Automatically triggers CLIP training when thresholds met (10+ positives + 10+ negatives)
- Negatives counter updates on profile page
- Training happens automatically in background

**Benefits:**
- ✅ Can train CLIP to avoid false positives
- ✅ Example: Mark "cat" videos as negative for "hedgehog" profile
- ✅ Improves classification accuracy over time

### Change 4: Training Status Display
**File:** `src/static/profiles.html`

Enhanced CLIP profile training status section:

**Now shows:**
- **Model status**: trained / not trained
- **Last trained**: date and time of last training
- **Positives**: X / 10 needed (color-coded: green if ready, orange if not)
- **Negatives**: X / 10 needed (color-coded: green if ready, red if not)
- **New since training**: count of new positive examples
- **Ready to train alert**: visual indicator when both thresholds met

**"Refresh status" button:**
- Updates counts without reloading page
- Useful after adding training data via review

### Change 5: Confidence Metadata for Frames
**File:** `src/webapp.py`

Training frames now include confidence scores:

**Metadata saved with each frame:**
```json
{
  "source_video": "video_001.mp4",
  "profile_id": "hedgehog",
  "frame_index": 0,
  "confidence": 0.87,
  "timestamp": "2026-04-01T01:30:00"
}
```

**Benefits:**
- ✅ "Viz" button shows highest-confidence frames first
- ✅ Better quality visualization of training data
- ✅ Confidence pulled from video's CLIP/YOLO detection score

**Note:** Frames are still extracted at 1 FPS intervals from entire video. Future enhancement could filter to only frames with actual bounding box detections.

---

## Current Workflow Summary

### Reviewing Videos
1. Navigate to Review page
2. Select category (e.g., "dog", "cat", "bird")
3. Watch video to confirm/reject

### Assigning to Profile (Positive Examples)
1. Select profile from dropdown (two-tier: matching first, then others)
2. Check "Save rejects as negatives" if you want negative training
3. Click "Assign to Profile"
4. System:
   - Extracts frames → `/data/training/{profile}/confirmed/`
   - Copies video → `/data/sorted/{profile}/`
   - Updates positive count
   - Triggers training if 10+ positives AND 10+ negatives

### Rejecting Videos (Negative Examples)
1. Select profile from dropdown
2. Check "Save rejects as negatives" checkbox
3. Click "Reject"
4. System:
   - Extracts frames → `/data/training/{profile}/rejected/`
   - Deletes video from review
   - Updates negative count
   - Triggers training if 10+ positives AND 10+ negatives

### Viewing Training Status
1. Navigate to CLIP Profiles page
2. Each profile shows:
   - Model trained status
   - Last training date/time
   - Positive/negative frame counts with visual indicators
   - "Ready to train" alert when thresholds met
3. Click "Refresh status" to update counts
4. Click "Viz" to see top 10 highest-confidence training frames

---

## Recent Bug Fixes

### Fix 1: Missing shutil import (April 1, 2026)
**Commit:** `5fad4b2`

Fixed `NameError: name 'shutil' is not defined` in reject videos background task. Added `import shutil` to module-level imports in `webapp.py`.

### Fix 2: Rejected count not updating (March 31, 2026)
**Commit:** `cbe2ad9`

Fixed issue where using Reject button with "save as negatives" didn't update the `rejected_count`. Now properly increments counter and triggers training check.

---

## Testing Checklist (Updated)

After deployment, verify:
- [x] Review page loads without errors
- [x] Profile dropdown shows two groups when applicable
- [x] Can assign videos from any category to any profile
- [x] Videos appear in `/data/sorted/{profile}/` after assignment
- [x] Training frames extracted to `/data/training/{profile}/confirmed/`
- [x] Videos removed from `/data/review/{category}/` after assignment
- [x] "Save rejects as negatives" checkbox works
- [x] Negative frames extracted to `/data/training/{profile}/rejected/`
- [x] Negative count updates on profiles page
- [x] Training triggers automatically when 10+ positives AND 10+ negatives
- [x] "Refresh status" button updates training counts
- [x] "Last trained" date displays correctly
- [x] "Viz" button shows training frames sorted by confidence
- [x] Frame metadata includes confidence scores

---

## Future Enhancements

Potential improvements for consideration:
1. Filter training frames to only those with actual YOLO detections and bounding boxes
2. Draw bounding boxes on saved training frames for better visualization
3. Add frame-level confidence scores (currently using video-level confidence)
4. Show video count in `/data/sorted/{profile}/` on profile pages
5. Add "View Sorted Videos" page to browse kept videos
6. Option to move (instead of copy) video to sorted if disk space is concern
7. Bulk operations for assigning/rejecting multiple videos at once
