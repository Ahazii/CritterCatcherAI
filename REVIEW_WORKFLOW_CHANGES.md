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

## Future Enhancements

Potential improvements for consideration:
1. Add checkbox to optionally disable "keep video" copy
2. Show video count in `/data/sorted/{profile}/` on profile pages
3. Add "View Sorted Videos" page to browse kept videos
4. Option to move (instead of copy) video to sorted if disk space is concern
