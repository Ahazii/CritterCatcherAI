# Bug Fixes: Log Viewer Auto-Scroll & Downloads Folder Videos

## Date: 2025-12-03

## Bug 1: Log Viewer Auto-Scroll Issue

### Problem
The Logs tab auto-scroll feature was forcing the view to the bottom every 5 seconds (on refresh), making it impossible for users to scroll up and read older log entries.

### Root Cause
The `renderLogs()` function in `src/static/index.html` (line 3101-3104) was checking if the auto-scroll checkbox was enabled, but not checking if the user had manually scrolled up to read older logs.

### Solution
Modified the `renderLogs()` function to:
1. Check if user is already at/near bottom BEFORE rendering (within 50px tolerance)
2. Only auto-scroll if BOTH conditions are true:
   - Auto-scroll checkbox is enabled
   - User was already at bottom (hasn't scrolled up)

### Files Changed
- `src/static/index.html` (lines 3070-3109)

### Code Change
```javascript
// Before
if (document.getElementById('logAutoScroll').checked) {
    logViewer.scrollTop = logViewer.scrollHeight;
}

// After
const wasAtBottom = logViewer.scrollHeight - logViewer.scrollTop - logViewer.clientHeight < 50;
// ... render logs ...
if (document.getElementById('logAutoScroll').checked && wasAtBottom) {
    logViewer.scrollTop = logViewer.scrollHeight;
}
```

---

## Bug 2: Videos Not Moving Out of Downloads Folder

### Problem
Videos in the `/data/downloads/` folder were not being moved to review folders during processing when YOLO detected no objects. Only JSON metadata files remained in downloads, but the actual `.mp4` files stayed there.

### Root Cause
In `src/main.py` (lines 418-419), when `detected_objects` was empty (no YOLO detections), the code logged "No objects detected, skipping YOLO sorting and tracking" and did nothing else. The video file remained in the downloads folder, and only a JSON metadata file was created there.

Later code at lines 619-627 assumed the video had already been sorted, but for videos with no detections, `video_path` still pointed to downloads folder.

### Solution
Modified `src/main.py` to:
1. When no objects are detected, sort video to "unknown" category in `/data/review/unknown/`
2. Update `video_path` to the new location for subsequent processing
3. Set `yolo_category = "unknown"` and `yolo_confidence = 0.0` for metadata

Also modified `src/video_sorter.py` to:
1. Clean up orphaned JSON metadata files from downloads folder when moving videos
2. Delete old metadata before moving video to prevent accumulation

### Files Changed
- `src/main.py` (lines 418-439)
- `src/video_sorter.py` (lines 204-213)

### Code Changes

**main.py:**
```python
# Before
else:
    logger.debug("No objects detected, skipping YOLO sorting and tracking")

# After
else:
    # No objects detected - move to "unknown" category in review
    logger.info("No objects detected - sorting to 'unknown' category")
    # ... progress update ...
    
    yolo_sorted_path = video_sorter.sort_by_yolo_category(
        video_path,
        yolo_category="unknown",
        confidence=0.0,
        metadata={"all_detections": {}}
    )
    logger.info("Video sorted to /data/review/unknown/")
    
    # Update video_path to the new location
    video_path = yolo_sorted_path
    yolo_category = "unknown"
    yolo_confidence = 0.0
```

**video_sorter.py:**
```python
try:
    # Clean up any existing JSON metadata from downloads folder
    old_metadata = video_path.with_suffix(video_path.suffix + ".json")
    if old_metadata.exists():
        try:
            old_metadata.unlink()
            logger.debug(f"Cleaned up old metadata file: {old_metadata.name}")
        except Exception as cleanup_err:
            logger.warning(f"Failed to cleanup old metadata: {cleanup_err}")
    
    # Move video to review folder
    shutil.move(str(video_path), str(dest_path))
    ...
```

---

## Testing Plan

### Test 1: Log Viewer Auto-Scroll
1. Navigate to Logs tab
2. Ensure auto-scroll is enabled
3. Let logs refresh while at bottom - should auto-scroll ✓
4. Scroll up manually to read older logs
5. Wait for refresh (5 seconds) - should NOT jump to bottom ✓
6. Scroll back to bottom
7. Wait for refresh - should auto-scroll again ✓

### Test 2: Videos in Downloads Folder
1. Check current downloads folder for stranded videos
2. Trigger "Process Now" or wait for scheduled run
3. Verify all videos move to review folders (including "unknown" category)
4. Verify orphaned JSON files are cleaned up from downloads
5. Check `/data/review/unknown/` for videos with no detections

---

## Impact

### Positive
- Users can now scroll up in logs without being forced back to bottom
- All videos are properly moved out of downloads folder
- No orphaned metadata files accumulate in downloads
- Videos with no detections are now properly organized in "unknown" category
- Review UI will show "unknown" category for manual inspection

### No Breaking Changes
- Existing workflow unchanged for videos with detections
- Log viewer still auto-scrolls when user is at bottom
- All metadata handling remains compatible

---

## Related Plan Items
- Plan: ce91ab91-e2c2-432d-8df8-8c9e44d1bee5
- Priority: High (user-reported bugs affecting usability)
