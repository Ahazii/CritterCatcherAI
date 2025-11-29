# CritterCatcherAI Development Context - November 28, 2025

## Project Overview
CritterCatcherAI runs in a Docker container on an Unraid server at IP `192.168.1.55`. The system downloads video files from Ring cameras, uses AI to detect specific animals/objects, isolates relevant videos into folders on a Plex server, and deletes others.

**GitHub Repository:** https://github.com/Ahazii/CritterCatcherAI.git  
**Container:** `ghcr.io/ahazii/crittercatcherai:latest`  
**Container Name:** `crittercatcher`  
**Web Interface:** Running on port exposed by container  

## Unraid Server Details
- **IP Address:** 192.168.1.55
- **SSH Access:** `ssh root@192.168.1.55`
- **Container Path:** `/mnt/user/appdata/crittercatcher`
- **Docker Management:** Via Unraid web UI (dockerman) - NO docker-compose installed
- **Note:** Source code is NOT cloned locally on the server - container runs from pre-built image

## Recent Development Session (Nov 28, 2025)

### Issues Addressed

#### Issue 1: Rejected Videos Being Re-Downloaded
**Problem:** Videos that were rejected during processing were being downloaded again on subsequent download runs.

**Root Cause:** When videos were rejected, they were deleted from `/data/review/` but NOT marked in the download tracker database, so the downloader didn't know they had been processed.

**Solution (Commit `afcddaf`):**
- Modified `src/webapp.py` to initialize download_tracker on startup
- Updated `reject_videos_background()` function to extract event_id from filename (format: `CameraName_timestamp_eventid.mp4`)
- Call `download_tracker.update_status(event_id, 'rejected')` for each rejected video
- Download logic already checks database, so rejected videos are now skipped

**Database:** `/data/download_history.db`  
**Status values:** 'downloaded', 'processed', 'rejected', 'failed', '404'

#### Issue 2: Blocking Progress Modal
**Problem:** When running background tasks (Confirm as Person, Download All, etc.), a modal would block the entire UI, preventing navigation.

**User Request:** Replace blocking modal with a non-blocking status widget that allows navigation while tasks run in background.

**Solution - Non-Blocking Status Widget:**

Created `src/static/status_widget.html` (413 lines) with:
- **Fixed position widget** in top-right corner (top: 70px, right: 20px, z-index: 9998)
- **Purple gradient header** with minimize/close buttons
- **Progress bars** showing percentage, message, and details for each task
- **Small indicator badge** (z-index: 9997) when widget closed but tasks active
- **Collapsible and draggable** interface
- **Multiple concurrent tasks** supported via `window.activeTasks` Map

**Modified Files:**
- `src/static/review.html` - Added widget loading code, updated button handlers
- `src/static/index.html` - Added widget loading code for dashboard

**Widget Loading Pattern:**
```javascript
fetch('/static/status_widget.html')
  .then(response => response.text())
  .then(html => {
    // Extract script content since innerHTML doesn't execute scripts
    const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
    const htmlWithoutScript = html.replace(/<script>[\s\S]*?<\/script>/, '');
    container.innerHTML = htmlWithoutScript;
    
    // Execute script separately in global context
    const scriptEl = document.createElement('script');
    scriptEl.textContent = scriptMatch[1];
    document.body.appendChild(scriptEl);
  });
```

**Button Handler Updates:**
- `confirmPerson()`, `confirmVideos()`, `rejectVideos()` now call `startTaskTracking(taskId, taskName)`
- Removed old modal functions: `showProgressModal()`, `closeProgressModal()`

### Issue 3: Widget Not Persisting Across Pages

**Problem:** Widget appeared when task started, but disappeared when navigating between Dashboard and Review pages. Tasks were still running but invisible.

**Solution - sessionStorage Persistence (Commits `fdff936`, `902de65`):**

**Added Functions:**
- `loadTasksFromStorage()` - Restores tasks from sessionStorage on page load
- `saveTasksToStorage()` - Saves task state after every update
- `startPolling(taskId)` - Resumes polling for active tasks after page load

**sessionStorage Format:**
```json
[
  {
    "id": "task-uuid",
    "name": "Extracting Faces",
    "status": "running",
    "progress": 45,
    "message": "Processing frame 45/100",
    "details": "Detected 3 faces",
    "error": null
  }
]
```

**Auto-Restore Logic:**
- On page load, check sessionStorage for active tasks
- If task status is 'running' or 'pending', resume polling
- Show widget/indicator automatically if active tasks exist

**Task Lifecycle:**
1. Task starts → added to `window.activeTasks` Map
2. Every progress update → saved to sessionStorage
3. Task completes/fails → removed from Map after 5 seconds
4. Page navigation → tasks loaded from sessionStorage and polling resumes

### Issue 4: sessionStorage Not Loading (FINAL FIX)

**Problem:** Even after implementing sessionStorage persistence, tasks were not restoring on page navigation.

**Root Cause:** Widget HTML is loaded dynamically via `fetch()` AFTER the main page's `DOMContentLoaded` event has already fired. The widget script was waiting for `DOMContentLoaded` which never came.

**Solution (Commit `902de65`):**
Changed from:
```javascript
document.addEventListener('DOMContentLoaded', () => {
    loadTasksFromStorage();
    // ... show widget if active tasks
});
```

To immediate execution:
```javascript
(function() {
    loadTasksFromStorage();
    // ... show widget if active tasks
})();
```

**Status:** Code committed and pushed. Container needs rebuild to apply fix.

## Current State

### Git Commits (Most Recent First)
- `902de65` - Fix sessionStorage loading - execute immediately instead of DOMContentLoaded
- `fdff936` - Implemented global persistence with sessionStorage  
- `13b60a4` - Added widget to dashboard (index.html)
- `6aaf0e7` - Fixed script execution via separate script element
- `1e39c80` - Fixed race condition with fallback function
- `afcddaf` - Fix download tracker and implement non-blocking status widget

### Pending Actions
**Container rebuild required** to apply commit `902de65` fix.

**Rebuild Steps:**
1. Go to Unraid Docker management UI
2. Find CritterCatcherAI container
3. Force update/rebuild (pulls latest image)

OR via SSH:
```bash
ssh root@192.168.1.55
docker stop crittercatcher
docker pull ghcr.io/ahazii/crittercatcherai:latest
docker start crittercatcher
```

### Testing Checklist (After Rebuild)
1. ✅ Start a task on Review page (e.g., "Confirm as Person")
2. ✅ Verify widget appears with progress
3. ✅ Navigate to Dashboard - widget should persist
4. ✅ Navigate back to Review - widget should still show
5. ✅ Check browser console for: `"Restored X task(s) from storage"`
6. ✅ Verify `sessionStorage.getItem('activeTasks')` contains task JSON
7. ✅ Start "Download All" task from Dashboard - widget should appear
8. ✅ Let task complete - widget should auto-close after 7 seconds (5s delay + 2s fade)

## Key Technical Details

### File Structure
```
src/
├── webapp.py                    # Main Flask app (download tracker integration)
├── download_tracker.py          # Database tracking of downloaded videos
├── static/
│   ├── status_widget.html       # Non-blocking status widget component
│   ├── review.html              # Review page (widget integrated)
│   └── index.html               # Dashboard (widget integrated)
└── templates/
    └── ...
```

### Database Schema
**File:** `/data/download_history.db`

**Table:** `download_history`
- `event_id` (TEXT PRIMARY KEY) - Ring event ID from filename
- `camera_name` (TEXT)
- `timestamp` (TEXT)
- `download_date` (TEXT)
- `status` (TEXT) - Values: 'downloaded', 'processed', 'rejected', 'failed', '404'

### API Endpoints Used by Widget
- `POST /api/process/confirm_person` → Returns task_id
- `POST /api/process/reject` → Returns task_id
- `POST /api/process/download_all` → Returns task_id
- `GET /api/progress/{task_id}` → Returns progress data
  ```json
  {
    "status": "running|completed|failed",
    "progress_percentage": 0-100,
    "message": "Current operation",
    "details": "Additional info",
    "error": "Error message if failed"
  }
  ```

### Widget Styling
- **Widget:** Fixed position, top: 70px, right: 20px, z-index: 9998
- **Indicator:** Fixed position, top: 70px, right: 20px, z-index: 9997
- **Colors:** Purple gradient header (#8b5cf6 to #7c3aed), white text
- **Animations:** Smooth transitions, spinner animation for active tasks
- **Responsive:** Auto-hides when no tasks, shows indicator when closed with active tasks

### Known Behaviors
- Tasks auto-remove from widget 5 seconds after completion/failure
- Widget auto-closes 2 seconds after last task removed
- Polling interval: 500ms (2 requests per second per active task)
- Widget persists during browser session (sessionStorage cleared on tab close)
- Multiple tasks can run concurrently, each tracked independently

## Development Environment

### Local Machine (Windows)
- **Path:** `C:\Coding\CritterCatcherAI`
- **Shell:** PowerShell 5.1
- **Git:** Local repository, pushes to GitHub
- **Note:** PowerShell doesn't support `&&` operator - use separate commands

### Docker Container (Unraid)
- **Container Name:** `crittercatcher`
- **Image:** `ghcr.io/ahazii/crittercatcherai:latest`
- **Management:** Unraid web UI (no docker-compose)
- **Data Persistence:** `/mnt/user/appdata/crittercatcher` mounted to `/data` in container

## Troubleshooting

### Widget Not Appearing
1. Check browser console for "Status widget loaded successfully"
2. Check `sessionStorage.getItem('activeTasks')` for saved tasks
3. Verify script execution: look for "Restored X task(s) from storage"
4. Check widget HTML loaded: `document.getElementById('statusWidget')`

### Tasks Not Persisting Across Pages
1. Verify commit `902de65` is deployed (check container image date)
2. Check browser console for IIFE execution (should run immediately)
3. Inspect sessionStorage manually in DevTools → Application → Storage

### Download Tracker Issues
1. Check database exists: `ls -la /data/download_history.db` (in container)
2. Verify `download_tracker` initialized in webapp.py startup
3. Check logs for "update_status" calls during rejection
4. Query database: `sqlite3 /data/download_history.db "SELECT * FROM download_history WHERE status='rejected';"`

## User Configuration Rules
1. **Git Commits:** Always push and commit after Docker container changes (use short commands)
2. **Logging:** All features should have configurable logging to maintain consistency
3. **Permissions:** Folders created by AI should have read/write for all users (applies to local and server)
4. **Project Nature:** AI-powered video processing system for wildlife detection on Ring cameras

## Next Steps (When Resuming)
1. Rebuild container on Unraid server to apply commit `902de65`
2. Test complete widget persistence workflow
3. Verify download tracker prevents re-downloading rejected videos
4. Optional: Add user preferences for widget position/behavior
5. Optional: Add ability to cancel running tasks from widget

---

**Document Created:** November 28, 2025  
**Last Commit:** `902de65` - Fix sessionStorage loading  
**Status:** Ready for container rebuild and testing
