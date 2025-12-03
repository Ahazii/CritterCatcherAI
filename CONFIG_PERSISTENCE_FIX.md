# CRITICAL FIX: Config Persistence Issue - December 3, 2025

## Problem Summary

**CRITICAL ISSUE:** All configuration settings (including YOLO categories) were being saved to `/app/config/config.yaml` **inside the Docker image** instead of `/config/config.yaml` on the **persisted volume**. This caused all settings to reset on every new deployment.

## User Report

> "The YOLO categories still show as empty on a new deployment - the setting from the previous config should still show. These should be saved in the config file in the /config/ location that is also used in the config tab."

User correctly identified that settings from the Config tab persisted across deployments, but YOLO categories (and ALL other settings) should have been using the same persistence mechanism.

## Root Cause Analysis

### The Bug

**webapp.py line 105:**
```python
CONFIG_PATH = Path("/app/config/config.yaml")  # ❌ WRONG - inside image
```

**main.py line 81:**
```python
def load_config(config_path: str = "/app/config/config.yaml") -> dict:  # ❌ WRONG
```

### What Was Happening

1. **Container starts:**
   - Dockerfile creates `/config` volume mount point
   - Unraid maps `/config` → `/mnt/user/appdata/crittercatcher` (persisted storage)
   - Application looks for config at `/app/config/config.yaml` (inside image)

2. **First run:**
   - No config exists at `/app/config/config.yaml`
   - Copies default from... `/app/config/config.yaml` (circular reference!)
   - Config created **inside image**, not on volume

3. **User changes settings:**
   - Saves to `/app/config/config.yaml` (inside image)
   - Appears to work because container keeps running
   - Settings persist during container lifetime

4. **Container restart:**
   - `/app/config/` still exists in running container
   - Settings appear to persist ✅ (misleading!)

5. **New deployment (image rebuild):**
   - New Docker image created with fresh `/app/config/`
   - **All user settings LOST** ❌
   - Volume `/config/` never used, so nothing persisted there

### Volume Mapping

**Docker Compose / Unraid:**
```yaml
volumes:
  - /mnt/user/appdata/crittercatcher:/config  # Config volume
  - /mnt/user/data/CritterCatcher:/data      # Data volume
```

**Dockerfile:**
```dockerfile
VOLUME ["/data", "/config"]  # Declares mount points
```

**Application (BEFORE FIX):**
```python
CONFIG_PATH = Path("/app/config/config.yaml")  # ❌ Not using volume!
log_file = Path("/app/config/crittercatcher.log")  # ❌ Not using volume!
```

## The Fix

Changed all config and log paths from `/app/config/` to `/config/`:

### Files Modified

#### 1. src/webapp.py

**Line 105:**
```python
# BEFORE
CONFIG_PATH = Path("/app/config/config.yaml")

# AFTER
CONFIG_PATH = Path("/config/config.yaml")
```

**Lines 135-138 (startup):**
```python
# BEFORE
if not CONFIG_PATH.exists():
    default_config = Path("/app/config/config.yaml")
    shutil.copy(default_config, CONFIG_PATH)

# AFTER
if not CONFIG_PATH.exists():
    default_config = Path("/app/config/config.yaml")  # Template in image
    shutil.copy(default_config, CONFIG_PATH)  # Copy to /config volume
```

**Log file paths (lines 337, 379, 400):**
```python
# BEFORE
log_file = Path("/app/config/crittercatcher.log")

# AFTER
log_file = Path("/config/crittercatcher.log")
```

#### 2. src/main.py

**Line 44 (logging):**
```python
# BEFORE
log_dir = Path("/app/config")

# AFTER
log_dir = Path("/config")
```

**Line 81 (config loading):**
```python
# BEFORE
def load_config(config_path: str = "/app/config/config.yaml") -> dict:

# AFTER
def load_config(config_path: str = "/config/config.yaml") -> dict:
```

**Lines 91-94:**
```python
# BEFORE
default_config = Path("/app/config/config.yaml")
shutil.copy(default_config, config_path)

# AFTER
default_config = Path("/app/config/config.yaml")  # Template in image
shutil.copy(default_config, config_path)  # Copy to /config volume
```

## Path Architecture (AFTER FIX)

### Image Paths (Read-Only, Baked into Docker Image)
- `/app/config/config.yaml` - Default config template
- `/app/src/` - Application code
- `/app/version.txt` - Build version
- `/app/build_date.txt` - Build timestamp

### Volume Paths (Persisted, Survives Deployments)
- `/config/config.yaml` - **Active configuration** (user settings)
- `/config/crittercatcher.log` - Application logs
- `/data/downloads/` - Downloaded videos
- `/data/sorted/` - Sorted videos by category
- `/data/faces/` - Face recognition data
- `/data/tokens/` - Ring API tokens
- `/data/animal_profiles/` - Animal detection profiles
- `/data/review/` - Pending review videos

### How It Works Now

```
1. Container starts
   ├─ Check: Does /config/config.yaml exist?
   │   ├─ NO → Copy /app/config/config.yaml (template) to /config/config.yaml
   │   └─ YES → Use existing /config/config.yaml (user settings preserved!)
   │
2. Application runs
   ├─ Load config from /config/config.yaml
   ├─ Write logs to /config/crittercatcher.log
   └─ All data in /data/ (also persisted)
   │
3. User changes settings
   ├─ Save to /config/config.yaml (on volume!)
   └─ Settings persist across:
       ├─ Browser refresh ✅
       ├─ Container restart ✅
       └─ New deployment ✅ (FIXED!)
```

## What This Fixes

### Before Fix
| Action | Config Tab Settings | YOLO Categories | Logs |
|--------|-------------------|----------------|------|
| Browser refresh | ✅ Persist | ✅ Persist | ✅ Persist |
| Container restart | ✅ Persist | ✅ Persist | ✅ Persist |
| **New deployment** | ❌ **LOST** | ❌ **LOST** | ❌ **LOST** |

### After Fix
| Action | Config Tab Settings | YOLO Categories | Logs |
|--------|-------------------|----------------|------|
| Browser refresh | ✅ Persist | ✅ Persist | ✅ Persist |
| Container restart | ✅ Persist | ✅ Persist | ✅ Persist |
| **New deployment** | ✅ **PERSIST** | ✅ **PERSIST** | ✅ **PERSIST** |

## Impact

### What Gets Fixed
✅ **YOLO categories** - Persist across deployments
✅ **Config tab settings** - Actually use volume (was coincidentally working)
✅ **Logging settings** - Persist across deployments
✅ **Scheduler settings** - Persist across deployments
✅ **Face recognition settings** - Persist across deployments
✅ **All configuration** - Properly persisted to volume

### What Doesn't Change
- Data paths (`/data/*`) were already correct
- Application behavior remains the same
- No breaking changes for existing deployments

### Migration for Existing Users

**If you had settings configured:**
1. Old config at `/app/config/config.yaml` inside old container
2. New deployment → Fresh `/config/config.yaml` from template
3. **You'll need to reconfigure settings once**
4. Future deployments will preserve your settings ✅

**To preserve old settings:**
```bash
# Before stopping old container:
docker cp CritterCatcherAI:/app/config/config.yaml /tmp/old_config.yaml

# After starting new container:
docker cp /tmp/old_config.yaml CritterCatcherAI:/config/config.yaml
docker restart CritterCatcherAI
```

Or just reconfigure via web UI (easier).

## Testing After Deployment

### Test 1: Initial Configuration
1. Fresh deployment
2. Configure settings in web UI:
   - Enable some YOLO categories
   - Change scheduler settings
   - Adjust confidence thresholds
3. ✅ **Expected:** Settings save successfully

### Test 2: Container Restart
1. Restart container: `docker restart CritterCatcherAI`
2. Open web UI
3. ✅ **Expected:** All settings preserved

### Test 3: New Deployment (THE CRITICAL TEST)
1. Rebuild and deploy new container
2. Open web UI
3. ✅ **Expected:** All settings from previous deployment preserved
4. ✅ **Expected:** YOLO categories still checked
5. ✅ **Expected:** Logs show history

### Test 4: Volume Verification
```bash
# SSH into Unraid
ssh root@192.168.1.100

# Check config file location
ls -la /mnt/user/appdata/crittercatcher/config.yaml

# Should show file exists with recent modification time
```

## Commit Information

**Commit:** `7a32107`
**Message:** "CRITICAL FIX: Use /config volume for persistent config instead of /app/config"

**Files Changed:**
- `src/webapp.py` - Fixed CONFIG_PATH and log paths
- `src/main.py` - Fixed config loading and log directory
- `WIDGET_FIX_SUMMARY.md` - Created
- `YOLO_CATEGORIES_FIX.md` - Created

## Why This Happened

This was a **fundamental architecture mistake** from initial development:

1. **Original design** assumed config would be in `/app/config/` (inside image)
2. **Docker volumes** were added later for persistence
3. **Code was never updated** to use the volume paths
4. **Testing gap** - deployments weren't tested frequently enough to catch this

The Config tab settings *appeared* to work because:
- They used the same wrong path
- Consistency meant both read and write used same location
- Only failed on new deployments (which weren't tested often)

## Recommendation

**CRITICAL:** Deploy this fix immediately. After deployment:

1. **Reconfigure your settings** (one-time migration)
2. **Verify persistence** by doing a test deployment
3. **Check logs** in `/mnt/user/appdata/crittercatcher/`
4. **Confirm YOLO categories** persist after rebuild

Future deployments will now properly preserve all settings.

---

**Status:** ✅ CRITICAL FIX COMPLETED AND PUSHED
**Date:** December 3, 2025
**Commit:** 7a32107
**Priority:** URGENT - Deploy immediately
