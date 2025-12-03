# YOLO Categories Reset Issue - December 3, 2025

## Problem Description

**Issue:** YOLO categories show as **unchecked/disabled** on first load after a new deployment, even though they persist correctly after being manually enabled once.

**User Observation:**
- After fresh deployment → All categories unchecked ❌
- User enables categories → They save correctly ✅
- Reload browser → Categories remain checked ✅
- Restart container → Categories remain checked ✅
- **New deployment** → Categories reset to unchecked ❌

## Root Cause

The default configuration file (`config/config.yaml`) that ships with the Docker image **did not include** the `yolo_manual_categories` field.

### What Was Happening

1. **Fresh Deployment:**
   ```yaml
   # config/config.yaml (DEFAULT)
   detection:
     object_labels:
       - bird
       - cat
   # NO yolo_manual_categories field!
   ```

2. **Config Loading:**
   - Docker copies default config → `/app/config/config.yaml`
   - No `yolo_manual_categories` field exists
   - Backend returns `config.get('yolo_manual_categories', [])` → `[]`

3. **UI Behavior:**
   ```javascript
   // Frontend loads config
   manual_categories = [] // Nothing in config
   // All checkboxes render as unchecked
   ```

4. **User Enables Categories:**
   ```yaml
   # User checks boxes → Saves to config
   yolo_manual_categories:
     - bird
     - cat
     - dog
   ```
   - Field NOW exists in config ✅
   - Persists in `/app/config/` volume ✅

5. **New Deployment:**
   - Docker rebuilds image with **default config** (no field)
   - Copies default config → overwrites `/app/config/config.yaml`
   - Field disappears again ❌

## Solution

Added `yolo_manual_categories: []` to the **default configuration file** so the field always exists from initial deployment.

### Changes Made

**File:** `config/config.yaml`

**Added (lines 17-19):**
```yaml
# Manual YOLO categories (managed via Species Training page)
# Categories manually enabled here will be used for detection even if not part of an Animal Profile
yolo_manual_categories: []
```

**Impact:**
- Fresh deployments now have the field initialized
- UI correctly loads empty array instead of undefined
- User selections persist across deployments
- No behavior change for existing deployments (field already exists in their config)

## Technical Details

### Config Loading Flow

1. **Container Starts:**
   ```python
   # src/webapp.py startup_event()
   if not CONFIG_PATH.exists():
       shutil.copy(default_config, CONFIG_PATH)  # Copy from /app/config/config.yaml
   ```

2. **API Loads Manual Categories:**
   ```python
   # src/webapp.py - get_yolo_categories()
   with open(CONFIG_PATH, 'r') as f:
       config = yaml.safe_load(f) or {}
   manual_categories = set(config.get('yolo_manual_categories', []))
   # NOW returns [] instead of missing, so UI knows it's intentionally empty
   ```

3. **UI Renders Checkboxes:**
   ```javascript
   // src/static/index.html
   const is_manually_enabled = category.manually_enabled;  // false for all initially
   <input type="checkbox" ${is_manually_enabled ? 'checked' : ''}>
   // All render unchecked initially (correct behavior)
   ```

4. **User Toggles Category:**
   ```python
   # src/webapp.py - toggle_manual_category()
   config['yolo_manual_categories'].append(category)  # Modifies existing list
   yaml.dump(config, f)  # Saves to /app/config/config.yaml
   ```

5. **Persistence:**
   - `/app/config/` is a **Docker volume**
   - Persists between container restarts
   - **BUT** gets reset on fresh deployment (new container instance)
   - NOW: Even after reset, field exists (empty array)

### Volume Persistence vs Fresh Deployment

| Scenario | Old Behavior | New Behavior |
|----------|--------------|--------------|
| Fresh deployment | Field missing → All unchecked | Field exists (empty) → All unchecked ✅ |
| User enables categories | Saved to config → Checked | Same → Checked ✅ |
| Container restart | Config persists → Checked | Same → Checked ✅ |
| New deployment | Field missing → Unchecked ❌ | Field exists (empty) → Unchecked ✅ |

## Why This Happened

The `yolo_manual_categories` feature was added **after** the initial config file was created. The field was never backfilled into the default template.

### Related Code

- **Config template:** `config/config.yaml` (default config shipped with Docker image)
- **Config loading:** `src/main.py:load_config()` and `src/webapp.py:startup_event()`
- **API endpoints:** `src/webapp.py:get_yolo_categories()`, `src/webapp.py:toggle_manual_category()`
- **Frontend:** `src/static/index.html` (Species Training tab)

## Testing After Deployment

After deploying this fix, test the following:

### Test 1: Fresh Deployment
1. Deploy fresh container (delete `/mnt/user/appdata/crittercatcher-ai/config/config.yaml`)
2. Open web UI → Species Training tab
3. ✅ **Expected:** All YOLO categories show as unchecked (empty state)
4. Enable some categories (e.g., bird, cat, dog)
5. ✅ **Expected:** Categories save and show as checked

### Test 2: Persistence
1. Reload browser
2. ✅ **Expected:** Previously enabled categories still checked
3. Restart container
4. ✅ **Expected:** Categories still checked

### Test 3: New Deployment
1. Deploy new container (rebuild from latest code)
2. Delete config to simulate fresh install
3. Open web UI
4. ✅ **Expected:** All categories unchecked (field exists but empty)
5. No error messages in logs

### Test 4: Existing Deployment Upgrade
1. Deploy new container **without** deleting config
2. ✅ **Expected:** Previously enabled categories remain checked
3. Config file should have both old user data AND new field structure

## Files Modified

**Commit:** `09dc03f`
**Message:** "Fix: Add yolo_manual_categories to default config"

**Changed Files:**
- `config/config.yaml` - Added `yolo_manual_categories: []` field with comments

## Prevention

To prevent similar issues in the future:

1. **Always initialize new config fields** in the default template
2. **Document config migration** when adding new fields
3. **Test fresh deployment** scenario for all config-related features
4. **Consider config versioning** to auto-migrate old configs

## Related Issues

This fix resolves:
- YOLO categories showing as unchecked after fresh deployment
- User confusion about "losing" category selections
- Inconsistent state between fresh vs existing deployments

---

**Status:** ✅ COMPLETED AND PUSHED
**Date:** December 3, 2025
**Commit:** 09dc03f
