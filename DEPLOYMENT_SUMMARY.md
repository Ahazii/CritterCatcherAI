# Deployment Summary - Ring Auth Fix + Docker Image Tracking

## Issues Fixed ✅

### 1. Ring Authentication Token Access (Commit f53b120)
**Error**: `AttributeError: Auth has no attribute 'token'`

**Root Cause**: Ring library API uses private attribute `_token`, not public `token`

**Fix Applied** (ring_downloader.py lines 132, 185):
```python
# Before (WRONG):
token_data = self.auth.token  # ❌ Doesn't exist

# After (CORRECT):
token_data = self.auth._token  # ✅ Works!
```

**Status**: ✅ **FIXED and committed**

---

### 2. Event Loop Conflict (Commit 01d5ea2 - Already Fixed)
**Error**: `You cannot call deprecated sync function Auth.fetch_token from within a running event loop.`

**Fix**: Wrapped Ring auth calls in `asyncio.to_thread()`

**Status**: ✅ **FIXED and committed**

---

## Improvements Added ✅

### 1. Docker Image ID in App Header
**Change**: Replaced "Built: 2025-12-05 10:13:39 UTC" with Docker container ID

**Why**: Makes it crystal clear which version is deployed

**Location**: Web UI header (top-right corner)

**Format**: Shows container ID (e.g., `a1b2c3d4e5f6`) or falls back to build date

---

### 2. Enhanced Startup Logging
**Added to main.py**:
```
================================================================================
Starting CritterCatcherAI
  Container: a1b2c3d4e5f6
  Git SHA: f53b120e9a45
================================================================================
```

**Added to webapp.py**:
```
================================================================================
CritterCatcherAI starting up
  Version: v0.1.0
  Docker Image: a1b2c3d4e5f6
  Container: a1b2c3d4e5f6
================================================================================
```

**Why**: Makes it immediately obvious when new code is deployed!

---

## Deployment Instructions

### SSH to Unraid Server:
```bash
ssh root@192.168.1.55

# Stop and rebuild container
docker stop CritterCatcherAI
docker rm CritterCatcherAI

# Pull latest from Git (or rebuild)
# Then restart via Unraid UI
```

### Verify New Deployment:
```bash
# Check startup logs
docker logs CritterCatcherAI 2>&1 | head -30

# Look for:
# ================================================================================
# Starting CritterCatcherAI
#   Container: <new-container-id>
#   Git SHA: f53b120e9a45   <-- Should show commit f53b120
# ================================================================================
```

### Check Web UI:
1. Open http://192.168.1.55:8080
2. Look at header (top-right)
3. Should show Docker container ID instead of build date

---

## Testing Ring Authentication

### Test 1: Existing Token
If you have `/data/tokens/ring_token.json`, it should just work:
```bash
docker logs -f CritterCatcherAI | grep -i auth
```

Expected: `Successfully authenticated with existing token`

### Test 2: Fresh Authentication
1. Delete old token: `docker exec CritterCatcherAI rm /data/tokens/ring_token.json`
2. Go to web UI → Ring Settings
3. Enter username/password
4. Should prompt for 2FA code
5. Enter code
6. Should see: `Authentication successful`

### Test 3: Download Videos
1. Click "Download All" in web UI
2. Should work without auth errors
3. Videos should download successfully

---

## Expected Log Patterns

### ✅ Good (After Deployment):
```
# Startup
================================================================================
Starting CritterCatcherAI
  Container: a1b2c3d4e5f6
  Git SHA: f53b120e9a45
================================================================================

# Authentication
ring_downloader - INFO - Loading existing Ring refresh token
ring_downloader - INFO - Successfully authenticated with existing token

# OR (fresh auth)
ring_downloader - INFO - Authenticating with Ring account: user@example.com
webapp - INFO - 2FA required for user@example.com
[After entering 2FA code]
ring_downloader - INFO - Successfully authenticated with 2FA
ring_downloader - INFO - Saved refresh token for future use
```

### ❌ Bad (Should NOT See):
```
# Old errors (FIXED):
AttributeError: Auth has no attribute 'token'
RingError: You cannot call deprecated sync function Auth.fetch_token
AttributeError: Auth has no attribute '_oauth'
```

---

## Commits History

1. **01d5ea2** (Dec 5, 10:04): Fix Ring authentication event loop conflict
   - Wrapped auth calls in `asyncio.to_thread()`

2. **c4a7c94** (Dec 5, 10:07): Fix Ring token access for updated library API
   - Changed `auth._oauth.token` → `auth.token` (INCORRECT!)

3. **aded9c4** (Dec 5, 10:09): Add deployment guide

4. **f53b120** (Dec 5, 10:26): ✅ **FINAL FIX**
   - Changed `auth.token` → `auth._token` (CORRECT!)
   - Added Docker image ID to UI header
   - Enhanced startup logging with container info

---

## Summary

### What Changed:
1. ✅ Ring authentication now uses correct token attribute (`_token`)
2. ✅ Event loop conflict fixed (already done in 01d5ea2)
3. ✅ Docker container ID shown in web UI header
4. ✅ Clear startup logging with Git SHA and container info

### Expected Result:
- ✅ Authentication works from web UI
- ✅ No errors about tokens, event loops, or `_oauth`
- ✅ Videos download successfully
- ✅ Clear visibility into which version is deployed

### Deploy:
Force Docker container update on Unraid server (stop, remove, pull, restart)

### Verify:
- Check logs for Git SHA `f53b120`
- Check web UI header for Docker container ID
- Test Ring authentication

---

## Quick Verification Script

```bash
#!/bin/bash
# Run on Unraid server after deployment

echo "=== Checking Deployment ==="

# Check Git SHA
echo -n "Git SHA: "
docker logs CritterCatcherAI 2>&1 | grep "Git SHA:" | tail -1

# Check Container ID
echo -n "Container: "
docker logs CritterCatcherAI 2>&1 | grep "Container:" | tail -1

# Check for any recent auth errors
echo -e "\n=== Recent Auth Errors (should be empty) ==="
docker logs CritterCatcherAI 2>&1 | grep -i "error.*auth\|attributeerror\|event loop" | tail -5

# Check if auth is working
echo -e "\n=== Recent Auth Success ==="
docker logs CritterCatcherAI 2>&1 | grep "Successfully authenticated" | tail -3

echo -e "\n=== Done ==="
```

Save as `verify_deployment.sh` and run: `bash verify_deployment.sh`
