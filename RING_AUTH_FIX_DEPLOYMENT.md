# Ring Authentication Fix - Deployment Guide

## Issues Fixed

### Issue 1: Event Loop Conflict ✅ FIXED (Commit 01d5ea2)
**Error**: `You cannot call deprecated sync function Auth.fetch_token from within a running event loop.`

**Root Cause**: 
- FastAPI runs in async event loop
- Ring library's `Auth.fetch_token()` is synchronous and tries to use `asyncio.run()`
- Cannot call `asyncio.run()` from within already-running event loop

**Fix Applied** (src/webapp.py lines 649, 666):
```python
# Before:
if rd.authenticate(username, password):
    ...

# After:
auth_result = await asyncio.to_thread(rd.authenticate, username, password)
if auth_result:
    ...
```

---

### Issue 2: Ring Library API Change ✅ FIXED (Commit c4a7c94)
**Error**: `AttributeError: Auth has no attribute '_oauth'`

**Root Cause**:
- Ring library updated their API
- Token access changed from `self.auth._oauth.token` to `self.auth.token`
- Old code was using private/internal attribute that no longer exists

**Fix Applied** (src/ring_downloader.py lines 132, 185):
```python
# Before:
token_data = self.auth._oauth.token

# After:
token_data = self.auth.token
```

---

## Deployment Steps

### On Unraid Server (192.168.1.55)

1. **SSH into Unraid**:
   ```bash
   ssh root@192.168.1.55
   ```

2. **Stop and remove old container**:
   ```bash
   docker stop CritterCatcherAI
   docker rm CritterCatcherAI
   ```

3. **Pull latest image** (if auto-built from GitHub):
   ```bash
   docker pull <your-docker-registry>/crittercatcherai:latest
   ```

4. **Restart container** via Unraid UI or docker command
   - The container should pull the new code automatically
   - OR force update in Unraid Docker management interface

5. **Verify deployment**:
   ```bash
   docker logs -f CritterCatcherAI
   ```
   
   Look for:
   - Build date showing recent timestamp (after 2025-12-05 10:04:00 UTC)
   - No more "event loop" errors during authentication
   - No more "_oauth" attribute errors

---

## Testing After Deployment

### Test 1: Authentication with Existing Token
1. If you have a valid token at `/data/tokens/ring_token.json`, it should work
2. Check logs for: `Successfully authenticated with existing token`
3. No errors about event loops or missing attributes

### Test 2: Fresh Authentication (if needed)
1. Delete old token if expired: `rm /data/tokens/ring_token.json`
2. Go to web UI → Ring Settings
3. Enter username and password
4. Should see: `2FA code required` message (if 2FA enabled)
5. Enter 2FA code when prompted
6. Should see: `Authentication successful`
7. Token should be saved to `/data/tokens/ring_token.json`

### Test 3: Download Videos
1. Click "Download All" in web UI
2. Should not see any authentication errors
3. Should successfully download videos from Ring devices

---

## Log Patterns to Watch For

### ✅ Good (After Fix):
```
ring_downloader - INFO - Loading existing Ring refresh token
ring_downloader - INFO - Successfully authenticated with existing token
```

OR (for fresh auth):
```
ring_downloader - INFO - Authenticating with Ring account: user@example.com
webapp - INFO - 2FA required for user@example.com
[After entering code]
ring_downloader - INFO - Successfully authenticated with 2FA
ring_downloader - INFO - Saved refresh token for future use
```

### ❌ Bad (Before Fix - Should NOT See These):
```
# Event loop error (Issue 1):
RingError: You cannot call deprecated sync function Auth.fetch_token from within a running event loop.

# API error (Issue 2):
AttributeError: Auth has no attribute '_oauth'
```

---

## Rollback Plan (If Issues Occur)

If the new code causes problems:

1. **Revert to previous commit**:
   ```bash
   # On your dev machine
   cd C:\Coding\CritterCatcherAI
   git revert c4a7c94  # Revert token fix
   git revert 01d5ea2  # Revert event loop fix
   git push
   ```

2. **Force Unraid to pull old version**:
   ```bash
   ssh root@192.168.1.55
   docker stop CritterCatcherAI
   docker rm CritterCatcherAI
   docker pull <registry>/crittercatcherai:<old-tag>
   # Restart container
   ```

3. **Report issue** with logs from:
   ```bash
   docker logs CritterCatcherAI > /tmp/critter_logs.txt
   cat /config/crittercatcher.log
   ```

---

## Additional Notes

### Token File Format
The token file at `/data/tokens/ring_token.json` should look like:
```json
{
  "access_token": "...",
  "refresh_token": "...",
  "expires_in": 3600,
  "token_type": "Bearer"
}
```

### Token Expiration
- If you see `MissingTokenError` or `TokenExpiredError` in logs
- The token has expired (usually after 30 days)
- Delete `/data/tokens/ring_token.json` and re-authenticate via web UI

### Rate Limiting
If you see rate limit errors:
- Ring has rate limits on API calls
- The code has built-in retry logic with exponential backoff
- Wait a few minutes and try again

---

## Commits

- **01d5ea2**: Fix Ring authentication event loop conflict
- **c4a7c94**: Fix Ring token access for updated ring-doorbell library API

Both fixes are required for authentication to work properly!

---

## Summary

**What Changed**:
1. Ring authentication now runs in thread pool (fixes event loop conflict)
2. Token access uses new Ring library API (fixes AttributeError)

**Expected Result**:
- Authentication works from web UI
- No event loop errors
- No AttributeError about _oauth
- Videos download successfully

**Deploy**: Force Docker container update on Unraid server to pull latest code
