# Ring 2FA Authentication Setup

## Overview
CritterCatcherAI now requires Ring authentication to be completed through the **web interface** on first run. This ensures a smooth 2FA experience without needing SSH access.

## What Changed

### 1. **Startup Behavior**
- On first startup (no token file exists), the app **will not attempt authentication**
- Instead, it logs a clear message directing you to the web interface
- The main processing loop waits for authentication before downloading videos

### 2. **Web GUI Authentication**
- Navigate to the **Ring Setup tab** in the web interface
- Enter your Ring credentials (or they'll be pre-filled from environment variables)
- Click "Authenticate with Ring"
- **If 2FA is required:**
  - You'll see a green success message: "✓ Ring has sent a verification code!"
  - A 2FA code input field will appear
  - Check your phone/email for the code
  - Enter the code and click "Submit 2FA Code"
- Once authenticated, the token is saved and processing begins automatically

### 3. **Alert Banner**
- If Ring is not authenticated, a warning banner appears at the top of the dashboard
- Click the link to go directly to the Ring Setup tab
- The banner disappears once authentication succeeds

## Step-by-Step Guide

### First Time Setup
1. **Start the container** (it will idle waiting for authentication)
2. **Open the web interface** at `http://your-server-ip:8080`
3. You'll see a warning banner: "⚠ Ring Not Authenticated"
4. **Click "Ring Setup tab"** (or navigate manually)
5. **Enter your Ring credentials**:
   - Email address
   - Password
   - (Leave 2FA code blank for now)
6. **Click "Authenticate with Ring"**
7. **You'll see**: "✓ Ring has sent a verification code!"
8. **Check your phone/email** for the 6-digit code from Ring
9. **Enter the code** in the 2FA field that appeared
10. **Click "Submit 2FA Code"**
11. **Success!** The token is saved and video processing begins

### Token Expiration
If your Ring token expires:
- The container will log an error on startup
- The old token file is automatically deleted
- Follow the "First Time Setup" steps again to re-authenticate

## Environment Variables
You can pre-configure credentials via environment variables:
- `RING_USERNAME` - Your Ring email address
- `RING_PASSWORD` - Your Ring password

If set, these will be pre-filled in the web interface, making setup even easier.

## Troubleshooting

### "Authentication failed" error
- Double-check your Ring credentials
- Make sure you're entering the current password (not an old one)

### 2FA code not appearing
- Make sure you clicked "Authenticate with Ring" first (without entering a code)
- The app needs to make the initial request to trigger Ring sending the code

### 2FA code rejected
- Ring codes expire quickly - request a new one if needed
- Make sure you're entering the code correctly (no spaces)
- Try copying/pasting the code directly from your email/SMS

### Container keeps restarting
- The container will idle (not crash) if not authenticated
- Check logs: `docker logs crittercatcher-ai`
- Look for the authentication warning message

## Technical Details

### Token Storage
- Token file: `/data/tokens/ring_token.json`
- This is mapped to your Unraid appdata: `/mnt/user/appdata/crittercatcher/tokens/`
- The token persists across container restarts

### 2FA Flow
1. Initial auth request with username/password → Ring returns 412 status
2. Backend detects `Requires2FAError` and returns `needs_2fa` status
3. Frontend shows 2FA input field
4. User submits code → Backend calls `authenticate_with_2fa()`
5. Token is saved and processing begins

### Logs
The log output will show:
```
WARNING - NO RING TOKEN FOUND - AUTHENTICATION REQUIRED
WARNING - Please complete Ring authentication using the web interface:
WARNING - 
WARNING -   1. Open the web interface in your browser
WARNING -   2. Navigate to the 'Ring Setup' tab
WARNING -   3. Enter your Ring credentials
WARNING -   4. Complete 2FA verification when prompted
```

Once authenticated:
```
INFO - Authenticating with Ring using saved token
INFO - Successfully authenticated with existing token
```

## Benefits of This Approach
✅ No SSH required for 2FA  
✅ Clear visual feedback in the browser  
✅ Works from any device with web access  
✅ Token persists - only authenticate once  
✅ Graceful handling of expired tokens  

## Next Steps
After successful authentication:
- Videos will download automatically based on your schedule
- Check the Dashboard tab for processing statistics
- View sorted videos in the Videos tab
- Train face recognition in the Face Training tab
