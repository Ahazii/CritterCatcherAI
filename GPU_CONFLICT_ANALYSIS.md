# GPU Conflict Issue - Root Cause Analysis & Solution

**Date:** December 9, 2025  
**Status:** CRITICAL - Container won't start when VM is using RTX 3080

---

## 🔴 PROBLEM SUMMARY

**Symptom:** CritterCatcherAI container fails to start when Windows VM is running (VM uses RTX 3080)

**Expected Behavior:** Container should use Quadro RTX 4000 (device 1), VM uses RTX 3080 (device 0)

**Actual Behavior:** Container startup fails when VM is active, works fine when VM is stopped

**Current Configuration:**
- Unraid template: `--gpus device=1` (Extra Params)
- Unraid template: `NVIDIA_VISIBLE_DEVICES=""` (empty environment variable)
- Docker inspect shows: `"DeviceIDs": ["1"]` ✓

---

## 🔍 ROOT CAUSE ANALYSIS

### Issue #1: PyTorch Always Uses GPU 0 by Default ⚠️

**The Code Problem:**

```python
# src/object_detector.py line 75
gpu_name = torch.cuda.get_device_name(0)  # ❌ HARDCODED to device 0!

# src/object_detector.py line 83
self.model.to(device)  # device='cuda' means CUDA:0 by default!
```

```python
# src/clip_vit_classifier.py line 26
self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # "cuda" = CUDA:0!

# src/clip_vit_classifier.py line 32
self.model = self.model.to(self.device)  # ❌ Goes to CUDA:0!
```

**What Happens:**
1. Docker restricts container to device 1 (Quadro) via `--gpus device=1`
2. Inside container, device 1 appears as `cuda:0` (remapped by Docker)
3. **BUT**: When VM holds physical device 0 (RTX 3080), Docker CANNOT remap properly
4. PyTorch tries to access `cuda:0` which points to the RTX 3080
5. GPU is locked by VM → Container startup FAILS

### Issue #2: Docker Device Remapping Conflict

**System GPU Layout:**
```
Host (Unraid):
├── GPU 0: RTX 3080 (nvidia-smi shows first)
└── GPU 1: Quadro RTX 4000 (nvidia-smi shows second)
```

**When VM is Running:**
```
GPU 0 (RTX 3080): ❌ LOCKED by VM (vfio-pci passthrough)
GPU 1 (Quadro):   ✓ Available for Docker
```

**Docker Behavior with `--gpus device=1`:**
- **When VM OFF**: Maps Quadro as `cuda:0` inside container ✓
- **When VM ON**: Cannot map because device 0 is locked → FAILS ❌

### Issue #3: Environment Variable Not Set

The Unraid template has `NVIDIA_VISIBLE_DEVICES` but it's **empty** by default!

```xml
<!-- Line 27 in my-CritterCatcherAI.xml -->
<Config Name="NVIDIA_VISIBLE_DEVICES" Target="NVIDIA_VISIBLE_DEVICES" Default="" Mode="" ... />
```

This variable is **critical** for NVIDIA container runtime but is not populated.

---

## ✅ SOLUTION

### Step 1: Get GPU UUID (One-Time Setup)

SSH into Unraid and get the Quadro UUID:

```bash
ssh root@192.168.1.55
nvidia-smi -L
```

**Output:**
```
GPU 0: Quadro RTX 4000 (UUID: GPU-cfa298ff-61ff-fa5c-06ae-6f01e9cc4cda)
```

**Note:** There's only 1 GPU showing - this means the RTX 3080 is already passed through to the VM!

### Step 2: Update Unraid Template

Edit `my-CritterCatcherAI.xml`:

```xml
<!-- CHANGE THIS LINE (line 17): -->
<ExtraParams>--restart=unless-stopped --gpus device=1</ExtraParams>

<!-- TO THIS: -->
<ExtraParams>--restart=unless-stopped --gpus '"device=GPU-cfa298ff-61ff-fa5c-06ae-6f01e9cc4cda"'</ExtraParams>

<!-- AND CHANGE THIS LINE (line 27): -->
<Config Name="NVIDIA_VISIBLE_DEVICES" Target="NVIDIA_VISIBLE_DEVICES" Default="" ... />

<!-- TO THIS: -->
<Config Name="NVIDIA_VISIBLE_DEVICES" Target="NVIDIA_VISIBLE_DEVICES" Default="GPU-cfa298ff-61ff-fa5c-06ae-6f01e9cc4cda" ... />
```

### Step 3: Fix Python Code to Respect NVIDIA_VISIBLE_DEVICES

#### Fix #1: object_detector.py

```python
# src/object_detector.py lines 71-79
# CHANGE FROM:
if torch.cuda.is_available():
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)  # ❌ Hardcoded
    logger.info(f"CUDA available - using GPU: {gpu_name}")

# CHANGE TO:
if torch.cuda.is_available():
    device = 'cuda'  # PyTorch will respect CUDA_VISIBLE_DEVICES
    # Get first available GPU (which is the one NVIDIA_VISIBLE_DEVICES allows)
    gpu_name = torch.cuda.get_device_name(0)  # Now 0 = first visible device
    logger.info(f"CUDA available - using GPU: {gpu_name}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
```

Wait - actually, the code is fine IF the environment variable is set correctly!

The real issue is **we need to ensure PyTorch sees the NVIDIA_VISIBLE_DEVICES variable**.

#### Fix #2: Add Environment Variable Passthrough

Add to `src/main.py` at the top (after imports):

```python
import os

# Ensure NVIDIA_VISIBLE_DEVICES is propagated to CUDA
if 'NVIDIA_VISIBLE_DEVICES' in os.environ and os.environ['NVIDIA_VISIBLE_DEVICES']:
    # PyTorch uses CUDA_VISIBLE_DEVICES, so set it based on NVIDIA_VISIBLE_DEVICES
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Docker remaps, so always use 0
        logger.info(f"Set CUDA_VISIBLE_DEVICES=0 (using GPU: {os.environ['NVIDIA_VISIBLE_DEVICES']})")
```

---

## 📝 RECOMMENDED CHANGES

### Change #1: Update Unraid Template (REQUIRED)

**File:** `my-CritterCatcherAI.xml`

```xml
<!-- Line 17: Use GPU UUID instead of device number -->
<ExtraParams>--restart=unless-stopped --gpus '"device=GPU-cfa298ff-61ff-fa5c-06ae-6f01e9cc4cda"'</ExtraParams>

<!-- Line 27: Set default GPU UUID -->
<Config Name="NVIDIA_VISIBLE_DEVICES" 
        Target="NVIDIA_VISIBLE_DEVICES" 
        Default="GPU-cfa298ff-61ff-fa5c-06ae-6f01e9cc4cda" 
        Mode="" 
        Description="GPU UUID for Quadro RTX 4000. This restricts the container to use only this GPU. Found via 'nvidia-smi -L' on host." 
        Type="Variable" 
        Display="advanced" 
        Required="false" 
        Mask="false">GPU-cfa298ff-61ff-fa5c-06ae-6f01e9cc4cda</Config>
```

### Change #2: Add Logging to Verify GPU Selection (RECOMMENDED)

**File:** `src/object_detector.py` line 76

```python
# ADD after gpu_name = torch.cuda.get_device_name(0):
logger.info(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}")
logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
```

**File:** `src/clip_vit_classifier.py` line 28

```python
# ADD after logger.info(f"Loading CLIP model: {model_name} on {self.device}"):
logger.info(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}")
```

### Change #3: Update Dockerfile Environment (OPTIONAL)

**File:** `Dockerfile` after line 92

```dockerfile
# Add environment variable for NVIDIA container runtime
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

---

## 🧪 TESTING PROCEDURE

### Test 1: With VM Stopped

```bash
# SSH to Unraid
ssh root@192.168.1.55

# Update container with new template
# (Force update in Unraid UI to pull changes)

# Start container
docker start CritterCatcherAI

# Check logs
docker logs CritterCatcherAI 2>&1 | grep -i "gpu\|cuda" | head -20

# Expected output:
# - "CUDA available - using GPU: Quadro RTX 4000" ✓
# - "NVIDIA_VISIBLE_DEVICES: GPU-cfa298ff-61ff-fa5c-06ae-6f01e9cc4cda" ✓
# - "GPU monitoring started for Quadro RTX 4000" ✓
```

### Test 2: With VM Running ← CRITICAL TEST

```bash
# Start Windows VM (which uses RTX 3080)
# Then try to start container

docker start CritterCatcherAI

# Should start successfully now! ✓
# Check logs to confirm using Quadro
docker logs CritterCatcherAI 2>&1 | grep -i "gpu" | head -10
```

---

## 🎯 WHY THIS FIXES IT

### Before Fix:
1. `--gpus device=1` tells Docker to use physical device 1
2. When VM is running, physical device 0 (RTX 3080) is locked
3. Docker tries to remap but **PyTorch doesn't know about the restriction**
4. Container startup fails because GPU 0 is unavailable

### After Fix:
1. `--gpus device=GPU-cfa298ff-...` explicitly specifies Quadro UUID
2. `NVIDIA_VISIBLE_DEVICES=GPU-cfa298ff-...` tells NVIDIA runtime which GPU to expose
3. Docker runtime **only exposes the Quadro** to the container
4. PyTorch sees only 1 GPU (the Quadro), which it calls `cuda:0`
5. No conflict with VM because we're using UUID, not device index ✓

---

## 📋 IMPLEMENTATION CHECKLIST

- [ ] Get Quadro GPU UUID from `nvidia-smi -L`
- [ ] Update `my-CritterCatcherAI.xml` with GPU UUID in ExtraParams
- [ ] Update `my-CritterCatcherAI.xml` with GPU UUID in NVIDIA_VISIBLE_DEVICES default
- [ ] (Optional) Add logging to object_detector.py and clip_vit_classifier.py
- [ ] Commit changes to git
- [ ] Push to GitHub
- [ ] Force update container in Unraid UI
- [ ] Test with VM stopped → Should work ✓
- [ ] Test with VM running → Should work ✓ (THIS IS THE KEY TEST)
- [ ] Verify GPU monitoring shows Quadro in web UI
- [ ] Verify YOLO processing uses Quadro

---

## 🔧 ALTERNATIVE SOLUTIONS (If Above Doesn't Work)

### Alternative 1: Use CUDA_VISIBLE_DEVICES in Python

Add to `src/main.py` before any CUDA operations:

```python
import os
# Force PyTorch to only see GPU 0 (which Docker maps to the Quadro)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

### Alternative 2: Remove --gpus flag, Use Only Environment Variable

```xml
<ExtraParams>--restart=unless-stopped</ExtraParams>
<Config Name="NVIDIA_VISIBLE_DEVICES" ... Default="GPU-cfa298ff-61ff-fa5c-06ae-6f01e9cc4cda" ...>
```

**Note:** This requires nvidia-container-runtime to be the default Docker runtime.

---

## 📊 DIAGNOSIS COMMANDS

```bash
# Check GPU availability on host
ssh root@192.168.1.55 "nvidia-smi -L"

# Check container GPU config
ssh root@192.168.1.55 "docker inspect CritterCatcherAI | grep -A 20 DeviceRequests"

# Check environment variables in container
ssh root@192.168.1.55 "docker exec CritterCatcherAI env | grep -i nvidia"

# Check what PyTorch sees
ssh root@192.168.1.55 "docker exec CritterCatcherAI python -c 'import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))'"
```

---

**Status:** Ready to implement  
**Priority:** CRITICAL  
**Estimated Time:** 30 minutes to implement and test

