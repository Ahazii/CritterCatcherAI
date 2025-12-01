import cv2
import numpy as np
import os

# Test H.264 codec
print("Testing H.264 (avc1) codec...")
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter('/tmp/test_h264.mp4', fourcc, 15, (640, 480))
h264_opened = out.isOpened()
print(f"H.264 (avc1) opened: {h264_opened}")

if h264_opened:
    # Write test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out.write(frame)
    out.release()
    size = os.path.getsize('/tmp/test_h264.mp4')
    print(f"Test file size: {size} bytes")
    
    # Check codec of created file
    os.system("ffprobe -v quiet -show_entries stream=codec_name -of default=noprint_wrappers=1 /tmp/test_h264.mp4")
else:
    out.release()
    # Try mp4v fallback
    print("\nTesting MPEG-4 (mp4v) codec...")
    fourcc2 = cv2.VideoWriter_fourcc(*'mp4v')
    out2 = cv2.VideoWriter('/tmp/test_mp4v.mp4', fourcc2, 15, (640, 480))
    print(f"MPEG-4 (mp4v) opened: {out2.isOpened()}")
    out2.release()
