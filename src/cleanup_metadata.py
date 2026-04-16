#!/usr/bin/env python3
"""Clean orphaned metadata files from review directories."""

from pathlib import Path

def main():
    review_path = Path('/data/review')
    deleted = 0
    
    print("Scanning for orphaned metadata files...")
    
    for json_file in review_path.rglob('*.json'):
        # Get corresponding video file (metadata files are named video.mp4.json)
        video_name = json_file.stem  # Removes .json
        video_file = json_file.parent / video_name
        
        if not video_file.exists():
            print(f"Deleting orphaned: {json_file.relative_to(review_path)}")
            json_file.unlink()
            deleted += 1
    
    print(f"\nDeleted {deleted} orphaned metadata files")

if __name__ == "__main__":
    main()
