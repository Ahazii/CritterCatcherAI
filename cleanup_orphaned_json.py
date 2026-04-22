#!/usr/bin/env python3
"""Clean up orphaned JSON files that don't have matching video files."""
from pathlib import Path

def cleanup_orphaned_json(base_path):
    """Remove JSON files that don't have matching video files."""
    base = Path(base_path)
    orphans = []
    
    # Find all JSON files
    for json_file in base.rglob('*.json'):
        # Check if the corresponding video file exists
        # JSON files are named like "video.mp4.json", so remove the .json suffix
        video_file = json_file.with_suffix('')
        
        if not video_file.exists():
            orphans.append(json_file)
            json_file.unlink()
    
    return len(orphans)

if __name__ == '__main__':
    review_count = cleanup_orphaned_json('/data/review')
    downloads_count = cleanup_orphaned_json('/data/downloads')
    sorted_count = cleanup_orphaned_json('/data/sorted')
    
    print(f'Removed {review_count} orphaned JSON files from review')
    print(f'Removed {downloads_count} orphaned JSON files from downloads')
    print(f'Removed {sorted_count} orphaned JSON files from sorted')
    print(f'Total: {review_count + downloads_count + sorted_count} orphaned JSON files removed')
