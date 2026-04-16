#!/usr/bin/env python3
"""
System health check script for CritterCatcherAI.
Tests configuration, directories, and core components.
"""
import sys
import yaml
from pathlib import Path

def main():
    print("=" * 60)
    print("CritterCatcherAI System Health Check")
    print("=" * 60)
    print()
    
    # Load config
    try:
        with open('/config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return 1
    
    print()
    print("=== Configuration ===")
    detection = config.get('detection', {})
    print(f"  Face model: {detection.get('face_model', 'N/A')}")
    print(f"  Face frames: {detection.get('face_frames', 'N/A')}")
    print(f"  Face tolerance: {detection.get('face_tolerance', 'N/A')}")
    print(f"  Object frames: {detection.get('object_frames', 'N/A')}")
    print(f"  YOLO model: {detection.get('yolo_model', 'yolov8n')}")
    print(f"  Face recognition: {config.get('face_recognition', {}).get('enabled', False)}")
    print(f"  GPU enabled: {not detection.get('force_cpu', False)}")
    
    print()
    print("=== Critical Directories ===")
    critical_paths = {
        '/data/downloads': 'Downloaded videos',
        '/data/review': 'Videos pending review',
        '/data/sorted': 'Confirmed sorted videos',
        '/data/training/faces/unassigned': 'Unknown faces for profile creation',
        '/data/faces': 'Face encodings',
        '/data/animal_profiles': 'Animal profile configs',
        '/data/face_profiles': 'Face profile configs',
        '/config': 'Configuration files'
    }
    
    all_dirs_ok = True
    for path, description in critical_paths.items():
        p = Path(path)
        if p.exists():
            print(f"  ✓ {path:<40} ({description})")
        else:
            print(f"  ✗ {path:<40} MISSING!")
            all_dirs_ok = False
    
    print()
    print("=== Review Categories ===")
    review_path = Path('/data/review')
    if review_path.exists():
        categories = sorted([d.name for d in review_path.iterdir() if d.is_dir()])
        if categories:
            for cat in categories:
                count = len(list((review_path / cat).glob('*.mp4')))
                metadata_count = len(list((review_path / cat).glob('*.json')))
                print(f"  {cat:20} {count:3} videos, {metadata_count:3} metadata files")
        else:
            print("  No categories found (empty)")
    else:
        print("  ✗ Review directory missing!")
    
    print()
    print("=== Animal Profiles ===")
    profiles_path = Path('/data/animal_profiles')
    if profiles_path.exists():
        profiles = list(profiles_path.glob('*.json'))
        if profiles:
            print(f"  Found {len(profiles)} profile(s):")
            for prof in sorted(profiles):
                print(f"    - {prof.stem}")
        else:
            print("  No profiles configured")
    else:
        print("  ✗ Animal profiles directory missing!")
    
    print()
    print("=== Face Profiles ===")
    face_profiles_path = Path('/data/face_profiles')
    if face_profiles_path.exists():
        face_profiles = list(face_profiles_path.glob('*.json'))
        if face_profiles:
            print(f"  Found {len(face_profiles)} profile(s):")
            for prof in sorted(face_profiles):
                print(f"    - {prof.stem}")
        else:
            print("  No face profiles configured")
    else:
        print("  Face profiles directory not created yet (normal if unused)")
    
    print()
    print("=== Storage Usage ===")
    import shutil
    storage_paths = {
        '/data/review': 'Review queue',
        '/data/sorted': 'Sorted videos',
        '/data/training': 'Training data',
        '/data/downloads': 'Downloads'
    }
    
    for path, description in storage_paths.items():
        p = Path(path)
        if p.exists():
            try:
                total, used, free = shutil.disk_usage(path)
                size_gb = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / (1024**3)
                print(f"  {description:20} {size_gb:6.2f} GB")
            except:
                print(f"  {description:20} (unable to calculate)")
        else:
            print(f"  {description:20} N/A (missing)")
    
    print()
    print("=== Component Tests ===")
    
    # Test imports
    try:
        sys.path.insert(0, '/app/src')
        from object_detector import ObjectDetector
        print("  ✓ ObjectDetector import")
    except Exception as e:
        print(f"  ✗ ObjectDetector import failed: {e}")
    
    try:
        from face_recognizer import FaceRecognizer
        print("  ✓ FaceRecognizer import")
    except Exception as e:
        print(f"  ✗ FaceRecognizer import failed: {e}")
    
    try:
        from animal_profile import AnimalProfileManager
        print("  ✓ AnimalProfileManager import")
    except Exception as e:
        print(f"  ✗ AnimalProfileManager import failed: {e}")
    
    try:
        from face_profile import FaceProfileManager
        print("  ✓ FaceProfileManager import")
    except Exception as e:
        print(f"  ✗ FaceProfileManager import failed: {e}")
    
    print()
    print("=" * 60)
    if all_dirs_ok:
        print("System Status: HEALTHY ✓")
    else:
        print("System Status: WARNING - Some directories missing")
    print("=" * 60)
    
    return 0 if all_dirs_ok else 1

if __name__ == "__main__":
    sys.exit(main())
