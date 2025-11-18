#!/usr/bin/env python
"""Test script for processing pipeline."""
import logging
from pathlib import Path
import sys
import tempfile
import shutil
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_image(filepath: Path, description: str = ""):
    """Create a simple test image."""
    width, height = 224, 224
    color = (100, 150, 200)
    image = Image.new('RGB', (width, height), color)
    image.save(filepath)
    if description:
        logger.info(f"  Created: {filepath.name} - {description}")


def test_processing_pipeline():
    """Test the processing pipeline."""
    print("=" * 60)
    print("Testing Processing Pipeline")
    print("=" * 60)
    
    # Import modules
    try:
        from animal_profile import AnimalProfile, AnimalProfileManager
        from processing_pipeline import FrameExtractor, ProcessingResult, TwoStageProcessor
        print("✓ Successfully imported pipeline modules")
    except ImportError as e:
        print(f"✗ Failed to import modules: {e}")
        return False
    
    # Create temporary test environment
    test_dir = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
    
    try:
        print("\n[Test 1] Creating test environment...")
        profiles_dir = test_dir / "profiles"
        data_dir = test_dir / "data"
        profiles_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)
        print(f"✓ Created test environment at {test_dir}")
        
        print("\n[Test 2] Creating test animal profiles...")
        profile_manager = AnimalProfileManager(data_dir)
        
        # Create test profiles
        hedgehog = profile_manager.create_profile(
            name="Hedgehog",
            yolo_categories=["cat", "dog", "sheep"],
            text_description="a small hedgehog"
        )
        print(f"✓ Created profile: {hedgehog.name} (ID: {hedgehog.id})")
        
        finch = profile_manager.create_profile(
            name="Finch",
            yolo_categories=["bird"],
            text_description="a small finch"
        )
        print(f"✓ Created profile: {finch.name} (ID: {finch.id})")
        
        print("\n[Test 3] Creating test frame images...")
        test_frames_dir = data_dir / "temp_frames"
        test_frames_dir.mkdir(exist_ok=True)
        
        frame_paths = []
        for i in range(5):
            frame_path = test_frames_dir / f"frame_{i:06d}.jpg"
            create_test_image(frame_path, f"Test frame {i}")
            frame_paths.append(str(frame_path))
        print(f"✓ Created {len(frame_paths)} test frames")
        
        print("\n[Test 4] Testing FrameExtractor...")
        extractor = FrameExtractor(target_fps=1)
        print(f"✓ Initialized FrameExtractor (target FPS: {extractor.target_fps})")
        
        print("\n[Test 5] Testing ProcessingResult...")
        result = ProcessingResult("test_profile", "Test Animal")
        result.sorted_count = 3
        result.review_count = 2
        result.sorted_frames = ["frame1.jpg", "frame2.jpg", "frame3.jpg"]
        result.review_frames = ["frame4.jpg", "frame5.jpg"]
        
        result_dict = result.to_dict()
        print(f"✓ ProcessingResult summary:")
        print(f"    Sorted: {result_dict['sorted_count']}")
        print(f"    Review: {result_dict['review_count']}")
        print(f"    Total: {result_dict['total_count']}")
        
        print("\n[Test 6] Testing TwoStageProcessor initialization...")
        processor = TwoStageProcessor(
            profile_manager=profile_manager,
            base_data_path=str(data_dir)
        )
        print(f"✓ Initialized TwoStageProcessor")
        print(f"    Base data path: {processor.base_data_path}")
        print(f"    Directories created successfully")
        
        print("\n[Test 7] Testing frame processing for a profile...")
        # Process frames for hedgehog profile (without YOLO Stage 1 since we don't have object_detector)
        try:
            result = processor.process_frames_for_profile(hedgehog, frame_paths)
            print(f"✓ Processed frames for {hedgehog.name}")
            print(f"    Sorted: {result.sorted_count}")
            print(f"    Review: {result.review_count}")
            print(f"    Errors: {len(result.errors)}")
            
            # Check directory organization
            sorted_dir = data_dir / "sorted" / hedgehog.id
            review_dir = data_dir / "review" / hedgehog.id
            
            if sorted_dir.exists():
                sorted_files = list(sorted_dir.glob("*.jpg"))
                print(f"    Sorted directory: {len(sorted_files)} frames")
            
            if review_dir.exists():
                review_files = list(review_dir.glob("*.jpg"))
                print(f"    Review directory: {len(review_files)} frames")
        
        except Exception as e:
            print(f"✗ Error processing frames: {e}")
            logger.exception("Frame processing error")
            return False
        
        print("\n[Test 8] Verifying directory structure...")
        expected_dirs = [
            data_dir / "sorted",
            data_dir / "review",
            data_dir / "training",
            data_dir / "models",
            data_dir / "temp"
        ]
        
        missing_dirs = [d for d in expected_dirs if not d.exists()]
        if missing_dirs:
            print(f"✗ Missing directories: {missing_dirs}")
            return False
        else:
            print(f"✓ All expected directories exist")
        
        print("\n[Test 9] Verifying metadata files...")
        metadata_files = list((data_dir / "review").rglob("*.json"))
        print(f"✓ Metadata files saved: {len(metadata_files)}")
        if metadata_files:
            # Check first metadata file
            import json
            with open(metadata_files[0], 'r') as f:
                metadata = json.load(f)
            print(f"    Sample metadata fields: {list(metadata.keys())}")
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
    
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
            logger.info(f"Cleaned up test directory: {test_dir}")


if __name__ == "__main__":
    success = test_processing_pipeline()
    sys.exit(0 if success else 1)
