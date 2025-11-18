#!/usr/bin/env python
"""Test script for review and feedback system."""
import logging
from pathlib import Path
import sys
import tempfile
import shutil
import json
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_image(filepath: Path):
    """Create a simple test image."""
    image = Image.new('RGB', (224, 224), (100, 150, 200))
    image.save(filepath)


def create_frame_metadata(filepath: Path, confidence: float, description: str = "a test animal"):
    """Create metadata file for a frame."""
    metadata = {
        "confidence": confidence,
        "description": description,
        "timestamp": "2024-01-01T12:00:00"
    }
    metadata_path = Path(str(filepath) + ".json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)


def test_review_feedback():
    """Test the review feedback system."""
    print("=" * 60)
    print("Testing Review & Feedback System")
    print("=" * 60)
    
    # Import modules
    try:
        from animal_profile import AnimalProfileManager
        from review_feedback import ReviewFrame, ReviewManager, ReviewSummary
        print("✓ Successfully imported review/feedback modules")
    except ImportError as e:
        print(f"✗ Failed to import modules: {e}")
        return False
    
    # Create temporary test environment
    test_dir = Path(tempfile.mkdtemp(prefix="review_test_"))
    
    try:
        print("\n[Test 1] Setting up test environment...")
        data_dir = test_dir / "data"
        data_dir.mkdir(exist_ok=True)
        print(f"✓ Created test environment at {test_dir}")
        
        print("\n[Test 2] Creating test profiles...")
        profile_manager = AnimalProfileManager(data_dir)
        
        hedgehog = profile_manager.create_profile(
            name="Hedgehog",
            yolo_categories=["cat", "dog"],
            text_description="a hedgehog"
        )
        print(f"✓ Created profile: {hedgehog.name}")
        
        print("\n[Test 3] Creating test review frames...")
        review_dir = data_dir / "review" / hedgehog.id
        review_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test frames with metadata
        frame_files = []
        for i in range(5):
            frame_path = review_dir / f"frame_{i:06d}.jpg"
            create_test_image(frame_path)
            confidence = 0.5 + (i * 0.1)  # 0.5, 0.6, 0.7, 0.8, 0.9
            create_frame_metadata(frame_path, confidence)
            frame_files.append(frame_path.name)
        
        print(f"✓ Created {len(frame_files)} test frames with metadata")
        
        print("\n[Test 4] Testing ReviewFrame...")
        review_frame = ReviewFrame(str(review_dir / frame_files[0]), hedgehog.id)
        print(f"✓ ReviewFrame loaded:")
        print(f"    Confidence: {review_frame.get_confidence():.2f}")
        print(f"    Description: {review_frame.get_description()}")
        
        print("\n[Test 5] Initializing ReviewManager...")
        review_manager = ReviewManager(profile_manager, str(data_dir))
        print(f"✓ ReviewManager initialized")
        
        print("\n[Test 6] Testing list_pending_reviews...")
        pending_frames = review_manager.list_pending_reviews(hedgehog.id)
        print(f"✓ Listed {len(pending_frames)} pending frames")
        
        print("\n[Test 7] Testing get_pending_count...")
        count = review_manager.get_pending_count(hedgehog.id)
        print(f"✓ Pending count: {count}")
        assert count == 5, f"Expected 5, got {count}"
        
        print("\n[Test 8] Testing confirm_frame...")
        initial_confirmed = hedgehog.confirmed_count
        review_manager.confirm_frame(hedgehog.id, frame_files[0])
        
        # Reload profile to check update
        updated_profile = profile_manager.get_profile(hedgehog.id)
        print(f"✓ Confirmed frame {frame_files[0]}")
        print(f"    Confirmed count: {initial_confirmed} → {updated_profile.confirmed_count}")
        assert updated_profile.confirmed_count == initial_confirmed + 1
        
        # Check frame moved
        training_confirmed_dir = data_dir / "training" / hedgehog.id / "confirmed"
        assert (training_confirmed_dir / frame_files[0]).exists(), "Frame should be in training/confirmed"
        print(f"    Frame moved to training/confirmed")
        
        print("\n[Test 9] Testing reject_frame...")
        initial_rejected = updated_profile.rejected_count
        review_manager.reject_frame(hedgehog.id, frame_files[1], save_as_negative=True)
        
        # Reload profile
        updated_profile = profile_manager.get_profile(hedgehog.id)
        print(f"✓ Rejected frame {frame_files[1]} (saved as negative)")
        print(f"    Rejected count: {initial_rejected} → {updated_profile.rejected_count}")
        assert updated_profile.rejected_count == initial_rejected + 1
        
        # Check frame moved to training/rejected
        training_rejected_dir = data_dir / "training" / hedgehog.id / "rejected"
        assert (training_rejected_dir / frame_files[1]).exists(), "Frame should be in training/rejected"
        print(f"    Frame moved to training/rejected")
        
        print("\n[Test 10] Testing bulk_confirm_frames...")
        frames_to_confirm = [frame_files[2], frame_files[3]]
        bulk_result = review_manager.bulk_confirm_frames(hedgehog.id, frames_to_confirm)
        print(f"✓ Bulk confirmed {len(bulk_result['confirmed'])} frames")
        print(f"    Failed: {len(bulk_result['failed'])}")
        print(f"    Updated accuracy: {bulk_result['updated_accuracy']:.1f}%")
        
        print("\n[Test 11] Testing get_review_stats...")
        stats = review_manager.get_review_stats(hedgehog.id)
        print(f"✓ Review stats for {stats['profile_name']}:")
        print(f"    Pending: {stats['pending_count']}")
        print(f"    Confirmed: {stats['confirmed_count']}")
        print(f"    Rejected: {stats['rejected_count']}")
        print(f"    Accuracy: {stats['accuracy_percentage']:.1f}%")
        print(f"    Avg confidence: {stats['average_confidence']:.3f}")
        
        print("\n[Test 12] Testing ReviewSummary...")
        summary = ReviewSummary(review_manager)
        dashboard = summary.get_dashboard_summary()
        print(f"✓ Dashboard summary:")
        print(f"    Total pending: {dashboard['total_pending']}")
        print(f"    Total confirmed: {dashboard['total_confirmed']}")
        print(f"    Total rejected: {dashboard['total_rejected']}")
        print(f"    Profiles with pending: {dashboard['profiles_with_pending']}")
        
        print("\n[Test 13] Testing move_confirmed_to_sorted...")
        confirmed_frames = [frame_files[2]]  # Already confirmed in bulk
        move_result = review_manager.move_confirmed_to_sorted(hedgehog.id, confirmed_frames)
        print(f"✓ Moved {len(move_result['moved'])} frames to sorted")
        
        sorted_dir = data_dir / "sorted" / hedgehog.id
        if (sorted_dir / frame_files[2]).exists():
            print(f"    Frame successfully in /sorted directory")
        
        print("\n[Test 14] Verifying directory structure...")
        dirs_to_check = [
            data_dir / "review" / hedgehog.id,
            data_dir / "training" / hedgehog.id / "confirmed",
            data_dir / "training" / hedgehog.id / "rejected",
            data_dir / "sorted" / hedgehog.id
        ]
        
        for dir_path in dirs_to_check:
            if dir_path.exists():
                files = list(dir_path.glob("*.jpg"))
                print(f"✓ {dir_path.name}: {len(files)} frames")
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)
            logger.info(f"Cleaned up test directory: {test_dir}")


if __name__ == "__main__":
    success = test_review_feedback()
    sys.exit(0 if success else 1)
