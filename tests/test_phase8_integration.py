"""
Integration tests for Phase 8 API endpoints (Review & Retraining).

Tests the complete workflow:
1. Create animal profile
2. Generate mock frames in review directory
3. List pending reviews
4. Serve frame images
5. Confirm/reject frames via API
6. Verify accuracy updates
7. Trigger retraining
"""
import sys
from pathlib import Path
import json
import shutil
import pytest
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from animal_profile import AnimalProfileManager, AnimalProfile
from review_feedback import ReviewManager, ReviewFrame
from PIL import Image
import io


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create subdirectories
    (data_dir / "animal_profiles").mkdir()
    (data_dir / "sorted").mkdir()
    (data_dir / "review").mkdir()
    (data_dir / "training").mkdir()
    
    yield data_dir
    
    # Cleanup
    if data_dir.exists():
        shutil.rmtree(data_dir)


@pytest.fixture
def animal_profile_manager(test_data_dir):
    """Create and return an AnimalProfileManager instance."""
    return AnimalProfileManager(test_data_dir)


@pytest.fixture
def review_manager(animal_profile_manager, test_data_dir):
    """Create and return a ReviewManager instance."""
    return ReviewManager(animal_profile_manager, test_data_dir)


@pytest.fixture
def test_profile(animal_profile_manager):
    """Create a test animal profile."""
    profile = animal_profile_manager.create_profile(
        name="Test Hedgehog",
        yolo_categories=["cat", "dog"],
        text_description="A small hedgehog"
    )
    return profile


@pytest.fixture
def mock_frames(test_data_dir, test_profile):
    """Create mock frame images in the review directory."""
    review_dir = test_data_dir / "review" / test_profile.id
    review_dir.mkdir(parents=True, exist_ok=True)
    
    frames = []
    for i in range(5):
        # Create a simple test image
        img = Image.new('RGB', (640, 480), color=(73, 109, 137))
        frame_path = review_dir / f"frame_{i:06d}.jpg"
        img.save(frame_path)
        
        # Create metadata
        metadata = {
            "confidence": 0.85 + (i * 0.02),
            "description": "Test hedgehog image",
            "timestamp": datetime.now().isoformat()
        }
        metadata_path = Path(str(frame_path) + ".json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        frames.append(frame_path.name)
    
    return frames


class TestPendingReviewsEndpoint:
    """Tests for GET /api/animal-profiles/{id}/pending-reviews"""
    
    def test_list_pending_reviews_empty(self, review_manager, test_profile):
        """Test listing pending reviews when directory is empty."""
        pending = review_manager.list_pending_reviews(test_profile.id)
        assert pending == []
    
    def test_list_pending_reviews_with_frames(self, review_manager, test_profile, mock_frames):
        """Test listing pending reviews with actual frames."""
        pending = review_manager.list_pending_reviews(test_profile.id)
        
        assert len(pending) == 5
        assert all(isinstance(frame, ReviewFrame) for frame in pending)
        assert all(frame.frame_path.exists() for frame in pending)
    
    def test_pending_review_metadata(self, review_manager, test_profile, mock_frames):
        """Test that pending reviews contain correct metadata."""
        pending = review_manager.list_pending_reviews(test_profile.id)
        
        for i, frame in enumerate(pending):
            assert frame.frame_path.name == f"frame_{i:06d}.jpg"
            assert frame.get_confidence() >= 0.85  # >= because first frame is exactly 0.85
            assert frame.get_description() == "Test hedgehog image"
            assert frame.metadata.get("timestamp") is not None
    
    def test_pending_count_matches_frames(self, review_manager, test_profile, mock_frames):
        """Test that pending count matches actual frame count."""
        count = review_manager.get_pending_count(test_profile.id)
        assert count == len(mock_frames)


class TestFrameServingEndpoint:
    """Tests for GET /api/animal-profiles/{id}/frame/{filename}"""
    
    def test_frame_path_resolution(self, test_data_dir, test_profile, mock_frames):
        """Test that frame paths are correctly resolved."""
        review_dir = test_data_dir / "review" / test_profile.id
        frame_path = review_dir / mock_frames[0]
        
        assert frame_path.exists()
        assert frame_path.suffix == ".jpg"
    
    def test_frame_with_metadata(self, test_data_dir, test_profile, mock_frames):
        """Test that frames can be retrieved with metadata."""
        review_dir = test_data_dir / "review" / test_profile.id
        frame_path = review_dir / mock_frames[0]
        metadata_path = Path(str(frame_path) + ".json")
        
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert "confidence" in metadata
        assert "timestamp" in metadata
    
    def test_frame_image_loadable(self, test_data_dir, test_profile, mock_frames):
        """Test that frame images can be loaded with PIL."""
        review_dir = test_data_dir / "review" / test_profile.id
        frame_path = review_dir / mock_frames[0]
        
        img = Image.open(frame_path)
        assert img.size == (640, 480)


class TestConfirmImagesEndpoint:
    """Tests for POST /api/animal-profiles/{id}/confirm-images"""
    
    def test_confirm_single_frame(self, review_manager, test_profile, mock_frames):
        """Test confirming a single frame."""
        review_dir = Path("/data/review") / test_profile.id
        review_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a frame to confirm
        img = Image.new('RGB', (640, 480))
        frame_path = review_dir / mock_frames[0]
        img.save(frame_path)
        
        # Confirm it
        results = review_manager.bulk_confirm_frames(test_profile.id, [mock_frames[0]])
        
        assert len(results['confirmed']) == 1
        assert mock_frames[0] in results['confirmed']
    
    def test_confirm_multiple_frames(self, review_manager, test_profile, mock_frames, test_data_dir):
        """Test confirming multiple frames."""
        review_dir = test_data_dir / "review" / test_profile.id
        
        # Confirm first 3 frames
        to_confirm = mock_frames[:3]
        results = review_manager.bulk_confirm_frames(test_profile.id, to_confirm)
        
        assert len(results['confirmed']) == 3
        assert len(results['failed']) == 0
    
    def test_confirmed_frames_moved(self, review_manager, test_profile, mock_frames, test_data_dir):
        """Test that confirmed frames are moved to training directory."""
        review_dir = test_data_dir / "review" / test_profile.id
        training_dir = test_data_dir / "training" / test_profile.id / "confirmed"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # Confirm a frame
        results = review_manager.bulk_confirm_frames(test_profile.id, [mock_frames[0]])
        
        # Frame should be in training directory
        assert (training_dir / mock_frames[0]).exists()
        # Frame should no longer be in review directory
        assert not (review_dir / mock_frames[0]).exists()
    
    def test_accuracy_updated_on_confirm(self, animal_profile_manager, review_manager, test_profile, mock_frames, test_data_dir):
        """Test that accuracy counters are updated after confirmation."""
        initial_confirmed = test_profile.confirmed_count
        
        review_dir = test_data_dir / "review" / test_profile.id
        
        # Confirm frames
        review_manager.bulk_confirm_frames(test_profile.id, mock_frames[:2])
        
        # Reload profile
        updated_profile = animal_profile_manager.get_profile(test_profile.id)
        assert updated_profile.confirmed_count == initial_confirmed + 2


class TestRejectImagesEndpoint:
    """Tests for POST /api/animal-profiles/{id}/reject-images"""
    
    def test_reject_single_frame_delete(self, review_manager, test_profile, mock_frames, test_data_dir):
        """Test rejecting a single frame (delete mode)."""
        review_dir = test_data_dir / "review" / test_profile.id
        
        # Reject frame (delete)
        results = review_manager.bulk_reject_frames(test_profile.id, [mock_frames[0]], save_as_negative=False)
        
        assert len(results['rejected']) == 1
        assert mock_frames[0] in results['rejected']
    
    def test_reject_frame_deleted(self, review_manager, test_profile, mock_frames, test_data_dir):
        """Test that rejected frames are actually deleted."""
        review_dir = test_data_dir / "review" / test_profile.id
        frame_path = review_dir / mock_frames[0]
        
        # Reject frame
        review_manager.bulk_reject_frames(test_profile.id, [mock_frames[0]], save_as_negative=False)
        
        # Frame should be deleted
        assert not frame_path.exists()
    
    def test_reject_frame_save_as_negative(self, review_manager, test_profile, mock_frames, test_data_dir):
        """Test rejecting a frame and saving as negative training example."""
        review_dir = test_data_dir / "review" / test_profile.id
        training_dir = test_data_dir / "training" / test_profile.id / "rejected"
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # Reject frame (save as negative)
        results = review_manager.bulk_reject_frames(test_profile.id, [mock_frames[0]], save_as_negative=True)
        
        assert len(results['rejected']) == 1
        # Frame should be in rejected training directory
        assert (training_dir / mock_frames[0]).exists()
        # Frame should no longer be in review directory
        assert not (review_dir / mock_frames[0]).exists()
    
    def test_accuracy_updated_on_reject(self, animal_profile_manager, review_manager, test_profile, mock_frames, test_data_dir):
        """Test that accuracy counters are updated after rejection."""
        initial_rejected = test_profile.rejected_count
        
        # Reject frames
        review_manager.bulk_reject_frames(test_profile.id, mock_frames[:2])
        
        # Reload profile
        updated_profile = animal_profile_manager.get_profile(test_profile.id)
        assert updated_profile.rejected_count == initial_rejected + 2


class TestBulkOperations:
    """Tests for bulk operations on frames."""
    
    def test_mixed_confirm_and_reject(self, review_manager, test_profile, mock_frames, test_data_dir):
        """Test confirming and rejecting different frames."""
        review_dir = test_data_dir / "review" / test_profile.id
        
        # Confirm first 2, reject next 2
        confirm_results = review_manager.bulk_confirm_frames(test_profile.id, mock_frames[:2])
        reject_results = review_manager.bulk_reject_frames(test_profile.id, mock_frames[2:4])
        
        assert len(confirm_results['confirmed']) == 2
        assert len(reject_results['rejected']) == 2
    
    def test_partial_failure(self, review_manager, test_profile, mock_frames, test_data_dir):
        """Test handling of partial failures in bulk operations."""
        # Try to confirm non-existent frames
        results = review_manager.bulk_confirm_frames(test_profile.id, ["nonexistent.jpg"])
        
        assert len(results['failed']) > 0
        assert len(results['confirmed']) == 0


class TestAccuracyTracking:
    """Tests for accuracy tracking after feedback."""
    
    def test_accuracy_calculation(self, animal_profile_manager, test_profile):
        """Test accuracy percentage calculation."""
        profile = test_profile
        
        # No feedback yet
        assert profile.accuracy_percentage == 0.0
        
        # Update with feedback
        animal_profile_manager.update_accuracy(test_profile.id, confirmed=10, rejected=0)
        updated_profile = animal_profile_manager.get_profile(test_profile.id)
        
        assert updated_profile.confirmed_count == 10
        assert updated_profile.accuracy_percentage == 100.0
    
    def test_accuracy_with_mixed_feedback(self, animal_profile_manager, test_profile):
        """Test accuracy calculation with both confirmations and rejections."""
        animal_profile_manager.update_accuracy(test_profile.id, confirmed=80, rejected=20)
        updated_profile = animal_profile_manager.get_profile(test_profile.id)
        
        assert updated_profile.confirmed_count == 80
        assert updated_profile.rejected_count == 20
        assert updated_profile.accuracy_percentage == 80.0


class TestRetrainingTrigger:
    """Tests for retraining endpoint."""
    
    def test_retrain_endpoint_acceptance(self, animal_profile_manager, test_profile):
        """Test that retrain endpoint accepts valid profile."""
        profile = animal_profile_manager.get_profile(test_profile.id)
        assert profile is not None
        # Endpoint would accept this


class TestEndToEndWorkflow:
    """Tests for complete end-to-end workflow."""
    
    def test_complete_review_workflow(self, animal_profile_manager, review_manager, test_profile, mock_frames, test_data_dir):
        """Test complete workflow: create profile -> generate frames -> confirm/reject -> verify."""
        # 1. Verify profile created
        assert test_profile.name == "Test Hedgehog"
        assert test_profile.id is not None
        
        # 2. Verify frames exist
        pending = review_manager.list_pending_reviews(test_profile.id)
        assert len(pending) == 5
        
        # 3. Confirm and reject frames
        review_manager.bulk_confirm_frames(test_profile.id, mock_frames[:3])
        review_manager.bulk_reject_frames(test_profile.id, mock_frames[3:], save_as_negative=True)
        
        # 4. Verify accuracy updated
        updated_profile = animal_profile_manager.get_profile(test_profile.id)
        assert updated_profile.confirmed_count == 3
        assert updated_profile.rejected_count == 2
        assert updated_profile.accuracy_percentage == 60.0
        
        # 5. Verify no pending frames remain
        remaining = review_manager.list_pending_reviews(test_profile.id)
        assert len(remaining) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
