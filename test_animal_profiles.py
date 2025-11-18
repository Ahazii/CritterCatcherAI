#!/usr/bin/env python
"""Test script for animal profile API endpoints."""
import json
import asyncio
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from animal_profile import AnimalProfile, AnimalProfileManager


def test_animal_profile_manager():
    """Test the AnimalProfileManager class."""
    print("=" * 60)
    print("Testing AnimalProfileManager")
    print("=" * 60)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        manager = AnimalProfileManager(tmpdir)
        print(f"✓ Created profile manager at {manager.base_path}")
        
        # Test 1: Create profile
        print("\n[Test 1] Creating profile 'Hedgehog'...")
        profile = manager.create_profile(
            name="Hedgehog",
            yolo_categories=["cat", "dog", "sheep"],
            text_description="a small hedgehog"
        )
        assert profile.id == "hedgehog", f"Expected id 'hedgehog', got '{profile.id}'"
        assert profile.name == "Hedgehog"
        assert profile.accuracy_percentage == 0.0, "New profile should have 0% accuracy"
        print(f"✓ Created profile: {profile.name} (ID: {profile.id})")
        
        # Test 2: Get profile
        print("\n[Test 2] Retrieving profile...")
        retrieved = manager.get_profile("hedgehog")
        assert retrieved is not None, "Profile should exist"
        assert retrieved.name == "Hedgehog"
        print(f"✓ Retrieved profile: {retrieved.name}")
        
        # Test 3: List profiles
        print("\n[Test 3] Creating additional profiles and listing...")
        manager.create_profile("Finch", ["bird"])
        manager.create_profile("Fox", ["dog", "cat"])
        profiles = manager.list_profiles()
        assert len(profiles) == 3, f"Expected 3 profiles, got {len(profiles)}"
        print(f"✓ Listed {len(profiles)} profiles: {[p.name for p in profiles]}")
        
        # Test 4: Update profile
        print("\n[Test 4] Updating profile settings...")
        updated = manager.update_profile(
            "hedgehog",
            confidence_threshold=0.75,
            auto_approval_enabled=False,
            retraining_threshold=0.90
        )
        assert updated.confidence_threshold == 0.75
        assert updated.auto_approval_enabled == False
        assert updated.retraining_threshold == 0.90
        print(f"✓ Updated profile confidence_threshold to {updated.confidence_threshold}")
        
        # Test 5: Update accuracy
        print("\n[Test 5] Updating accuracy counters...")
        manager.update_accuracy("hedgehog", confirmed=40, rejected=10)
        profile = manager.get_profile("hedgehog")
        assert profile.confirmed_count == 40
        assert profile.rejected_count == 10
        expected_accuracy = (40 / 50) * 100
        assert profile.accuracy_percentage == expected_accuracy
        print(f"✓ Updated accuracy: {profile.confirmed_count} confirmed, "
              f"{profile.rejected_count} rejected = {profile.accuracy_percentage:.1f}%")
        
        # Test 6: Check retraining recommendation (at count threshold)
        print("\n[Test 6] Testing retraining recommendations...")
        should_retrain, message = profile.should_recommend_retraining
        print(f"  Accuracy: {profile.accuracy_percentage:.1f}% (threshold: {profile.retraining_threshold * 100:.0f}%)")
        print(f"  Confirmations: {profile.confirmed_count} (recommendation: {profile.confirmation_count_recommendation})")
        print(f"  Should retrain: {should_retrain}")
        print(f"  Message: {message}")
        # At 50 confirmations, should recommend retraining
        assert should_retrain == True, "Should recommend when count reaches threshold"
        print("✓ Retraining logic working correctly (recommends at count threshold)")
        
        # Test 7: Delete profile
        print("\n[Test 7] Deleting profile...")
        deleted = manager.delete_profile("finch")
        assert deleted == True, "Profile should be deleted"
        retrieved = manager.get_profile("finch")
        assert retrieved is None, "Deleted profile should not exist"
        print(f"✓ Deleted profile 'finch'")
        
        # Test 8: Duplicate profile check
        print("\n[Test 8] Testing duplicate prevention...")
        try:
            manager.create_profile("Hedgehog", ["dog"])
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"✓ Correctly prevented duplicate: {e}")
        
        # Test 9: Verify persistence
        print("\n[Test 9] Testing persistence to disk...")
        profiles_before = manager.list_profiles()
        profile_names_before = [p.name for p in profiles_before]
        
        # Create new manager instance from same directory
        manager2 = AnimalProfileManager(tmpdir)
        profiles_after = manager2.list_profiles()
        profile_names_after = [p.name for p in profiles_after]
        
        assert profile_names_before == profile_names_after, "Profiles should persist"
        print(f"✓ Profiles persisted correctly: {profile_names_after}")


def test_animal_profile_dataclass():
    """Test the AnimalProfile dataclass."""
    print("\n" + "=" * 60)
    print("Testing AnimalProfile Dataclass")
    print("=" * 60)
    
    # Create profile
    profile = AnimalProfile(
        id="hedgehog",
        name="Hedgehog",
        yolo_categories=["cat", "dog"],
        text_description="a hedgehog",
        confirmed_count=85,
        rejected_count=15,
        retraining_threshold=0.85,
        confirmation_count_recommendation=50
    )
    
    # Test accuracy calculation
    print("\n[Test 1] Accuracy calculation...")
    expected_accuracy = (85 / 100) * 100
    assert profile.accuracy_percentage == expected_accuracy
    print(f"✓ Accuracy: {profile.accuracy_percentage:.1f}%")
    
    # Test retraining recommendation by accuracy
    print("\n[Test 2] Retraining recommendation (accuracy below threshold)...")
    should_retrain, message = profile.should_recommend_retraining
    print(f"  Accuracy {profile.accuracy_percentage:.1f}% < Threshold {profile.retraining_threshold*100:.0f}%: {should_retrain}")
    print(f"  Message: {message}")
    assert should_retrain == True, "Should recommend retraining (accuracy too low)"
    print("✓ Correctly recommends retraining due to accuracy")
    
    # Test retraining recommendation by count (not by accuracy)
    print("\n[Test 3] Retraining recommendation (count threshold not reached)...")
    profile2 = AnimalProfile(
        id="finch",
        name="Finch",
        yolo_categories=["bird"],
        text_description="a finch",
        confirmed_count=45,  # 45/49 = 91.8%, well above 90% threshold
        rejected_count=4,    # Total = 49, below 50
        retraining_threshold=0.90,  # So accuracy check won't trigger
        confirmation_count_recommendation=50
    )
    should_retrain, message = profile2.should_recommend_retraining
    total = profile2.confirmed_count + profile2.rejected_count
    accuracy = profile2.accuracy_percentage
    print(f"  Accuracy: {accuracy:.1f}% (threshold: {profile2.retraining_threshold*100:.0f}%)")
    print(f"  Count: {total} < Recommendation {profile2.confirmation_count_recommendation}")
    assert should_retrain == False, "Should not recommend when count not reached AND accuracy good"
    print("✓ Correctly doesn't recommend retraining")
    
    # Test zero feedback
    print("\n[Test 4] Edge case - no feedback yet...")
    profile3 = AnimalProfile(
        id="fox",
        name="Fox",
        yolo_categories=["dog"],
        text_description="a fox"
    )
    assert profile3.accuracy_percentage == 0.0
    should_retrain, message = profile3.should_recommend_retraining
    assert should_retrain == False
    print(f"✓ New profile: accuracy={profile3.accuracy_percentage:.1f}%, should_retrain={should_retrain}")


def run_all_tests():
    """Run all tests."""
    try:
        test_animal_profile_dataclass()
        test_animal_profile_manager()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
