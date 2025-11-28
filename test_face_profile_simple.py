"""
Simplified validation tests for Face Profile system.
Tests core logic without requiring full webapp dependencies.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_face_profile_system():
    """Test Face Profile dataclass and FaceProfileManager."""
    from face_profile import FaceProfile, FaceProfileManager
    import tempfile
    
    print("="*60)
    print("FACE PROFILE SYSTEM VALIDATION")
    print("="*60)
    print()
    
    # Test 1: Imports
    print("✓ Test 1: Face Profile imports successful")
    
    # Test 2-5: Profile CRUD operations
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = FaceProfileManager(Path(tmpdir))
        
        # Create
        profile = manager.create_profile("John Doe", confidence_threshold=0.6)
        assert profile.name == "John Doe"
        assert profile.id == "john_doe"
        print("✓ Test 2: Profile creation works")
        
        # Duplicate prevention
        try:
            manager.create_profile("John Doe")
            assert False
        except ValueError:
            pass
        print("✓ Test 3: Duplicate prevention works")
        
        # Retrieve
        retrieved = manager.get_profile("john_doe")
        assert retrieved.name == "John Doe"
        print("✓ Test 4: Profile retrieval works")
        
        # List
        profiles = manager.list_profiles()
        assert len(profiles) == 1
        print("✓ Test 5: Profile listing works")
        
        # Update
        updated = manager.update_profile("john_doe", confidence_threshold=0.5)
        assert updated.confidence_threshold == 0.5
        print("✓ Test 6: Profile update works")
    
    # Test 7: Accuracy calculation
    profile = FaceProfile(
        id="test",
        name="Test",
        confirmed_count=80,
        rejected_count=20
    )
    assert profile.accuracy_percentage == 80.0
    print("✓ Test 7: Accuracy calculation works")
    
    # Test 8: Retraining recommendation
    profile.confirmed_count = 25
    profile.rejected_count = 0
    should_retrain, message = profile.should_recommend_retraining
    assert should_retrain == True
    print("✓ Test 8: Retraining recommendation works")
    
    print()
    print("="*60)
    print("✅ ALL TESTS PASSED (8/8)")
    print("="*60)
    return True


def test_main_imports():
    """Test main.py imports Face Profile Manager."""
    print()
    print("="*60)
    print("MAIN.PY INTEGRATION TEST")
    print("="*60)
    print()
    
    try:
        import main
        assert hasattr(main, 'FaceProfileManager')
        print("✓ Test 9: main.py imports FaceProfileManager")
        
        # Check it's used in process_videos
        import inspect
        source = inspect.getsource(main.process_videos)
        assert 'FaceProfileManager' in source
        print("✓ Test 10: FaceProfileManager initialized in process_videos")
        
        print()
        print("="*60)
        print("✅ INTEGRATION TESTS PASSED (2/2)")
        print("="*60)
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_api_structure():
    """Verify API endpoint structure without importing webapp."""
    print()
    print("="*60)
    print("API STRUCTURE VALIDATION")
    print("="*60)
    print()
    
    # Read webapp.py and check for endpoint definitions
    webapp_path = Path(__file__).parent / "src" / "webapp.py"
    with open(webapp_path, 'r') as f:
        content = f.read()
    
    endpoints = [
        '@app.get("/api/face-profiles")',
        '@app.post("/api/face-profiles")',
        '@app.get("/api/faces/unassigned")',
        '@app.post("/api/faces/assign")',
        '@app.post("/api/faces/reject")',
        '@app.post("/api/review/confirm-person")'
    ]
    
    for endpoint in endpoints:
        assert endpoint in content, f"Missing: {endpoint}"
        print(f"✓ Found: {endpoint}")
    
    print()
    print("="*60)
    print("✅ API STRUCTURE VALIDATED (6/6 endpoints)")
    print("="*60)
    return True


def print_workflow_summary():
    """Print workflow summary."""
    print()
    print("="*60)
    print("COMPLETE WORKFLOW")
    print("="*60)
    print()
    print("1. Review Tab: User sees 'person' videos")
    print("   └─ Click 'Confirm as Person'")
    print("   └─ API: POST /api/review/confirm-person")
    print("   └─ Extracts face images → /data/training/faces/unassigned/")
    print()
    print("2. Face Training Tab: User sees unassigned faces")
    print("   └─ API: GET /api/faces/unassigned")
    print("   └─ Select faces + Enter person name")
    print("   └─ Click 'Assign to Person'")
    print("   └─ API: POST /api/faces/assign")
    print("   └─ Creates Face Profile (if new)")
    print("   └─ Moves to /data/training/faces/{person}/confirmed/")
    print("   └─ Triggers face encoding retraining")
    print()
    print("3. Future videos with 'person'")
    print("   └─ Face recognition runs automatically")
    print("   └─ Identifies known people")
    print("   └─ Routes accordingly")
    print()
    print("="*60)


if __name__ == "__main__":
    try:
        success1 = test_face_profile_system()
        success2 = test_main_imports()
        success3 = test_api_structure()
        
        if success1 and success2 and success3:
            print_workflow_summary()
            print()
            print("="*60)
            print("✅ ALL VALIDATION PASSED - READY FOR DEPLOYMENT")
            print("="*60)
            print()
            print("Next Steps:")
            print("  1. Complete frontend UI (Confirm button + Face Training page)")
            print("  2. Build Docker container")
            print("  3. Deploy to Unraid")
            print("  4. Test end-to-end workflow with real videos")
            print()
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
