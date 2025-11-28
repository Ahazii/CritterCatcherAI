"""Face profile management for face recognition routing."""
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class FaceProfile:
    """Configuration for face recognition."""
    id: str
    name: str
    confidence_threshold: float = 0.6  # face_recognition tolerance (lower = stricter)
    auto_approval_enabled: bool = False
    enabled: bool = True
    confirmed_count: int = 0
    rejected_count: int = 0
    training_images_path: Optional[str] = None
    
    @property
    def accuracy_percentage(self) -> float:
        """Calculate accuracy percentage."""
        total = self.confirmed_count + self.rejected_count
        if total == 0:
            return 0.0
        return (self.confirmed_count / total) * 100
    
    @property
    def should_recommend_retraining(self) -> tuple:
        """Check if retraining should be recommended. Returns (bool, message)."""
        total = self.confirmed_count + self.rejected_count
        
        # By accuracy (target 85%)
        if self.accuracy_percentage < 85.0 and total > 10:
            return True, f"Model accuracy {self.accuracy_percentage:.1f}% below target 85%"
        
        # By count (recommend retraining every 20 confirmations)
        if total >= 20:
            return True, f"Reached {total} confirmations (recommendation: retrain every 20)"
        
        return False, ""


class FaceProfileManager:
    """Manage face profiles storage and retrieval."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path) / "faces" / "profiles"
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized face profiles at {self.base_path}")
    
    def create_profile(self, name: str, confidence_threshold: float = 0.6) -> FaceProfile:
        """Create new face profile."""
        profile_id = name.lower().replace(" ", "_")
        
        if self._profile_exists(profile_id):
            raise ValueError(f"Face profile '{name}' already exists")
        
        training_path = f"/data/training/faces/{profile_id}/confirmed"
        
        profile = FaceProfile(
            id=profile_id,
            name=name,
            confidence_threshold=confidence_threshold,
            training_images_path=training_path
        )
        
        # Create training directory
        Path(training_path).mkdir(parents=True, exist_ok=True)
        
        self._save_profile(profile)
        logger.info(f"Created face profile: {name}")
        return profile
    
    def get_profile(self, profile_id: str) -> Optional[FaceProfile]:
        """Get profile by ID."""
        profile_file = self.base_path / f"{profile_id}.json"
        if not profile_file.exists():
            return None
        
        with open(profile_file, 'r') as f:
            data = json.load(f)
        return FaceProfile(**data)
    
    def get_profile_by_name(self, name: str) -> Optional[FaceProfile]:
        """Get profile by person name."""
        profile_id = name.lower().replace(" ", "_")
        return self.get_profile(profile_id)
    
    def list_profiles(self) -> List[FaceProfile]:
        """List all face profiles."""
        profiles = []
        for profile_file in self.base_path.glob("*.json"):
            with open(profile_file, 'r') as f:
                data = json.load(f)
            profiles.append(FaceProfile(**data))
        return sorted(profiles, key=lambda p: p.name)
    
    def update_profile(self, profile_id: str, **kwargs) -> FaceProfile:
        """Update profile fields."""
        profile = self.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Face profile '{profile_id}' not found")
        
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        self._save_profile(profile)
        logger.info(f"Updated face profile: {profile_id}")
        return profile
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete profile and associated data."""
        profile_file = self.base_path / f"{profile_id}.json"
        if profile_file.exists():
            profile_file.unlink()
            logger.info(f"Deleted face profile: {profile_id}")
            return True
        return False
    
    def update_accuracy(self, profile_id: str, confirmed: int, rejected: int):
        """Update accuracy tracking."""
        profile = self.get_profile(profile_id)
        if profile:
            profile.confirmed_count = confirmed
            profile.rejected_count = rejected
            self._save_profile(profile)
    
    def _profile_exists(self, profile_id: str) -> bool:
        """Check if profile exists."""
        return (self.base_path / f"{profile_id}.json").exists()
    
    def _save_profile(self, profile: FaceProfile):
        """Save profile to disk."""
        profile_file = self.base_path / f"{profile.id}.json"
        with open(profile_file, 'w') as f:
            json.dump(asdict(profile), f, indent=2)
