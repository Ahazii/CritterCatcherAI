"""Animal profile management for two-stage detection."""
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnimalProfile:
    """Configuration for target animal detection."""
    id: str
    name: str
    yolo_categories: List[str]
    text_description: str
    confidence_threshold: float = 0.80
    auto_approval_enabled: bool = True
    requires_manual_confirmation: bool = True
    enabled: bool = True
    confirmed_count: int = 0
    rejected_count: int = 0
    retraining_threshold: float = 0.85
    confirmation_count_recommendation: int = 50
    last_training_date: Optional[str] = None  # ISO format datetime
    training_manually_completed: bool = False  # User marked training as done
    
    @property
    def accuracy_percentage(self) -> float:
        """Calculate model accuracy percentage."""
        total = self.confirmed_count + self.rejected_count
        if total == 0:
            return 0.0
        return (self.confirmed_count / total) * 100
    
    @property
    def should_recommend_retraining(self) -> tuple:
        """Check if retraining should be recommended. Returns (bool, message)."""
        # Don't recommend if user manually marked training as complete
        if self.training_manually_completed:
            return False, ""
        
        total = self.confirmed_count + self.rejected_count
        
        # By accuracy
        if self.accuracy_percentage < (self.retraining_threshold * 100) and total > 10:
            return True, f"Model accuracy {self.accuracy_percentage:.1f}% below target {self.retraining_threshold*100:.0f}%"
        
        # By count
        if total >= self.confirmation_count_recommendation:
            return True, f"Reached {total} confirmations (recommendation: {self.confirmation_count_recommendation})"
        
        return False, ""


class AnimalProfileManager:
    """Manage animal profiles storage and retrieval."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path) / "animal_profiles"
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized animal profiles at {self.base_path}")
    
    def create_profile(self, name: str, yolo_categories: List[str], 
                      text_description: Optional[str] = None) -> AnimalProfile:
        """Create new animal profile."""
        profile_id = name.lower().replace(" ", "_")
        
        if self._profile_exists(profile_id):
            raise ValueError(f"Profile '{name}' already exists")
        
        description = text_description or f"a {name.lower()}"
        
        # Validate text description length (CLIP has 77 token limit)
        token_count = self._count_tokens(description)
        if token_count > 77:
            raise ValueError(
                f"Text description is too long ({token_count} tokens). "
                f"CLIP model has a 77 token limit. Please shorten your description. "
                f"Example: 'a hedgehog with spines' instead of a long paragraph."
            )
        
        profile = AnimalProfile(
            id=profile_id,
            name=name,
            yolo_categories=yolo_categories,
            text_description=description
        )
        
        self._save_profile(profile)
        logger.info(f"Created animal profile: {name} (text: {token_count} tokens)")
        return profile
    
    def get_profile(self, profile_id: str) -> Optional[AnimalProfile]:
        """Get profile by ID."""
        profile_file = self.base_path / f"{profile_id}.json"
        if not profile_file.exists():
            return None
        
        with open(profile_file, 'r') as f:
            data = json.load(f)
        return AnimalProfile(**data)
    
    def list_profiles(self) -> List[AnimalProfile]:
        """List all profiles."""
        profiles = []
        for profile_file in self.base_path.glob("*.json"):
            with open(profile_file, 'r') as f:
                data = json.load(f)
            profiles.append(AnimalProfile(**data))
        return sorted(profiles, key=lambda p: p.name)
    
    def update_profile(self, profile_id: str, **kwargs) -> AnimalProfile:
        """Update profile fields."""
        profile = self.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Profile '{profile_id}' not found")
        
        # Validate text_description if being updated
        if 'text_description' in kwargs:
            token_count = self._count_tokens(kwargs['text_description'])
            if token_count > 77:
                raise ValueError(
                    f"Text description is too long ({token_count} tokens). "
                    f"CLIP model has a 77 token limit. Please shorten your description. "
                    f"Example: 'a small brown hedgehog' instead of a long paragraph."
                )
        
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        self._save_profile(profile)
        logger.info(f"Updated profile: {profile_id}")
        return profile
    
    def delete_profile(self, profile_id: str) -> bool:
        """Delete profile and associated data."""
        profile_file = self.base_path / f"{profile_id}.json"
        if profile_file.exists():
            profile_file.unlink()
            logger.info(f"Deleted profile: {profile_id}")
            return True
        return False
    
    def update_accuracy(self, profile_id: str, confirmed: int, rejected: int):
        """Update model accuracy tracking."""
        profile = self.get_profile(profile_id)
        if profile:
            profile.confirmed_count = confirmed
            profile.rejected_count = rejected
            self._save_profile(profile)
    
    def _profile_exists(self, profile_id: str) -> bool:
        """Check if profile exists."""
        return (self.base_path / f"{profile_id}.json").exists()
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using CLIP's tokenizer approximation."""
        # Simple approximation: split on whitespace and punctuation
        # CLIP tokenizer is more complex, but this is close enough for validation
        import re
        # Split on whitespace and common punctuation
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())
        return len(tokens)
    
    def _save_profile(self, profile: AnimalProfile):
        """Save profile to disk."""
        profile_file = self.base_path / f"{profile.id}.json"
        with open(profile_file, 'w') as f:
            json.dump(asdict(profile), f, indent=2)
