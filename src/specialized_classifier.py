"""
Specialized classifier for fine-grained species identification (Stage 2).
Uses custom-trained models to identify specific animals from YOLO detections.
"""
import logging
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class SpeciesClassifier:
    """Individual species classifier model wrapper."""
    
    def __init__(self, species_name: str, model_path: Path, 
                 confidence_threshold: float = 0.75, device: str = 'cpu'):
        """
        Initialize a species classifier.
        
        Args:
            species_name: Name of the species (e.g., 'hedgehog', 'finch')
            model_path: Path to the trained model file
            confidence_threshold: Minimum confidence for positive detection
            device: 'cpu' or 'cuda'
        """
        self.species_name = species_name
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load model if path exists
        if model_path.exists():
            self._load_model()
        else:
            logger.warning(f"Model not found for {species_name}: {model_path}")
    
    def _load_model(self):
        """Load the trained model from disk."""
        try:
            logger.info(f"Loading {self.species_name} classifier from {self.model_path}")
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create model architecture (ResNet18 for transfer learning)
            self.model = models.resnet18(pretrained=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 2)  # Binary: species vs not-species
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Successfully loaded {self.species_name} classifier")
            
        except Exception as e:
            logger.error(f"Failed to load model for {self.species_name}: {e}")
            self.model = None
    
    def classify(self, image: Image.Image) -> Tuple[bool, float]:
        """
        Classify an image as species or not-species.
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Tuple of (is_species: bool, confidence: float)
        """
        if self.model is None:
            return False, 0.0
        
        try:
            # Preprocess image
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][1].item()  # Probability of being the species
            
            is_species = confidence >= self.confidence_threshold
            
            logger.debug(f"{self.species_name} classification: {confidence:.3f} "
                        f"({'MATCH' if is_species else 'no match'})")
            
            return is_species, confidence
            
        except Exception as e:
            logger.error(f"Classification error for {self.species_name}: {e}")
            return False, 0.0


class SpecializedClassifier:
    """
    Manages multiple specialized species classifiers.
    Coordinates Stage 2 classification after YOLO detection.
    """
    
    def __init__(self, config: dict):
        """
        Initialize specialized classifier system.
        
        Args:
            config: Configuration dictionary with specialized_detection section
        """
        self.config = config
        self.enabled = config.get('specialized_detection', {}).get('enabled', False)
        self.species_config = config.get('specialized_detection', {}).get('species', [])
        
        # Determine device (GPU if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"SpecializedClassifier using device: {self.device}")
        
        # Load all species classifiers
        self.classifiers: Dict[str, SpeciesClassifier] = {}
        self.parent_class_mapping: Dict[str, List[str]] = {}  # YOLO class -> species list
        
        if self.enabled:
            self._load_classifiers()
        else:
            logger.info("Specialized detection is disabled in config")
    
    def _load_classifiers(self):
        """Load all configured species classifiers."""
        for species_cfg in self.species_config:
            species_name = species_cfg['name']
            model_path = Path(species_cfg['model_path'])
            confidence_threshold = species_cfg.get('confidence_threshold', 0.75)
            parent_yolo_classes = species_cfg.get('parent_yolo_class', [])
            
            # Create classifier
            classifier = SpeciesClassifier(
                species_name=species_name,
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=self.device
            )
            
            self.classifiers[species_name] = classifier
            
            # Build parent class mapping
            for parent_class in parent_yolo_classes:
                if parent_class not in self.parent_class_mapping:
                    self.parent_class_mapping[parent_class] = []
                self.parent_class_mapping[parent_class].append(species_name)
            
            logger.info(f"Loaded {species_name} classifier (parent classes: {parent_yolo_classes})")
    
    def classify_detections(self, video_path: Path, yolo_detections: Dict[str, float],
                           detected_objects_path: Path) -> Dict[str, float]:
        """
        Run specialized classification on YOLO detections.
        
        Args:
            video_path: Path to the video file being processed
            yolo_detections: YOLO detection results {label: confidence}
            detected_objects_path: Path where YOLO saved detection images
            
        Returns:
            Dictionary of {species_name: confidence} for specialized detections
        """
        if not self.enabled or not self.classifiers:
            return {}
        
        species_results = {}
        video_name = video_path.stem
        
        logger.info(f"Running specialized classification for {video_name}")
        logger.debug(f"YOLO detections: {yolo_detections}")
        
        # For each YOLO detection, check if we have specialized classifiers
        for yolo_class, yolo_confidence in yolo_detections.items():
            if yolo_class not in self.parent_class_mapping:
                continue
            
            # Get specialized classifiers for this YOLO class
            species_to_check = self.parent_class_mapping[yolo_class]
            logger.debug(f"YOLO detected '{yolo_class}', checking species: {species_to_check}")
            
            # Find detection images for this YOLO class
            label_dir = detected_objects_path / yolo_class.replace(" ", "_")
            if not label_dir.exists():
                logger.warning(f"Detection images not found for {yolo_class}")
                continue
            
            # Get recent detection images for this video
            detection_images = []
            for img_file in label_dir.glob("*.jpg"):
                # Check if this image is from the current video
                if video_name in img_file.name:
                    detection_images.append(img_file)
            
            if not detection_images:
                logger.debug(f"No detection images found for {yolo_class} in {video_name}")
                continue
            
            logger.debug(f"Found {len(detection_images)} detection images for {yolo_class}")
            
            # Run each specialized classifier on the detection images
            for species_name in species_to_check:
                classifier = self.classifiers[species_name]
                
                if classifier.model is None:
                    logger.warning(f"No model loaded for {species_name}, skipping")
                    continue
                
                # Classify each detection image
                max_confidence = 0.0
                best_match_image = None
                
                for img_path in detection_images:
                    try:
                        image = Image.open(img_path).convert('RGB')
                        is_species, confidence = classifier.classify(image)
                        
                        if confidence > max_confidence:
                            max_confidence = confidence
                            best_match_image = img_path
                        
                        if is_species:
                            logger.info(f"✓ {species_name} detected in {img_path.name} "
                                      f"(confidence: {confidence:.3f})")
                    
                    except Exception as e:
                        logger.error(f"Error classifying {img_path}: {e}")
                
                # Store best result for this species
                if max_confidence > 0:
                    species_results[species_name] = max_confidence
                    
                    # Save specialized detection metadata
                    if max_confidence >= classifier.confidence_threshold:
                        self._save_specialized_detection(
                            video_path, species_name, max_confidence, 
                            best_match_image, yolo_class
                        )
        
        logger.info(f"Specialized detection results for {video_name}: {species_results}")
        return species_results
    
    def _save_specialized_detection(self, video_path: Path, species_name: str,
                                   confidence: float, detection_image: Path,
                                   yolo_class: str):
        """
        Save metadata for a successful specialized detection.
        
        Args:
            video_path: Source video path
            species_name: Detected species name
            confidence: Detection confidence
            detection_image: Path to the detection image
            yolo_class: Original YOLO class that triggered this check
        """
        try:
            # Create specialized detections directory
            specialized_dir = Path("/data/objects/specialized")
            specialized_dir.mkdir(parents=True, exist_ok=True)
            
            species_dir = specialized_dir / species_name.replace(" ", "_")
            species_dir.mkdir(parents=True, exist_ok=True)
            
            # Create metadata
            metadata = {
                "species": species_name,
                "confidence": confidence,
                "video_name": video_path.name,
                "detection_image": str(detection_image),
                "yolo_class": yolo_class,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save metadata
            metadata_file = species_dir / f"{video_path.stem}_{species_name}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Saved specialized detection metadata: {metadata_file}")
            
        except Exception as e:
            logger.error(f"Failed to save specialized detection metadata: {e}")
    
    def get_target_species(self) -> List[str]:
        """Get list of all configured target species."""
        return list(self.classifiers.keys())
    
    def is_model_available(self, species_name: str) -> bool:
        """Check if a trained model exists for a species."""
        if species_name not in self.classifiers:
            return False
        return self.classifiers[species_name].model is not None
