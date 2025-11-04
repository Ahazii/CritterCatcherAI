"""
Specialized classifier for fine-grained species identification (Stage 2).
Uses custom-trained models to identify specific animals from YOLO detections.
Supports hierarchical taxonomy for multi-level classification.
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

from taxonomy_tree import TaxonomyTree, TaxonomyNode

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
    Manages hierarchical species classifiers.
    Coordinates Stage 2+ classification after YOLO detection using taxonomy tree.
    """
    
    def __init__(self, config: dict, taxonomy_tree: Optional[TaxonomyTree] = None):
        """
        Initialize specialized classifier system.
        
        Args:
            config: Configuration dictionary with specialized_detection section
            taxonomy_tree: Hierarchical taxonomy tree (optional, for backward compatibility)
        """
        self.config = config
        self.enabled = config.get('specialized_detection', {}).get('enabled', False)
        self.taxonomy_tree = taxonomy_tree
        
        # Determine device (GPU if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"SpecializedClassifier using device: {self.device}")
        
        # Cache for loaded classifiers (node_id -> SpeciesClassifier)
        self.classifier_cache: Dict[str, SpeciesClassifier] = {}
        
        if self.enabled:
            if self.taxonomy_tree:
                logger.info("Using hierarchical taxonomy tree for classification")
            else:
                logger.warning("Taxonomy tree not provided, specialized detection may not work")
        else:
            logger.info("Specialized detection is disabled in config")
    
    def _get_or_load_classifier(self, node: TaxonomyNode) -> Optional[SpeciesClassifier]:
        """Get classifier from cache or load it."""
        if node.id in self.classifier_cache:
            return self.classifier_cache[node.id]
        
        if not node.has_model():
            return None
        
        # Load classifier
        try:
            classifier = SpeciesClassifier(
                species_name=node.name,
                model_path=Path(node.model_path),
                confidence_threshold=node.confidence_threshold,
                device=self.device
            )
            self.classifier_cache[node.id] = classifier
            logger.info(f"Loaded classifier for {node.name}")
            return classifier
        except Exception as e:
            logger.error(f"Failed to load classifier for {node.name}: {e}")
            return None
    
    def classify_detections(self, video_path: Path, yolo_detections: Dict[str, float],
                           detected_objects_path: Path) -> Dict[str, Tuple[float, List[str]]]:
        """
        Run hierarchical classification on YOLO detections.
        
        Args:
            video_path: Path to the video file being processed
            yolo_detections: YOLO detection results {label: confidence}
            detected_objects_path: Path where YOLO saved detection images
            
        Returns:
            Dictionary of {species_path: (confidence, path_list)} where:
            - species_path: Full path string (e.g. "bird/finch/goldfinch")
            - confidence: Detection confidence
            - path_list: List of taxonomy nodes from root to leaf
        """
        if not self.enabled or not self.taxonomy_tree:
            return {}
        
        results = {}
        video_name = video_path.stem
        
        logger.info(f"Running hierarchical classification for {video_name}")
        logger.info(f"YOLO Stage 1 detections: {yolo_detections}")
        
        if not yolo_detections:
            logger.warning(f"⚠️ No YOLO detections for {video_name} - Stage 2 cannot run without Stage 1 detections")
        
        # For each YOLO detection, traverse the taxonomy tree
        for yolo_class, yolo_confidence in yolo_detections.items():
            # Get classifier chain for this YOLO class
            logger.info(f"  → Processing YOLO class '{yolo_class}' (confidence: {yolo_confidence:.2f})")
            classifier_chain = self.taxonomy_tree.get_classifier_chain(yolo_class)
            
            if not classifier_chain:
                logger.debug(f"    No trained species models for {yolo_class}")
                logger.debug(f"No specialized classifiers configured for '{yolo_class}'")
                continue
            
            logger.info(f"    Found {len(classifier_chain)} trained species models for '{yolo_class}': {[n.name for n in classifier_chain]}")
            
            # Find detection images for this YOLO class
            label_dir = detected_objects_path / yolo_class.replace(" ", "_")
            if not label_dir.exists():
                logger.warning(f"    ⚠️ Detection images directory not found: {label_dir}")
                continue
            
            # Get detection images for this video
            detection_images = []
            for img_file in label_dir.glob("*.jpg"):
                if video_name in img_file.name:
                    detection_images.append(img_file)
            
            if not detection_images:
                logger.warning(f"    ⚠️ No detection images found for {yolo_class} in {video_name}")
                logger.debug(f"    Searched in: {label_dir}")
                continue
            
            logger.info(f"    Found {len(detection_images)} detection images for Stage 2 classification")
            
            # Run hierarchical classification
            classification_result = self._classify_hierarchical(
                detection_images, 
                classifier_chain,
                yolo_class
            )
            
            if classification_result:
                node, confidence, best_image = classification_result
                path = self.taxonomy_tree.get_node_path(node.id)
                path_str = "/".join(path)
                
                results[path_str] = (confidence, path)
                
                logger.info(f"✓ Hierarchical match: {path_str} (confidence: {confidence:.3f})")
                
                # Save specialized detection metadata
                self._save_specialized_detection(
                    video_path, path_str, confidence,
                    best_image, yolo_class
                )
        
        logger.info(f"Hierarchical classification results for {video_name}: {list(results.keys())}")
        return results
    
    def _classify_hierarchical(self, detection_images: List[Path], 
                              classifier_chain: List[TaxonomyNode],
                              yolo_class: str) -> Optional[Tuple[TaxonomyNode, float, Path]]:
        """
        Run hierarchical classification cascade through the tree.
        
        Args:
            detection_images: List of detection image paths
            classifier_chain: Ordered list of classifier nodes to check
            yolo_class: Original YOLO class
            
        Returns:
            Tuple of (matched_node, confidence, best_image) or None
        """
        best_match = None
        best_confidence = 0.0
        best_image = None
        
        # Try each classifier in the chain
        for node in classifier_chain:
            classifier = self._get_or_load_classifier(node)
            
            if classifier is None:
                logger.debug(f"No model available for {node.name}, skipping")
                continue
            
            # Classify each detection image with this classifier
            max_conf = 0.0
            max_img = None
            
            for img_path in detection_images:
                try:
                    image = Image.open(img_path).convert('RGB')
                    is_match, confidence = classifier.classify(image)
                    
                    if confidence > max_conf:
                        max_conf = confidence
                        max_img = img_path
                
                except Exception as e:
                    logger.error(f"Error classifying {img_path}: {e}")
            
            # Check if this classifier matched
            if max_conf >= node.confidence_threshold:
                logger.info(f"      ✓ MATCH: {node.name} (confidence: {max_conf:.3f} >= threshold: {node.confidence_threshold})")
                
                # Keep track of deepest match in tree
                if max_conf > best_confidence:
                    best_match = node
                    best_confidence = max_conf
                    best_image = max_img
            else:
                logger.info(f"      ✗ NO MATCH: {node.name} (confidence: {max_conf:.3f} < threshold: {node.confidence_threshold})")
        
        if best_match:
            return (best_match, best_confidence, best_image)
        
        return None
    
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
        """Get list of all species in taxonomy tree with trained models."""
        if not self.taxonomy_tree:
            return []
        
        species_list = []
        for node in self.taxonomy_tree._node_index.values():
            if node.level != 'yolo' and node.has_model():
                path = self.taxonomy_tree.get_node_path(node.id)
                species_list.append("/".join(path))
        
        return species_list
    
    def is_model_available(self, node_id: str) -> bool:
        """Check if a trained model exists for a taxonomy node."""
        if not self.taxonomy_tree:
            return False
        
        node = self.taxonomy_tree.get_node(node_id)
        if not node:
            return False
        
        return node.has_model()
