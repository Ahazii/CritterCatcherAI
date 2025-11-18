"""CLIP/ViT-based second-stage classifier for animal identification."""
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import json

logger = logging.getLogger(__name__)


class CLIPVitClassifier:
    """Text-based image classifier using CLIP for animal identification."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        """
        Initialize CLIP classifier.
        
        Args:
            model_name: HuggingFace model identifier (default: OpenAI CLIP-ViT base)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"CLIP model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def score_image(self, image_path: str, text_description: str) -> float:
        """
        Score an image against a text description.
        
        Args:
            image_path: Path to image file
            text_description: Text description of target animal (e.g., "a hedgehog")
        
        Returns:
            Confidence score (0.0 to 1.0)
        
        Raises:
            FileNotFoundError: If image doesn't exist
            RuntimeError: If scoring fails
        """
        try:
            # Load and process image
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs
            inputs = self.processor(
                text=text_description,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get logits and convert to probability
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Return confidence as float between 0 and 1
            confidence = float(probs[0, 0].cpu().numpy())
            
            logger.debug(f"Scored {image_path.name}: {confidence:.3f} - {text_description}")
            return confidence
        
        except FileNotFoundError as e:
            logger.error(f"Image file error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error scoring image {image_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to score image: {e}")
    
    def score_batch(self, image_paths: list, text_description: str) -> list:
        """
        Score multiple images against a text description.
        
        Args:
            image_paths: List of image file paths
            text_description: Text description of target animal
        
        Returns:
            List of confidence scores (same order as input)
        """
        scores = []
        errors = []
        
        for image_path in image_paths:
            try:
                score = self.score_image(image_path, text_description)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Failed to score {image_path}: {e}")
                scores.append(0.0)  # Default to 0 on error
                errors.append((image_path, str(e)))
        
        if errors:
            logger.warning(f"Encountered {len(errors)} errors during batch scoring")
        
        return scores
    
    def compare_descriptions(self, image_path: str, descriptions: list) -> dict:
        """
        Score an image against multiple text descriptions.
        
        Args:
            image_path: Path to image file
            descriptions: List of text descriptions to compare
        
        Returns:
            Dict mapping descriptions to confidence scores
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs
            inputs = self.processor(
                text=descriptions,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get logits and convert to probabilities
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Create result dict
            result = {}
            for i, desc in enumerate(descriptions):
                result[desc] = float(probs[0, i].cpu().numpy())
            
            return result
        
        except Exception as e:
            logger.error(f"Error comparing descriptions for {image_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to compare descriptions: {e}")


class AnimalIdentifier:
    """High-level API for identifying animals using CLIP."""
    
    def __init__(self, classifier: Optional[CLIPVitClassifier] = None):
        """
        Initialize animal identifier.
        
        Args:
            classifier: CLIPVitClassifier instance (creates default if None)
        """
        self.classifier = classifier or CLIPVitClassifier()
    
    def identify_animal(
        self,
        image_path: str,
        animal_description: str,
        threshold: float = 0.80,
        negative_descriptions: Optional[list] = None
    ) -> Tuple[bool, float]:
        """
        Identify if an image contains the target animal.
        
        Args:
            image_path: Path to image
            animal_description: Description of target animal (e.g., "a hedgehog")
            threshold: Confidence threshold (0.0-1.0)
            negative_descriptions: Optional list of negative descriptions to filter
        
        Returns:
            Tuple of (is_target_animal: bool, confidence: float)
        """
        try:
            # Get main confidence
            main_confidence = self.classifier.score_image(image_path, animal_description)
            
            # Check against negative descriptions if provided
            if negative_descriptions:
                neg_scores = self.classifier.compare_descriptions(
                    image_path,
                    negative_descriptions
                )
                max_neg_confidence = max(neg_scores.values())
                
                # If negative description scores higher, reduce confidence
                if max_neg_confidence > main_confidence:
                    logger.debug(
                        f"Negative description '{max(neg_scores, key=neg_scores.get)}' "
                        f"scored higher ({max_neg_confidence:.3f}) than target"
                    )
                    main_confidence = main_confidence * 0.5  # Penalty for conflicting match
            
            is_target = main_confidence >= threshold
            return is_target, main_confidence
        
        except Exception as e:
            logger.error(f"Error identifying animal in {image_path}: {e}")
            return False, 0.0
    
    def process_frames(
        self,
        frame_paths: list,
        animal_description: str,
        threshold: float = 0.80
    ) -> dict:
        """
        Process multiple frames and return results.
        
        Args:
            frame_paths: List of frame image paths
            animal_description: Description of target animal
            threshold: Confidence threshold
        
        Returns:
            Dict with:
            - 'high_confidence': list of (path, score) for scores >= threshold
            - 'low_confidence': list of (path, score) for scores < threshold
            - 'stats': dict with summary statistics
        """
        scores = self.classifier.score_batch(frame_paths, animal_description)
        
        high_confidence = []
        low_confidence = []
        
        for path, score in zip(frame_paths, scores):
            if score >= threshold:
                high_confidence.append((path, score))
            else:
                low_confidence.append((path, score))
        
        # Sort by confidence (descending)
        high_confidence.sort(key=lambda x: x[1], reverse=True)
        low_confidence.sort(key=lambda x: x[1], reverse=True)
        
        stats = {
            "total_frames": len(frame_paths),
            "high_confidence_count": len(high_confidence),
            "low_confidence_count": len(low_confidence),
            "average_score": float(np.mean(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "threshold": threshold
        }
        
        logger.info(
            f"Processed {stats['total_frames']} frames: "
            f"{stats['high_confidence_count']} high confidence, "
            f"{stats['low_confidence_count']} low confidence"
        )
        
        return {
            "high_confidence": high_confidence,
            "low_confidence": low_confidence,
            "stats": stats
        }


def load_or_create_classifier(device: Optional[str] = None) -> CLIPVitClassifier:
    """
    Load or create a CLIP classifier instance.
    
    Args:
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
    
    Returns:
        CLIPVitClassifier instance
    """
    logger.info("Initializing CLIP/ViT classifier")
    try:
        classifier = CLIPVitClassifier(device=device)
        logger.info("CLIP/ViT classifier ready")
        return classifier
    except Exception as e:
        logger.error(f"Failed to initialize CLIP classifier: {e}")
        raise
