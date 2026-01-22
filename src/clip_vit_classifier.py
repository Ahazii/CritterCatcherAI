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
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        force_cpu: bool = False
    ):
        """
        Initialize CLIP classifier.
        
        Args:
            model_name: HuggingFace model identifier (default: OpenAI CLIP-ViT base)
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name
        if force_cpu:
            self.device = "cpu"
        else:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading CLIP model: {model_name} on {self.device}")
        import os
        logger.info(f"NVIDIA_VISIBLE_DEVICES: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}")
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

    def get_image_embeddings(self, image_paths: list, batch_size: int = 8) -> np.ndarray:
        """Extract normalized CLIP image embeddings for a list of image paths."""
        embeddings = []
        valid_paths = []
        
        for image_path in image_paths:
            path_obj = Path(image_path)
            if path_obj.exists():
                valid_paths.append(path_obj)
            else:
                logger.warning(f"Image not found for embedding: {image_path}")
        
        if not valid_paths:
            return np.zeros((0, 0), dtype=np.float32)
        
        for i in range(0, len(valid_paths), batch_size):
            batch_paths = valid_paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy())
        
        return np.vstack(embeddings)

    def train_classifier(
        self,
        positive_paths: list,
        negative_paths: list,
        model_path: Path,
        epochs: int = 200,
        learning_rate: float = 0.1,
        l2: float = 0.001,
        batch_size: int = 8
    ) -> dict:
        """Train a lightweight binary classifier on CLIP embeddings."""
        pos_embeddings = self.get_image_embeddings(positive_paths, batch_size=batch_size)
        neg_embeddings = self.get_image_embeddings(negative_paths, batch_size=batch_size)
        
        if pos_embeddings.size == 0 or neg_embeddings.size == 0:
            raise ValueError("Need both positive and negative embeddings for training")
        
        X = np.vstack([pos_embeddings, neg_embeddings]).astype(np.float32)
        y = np.concatenate([
            np.ones(pos_embeddings.shape[0], dtype=np.float32),
            np.zeros(neg_embeddings.shape[0], dtype=np.float32)
        ])
        
        # Initialize weights
        weights = np.zeros(X.shape[1], dtype=np.float32)
        bias = 0.0
        
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))
        
        for _ in range(epochs):
            logits = X.dot(weights) + bias
            probs = sigmoid(logits)
            error = probs - y
            
            grad_w = (X.T.dot(error) / len(y)) + (l2 * weights)
            grad_b = float(np.mean(error))
            
            weights -= learning_rate * grad_w
            bias -= learning_rate * grad_b
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "weights": weights.tolist(),
            "bias": bias,
            "threshold": 0.5,
            "positive_count": int(pos_embeddings.shape[0]),
            "negative_count": int(neg_embeddings.shape[0]),
        }
        
        with open(model_path, 'w') as f:
            json.dump(model_data, f)
        
        return model_data

    @staticmethod
    def load_classifier(model_path: Path) -> Optional[dict]:
        """Load a classifier from disk."""
        if not model_path.exists():
            return None
        try:
            with open(model_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load classifier {model_path}: {e}")
            return None

    def score_with_classifier(self, image_paths: list, model_data: dict, batch_size: int = 8) -> list:
        """Score images using a trained classifier."""
        embeddings = self.get_image_embeddings(image_paths, batch_size=batch_size)
        if embeddings.size == 0:
            return []
        
        weights = np.array(model_data["weights"], dtype=np.float32)
        bias = float(model_data["bias"])
        
        logits = embeddings.dot(weights) + bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs.tolist()


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
