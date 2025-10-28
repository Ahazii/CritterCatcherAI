"""
Training manager for specialized species classifiers.
Handles dataset preparation, model training, and evaluation using transfer learning.
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json
import shutil
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class SpeciesDataset(Dataset):
    """Dataset for binary species classification (species vs not-species)."""
    
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of labels (1 = species, 0 = not-species)
            transform: Optional image transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        if len(image_paths) != len(labels):
            raise ValueError("Number of images and labels must match")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, 0


class TrainingManager:
    """Manages training of specialized species classifiers."""
    
    def __init__(self, config: dict):
        """
        Initialize training manager.
        
        Args:
            config: Configuration dictionary with training settings
        """
        self.config = config
        self.training_config = config.get('specialized_detection', {}).get('training', {})
        
        # Training hyperparameters
        self.epochs = self.training_config.get('epochs', 50)
        self.batch_size = self.training_config.get('batch_size', 32)
        self.learning_rate = self.training_config.get('learning_rate', 0.001)
        self.validation_split = self.training_config.get('validation_split', 0.2)
        self.data_augmentation = self.training_config.get('data_augmentation', True)
        
        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"TrainingManager initialized - Device: {self.device}")
        
        # Paths
        self.training_data_path = Path("/data/training_data")
        self.models_path = Path("/data/models")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_training = None
        self.training_history = []
    
    def get_transforms(self, augment: bool = False) -> transforms.Compose:
        """
        Get image transformations.
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            Composed transforms
        """
        if augment and self.data_augmentation:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def prepare_dataset(self, species_name: str, negative_samples_ratio: float = 1.0) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets for a species.
        
        Args:
            species_name: Name of the species to train
            negative_samples_ratio: Ratio of negative to positive samples
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        logger.info(f"Preparing dataset for {species_name}")
        
        species_dir = self.training_data_path / species_name.replace(" ", "_")
        
        # Collect positive samples (images of the species)
        positive_dir = species_dir / "train"
        if not positive_dir.exists():
            raise FileNotFoundError(f"Training data not found: {positive_dir}")
        
        positive_images = list(positive_dir.glob("*.jpg")) + list(positive_dir.glob("*.png"))
        if len(positive_images) == 0:
            raise ValueError(f"No training images found in {positive_dir}")
        
        logger.info(f"Found {len(positive_images)} positive samples")
        
        # Collect negative samples (images of other animals)
        negative_images = []
        negative_dir = species_dir / "negative"
        
        if negative_dir.exists():
            negative_images = list(negative_dir.glob("*.jpg")) + list(negative_dir.glob("*.png"))
            logger.info(f"Found {len(negative_images)} negative samples in negative folder")
        
        # If not enough negative samples, use YOLO detections from other classes
        if len(negative_images) < len(positive_images) * negative_samples_ratio:
            logger.info("Collecting additional negative samples from YOLO detections")
            detected_objects_path = Path("/data/objects/detected")
            
            if detected_objects_path.exists():
                for label_dir in detected_objects_path.iterdir():
                    if label_dir.is_dir() and label_dir.name != species_name.replace(" ", "_"):
                        label_images = list(label_dir.glob("*.jpg"))[:50]  # Limit per class
                        negative_images.extend(label_images)
        
        # Balance dataset
        target_negative_count = int(len(positive_images) * negative_samples_ratio)
        if len(negative_images) > target_negative_count:
            import random
            negative_images = random.sample(negative_images, target_negative_count)
        
        logger.info(f"Using {len(negative_images)} negative samples")
        
        # Create labels
        all_images = positive_images + negative_images
        all_labels = [1] * len(positive_images) + [0] * len(negative_images)
        
        # Shuffle
        import random
        combined = list(zip(all_images, all_labels))
        random.shuffle(combined)
        all_images, all_labels = zip(*combined)
        all_images = list(all_images)
        all_labels = list(all_labels)
        
        # Split into train/val
        split_idx = int(len(all_images) * (1 - self.validation_split))
        
        train_images = all_images[:split_idx]
        train_labels = all_labels[:split_idx]
        val_images = all_images[split_idx:]
        val_labels = all_labels[split_idx:]
        
        logger.info(f"Dataset split: {len(train_images)} train, {len(val_images)} validation")
        
        # Create datasets
        train_dataset = SpeciesDataset(
            train_images, 
            train_labels, 
            transform=self.get_transforms(augment=True)
        )
        
        val_dataset = SpeciesDataset(
            val_images,
            val_labels,
            transform=self.get_transforms(augment=False)
        )
        
        return train_dataset, val_dataset
    
    def create_model(self, pretrained: bool = True) -> nn.Module:
        """
        Create a ResNet18 model for binary classification.
        
        Args:
            pretrained: Whether to use pretrained weights
            
        Returns:
            PyTorch model
        """
        model = models.resnet18(pretrained=pretrained)
        
        # Replace final layer for binary classification
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)  # 2 classes: species vs not-species
        
        model = model.to(self.device)
        logger.info(f"Created ResNet18 model (pretrained={pretrained})")
        
        return model
    
    def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, model: nn.Module, dataloader: DataLoader, 
                criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train_species_classifier(self, species_name: str, 
                                progress_callback: Optional[Callable] = None) -> Dict:
        """
        Train a classifier for a specific species.
        
        Args:
            species_name: Name of the species
            progress_callback: Optional callback function(epoch, metrics)
            
        Returns:
            Training results dictionary
        """
        logger.info(f"=" * 80)
        logger.info(f"Starting training for {species_name}")
        logger.info(f"=" * 80)
        
        start_time = time.time()
        
        # Prepare datasets
        try:
            train_dataset, val_dataset = self.prepare_dataset(species_name)
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            return {"success": False, "error": str(e)}
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Create model
        model = self.create_model(pretrained=True)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # Training loop
        best_val_acc = 0.0
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Epoch {epoch+1}/{self.epochs} ({epoch_time:.1f}s) - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Callback for progress updates
            if progress_callback:
                progress_callback(epoch + 1, {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = self.models_path / f"{species_name.replace(' ', '_')}_classifier.pt"
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': history
                }, model_path)
                
                logger.info(f"✓ Saved best model (val_acc: {best_val_acc:.2f}%)")
        
        # Training complete
        total_time = time.time() - start_time
        
        logger.info(f"=" * 80)
        logger.info(f"Training complete for {species_name}")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        logger.info(f"Total training time: {total_time/60:.1f} minutes")
        logger.info(f"=" * 80)
        
        # Save training metadata
        metadata = {
            "species": species_name,
            "best_val_acc": best_val_acc,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "training_time_seconds": total_time,
            "timestamp": datetime.now().isoformat(),
            "history": history
        }
        
        metadata_path = self.models_path / f"{species_name.replace(' ', '_')}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "success": True,
            "species": species_name,
            "best_val_acc": best_val_acc,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "training_time": total_time
        }
    
    def get_training_status(self) -> Dict:
        """Get current training status."""
        return {
            "is_training": self.current_training is not None,
            "current_species": self.current_training.get("species") if self.current_training else None,
            "device": self.device,
            "available_models": self.list_trained_models()
        }
    
    def list_trained_models(self) -> List[Dict]:
        """List all trained models with metadata."""
        models = []
        
        for model_file in self.models_path.glob("*_classifier.pt"):
            species_name = model_file.stem.replace("_classifier", "").replace("_", " ")
            metadata_file = self.models_path / f"{model_file.stem.replace('_classifier', '')}_metadata.json"
            
            model_info = {
                "species": species_name,
                "model_path": str(model_file),
                "created": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
            }
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    model_info.update(metadata)
            
            models.append(model_info)
        
        return models
