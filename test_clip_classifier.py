#!/usr/bin/env python
"""Test script for CLIP/ViT classifier."""
import logging
from pathlib import Path
import sys
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_image(filepath: Path, color_rgb: tuple, text: str = ""):
    """Create a simple test image."""
    # Create a random image with dominant color
    width, height = 224, 224
    image = Image.new('RGB', (width, height), color_rgb)
    image.save(filepath)
    logger.info(f"Created test image: {filepath} with color {color_rgb}")
    if text:
        logger.info(f"  Description: {text}")


def test_clip_classifier():
    """Test the CLIP classifier."""
    print("=" * 60)
    print("Testing CLIP/ViT Classifier")
    print("=" * 60)
    
    # Import classifier
    try:
        from clip_vit_classifier import CLIPVitClassifier, AnimalIdentifier
        print("✓ Successfully imported CLIP classifier modules")
    except ImportError as e:
        print(f"✗ Failed to import CLIP classifier: {e}")
        print("  Note: CLIP requires PyTorch and transformers library")
        print("  Install with: pip install torch transformers pillow")
        return False
    
    # Create test images directory
    test_dir = Path("/tmp/clip_test")
    test_dir.mkdir(exist_ok=True)
    
    try:
        print("\n[Test 1] Creating test images...")
        # Create some test images
        create_test_image(test_dir / "hedgehog.jpg", (100, 80, 60), "brown/earthy colored image")
        create_test_image(test_dir / "bird.jpg", (135, 206, 235), "sky blue colored image")
        create_test_image(test_dir / "dog.jpg", (160, 82, 45), "brown colored image")
        print(f"✓ Created test images in {test_dir}")
        
        print("\n[Test 2] Initializing CLIP classifier...")
        print("  (First time will download model weights, may take time...)")
        classifier = CLIPVitClassifier()
        print(f"✓ CLIP classifier initialized on {classifier.device}")
        
        print("\n[Test 3] Scoring single image...")
        image_path = str(test_dir / "hedgehog.jpg")
        score = classifier.score_image(image_path, "a hedgehog")
        print(f"✓ Scored image: {score:.3f}")
        print(f"  Image: hedgehog.jpg")
        print(f"  Description: 'a hedgehog'")
        print(f"  Confidence: {score:.1%}")
        
        print("\n[Test 4] Comparing multiple descriptions...")
        descriptions = [
            "a hedgehog",
            "a dog",
            "a bird",
            "a cat",
            "a small spiky animal"
        ]
        results = classifier.compare_descriptions(image_path, descriptions)
        print(f"✓ Compared descriptions for hedgehog.jpg:")
        for desc, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"    {score:.3f} - {desc}")
        
        print("\n[Test 5] Batch scoring...")
        image_paths = [
            str(test_dir / "hedgehog.jpg"),
            str(test_dir / "bird.jpg"),
            str(test_dir / "dog.jpg"),
        ]
        scores = classifier.score_batch(image_paths, "a hedgehog")
        print(f"✓ Batch scored {len(scores)} images:")
        for path, score in zip(image_paths, scores):
            filename = Path(path).name
            print(f"    {score:.3f} - {filename}")
        
        print("\n[Test 6] High-level AnimalIdentifier API...")
        identifier = AnimalIdentifier(classifier)
        
        # Test single image identification
        is_target, confidence = identifier.identify_animal(
            str(test_dir / "hedgehog.jpg"),
            "a hedgehog",
            threshold=0.5
        )
        print(f"✓ Identified hedgehog.jpg as target: {is_target} (confidence: {confidence:.3f})")
        
        # Test batch processing
        print("\n[Test 7] Batch frame processing...")
        results = identifier.process_frames(
            image_paths,
            "a hedgehog",
            threshold=0.5
        )
        print(f"✓ Processed {results['stats']['total_frames']} frames:")
        print(f"    High confidence: {results['stats']['high_confidence_count']}")
        print(f"    Low confidence: {results['stats']['low_confidence_count']}")
        print(f"    Average score: {results['stats']['average_score']:.3f}")
        print(f"    Max score: {results['stats']['max_score']:.3f}")
        print(f"    Min score: {results['stats']['min_score']:.3f}")
        
        if results['high_confidence']:
            print(f"  High confidence matches:")
            for path, score in results['high_confidence'][:3]:
                print(f"    {score:.3f} - {Path(path).name}")
        
        if results['low_confidence']:
            print(f"  Low confidence matches:")
            for path, score in results['low_confidence'][:3]:
                print(f"    {score:.3f} - {Path(path).name}")
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return True
    
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
            logger.info(f"Cleaned up test directory: {test_dir}")


if __name__ == "__main__":
    success = test_clip_classifier()
    sys.exit(0 if success else 1)
