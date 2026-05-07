"""Simple demo script for zero-shot learning."""

import logging
import sys
from pathlib import Path

import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_dataset import SyntheticDataset
from src.models.clip_model import CLIPZeroShotModel
from src.utils.device import get_device, set_seed


def create_demo_image():
    """Create a simple demo image."""
    import numpy as np
    
    # Create a simple colored square
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[50:150, 50:150] = [255, 0, 0]  # Red square
    
    return Image.fromarray(image)


def run_demo():
    """Run a simple demo of zero-shot learning."""
    print("🔍 Zero-Shot Learning Demo")
    print("=" * 40)
    
    # Set up
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Class descriptions
    class_descriptions = [
        "A photo of a cat",
        "A photo of a dog",
        "A picture of a car",
        "A picture of a tree",
        "An image of a bird",
    ]
    
    print(f"Class descriptions: {class_descriptions}")
    
    # Create synthetic dataset
    print("\n📊 Creating synthetic dataset...")
    dataset = SyntheticDataset(
        class_descriptions=class_descriptions,
        num_samples_per_class=5,
        image_size=(224, 224),
        split="test",
        seed=42,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test with synthetic data
    print("\n🎲 Testing with synthetic data...")
    for i in range(min(3, len(dataset))):
        image, true_label = dataset[i]
        true_class = class_descriptions[true_label]
        
        print(f"\nSample {i+1}:")
        print(f"  True class: {true_class}")
        print(f"  True label: {true_label}")
        
        # Convert tensor to PIL Image for display
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype('uint8')
        pil_image = Image.fromarray(image_np)
        
        # Save image for inspection
        output_path = Path("assets/demo_images")
        output_path.mkdir(parents=True, exist_ok=True)
        pil_image.save(output_path / f"sample_{i+1}.png")
        print(f"  Image saved to: {output_path / f'sample_{i+1}.png'}")
    
    # Test CLIP model (if available)
    print("\n🤖 Testing CLIP model...")
    try:
        model = CLIPZeroShotModel(
            model_name="ViT-B/32",
            device=device,
            freeze_vision=True,
            freeze_text=True,
        )
        
        print("✅ CLIP model loaded successfully")
        
        # Test with first sample
        if len(dataset) > 0:
            image, true_label = dataset[0]
            true_class = class_descriptions[true_label]
            
            print(f"\nTesting prediction on: {true_class}")
            
            # Make prediction
            with torch.no_grad():
                logits = model(image.unsqueeze(0), class_descriptions)
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1)
                
                predicted_class = class_descriptions[prediction.item()]
                confidence = probabilities.max().item()
                
                print(f"  Predicted class: {predicted_class}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Correct: {'✅' if prediction.item() == true_label else '❌'}")
                
                # Show all probabilities
                print("  All probabilities:")
                for i, (desc, prob) in enumerate(zip(class_descriptions, probabilities[0])):
                    print(f"    {desc}: {prob:.3f}")
        
    except Exception as e:
        print(f"⚠️ CLIP model test skipped: {e}")
    
    # Test baseline model
    print("\n📈 Testing baseline model...")
    try:
        from src.models.baseline_models import BaselineZeroShotModel
        
        baseline_model = BaselineZeroShotModel(
            input_size=224 * 224 * 3,
            hidden_size=128,
            num_classes=len(class_descriptions),
        )
        
        print("✅ Baseline model created successfully")
        
        # Test forward pass
        if len(dataset) > 0:
            image, true_label = dataset[0]
            
            with torch.no_grad():
                output = baseline_model(image.unsqueeze(0))
                prediction = torch.argmax(output, dim=-1)
                
                predicted_class = class_descriptions[prediction.item()]
                confidence = torch.softmax(output, dim=-1).max().item()
                
                print(f"  Predicted class: {predicted_class}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Correct: {'✅' if prediction.item() == true_label else '❌'}")
        
    except Exception as e:
        print(f"⚠️ Baseline model test skipped: {e}")
    
    print("\n🎉 Demo completed!")
    print("\nTo run the full training pipeline:")
    print("  python src/train/trainer.py")
    print("\nTo run the interactive demo:")
    print("  streamlit run demo/streamlit_app.py")


if __name__ == "__main__":
    run_demo()
