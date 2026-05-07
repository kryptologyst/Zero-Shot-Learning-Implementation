"""Simple demo script for zero-shot learning (without CLIP)."""

import logging
import sys
from pathlib import Path

import torch
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_dataset import SyntheticDataset
from src.models.baseline_models import BaselineZeroShotModel
from src.utils.device import get_device, set_seed


def run_simple_demo():
    """Run a simple demo of zero-shot learning without CLIP."""
    print("🔍 Zero-Shot Learning Demo (Baseline Models)")
    print("=" * 50)
    
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
    
    # Test baseline model
    print("\n📈 Testing baseline model...")
    try:
        baseline_model = BaselineZeroShotModel(
            input_size=224 * 224 * 3,
            hidden_size=128,
            num_classes=len(class_descriptions),
        )
        
        print("✅ Baseline model created successfully")
        print(f"  Parameters: {baseline_model._count_parameters():,}")
        
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
        print(f"⚠️ Baseline model test failed: {e}")
    
    # Test classical baselines
    print("\n🔬 Testing classical baselines...")
    try:
        from src.models.baseline_models import ClassicalBaseline
        
        # Generate dummy data
        X = torch.randn(50, 3, 224, 224).numpy()
        y = torch.randint(0, len(class_descriptions), (50,)).numpy()
        
        # Test logistic regression
        lr_baseline = ClassicalBaseline("logistic_regression")
        lr_baseline.fit(X, y)
        lr_predictions = lr_baseline.predict(X[:10])
        lr_accuracy = (lr_predictions == y[:10]).mean()
        
        print(f"  Logistic Regression Accuracy: {lr_accuracy:.3f}")
        
        # Test random forest
        rf_baseline = ClassicalBaseline("random_forest")
        rf_baseline.fit(X, y)
        rf_predictions = rf_baseline.predict(X[:10])
        rf_accuracy = (rf_predictions == y[:10]).mean()
        
        print(f"  Random Forest Accuracy: {rf_accuracy:.3f}")
        
    except Exception as e:
        print(f"⚠️ Classical baseline test failed: {e}")
    
    print("\n🎉 Demo completed!")
    print("\nTo install CLIP and run the full demo:")
    print("  pip install clip-by-openai")
    print("  python run.py demo")
    print("\nTo run the interactive demo:")
    print("  streamlit run demo/streamlit_app.py")


if __name__ == "__main__":
    run_simple_demo()
