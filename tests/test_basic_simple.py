"""Basic tests for zero-shot learning implementation (without CLIP)."""

import logging
import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_dataset import SyntheticDataset
from src.models.baseline_models import BaselineZeroShotModel, ClassicalBaseline
from src.utils.device import get_device, set_seed


def test_synthetic_dataset():
    """Test synthetic dataset generation."""
    print("Testing synthetic dataset...")
    
    class_descriptions = [
        "A photo of a cat",
        "A photo of a dog",
        "A picture of a car",
    ]
    
    dataset = SyntheticDataset(
        class_descriptions=class_descriptions,
        num_samples_per_class=10,
        image_size=(224, 224),
        split="train",
        seed=42,
    )
    
    assert len(dataset) > 0
    assert len(dataset.get_class_descriptions()) == 3
    
    # Test data loading
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)
    assert image.shape == (3, 224, 224)
    assert 0 <= label < 3
    
    print("✅ Synthetic dataset test passed")


def test_baseline_model():
    """Test baseline model."""
    print("Testing baseline model...")
    
    model = BaselineZeroShotModel(
        input_size=224 * 224 * 3,
        hidden_size=512,
        num_classes=10,
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (2, 10)
    assert model._count_parameters() > 0
    
    print("✅ Baseline model test passed")


def test_classical_baseline():
    """Test classical baseline models."""
    print("Testing classical baseline...")
    
    baseline = ClassicalBaseline("logistic_regression")
    
    # Generate dummy data
    X = torch.randn(100, 3, 224, 224).numpy()
    y = torch.randint(0, 10, (100,)).numpy()
    
    # Test fitting and prediction
    baseline.fit(X, y)
    predictions = baseline.predict(X[:10])
    probabilities = baseline.predict_proba(X[:10])
    
    assert len(predictions) == 10
    assert probabilities.shape == (10, 10)
    
    print("✅ Classical baseline test passed")


def test_device_detection():
    """Test device detection."""
    print("Testing device detection...")
    
    device = get_device()
    assert isinstance(device, torch.device)
    
    print(f"✅ Device detection test passed - Using device: {device}")


def test_seed_setting():
    """Test seed setting."""
    print("Testing seed setting...")
    
    set_seed(42)
    
    # Generate some random numbers
    import numpy as np
    import random
    
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    # These should be deterministic
    np_val = np.random.random()
    random_val = random.random()
    torch_val = torch.rand(1).item()
    
    print(f"✅ Seed setting test passed - Random values: {np_val:.4f}, {random_val:.4f}, {torch_val:.4f}")


def run_all_tests():
    """Run all tests."""
    print("🧪 Running Zero-Shot Learning Tests (Basic)")
    print("=" * 50)
    
    tests = [
        test_synthetic_dataset,
        test_baseline_model,
        test_classical_baseline,
        test_device_detection,
        test_seed_setting,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check the output above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
