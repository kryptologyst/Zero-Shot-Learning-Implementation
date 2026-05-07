# Zero-Shot Learning Implementation

A comprehensive implementation of zero-shot learning using CLIP (Contrastive Language-Image Pre-training) for image classification. This project demonstrates how models can classify images from classes they have never seen during training, using only textual descriptions.

## Features

- **CLIP-based Zero-Shot Learning**: Implementation using OpenAI CLIP and OpenCLIP models
- **Synthetic Dataset Generation**: Automatically generates synthetic images for testing
- **Comprehensive Evaluation**: Multiple metrics including accuracy, F1, AUROC, AUPRC, and calibration error
- **Interactive Demo**: Streamlit web application for real-time inference
- **Modern Architecture**: Clean, typed, and reproducible codebase
- **Device Support**: Automatic device detection (CUDA → MPS → CPU)
- **Configuration Management**: Hydra-based configuration system

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA/MPS support (optional but recommended)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Zero-Shot-Learning-Implementation
   cd Zero-Shot-Learning-Implementation
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

   Or install with optional dependencies:
   ```bash
   pip install -e ".[dev,advanced]"
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

## Quick Start

### 1. Run the Interactive Demo

Launch the Streamlit demo application:

```bash
streamlit run demo/streamlit_app.py
```

This will open a web interface where you can:
- Upload images for zero-shot classification
- Generate synthetic test data
- Modify class descriptions
- View detailed prediction results

### 2. Train and Evaluate

Run the training pipeline:

```bash
python src/train/trainer.py
```

This will:
- Generate synthetic training data
- Train the CLIP model
- Evaluate on test set
- Save results and checkpoints

### 3. Custom Configuration

Modify the configuration in `configs/config.yaml`:

```yaml
# Model configuration
model:
  model_name: "ViT-B/32"  # or "ViT-L/14"
  temperature: 0.01
  freeze_vision: false
  freeze_text: false

# Data configuration
data:
  num_classes: 10
  num_samples_per_class: 100
  batch_size: 32

# Training configuration
training:
  max_epochs: 10
  learning_rate: 1e-4
  optimizer: "adamw"
```

## Dataset Schema

The synthetic dataset generates images with distinct patterns for each class:

- **Cat**: Orange oval body with triangular ears
- **Dog**: Brown rectangular body with circular head
- **Car**: Blue rectangular body with black wheels
- **Tree**: Brown trunk with green circular leaves
- **Bird**: Yellow oval body with wing patterns
- **Horse**: Brown rectangular body with head
- **Ship**: Gray hull with brown mast
- **Truck**: Red rectangular body with wheels
- **Bicycle**: Black frame with circular wheels
- **Person**: Skin-colored head with blue shirt

### Data Splits

- **Training**: 70% of data
- **Validation**: 15% of data  
- **Test**: 15% of data

## Evaluation Metrics

The implementation provides comprehensive evaluation metrics:

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Macro, micro, and weighted F1 scores
- **AUROC**: Area Under ROC Curve (macro and micro)
- **AUPRC**: Area Under Precision-Recall Curve
- **Top-K Accuracy**: Top-1 and Top-5 accuracy

### Calibration Metrics
- **Calibration Error**: Expected Calibration Error (ECE)
- **Confidence**: Prediction confidence scores

### Similarity Metrics
- **Mean Similarity**: Average similarity scores
- **Similarity Statistics**: Min, max, median, std

## Project Structure

```
zero-shot-learning/
├── src/                          # Source code
│   ├── data/                     # Data handling
│   │   └── synthetic_dataset.py  # Synthetic dataset generation
│   ├── models/                   # Model implementations
│   │   └── clip_model.py         # CLIP zero-shot model
│   ├── metrics/                  # Evaluation metrics
│   │   └── zero_shot_metrics.py # Zero-shot evaluation metrics
│   ├── train/                    # Training scripts
│   │   └── trainer.py           # Main trainer
│   ├── utils/                    # Utility functions
│   │   ├── device.py            # Device management
│   │   └── logging.py           # Logging utilities
│   └── eval/                     # Evaluation scripts
├── configs/                       # Configuration files
│   ├── config.yaml              # Main configuration
│   ├── model/                    # Model configurations
│   ├── data/                     # Data configurations
│   └── training/                 # Training configurations
├── demo/                         # Demo applications
│   └── streamlit_app.py         # Streamlit demo
├── tests/                        # Unit tests
├── assets/                       # Generated assets
├── checkpoints/                  # Model checkpoints
├── logs/                         # Training logs
└── outputs/                      # Output results
```

## Configuration

### Model Configuration

```yaml
model:
  model_name: "ViT-B/32"           # CLIP model variant
  device: "auto"                   # Device selection
  freeze_vision: false             # Freeze vision encoder
  freeze_text: false               # Freeze text encoder
  temperature: 0.01                # Temperature scaling
  normalize_features: true         # Feature normalization
  prompt_template: "A photo of a {class_name}"
  use_ensemble_prompts: false      # Use multiple prompts
```

### Data Configuration

```yaml
data:
  dataset_name: "synthetic_zero_shot"
  num_classes: 10
  num_samples_per_class: 100
  image_size: [224, 224]
  batch_size: 32
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

### Training Configuration

```yaml
training:
  max_epochs: 10
  learning_rate: 1e-4
  weight_decay: 1e-4
  optimizer: "adamw"
  scheduler: "cosine"
  gradient_clip_val: 1.0
  early_stopping:
    enabled: true
    patience: 5
```

## Expected Results

### Performance Benchmarks

On the synthetic dataset with 10 classes:

| Metric | Expected Range |
|--------|----------------|
| Accuracy | 0.85 - 0.95 |
| F1 Macro | 0.80 - 0.90 |
| AUROC Macro | 0.90 - 0.98 |
| AUPRC Macro | 0.85 - 0.95 |
| Calibration Error | 0.05 - 0.15 |

### Model Variants

| Model | Parameters | Accuracy | Speed |
|-------|------------|----------|-------|
| CLIP ViT-B/32 | 151M | ~0.90 | Fast |
| CLIP ViT-L/14 | 428M | ~0.92 | Medium |
| OpenCLIP ViT-B-32 | 151M | ~0.88 | Fast |

## Advanced Usage

### Custom Class Descriptions

Modify class descriptions in the configuration:

```yaml
data:
  class_descriptions:
    - "A photo of a cat"
    - "A photo of a dog"
    - "A picture of a car"
    # ... add more classes
```

### Ensemble Prompts

Enable ensemble prompting for improved performance:

```yaml
model:
  use_ensemble_prompts: true
  ensemble_prompts:
    - "A photo of a {class_name}"
    - "A picture of a {class_name}"
    - "An image of a {class_name}"
    - "A {class_name} in the image"
```

### Custom Datasets

To use your own dataset:

1. Create a custom dataset class inheriting from `torch.utils.data.Dataset`
2. Implement the `get_class_descriptions()` method
3. Update the configuration to use your dataset

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
pytest tests/ -m unit          # Unit tests
pytest tests/ -m integration  # Integration tests
pytest tests/ -m "not slow"   # Skip slow tests
```

## Development

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Safety and Limitations

### Safety Disclaimers

- **Research/Educational Use Only**: This implementation is for research and educational purposes
- **Not for Production**: Do not use for production decisions or control systems
- **Performance Variability**: Model performance may vary significantly across domains
- **Privacy Considerations**: Ensure proper consent and privacy protection with real data

### Known Limitations

- **Domain Gap**: Performance may degrade on real images vs. synthetic data
- **Class Imbalance**: May struggle with highly imbalanced datasets
- **Computational Requirements**: Large models require significant computational resources
- **Prompt Sensitivity**: Performance sensitive to prompt engineering

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**kryptologyst**

- GitHub: [https://github.com/kryptologyst](https://github.com/kryptologyst)

## Acknowledgments

- OpenAI for the CLIP model
- Hugging Face for the Transformers library
- Streamlit for the demo framework
- The open-source community for various tools and libraries

## References

1. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
2. Schick, T., & Schütze, H. "Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference." EACL 2021.
3. Zhou, K., et al. "Learning to Prompt for Vision-Language Models." IJCV 2022.

---

**⚠️ Disclaimer**: This is a research/educational demo. Not for production decisions or control. Model performance may vary significantly across different domains and distributions. Ensure proper consent and privacy protection when using with real data.
# Zero-Shot-Learning-Implementation
