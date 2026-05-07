"""Streamlit demo app for zero-shot learning."""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
import torch
from PIL import Image
from omegaconf import OmegaConf

from src.data.synthetic_dataset import SyntheticDataset
from src.metrics.zero_shot_metrics import ZeroShotEvaluator
from src.models.clip_model import CLIPZeroShotModel, CLIPZeroShotModelConfig
from src.utils.device import get_device, set_seed


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Zero-Shot Learning Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(config_path: str = "configs/config.yaml") -> CLIPZeroShotModel:
    """Load the CLIP model.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Loaded CLIP model.
    """
    try:
        # Load configuration
        config = OmegaConf.load(config_path)
        
        # Set device
        device = get_device(config.device)
        
        # Initialize model
        model_config = config.model
        model_config.device = device
        
        model = CLIPZeroShotModelConfig.from_config(model_config)
        
        logger.info(f"Model loaded successfully on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def generate_synthetic_data(num_samples: int = 5) -> List[Dict[str, Any]]:
    """Generate synthetic data for demonstration.
    
    Args:
        num_samples: Number of samples to generate.
        
    Returns:
        List of synthetic data samples.
    """
    try:
        # Create synthetic dataset
        class_descriptions = [
            "A photo of a cat",
            "A photo of a dog", 
            "A picture of a car",
            "A picture of a tree",
            "An image of a bird",
            "A photo of a horse",
            "A picture of a ship",
            "A photo of a truck",
            "An image of a bicycle",
            "A picture of a person",
        ]
        
        dataset = SyntheticDataset(
            class_descriptions=class_descriptions,
            num_samples_per_class=1,
            image_size=(224, 224),
            split="test",
            seed=42,
        )
        
        # Generate samples
        samples = []
        for i in range(min(num_samples, len(dataset))):
            image, label = dataset[i]
            
            # Convert tensor to PIL Image
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            
            samples.append({
                "image": pil_image,
                "label": label,
                "class_name": class_descriptions[label],
            })
        
        return samples
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        return []


def predict_image(model: CLIPZeroShotModel, image: Image.Image, class_descriptions: List[str]) -> Dict[str, Any]:
    """Predict class for an image.
    
    Args:
        model: CLIP model.
        image: Input image.
        class_descriptions: List of class descriptions.
        
    Returns:
        Prediction results.
    """
    try:
        # Preprocess image
        image_tensor = model.preprocess_image(image)
        
        # Get predictions
        with torch.no_grad():
            logits = model(image_tensor, class_descriptions)
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
            
            # Get similarity scores
            similarity_scores = logits.squeeze(0)
        
        return {
            "prediction": prediction.item(),
            "predicted_class": class_descriptions[prediction.item()],
            "probabilities": probabilities.squeeze(0).cpu().numpy(),
            "similarity_scores": similarity_scores.cpu().numpy(),
            "confidence": probabilities.max().item(),
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Zero-Shot Learning Demo</h1>', unsafe_allow_html=True)
    
    # Safety disclaimer
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Safety Disclaimer</h4>
        <p><strong>This is a research/educational demo.</strong> Not for production decisions or control.</p>
        <p>Model performance may vary significantly across different domains and distributions.</p>
        <p>Ensure proper consent and privacy protection when using with real data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_option = st.selectbox(
            "Select Model",
            ["CLIP ViT-B/32", "CLIP ViT-L/14", "OpenCLIP ViT-B-32"],
            index=0,
        )
        
        # Class descriptions
        st.subheader("📝 Class Descriptions")
        st.write("Modify the class descriptions for zero-shot learning:")
        
        default_classes = [
            "A photo of a cat",
            "A photo of a dog",
            "A picture of a car",
            "A picture of a tree",
            "An image of a bird",
            "A photo of a horse",
            "A picture of a ship",
            "A photo of a truck",
            "An image of a bicycle",
            "A picture of a person",
        ]
        
        class_descriptions = []
        for i, default_class in enumerate(default_classes):
            class_desc = st.text_input(
                f"Class {i+1}",
                value=default_class,
                key=f"class_{i}",
            )
            class_descriptions.append(class_desc)
        
        # Advanced options
        st.subheader("🔧 Advanced Options")
        
        use_ensemble = st.checkbox("Use Ensemble Prompts", value=False)
        temperature = st.slider("Temperature", 0.001, 0.1, 0.01, 0.001)
        
        # Generate synthetic data button
        if st.button("🎲 Generate Synthetic Data"):
            st.session_state.synthetic_data = generate_synthetic_data(5)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Image Input")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"],
            help="Upload an image to classify using zero-shot learning",
        )
        
        # Use synthetic data if available
        if "synthetic_data" in st.session_state and st.session_state.synthetic_data:
            st.subheader("🎲 Synthetic Data")
            
            selected_sample = st.selectbox(
                "Select a synthetic sample",
                range(len(st.session_state.synthetic_data)),
                format_func=lambda x: f"Sample {x+1}: {st.session_state.synthetic_data[x]['class_name']}",
            )
            
            if st.button("Use Selected Sample"):
                st.session_state.selected_image = st.session_state.synthetic_data[selected_sample]["image"]
                st.session_state.true_label = st.session_state.synthetic_data[selected_sample]["label"]
    
    with col2:
        st.header("🔍 Predictions")
        
        # Load model
        with st.spinner("Loading model..."):
            model = load_model()
        
        if model is None:
            st.error("Failed to load model. Please check the configuration.")
            return
        
        # Make predictions
        if uploaded_file is not None:
            # Process uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            with st.spinner("Making prediction..."):
                results = predict_image(model, image, class_descriptions)
            
            if results:
                display_predictions(results, class_descriptions)
        
        elif "selected_image" in st.session_state:
            # Process selected synthetic image
            image = st.session_state.selected_image
            st.image(image, caption="Synthetic Image", use_column_width=True)
            
            # Make prediction
            with st.spinner("Making prediction..."):
                results = predict_image(model, image, class_descriptions)
            
            if results:
                display_predictions(results, class_descriptions)
                
                # Show true label if available
                if "true_label" in st.session_state:
                    true_label = st.session_state.true_label
                    true_class = class_descriptions[true_label]
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>True Label</h4>
                        <p><strong>Class:</strong> {true_class}</p>
                        <p><strong>Correct:</strong> {'✅ Yes' if results['prediction'] == true_label else '❌ No'}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:
            st.info("👆 Upload an image or generate synthetic data to get started!")
    
    # Model information
    st.header("📊 Model Information")
    
    model_info = model.get_model_info()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model", model_info["model_name"])
        st.metric("Device", model_info["device"])
    
    with col2:
        st.metric("Total Parameters", f"{model_info['total_parameters']:,}")
        st.metric("Trainable Parameters", f"{model_info['trainable_parameters']:,}")
    
    with col3:
        st.metric("Temperature", model_info["temperature"])
        st.metric("Normalize Features", "Yes" if model_info["normalize_features"] else "No")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><strong>Author:</strong> <a href="https://github.com/kryptologyst" target="_blank">kryptologyst</a></p>
        <p><strong>GitHub:</strong> <a href="https://github.com/kryptologyst" target="_blank">https://github.com/kryptologyst</a></p>
        <p><em>Zero-Shot Learning Implementation with CLIP</em></p>
    </div>
    """, unsafe_allow_html=True)


def display_predictions(results: Dict[str, Any], class_descriptions: List[str]) -> None:
    """Display prediction results.
    
    Args:
        results: Prediction results.
        class_descriptions: List of class descriptions.
    """
    # Main prediction
    st.markdown(f"""
    <div class="success-box">
        <h4>🎯 Prediction</h4>
        <p><strong>Predicted Class:</strong> {results['predicted_class']}</p>
        <p><strong>Confidence:</strong> {results['confidence']:.3f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability distribution
    st.subheader("📊 Probability Distribution")
    
    probabilities = results["probabilities"]
    
    # Create a bar chart of probabilities
    import pandas as pd
    
    prob_df = pd.DataFrame({
        "Class": [desc.replace("A photo of a ", "").replace("A picture of a ", "").replace("An image of a ", "") for desc in class_descriptions],
        "Probability": probabilities,
    })
    
    st.bar_chart(prob_df.set_index("Class"))
    
    # Detailed results
    st.subheader("🔍 Detailed Results")
    
    # Create a table of all predictions
    results_data = []
    for i, (desc, prob) in enumerate(zip(class_descriptions, probabilities)):
        results_data.append({
            "Rank": i + 1,
            "Class": desc,
            "Probability": f"{prob:.4f}",
            "Similarity": f"{results['similarity_scores'][i]:.4f}",
        })
    
    # Sort by probability
    results_data.sort(key=lambda x: float(x["Probability"]), reverse=True)
    
    st.table(pd.DataFrame(results_data))
    
    # Similarity scores
    st.subheader("📈 Similarity Scores")
    
    similarity_df = pd.DataFrame({
        "Class": [desc.replace("A photo of a ", "").replace("A picture of a ", "").replace("An image of a ", "") for desc in class_descriptions],
        "Similarity": results["similarity_scores"],
    })
    
    st.bar_chart(similarity_df.set_index("Class"))


if __name__ == "__main__":
    main()
