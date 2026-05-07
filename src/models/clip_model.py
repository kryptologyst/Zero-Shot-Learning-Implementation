"""CLIP-based zero-shot learning model implementation."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class CLIPZeroShotModel(nn.Module):
    """CLIP-based zero-shot learning model.
    
    This model uses CLIP (Contrastive Language-Image Pre-training) for
    zero-shot classification by comparing image and text embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[torch.device] = None,
        freeze_vision: bool = False,
        freeze_text: bool = False,
        temperature: float = 0.01,
        normalize_features: bool = True,
        use_open_clip: bool = False,
        open_clip_model: str = "ViT-B-32",
        open_clip_pretrained: str = "laion2b_s34b_b79k",
        prompt_template: str = "A photo of a {class_name}",
        use_ensemble_prompts: bool = False,
        ensemble_prompts: Optional[List[str]] = None,
    ):
        """Initialize CLIP zero-shot model.
        
        Args:
            model_name: CLIP model name.
            device: Device to run the model on.
            freeze_vision: Whether to freeze vision encoder.
            freeze_text: Whether to freeze text encoder.
            temperature: Temperature for similarity computation.
            normalize_features: Whether to normalize features.
            use_open_clip: Whether to use OpenCLIP instead of OpenAI CLIP.
            open_clip_model: OpenCLIP model name.
            open_clip_pretrained: OpenCLIP pretrained weights.
            prompt_template: Template for text prompts.
            use_ensemble_prompts: Whether to use ensemble of prompts.
            ensemble_prompts: List of prompt templates for ensemble.
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device or torch.device("cpu")
        self.freeze_vision = freeze_vision
        self.freeze_text = freeze_text
        self.temperature = temperature
        self.normalize_features = normalize_features
        self.use_open_clip = use_open_clip
        self.prompt_template = prompt_template
        self.use_ensemble_prompts = use_ensemble_prompts
        self.ensemble_prompts = ensemble_prompts or [
            "A photo of a {class_name}",
            "A picture of a {class_name}",
            "An image of a {class_name}",
            "A {class_name} in the image",
        ]
        
        # Load CLIP model
        self._load_model()
        
        # Freeze parameters if specified
        self._freeze_parameters()
        
        # Move to device
        self.to(self.device)
        
        logging.info(f"Loaded CLIP model: {model_name}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Vision encoder frozen: {freeze_vision}")
        logging.info(f"Text encoder frozen: {freeze_text}")
    
    def _load_model(self) -> None:
        """Load CLIP model."""
        if self.use_open_clip:
            try:
                import open_clip
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    self.open_clip_model,
                    pretrained=self.open_clip_pretrained,
                    device=self.device,
                )
                self.tokenizer = open_clip.get_tokenizer(self.open_clip_model)
                logging.info(f"Loaded OpenCLIP model: {self.open_clip_model}")
            except ImportError:
                logging.warning("OpenCLIP not available, falling back to OpenAI CLIP")
                self.use_open_clip = False
        
        if not self.use_open_clip:
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.tokenizer = clip.tokenize
            logging.info(f"Loaded OpenAI CLIP model: {self.model_name}")
    
    def _freeze_parameters(self) -> None:
        """Freeze model parameters."""
        if self.freeze_vision:
            for param in self.model.visual.parameters():
                param.requires_grad = False
            logging.info("Vision encoder parameters frozen")
        
        if self.freeze_text:
            for param in self.model.transformer.parameters():
                param.requires_grad = False
            logging.info("Text encoder parameters frozen")
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to features.
        
        Args:
            images: Input images tensor.
            
        Returns:
            Image features.
        """
        with torch.set_grad_enabled(not self.freeze_vision):
            image_features = self.model.encode_image(images)
            
            if self.normalize_features:
                image_features = F.normalize(image_features, p=2, dim=-1)
            
            return image_features
    
    def encode_text(self, text: Union[str, List[str], torch.Tensor]) -> torch.Tensor:
        """Encode text to features.
        
        Args:
            text: Input text (string, list of strings, or tokenized tensor).
            
        Returns:
            Text features.
        """
        with torch.set_grad_enabled(not self.freeze_text):
            if isinstance(text, str):
                text = [text]
            
            if isinstance(text, list):
                # Tokenize text
                text_tokens = self.tokenizer(text).to(self.device)
            else:
                text_tokens = text
            
            text_features = self.model.encode_text(text_tokens)
            
            if self.normalize_features:
                text_features = F.normalize(text_features, p=2, dim=-1)
            
            return text_features
    
    def forward(
        self,
        images: torch.Tensor,
        class_descriptions: List[str],
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass for zero-shot classification.
        
        Args:
            images: Input images.
            class_descriptions: List of class descriptions.
            return_features: Whether to return image and text features.
            
        Returns:
            Classification logits or (logits, image_features, text_features).
        """
        # Encode images
        image_features = self.encode_image(images)
        
        # Prepare text prompts
        if self.use_ensemble_prompts:
            text_prompts = []
            for desc in class_descriptions:
                for prompt_template in self.ensemble_prompts:
                    prompt = prompt_template.format(class_name=desc)
                    text_prompts.append(prompt)
        else:
            text_prompts = [
                self.prompt_template.format(class_name=desc)
                for desc in class_descriptions
            ]
        
        # Encode text
        text_features = self.encode_text(text_prompts)
        
        # Compute similarity
        if self.use_ensemble_prompts:
            # Average over ensemble prompts
            num_classes = len(class_descriptions)
            num_prompts_per_class = len(self.ensemble_prompts)
            
            # Reshape text features for ensemble averaging
            text_features = text_features.view(num_classes, num_prompts_per_class, -1)
            text_features = text_features.mean(dim=1)  # Average over prompts
        
        # Compute similarity scores
        similarity = torch.matmul(image_features, text_features.T)
        
        # Apply temperature scaling
        logits = similarity / self.temperature
        
        if return_features:
            return logits, image_features, text_features
        else:
            return logits
    
    def predict(
        self,
        images: torch.Tensor,
        class_descriptions: List[str],
        return_probabilities: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict class labels for images.
        
        Args:
            images: Input images.
            class_descriptions: List of class descriptions.
            return_probabilities: Whether to return class probabilities.
            
        Returns:
            Predicted class indices or (indices, probabilities).
        """
        with torch.no_grad():
            logits = self.forward(images, class_descriptions)
            
            # Get predicted class indices
            predicted_indices = torch.argmax(logits, dim=-1)
            
            if return_probabilities:
                # Convert logits to probabilities
                probabilities = F.softmax(logits, dim=-1)
                return predicted_indices, probabilities
            else:
                return predicted_indices
    
    def get_similarity_scores(
        self,
        images: torch.Tensor,
        class_descriptions: List[str],
    ) -> torch.Tensor:
        """Get similarity scores between images and class descriptions.
        
        Args:
            images: Input images.
            class_descriptions: List of class descriptions.
            
        Returns:
            Similarity scores tensor.
        """
        with torch.no_grad():
            logits = self.forward(images, class_descriptions)
            return logits
    
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get image features.
        
        Args:
            images: Input images.
            
        Returns:
            Image features.
        """
        return self.encode_image(images)
    
    def get_text_features(self, class_descriptions: List[str]) -> torch.Tensor:
        """Get text features for class descriptions.
        
        Args:
            class_descriptions: List of class descriptions.
            
        Returns:
            Text features.
        """
        return self.encode_text(class_descriptions)
    
    def preprocess_image(self, image) -> torch.Tensor:
        """Preprocess image for inference.
        
        Args:
            image: Input image (PIL Image or numpy array).
            
        Returns:
            Preprocessed image tensor.
        """
        if hasattr(self, 'preprocess'):
            return self.preprocess(image).unsqueeze(0).to(self.device)
        else:
            # Fallback preprocessing
            import torchvision.transforms as transforms
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            return transform(image).unsqueeze(0).to(self.device)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary containing model information.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "freeze_vision": self.freeze_vision,
            "freeze_text": self.freeze_text,
            "temperature": self.temperature,
            "normalize_features": self.normalize_features,
            "use_open_clip": self.use_open_clip,
            "use_ensemble_prompts": self.use_ensemble_prompts,
        }


class CLIPZeroShotModelConfig:
    """Configuration class for CLIP zero-shot model."""
    
    @staticmethod
    def from_config(config: DictConfig) -> CLIPZeroShotModel:
        """Create model from configuration.
        
        Args:
            config: Model configuration.
            
        Returns:
            Configured CLIP model.
        """
        return CLIPZeroShotModel(
            model_name=config.get("model_name", "ViT-B/32"),
            device=config.get("device"),
            freeze_vision=config.get("freeze_vision", False),
            freeze_text=config.get("freeze_text", False),
            temperature=config.get("temperature", 0.01),
            normalize_features=config.get("normalize_features", True),
            use_open_clip=config.get("use_open_clip", False),
            open_clip_model=config.get("open_clip_model", "ViT-B-32"),
            open_clip_pretrained=config.get("open_clip_pretrained", "laion2b_s34b_b79k"),
            prompt_template=config.get("prompt_template", "A photo of a {class_name}"),
            use_ensemble_prompts=config.get("use_ensemble_prompts", False),
            ensemble_prompts=config.get("ensemble_prompts"),
        )
