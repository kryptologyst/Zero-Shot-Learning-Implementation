"""Simple baseline models for zero-shot learning comparison."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class BaselineZeroShotModel(nn.Module):
    """Baseline model for zero-shot learning comparison.
    
    This model uses simple feature extraction and classification
    without any vision-language pre-training.
    """
    
    def __init__(
        self,
        input_size: int = 224 * 224 * 3,
        hidden_size: int = 512,
        num_classes: int = 10,
        dropout: float = 0.5,
    ):
        """Initialize baseline model.
        
        Args:
            input_size: Size of input features.
            hidden_size: Size of hidden layer.
            num_classes: Number of classes.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Simple MLP
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
        
        logging.info(f"Initialized baseline model with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output logits.
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def _count_parameters(self) -> int:
        """Count model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ClassicalBaseline:
    """Classical machine learning baselines for zero-shot learning."""
    
    def __init__(self, model_type: str = "logistic_regression"):
        """Initialize classical baseline.
        
        Args:
            model_type: Type of classical model.
        """
        self.model_type = model_type
        self.model = self._create_model()
        self.is_fitted = False
        
    def _create_model(self):
        """Create the specified model."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == "knn":
            return KNeighborsClassifier(n_neighbors=5)
        elif self.model_type == "svm":
            return SVC(random_state=42, probability=True)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model.
        
        Args:
            X: Training features.
            y: Training labels.
        """
        # Flatten features if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        logging.info(f"Fitted {self.model_type} model")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Flatten features if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features.
            
        Returns:
            Class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Flatten features if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return one-hot encoding
            predictions = self.predict(X)
            num_classes = len(np.unique(predictions))
            proba = np.zeros((len(predictions), num_classes))
            proba[np.arange(len(predictions)), predictions] = 1.0
            return proba


class RandomBaseline:
    """Random baseline for zero-shot learning."""
    
    def __init__(self, num_classes: int = 10):
        """Initialize random baseline.
        
        Args:
            num_classes: Number of classes.
        """
        self.num_classes = num_classes
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model (no-op for random baseline).
        
        Args:
            X: Training features.
            y: Training labels.
        """
        self.is_fitted = True
        logging.info("Fitted random baseline model")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make random predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Random predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return np.random.randint(0, self.num_classes, size=len(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict random class probabilities.
        
        Args:
            X: Input features.
            
        Returns:
            Random class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Generate random probabilities
        proba = np.random.random((len(X), self.num_classes))
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba


class MajorityBaseline:
    """Majority class baseline for zero-shot learning."""
    
    def __init__(self):
        """Initialize majority baseline."""
        self.majority_class = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model.
        
        Args:
            X: Training features.
            y: Training labels.
        """
        # Find majority class
        unique_classes, counts = np.unique(y, return_counts=True)
        self.majority_class = unique_classes[np.argmax(counts)]
        self.is_fitted = True
        
        logging.info(f"Fitted majority baseline model (majority class: {self.majority_class})")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make majority class predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Majority class predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return np.full(len(X), self.majority_class)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict majority class probabilities.
        
        Args:
            X: Input features.
            
        Returns:
            Majority class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create one-hot encoding with majority class
        num_classes = self.majority_class + 1  # Assume classes are 0-indexed
        proba = np.zeros((len(X), num_classes))
        proba[:, self.majority_class] = 1.0
        return proba


def create_baseline_models() -> Dict[str, Any]:
    """Create a dictionary of baseline models.
    
    Returns:
        Dictionary of baseline models.
    """
    return {
        "neural_network": BaselineZeroShotModel(),
        "logistic_regression": ClassicalBaseline("logistic_regression"),
        "random_forest": ClassicalBaseline("random_forest"),
        "knn": ClassicalBaseline("knn"),
        "svm": ClassicalBaseline("svm"),
        "random": RandomBaseline(),
        "majority": MajorityBaseline(),
    }
