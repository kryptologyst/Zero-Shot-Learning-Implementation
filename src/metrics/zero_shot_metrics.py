"""Evaluation metrics for zero-shot learning."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve


class ZeroShotMetrics:
    """Metrics calculator for zero-shot learning evaluation."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes.
            class_names: Names of classes for reporting.
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.similarity_scores = []
    
    def update(
        self,
        predictions: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
        probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None,
        similarity_scores: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> None:
        """Update metrics with new batch.
        
        Args:
            predictions: Predicted class indices.
            targets: True class indices.
            probabilities: Class probabilities.
            similarity_scores: Raw similarity scores.
        """
        # Convert to numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        if similarity_scores is not None and isinstance(similarity_scores, torch.Tensor):
            similarity_scores = similarity_scores.cpu().numpy()
        
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities)
        
        if similarity_scores is not None:
            self.similarity_scores.extend(similarity_scores)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary of computed metrics.
        """
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(targets, predictions)
        metrics["f1_macro"] = f1_score(targets, predictions, average="macro")
        metrics["f1_micro"] = f1_score(targets, predictions, average="micro")
        metrics["f1_weighted"] = f1_score(targets, predictions, average="weighted")
        
        # Per-class F1 scores
        f1_per_class = f1_score(targets, predictions, average=None)
        for i, f1 in enumerate(f1_per_class):
            metrics[f"f1_class_{i}"] = f1
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics["confusion_matrix"] = cm.tolist()
        
        # ROC AUC (one-vs-rest)
        if len(self.probabilities) > 0:
            probabilities = np.array(self.probabilities)
            
            try:
                # Multi-class ROC AUC
                if probabilities.ndim == 2 and probabilities.shape[1] > 2:
                    metrics["auroc_macro"] = roc_auc_score(
                        targets, probabilities, multi_class="ovr", average="macro"
                    )
                    metrics["auroc_micro"] = roc_auc_score(
                        targets, probabilities, multi_class="ovr", average="micro"
                    )
                
                # Per-class ROC AUC
                for i in range(self.num_classes):
                    try:
                        binary_targets = (targets == i).astype(int)
                        if len(np.unique(binary_targets)) > 1:  # Check if class exists
                            auroc = roc_auc_score(binary_targets, probabilities[:, i])
                            metrics[f"auroc_class_{i}"] = auroc
                    except ValueError:
                        # Skip if class doesn't exist in targets
                        pass
                
                # Precision-Recall AUC
                try:
                    metrics["auprc_macro"] = self._compute_auprc_macro(targets, probabilities)
                except Exception as e:
                    logging.warning(f"Could not compute AUPRC: {e}")
                
                # Calibration metrics
                try:
                    calibration_error = self._compute_calibration_error(targets, probabilities)
                    metrics["calibration_error"] = calibration_error
                except Exception as e:
                    logging.warning(f"Could not compute calibration error: {e}")
        
        return metrics
    
    def _compute_auprc_macro(self, targets: np.ndarray, probabilities: np.ndarray) -> float:
        """Compute macro-averaged AUPRC.
        
        Args:
            targets: True class indices.
            probabilities: Class probabilities.
            
        Returns:
            Macro-averaged AUPRC.
        """
        auprc_scores = []
        
        for i in range(self.num_classes):
            binary_targets = (targets == i).astype(int)
            if len(np.unique(binary_targets)) > 1:  # Check if class exists
                precision, recall, _ = precision_recall_curve(binary_targets, probabilities[:, i])
                # Compute AUPRC using trapezoidal rule
                auprc = np.trapz(precision, recall)
                auprc_scores.append(auprc)
        
        return np.mean(auprc_scores) if auprc_scores else 0.0
    
    def _compute_calibration_error(self, targets: np.ndarray, probabilities: np.ndarray) -> float:
        """Compute calibration error (ECE).
        
        Args:
            targets: True class indices.
            probabilities: Class probabilities.
            
        Returns:
            Expected Calibration Error.
        """
        # Get predicted probabilities for true classes
        true_class_probs = probabilities[np.arange(len(targets)), targets]
        
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            targets, true_class_probs, n_bins=10
        )
        
        # Compute ECE
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (true_class_probs > bin_lower) & (true_class_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (targets[in_bin] == np.argmax(probabilities[in_bin], axis=1)).mean()
                avg_confidence_in_bin = true_class_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def get_classification_report(self) -> str:
        """Get detailed classification report.
        
        Returns:
            Classification report string.
        """
        if not self.predictions:
            return "No predictions available."
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        return classification_report(
            targets, predictions, target_names=self.class_names, digits=4
        )
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix.
        
        Returns:
            Confusion matrix.
        """
        if not self.predictions:
            return np.array([])
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        return confusion_matrix(targets, predictions)
    
    def get_top_k_accuracy(self, k: int = 5) -> float:
        """Compute top-k accuracy.
        
        Args:
            k: Number of top predictions to consider.
            
        Returns:
            Top-k accuracy.
        """
        if not self.probabilities:
            return 0.0
        
        probabilities = np.array(self.probabilities)
        targets = np.array(self.targets)
        
        # Get top-k predictions
        top_k_predictions = np.argsort(probabilities, axis=1)[:, -k:]
        
        # Check if true class is in top-k
        correct = 0
        for i, target in enumerate(targets):
            if target in top_k_predictions[i]:
                correct += 1
        
        return correct / len(targets)
    
    def get_similarity_statistics(self) -> Dict[str, float]:
        """Get statistics about similarity scores.
        
        Returns:
            Dictionary of similarity statistics.
        """
        if not self.similarity_scores:
            return {}
        
        similarity_scores = np.array(self.similarity_scores)
        
        return {
            "mean_similarity": np.mean(similarity_scores),
            "std_similarity": np.std(similarity_scores),
            "min_similarity": np.min(similarity_scores),
            "max_similarity": np.max(similarity_scores),
            "median_similarity": np.median(similarity_scores),
        }


class ZeroShotEvaluator:
    """Comprehensive evaluator for zero-shot learning models."""
    
    def __init__(self, class_descriptions: List[str]):
        """Initialize evaluator.
        
        Args:
            class_descriptions: List of class descriptions.
        """
        self.class_descriptions = class_descriptions
        self.num_classes = len(class_descriptions)
        self.metrics = ZeroShotMetrics(self.num_classes, class_descriptions)
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Evaluate model on dataset.
        
        Args:
            model: Model to evaluate.
            dataloader: Data loader for evaluation.
            device: Device to run evaluation on.
            
        Returns:
            Dictionary of evaluation results.
        """
        model.eval()
        self.metrics.reset()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_similarity_scores = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(device)
                targets = targets.to(device)
                
                # Get model predictions
                logits = model(images, self.class_descriptions)
                probabilities = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_similarity_scores.extend(logits.cpu().numpy())
        
        # Update metrics
        self.metrics.update(
            predictions=np.array(all_predictions),
            targets=np.array(all_targets),
            probabilities=np.array(all_probabilities),
            similarity_scores=np.array(all_similarity_scores),
        )
        
        # Compute metrics
        results = self.metrics.compute()
        
        # Add additional metrics
        results["top_1_accuracy"] = results.get("accuracy", 0.0)
        results["top_5_accuracy"] = self.metrics.get_top_k_accuracy(k=5)
        results["similarity_stats"] = self.metrics.get_similarity_statistics()
        
        # Add classification report
        results["classification_report"] = self.metrics.get_classification_report()
        
        return results
    
    def evaluate_single_image(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        true_label: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate model on a single image.
        
        Args:
            model: Model to evaluate.
            image: Input image tensor.
            true_label: True label (optional).
            
        Returns:
            Dictionary of evaluation results.
        """
        model.eval()
        
        with torch.no_grad():
            # Get predictions
            logits = model(image.unsqueeze(0), self.class_descriptions)
            probabilities = F.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
            
            # Get similarity scores
            similarity_scores = logits.squeeze(0)
            
            results = {
                "prediction": prediction.item(),
                "predicted_class": self.class_descriptions[prediction.item()],
                "probabilities": probabilities.squeeze(0).cpu().numpy().tolist(),
                "similarity_scores": similarity_scores.cpu().numpy().tolist(),
                "confidence": probabilities.max().item(),
            }
            
            if true_label is not None:
                results["true_label"] = true_label
                results["true_class"] = self.class_descriptions[true_label]
                results["correct"] = prediction.item() == true_label
                results["accuracy"] = float(prediction.item() == true_label)
            
            return results
