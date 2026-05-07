"""Visualization utilities for zero-shot learning results."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Names of classes.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    
    # Rotate labels
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = ["accuracy", "f1_macro", "auroc_macro"],
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot metrics comparison across models.
    
    Args:
        metrics_dict: Dictionary of metrics for each model.
        metrics_to_plot: List of metrics to plot.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Prepare data
    models = list(metrics_dict.keys())
    metrics_data = {metric: [metrics_dict[model].get(metric, 0) for model in models] for metric in metrics_to_plot}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot metrics
    x = np.arange(len(models))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, metrics_data[metric], width, label=metric.replace("_", " ").title())
    
    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Metrics comparison saved to {save_path}")
    
    return fig


def plot_probability_distribution(
    probabilities: np.ndarray,
    class_names: List[str],
    title: str = "Class Probability Distribution",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot probability distribution across classes.
    
    Args:
        probabilities: Class probabilities.
        class_names: Names of classes.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot probabilities
    x = np.arange(len(class_names))
    bars = ax.bar(x, probabilities, color="skyblue", alpha=0.7)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    ax.set_xlabel("Classes", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("A photo of a ", "").replace("A picture of a ", "").replace("An image of a ", "") for name in class_names], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Probability distribution saved to {save_path}")
    
    return fig


def plot_similarity_scores(
    similarity_scores: np.ndarray,
    class_names: List[str],
    title: str = "Similarity Scores",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot similarity scores across classes.
    
    Args:
        similarity_scores: Similarity scores for each class.
        class_names: Names of classes.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot similarity scores
    x = np.arange(len(class_names))
    bars = ax.bar(x, similarity_scores, color="lightcoral", alpha=0.7)
    
    # Add value labels on bars
    for bar, score in zip(bars, similarity_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    ax.set_xlabel("Classes", fontsize=12)
    ax.set_ylabel("Similarity Score", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("A photo of a ", "").replace("A picture of a ", "").replace("An image of a ", "") for name in class_names], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Similarity scores saved to {save_path}")
    
    return fig


def plot_training_curves(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot training curves.
    
    Args:
        train_metrics: Training metrics history.
        val_metrics: Validation metrics history.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    if "loss" in train_metrics and "loss" in val_metrics:
        axes[0].plot(train_metrics["loss"], label="Train Loss", color="blue")
        axes[0].plot(val_metrics["loss"], label="Val Loss", color="red")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curves")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    if "accuracy" in train_metrics and "accuracy" in val_metrics:
        axes[1].plot(train_metrics["accuracy"], label="Train Accuracy", color="blue")
        axes[1].plot(val_metrics["accuracy"], label="Val Accuracy", color="red")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Curves")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Training curves saved to {save_path}")
    
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot calibration curve.
    
    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    from sklearn.calibration import calibration_curve
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Calibration curve saved to {save_path}")
    
    return fig


def create_results_summary(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
) -> str:
    """Create a text summary of results.
    
    Args:
        results: Results dictionary.
        save_path: Path to save the summary.
        
    Returns:
        Text summary.
    """
    summary = "Zero-Shot Learning Results Summary\n"
    summary += "=" * 40 + "\n\n"
    
    # Basic metrics
    if "accuracy" in results:
        summary += f"Accuracy: {results['accuracy']:.4f}\n"
    if "f1_macro" in results:
        summary += f"F1 Macro: {results['f1_macro']:.4f}\n"
    if "auroc_macro" in results:
        summary += f"AUROC Macro: {results['auroc_macro']:.4f}\n"
    if "auprc_macro" in results:
        summary += f"AUPRC Macro: {results['auprc_macro']:.4f}\n"
    if "calibration_error" in results:
        summary += f"Calibration Error: {results['calibration_error']:.4f}\n"
    
    summary += "\n"
    
    # Per-class metrics
    for key, value in results.items():
        if key.startswith("f1_class_"):
            class_idx = key.split("_")[-1]
            summary += f"F1 Class {class_idx}: {value:.4f}\n"
    
    summary += "\n"
    
    # Similarity statistics
    if "similarity_stats" in results:
        stats = results["similarity_stats"]
        summary += "Similarity Statistics:\n"
        for stat_name, stat_value in stats.items():
            summary += f"  {stat_name}: {stat_value:.4f}\n"
    
    # Save if path provided
    if save_path:
        with open(save_path, "w") as f:
            f.write(summary)
        logging.info(f"Results summary saved to {save_path}")
    
    return summary


def save_all_visualizations(
    results: Dict[str, Any],
    class_names: List[str],
    output_dir: str = "assets/results",
) -> None:
    """Save all visualizations to output directory.
    
    Args:
        results: Results dictionary.
        class_names: Names of classes.
        output_dir: Output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save confusion matrix
    if "confusion_matrix" in results:
        y_true = results.get("y_true", [])
        y_pred = results.get("y_pred", [])
        if y_true and y_pred:
            plot_confusion_matrix(
                np.array(y_true),
                np.array(y_pred),
                class_names,
                save_path=str(output_path / "confusion_matrix.png")
            )
    
    # Save probability distribution (example)
    if "probabilities" in results:
        probs = results["probabilities"]
        if len(probs) > 0:
            # Use first sample as example
            plot_probability_distribution(
                probs[0],
                class_names,
                save_path=str(output_path / "probability_distribution.png")
            )
    
    # Save similarity scores (example)
    if "similarity_scores" in results:
        scores = results["similarity_scores"]
        if len(scores) > 0:
            # Use first sample as example
            plot_similarity_scores(
                scores[0],
                class_names,
                save_path=str(output_path / "similarity_scores.png")
            )
    
    # Save results summary
    create_results_summary(results, save_path=str(output_path / "results_summary.txt"))
    
    logging.info(f"All visualizations saved to {output_dir}")
