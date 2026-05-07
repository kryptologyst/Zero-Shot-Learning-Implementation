"""Logging utilities for the zero-shot learning project."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    experiment_name: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_dir: Directory to save log files.
        log_level: Logging level.
        experiment_name: Name of the experiment for log file naming.
        
    Returns:
        logging.Logger: Configured logger.
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("zero_shot_learning")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        log_filename = f"{experiment_name}_{timestamp}.log"
    else:
        log_filename = f"experiment_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_path / log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_config(logger: logging.Logger, config: DictConfig) -> None:
    """Log configuration parameters.
    
    Args:
        logger: Logger instance.
        config: Configuration object.
    """
    logger.info("Configuration:")
    logger.info(f"  Project: {config.project_name}")
    logger.info(f"  Version: {config.version}")
    logger.info(f"  Author: {config.author}")
    logger.info(f"  Seed: {config.seed}")
    logger.info(f"  Device: {config.device}")
    
    if hasattr(config, "model"):
        logger.info(f"  Model: {config.model.model_name}")
    
    if hasattr(config, "data"):
        logger.info(f"  Dataset: {config.data.dataset_name}")
        logger.info(f"  Classes: {config.data.num_classes}")


def log_device_info(logger: logging.Logger) -> None:
    """Log device information.
    
    Args:
        logger: Logger instance.
    """
    logger.info("Device Information:")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        logger.info(f"  Current GPU: {torch.cuda.current_device()}")
        logger.info(f"  GPU name: {torch.cuda.get_device_name()}")
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("  MPS (Apple Silicon) available: True")


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    stage: str = "train",
    epoch: Optional[int] = None,
) -> None:
    """Log metrics in a formatted way.
    
    Args:
        logger: Logger instance.
        metrics: Dictionary of metrics to log.
        stage: Training stage (train/val/test).
        epoch: Current epoch number.
    """
    if epoch is not None:
        logger.info(f"{stage.capitalize()} Epoch {epoch} Metrics:")
    else:
        logger.info(f"{stage.capitalize()} Metrics:")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


def log_model_info(logger: logging.Logger, model: torch.nn.Module) -> None:
    """Log model information.
    
    Args:
        logger: Logger instance.
        model: PyTorch model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Log model architecture
    logger.info("Model Architecture:")
    logger.info(f"  {model}")


class MetricsLogger:
    """Simple metrics logger for tracking training progress."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics_history: Dict[str, list] = {}
    
    def log(self, metrics: Dict[str, float], stage: str = "train") -> None:
        """Log metrics and store in history.
        
        Args:
            metrics: Dictionary of metrics.
            stage: Training stage.
        """
        for key, value in metrics.items():
            full_key = f"{stage}_{key}"
            if full_key not in self.metrics_history:
                self.metrics_history[full_key] = []
            self.metrics_history[full_key].append(value)
        
        log_metrics(self.logger, metrics, stage)
    
    def get_best_metric(self, metric_name: str, stage: str = "val", mode: str = "max") -> float:
        """Get the best value for a metric.
        
        Args:
            metric_name: Name of the metric.
            stage: Training stage.
            mode: 'max' or 'min'.
            
        Returns:
            Best metric value.
        """
        full_key = f"{stage}_{metric_name}"
        if full_key not in self.metrics_history:
            return 0.0
        
        values = self.metrics_history[full_key]
        return max(values) if mode == "max" else min(values)
    
    def get_metric_history(self, metric_name: str, stage: str = "val") -> list:
        """Get the history of a metric.
        
        Args:
            metric_name: Name of the metric.
            stage: Training stage.
            
        Returns:
            List of metric values.
        """
        full_key = f"{stage}_{metric_name}"
        return self.metrics_history.get(full_key, [])
