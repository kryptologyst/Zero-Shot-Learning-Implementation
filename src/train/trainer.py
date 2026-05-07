"""Main training script for zero-shot learning."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.synthetic_dataset import SyntheticDataset
from src.metrics.zero_shot_metrics import ZeroShotEvaluator
from src.models.clip_model import CLIPZeroShotModel, CLIPZeroShotModelConfig
from src.utils.device import get_device, set_seed
from src.utils.logging import MetricsLogger, log_config, log_device_info, setup_logging


class ZeroShotTrainer:
    """Trainer for zero-shot learning models."""
    
    def __init__(self, config: DictConfig):
        """Initialize trainer.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        
        # Set up logging
        self.logger = setup_logging(
            log_dir=config.logging.log_dir,
            log_level=config.logging.level,
            experiment_name=config.experiment.name,
        )
        
        # Set random seed
        set_seed(config.seed)
        
        # Set device
        self.device = get_device(config.device)
        
        # Log configuration
        log_config(self.logger, config)
        log_device_info(self.logger)
        
        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(self.logger)
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize datasets
        self.train_dataset, self.val_dataset, self.test_dataset = self._initialize_datasets()
        
        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = self._initialize_dataloaders()
        
        # Initialize evaluator
        self.evaluator = ZeroShotEvaluator(self.train_dataset.get_class_descriptions())
        
        # Initialize optimizer and scheduler
        self.optimizer = self._initialize_optimizer()
        self.scheduler = self._initialize_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_model_path = None
        
        # Create output directories
        self._create_output_dirs()
    
    def _initialize_model(self) -> CLIPZeroShotModel:
        """Initialize the model.
        
        Returns:
            Initialized model.
        """
        model_config = self.config.model
        model_config.device = self.device
        
        model = CLIPZeroShotModelConfig.from_config(model_config)
        
        # Log model info
        model_info = model.get_model_info()
        self.logger.info("Model Information:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")
        
        return model
    
    def _initialize_datasets(self) -> tuple:
        """Initialize datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        data_config = self.config.data
        
        train_dataset = SyntheticDataset(
            class_descriptions=data_config.class_descriptions,
            num_samples_per_class=data_config.num_samples_per_class,
            image_size=tuple(data_config.image_size),
            num_channels=data_config.num_channels,
            split="train",
            train_split=data_config.train_split,
            val_split=data_config.val_split,
            test_split=data_config.test_split,
            augmentation=data_config.get("augmentation"),
            seed=self.config.seed,
        )
        
        val_dataset = SyntheticDataset(
            class_descriptions=data_config.class_descriptions,
            num_samples_per_class=data_config.num_samples_per_class,
            image_size=tuple(data_config.image_size),
            num_channels=data_config.num_channels,
            split="val",
            train_split=data_config.train_split,
            val_split=data_config.val_split,
            test_split=data_config.test_split,
            augmentation=None,  # No augmentation for validation
            seed=self.config.seed,
        )
        
        test_dataset = SyntheticDataset(
            class_descriptions=data_config.class_descriptions,
            num_samples_per_class=data_config.num_samples_per_class,
            image_size=tuple(data_config.image_size),
            num_channels=data_config.num_channels,
            split="test",
            train_split=data_config.train_split,
            val_split=data_config.val_split,
            test_split=data_config.test_split,
            augmentation=None,  # No augmentation for test
            seed=self.config.seed,
        )
        
        self.logger.info(f"Dataset sizes - Train: {len(train_dataset)}, "
                        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _initialize_dataloaders(self) -> tuple:
        """Initialize data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        data_config = self.config.data
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=data_config.batch_size,
            shuffle=data_config.shuffle,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
            pin_memory=data_config.pin_memory,
        )
        
        return train_loader, val_loader, test_loader
    
    def _initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize optimizer.
        
        Returns:
            Initialized optimizer.
        """
        training_config = self.config.training
        
        if training_config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )
        elif training_config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {training_config.optimizer}")
        
        return optimizer
    
    def _initialize_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Initialize learning rate scheduler.
        
        Returns:
            Initialized scheduler or None.
        """
        training_config = self.config.training
        
        if training_config.scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.max_epochs,
            )
        elif training_config.scheduler.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config.max_epochs // 3,
                gamma=0.1,
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _create_output_dirs(self) -> None:
        """Create output directories."""
        paths = self.config.paths
        
        Path(paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(paths.output_dir).mkdir(parents=True, exist_ok=True)
        Path(paths.assets_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        class_descriptions = self.train_dataset.get_class_descriptions()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            logits = self.model(images, class_descriptions)
            loss = nn.CrossEntropyLoss()(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_val,
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{total_correct / total_samples:.4f}",
            })
            
            # Log metrics periodically
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.metrics_logger.log({
                    "loss": loss.item(),
                    "accuracy": total_correct / total_samples,
                }, stage="train")
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        class_descriptions = self.val_dataset.get_class_descriptions()
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits = self.model(images, class_descriptions)
                loss = nn.CrossEntropyLoss()(logits, targets)
                
                # Update metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                total_correct += (predictions == targets).sum().item()
                total_samples += targets.size(0)
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
            "config": OmegaConf.to_yaml(self.config),
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.paths.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.paths.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            self.logger.info(f"Saved best model to {best_path}")
    
    def train(self) -> None:
        """Train the model."""
        self.logger.info("Starting training...")
        
        training_config = self.config.training
        
        for epoch in range(training_config.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            self.metrics_logger.log(train_metrics, stage="train", epoch=epoch)
            self.metrics_logger.log(val_metrics, stage="val", epoch=epoch)
            
            # Check if this is the best model
            is_best = val_metrics["accuracy"] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics["accuracy"]
            
            # Save checkpoint
            if epoch % training_config.save_checkpoint_every_n_epochs == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if training_config.early_stopping.enabled:
                if epoch - self.metrics_logger.get_metric_history("accuracy", "val").index(
                    self.best_val_accuracy
                ) >= training_config.early_stopping.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on test set.
        
        Returns:
            Dictionary of evaluation results.
        """
        self.logger.info("Evaluating on test set...")
        
        # Load best model if available
        if self.best_model_path and Path(self.best_model_path).exists():
            self.logger.info(f"Loading best model from {self.best_model_path}")
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Evaluate
        results = self.evaluator.evaluate(self.model, self.test_loader, self.device)
        
        # Log results
        self.logger.info("Test Results:")
        for key, value in results.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function.
    
    Args:
        config: Configuration object.
    """
    # Create trainer
    trainer = ZeroShotTrainer(config)
    
    # Train model
    trainer.train()
    
    # Evaluate model
    results = trainer.evaluate()
    
    # Save results
    results_path = Path(config.paths.output_dir) / "evaluation_results.yaml"
    OmegaConf.save(OmegaConf.create(results), results_path)
    
    logging.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
