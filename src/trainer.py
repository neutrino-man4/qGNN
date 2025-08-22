"""
Author: Aritra Bal, ETP
Date: XIII Idibus Sextilibus anno ab urbe condita MMDCCLXXVIII

Minimalistic trainer for Jet GNN classification with essential training components.
Handles training loop, validation, checkpointing, and progress logging.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from loguru import logger
from sklearn.metrics import roc_auc_score
import json
import time
from tqdm import tqdm


class EarlyStopping:
    """
    Early stopping utility to halt training when validation metric stops improving.
    """
    
    def __init__(self, patience: int, monitor: str = "val_auc", mode: str = "max", min_delta: float = 1e-4):
        """
        Args:
            patience: Number of epochs to wait for improvement
            monitor: Metric to monitor ('val_loss' or 'val_auc')
            mode: 'min' for loss, 'max' for accuracy/auc
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.epochs_without_improvement = 0
        
        logger.info(f"Early stopping: monitoring {monitor} with patience {patience}")
    
    def __call__(self, current_score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current epoch's monitored metric value
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
        
        # Check for improvement
        if self.mode == "max":
            improved = current_score > self.best_score + self.min_delta
        else:  # mode == "min"
            improved = current_score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = current_score
            self.epochs_without_improvement = 0
            logger.debug(f"Metric improved to {current_score:.4f}")
        else:
            self.epochs_without_improvement += 1
            logger.debug(f"No improvement for {self.epochs_without_improvement}/{self.patience} epochs")
        
        should_stop = self.epochs_without_improvement >= self.patience
        if should_stop:
            logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
        
        return should_stop


class JetGNNTrainer:
    """
    Trainer class for Jet GNN models with comprehensive training utilities.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Any,
        save_dir: Path,
        device: torch.device
    ):
        """
        Initialize trainer with model, data, and training configuration.
        
        Args:
            model: The JetGNN model to train
            train_loader: Training data loader
            val_loader: Validation data loader  
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler (optional)
            config: Configuration object with training parameters
            save_dir: Directory to save checkpoints and logs
            device: Device to run training on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.save_dir = Path(save_dir)
        self.device = device
        
        # Create subdirectories
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup early stopping
        if config.training.early_stopping.enabled:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping.patience,
                monitor=config.training.early_stopping.monitor,
                mode=config.training.early_stopping.mode,
                min_delta=config.training.early_stopping.min_delta
            )
        else:
            self.early_stopping = None
        
        # Training state tracking
        self.current_epoch = 0
        self.best_metric = float('-inf') if config.checkpointing.best_mode == "max" else float('inf')
        self.training_history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [], 'val_auc': [],
            'learning_rates': []
        }
        
        logger.info(f"Trainer initialized - Device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    def train(self) -> Dict[str, float]:
        """
        Main training loop.
        
        Returns:
            Final training metrics
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.training.num_epochs):
                self.current_epoch = epoch
                
                # Training epoch
                train_metrics = self.train_epoch()
                
                # Validation epoch
                val_metrics = self.validate_epoch()
                
                # Learning rate scheduling
                #import pdb;pdb.set_trace()
                self._update_history(train_metrics, val_metrics)
                
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics[self.config.training.early_stopping.monitor])
                    else:
                        self.scheduler.step()
                
                # Update training history
                
                # Log epoch results
                self._log_epoch_results(epoch, train_metrics, val_metrics)
                
                # Save checkpoint
                is_best = self._is_best_model(val_metrics)
                if epoch % self.config.checkpointing.save_frequency == 0 or is_best:
                    self._save_checkpoint(epoch, val_metrics, is_best)
                
                # Early stopping check
                if self.early_stopping is not None:
                    monitor_metric = val_metrics[self.config.training.early_stopping.monitor]
                    if self.early_stopping(monitor_metric):
                        logger.info(f"Training stopped early at epoch {epoch}")
                        break
            
            # Training completed
            training_time = time.time() - start_time
            logger.success(f"Training completed in {training_time/3600:.2f} hours")
            
            # Save final checkpoint
            self._save_checkpoint(self.current_epoch, val_metrics, is_final=True)
            
            return {
                'final_train_loss': train_metrics['loss'],
                'final_val_loss': val_metrics['loss'],
                'final_val_accuracy': val_metrics['accuracy'],
                'final_val_auc': val_metrics['auc'],
                'training_time_hours': training_time / 3600,
                'total_epochs': self.current_epoch + 1
            }
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self._save_checkpoint(self.current_epoch, val_metrics, is_interrupted=True)
            raise
        except Exception as e:
            logger.exception("Training failed")  # includes full traceback
            raise
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Execute one training epoch.
        
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches=len(self.train_loader)
        # Progress tracking
        log_frequency = self.config.logging.log_frequency
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", total=num_batches)):
            # Move batch to device
            batch = batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(batch)
            loss = F.cross_entropy(logits, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip_val
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == batch.y).sum().item()
            total_samples += batch.y.size(0)
            
            # Log progress
            if batch_idx % log_frequency == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.debug(
                    f"Epoch {self.current_epoch} [{batch_idx}/{num_batches}] "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                )
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Execute one validation epoch.
        
        Returns:
            Validation metrics for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_probabilities = []
        all_labels = []
        num_batches = len(self.val_loader)
        with torch.no_grad():
            # wrap in tqdm with total arg
            for batch in tqdm(self.val_loader, desc="Validation", total=num_batches):
                # Move batch to device
                batch = batch.to(self.device)
                
                # Forward pass
                logits = self.model(batch)
                loss = F.cross_entropy(logits, batch.y)
                
                # Update metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                probabilities = F.softmax(logits, dim=1)[:, 1]  # Probability of TTbar class
                
                total_correct += (predictions == batch.y).sum().item()
                total_samples += batch.y.size(0)
                
                # Store for AUC calculation
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_samples
        # Calculate AUC
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except ValueError:
            # Handle case where only one class is present
            logger.warning("Could not calculate AUC - only one class present in validation set")
            auc = 0.5
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc
        }
    
    def _update_history(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Update training history with current epoch metrics."""
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['train_accuracy'].append(train_metrics['accuracy'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['val_accuracy'].append(val_metrics['accuracy'])
        self.training_history['val_auc'].append(val_metrics['auc'])
        self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log comprehensive epoch results."""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.3f} | "
            f"Val AUC: {val_metrics['auc']:.3f} | "
            f"LR: {current_lr:.2e}"
        )
    
    def _is_best_model(self, val_metrics: Dict[str, float]) -> bool:
        """Check if current model is the best so far."""
        current_metric = val_metrics[self.config.checkpointing.best_metric]
        
        if self.config.checkpointing.best_mode == "max":
            is_best = current_metric > self.best_metric
        else:
            is_best = current_metric < self.best_metric
        
        if is_best:
            self.best_metric = current_metric
            logger.info(f"New best model! {self.config.checkpointing.best_metric}: {current_metric:.4f}")
        
        return is_best
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, 
                        is_final: bool = False, is_interrupted: bool = False) -> None:
        """Save model checkpoint with training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.config.checkpointing.save_optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler and self.config.checkpointing.save_scheduler else None,
            'metrics': metrics,
            'best_metric': self.best_metric,
            'training_history': self.training_history,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }
        
        # Regular checkpoint
        if not self.config.checkpointing.save_best_only or is_best or is_final:
            checkpoint_name = f"checkpoint_epoch_{epoch:03d}.pth"
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            torch.save(checkpoint, checkpoint_path)
            logger.debug(f"Saved checkpoint: {checkpoint_name}")
        
        # Best model checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model checkpoint")
        
        # Final/interrupted checkpoint
        if is_final:
            final_path = self.checkpoint_dir / "final_model.pth"
            torch.save(checkpoint, final_path)
            logger.info("Saved final model checkpoint")
        
        if is_interrupted:
            interrupted_path = self.checkpoint_dir / "interrupted_model.pth"
            torch.save(checkpoint, interrupted_path)
            logger.warning("Saved interrupted training checkpoint")
        
        # Save training history as JSON
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number to resume from
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if checkpoint.get('optimizer_state_dict') and self.config.resume.resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.debug("Loaded optimizer state")
        
        # Load scheduler state
        if checkpoint.get('scheduler_state_dict') and self.scheduler and self.config.resume.resume_scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.debug("Loaded scheduler state")
        
        # Load training state
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        resume_epoch = checkpoint['epoch'] + 1
        logger.success(f" Checkpoint loaded successfully. Resuming from epoch {resume_epoch}")
        
        return resume_epoch


def create_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Configuration object
        
    Returns:
        Configured optimizer
    """
    optimizer_name = config.training.optimizer.lower()
    lr = config.training.learning_rate
    weight_decay = config.training.weight_decay
    
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    logger.info(f"Created optimizer: {optimizer_name} (lr={lr}, weight_decay={weight_decay})")
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Any) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration object
        
    Returns:
        Configured scheduler or None
    """
    if not config.training.use_scheduler:
        return None
    
    scheduler_type = config.training.scheduler_type.lower()
    
    if scheduler_type == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=config.training.early_stopping.mode,
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            verbose=True,threshold=1e-2
        )
    elif scheduler_type == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config.training.scheduler_patience,
            gamma=config.training.scheduler_factor
        )
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    logger.info(f"Created scheduler: {scheduler_type}")
    return scheduler