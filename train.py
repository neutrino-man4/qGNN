#!/usr/bin/env python3
"""
Author: Aritra Bal, ETP
Date: XIII Idibus Sextilibus anno ab urbe condita MMDCCLXXVIII

Master training script for Jet GNN classification.
Orchestrates data loading, model creation, training, and checkpointing.
"""

import argparse
import sys
import torch
import numpy as np
import random
from pathlib import Path

# Import project modules
sys.path.append('.')
from configs.config import load_config
from data_utils.graph_dataloader import create_dataloaders
from src.gnn import create_jet_gnn
from src.trainer import JetGNNTrainer, create_optimizer, create_scheduler
from src.logs import setup_logging
from loguru import logger


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For newer PyTorch versions
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
    
    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")


def setup_device(device_config: str) -> torch.device:
    """
    Setup and return the appropriate device for training.
    
    Args:
        device_config: Device configuration from config file
        
    Returns:
        PyTorch device object
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
    else:
        device = torch.device(device_config)
        logger.info(f"Using specified device: {device}")
    
    return device


def setup_experiment_directory(config) -> Path:
    """
    Create experiment directory with seed-based naming.
    
    Args:
        config: Configuration object
        
    Returns:
        Path to experiment directory
    """
    exp_name = f"{config.experiment.name}_{config.experiment.seed}"
    exp_dir = Path(config.experiment.base_save_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config copy to experiment directory
    config_copy_path = exp_dir / "config.yaml"
    import shutil
    shutil.copy("./configs/base.yaml", config_copy_path)
    
    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir


def log_model_summary(model: torch.nn.Module, config):
    """Log model architecture summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=" * 60)
    logger.info("MODEL ARCHITECTURE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model Type: {config.model.type}")
    logger.info(f"Message Passing Layers: {config.model.num_mp_layers}")
    logger.info(f"MP Hidden Layers: {config.model.mp_hidden_layers}")
    logger.info(f"Classifier Hidden Layers: {config.model.classifier_hidden_layers}")
    logger.info(f"Pooling Type: {config.model.pooling}")
    logger.info(f"Activation: {config.model.activation}")
    logger.info(f"Residual Connections: {config.model.residual_connections}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Model Size: {trainable_params * 4 / (1024**2):.2f} MB")
    logger.info("=" * 60)


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Jet GNN for binary classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="./configs/base.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume training from"
    )
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        logger.success(f"Configuration loaded from {args.config}")
        
        # Setup experiment directory
        logger.info("Setting up experiment directory...")
        exp_dir = setup_experiment_directory(config)
        
        # Setup logging to file
        setup_logging(config, exp_dir)
        logger.info(f"Logging setup complete")
        
        # Set random seeds for reproducibility
        logger.info("Setting up reproducibility...")
        set_seed(
            config.reproducibility.random_seed, 
            config.reproducibility.deterministic
        )
        
        # Setup device
        logger.info("Setting up compute device...")
        device = setup_device(config.hardware.device)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_dataloaders(
            train_files=config.data.train_files,
            val_files=config.data.val_files,
            batch_size=config.data.batch_size,
            use_qfi_correlations=config.data.use_qfi_correlations,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory
        )
        
        logger.success(f"Data loaders created")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        logger.info(f"QFI correlations: {config.data.use_qfi_correlations}")
        
        # Create model
        logger.info("Creating GNN model...")
        model = create_jet_gnn(
            model_type=config.model.type,
            num_layers=config.model.num_mp_layers,
            mp_hidden_layers=config.model.mp_hidden_layers,
            classifier_hidden_layers=config.model.classifier_hidden_layers,
            pooling=config.model.pooling,
            activation=config.model.activation,
            residual=config.model.residual_connections
        )
        
        # Move model to device
        model = model.to(device)
        
        # Log model summary
        log_model_summary(model, config)
        
        # Create optimizer and scheduler
        logger.info("Setting up optimization...")
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = JetGNNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            save_dir=exp_dir,
            device=device
        )
        
        # Resume training if requested
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            resume_epoch = trainer.load_checkpoint(args.resume)
            trainer.current_epoch = resume_epoch
        elif config.resume.enabled and config.resume.checkpoint_path:
            logger.info(f"Resuming training from config: {config.resume.checkpoint_path}")
            resume_epoch = trainer.load_checkpoint(config.resume.checkpoint_path)
            trainer.current_epoch = resume_epoch
        
        # Start training
        logger.info("Starting training process...")
        logger.info("=" * 60)
        final_metrics = trainer.train()
        logger.info("=" * 60)
        
        # Log final results
        logger.success("Training completed successfully!")
        logger.info("📊 FINAL RESULTS:")
        for metric_name, metric_value in final_metrics.items():
            if isinstance(metric_value, float):
                logger.info(f"   {metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"   {metric_name}: {metric_value}")
        
        # Log experiment info
        logger.info(f"Results saved in: {exp_dir}")
        logger.info(f"🏆 Best model: {exp_dir}/checkpoints/best_model.pth")
        logger.info(f"Training history: {exp_dir}/training_history.json")
        
        logger.success("All tasks completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (Ctrl+C)")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please check your file paths in the configuration")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("Check the error details above for debugging information")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())