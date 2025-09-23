"""
Author: Aritra Bal, ETP
Date: ante diem quartum Nonas Ianuarias anno ab urbe condita MMDCCLXXVIII

Master training script for QFI-based Jet GNN classification.
Orchestrates data loading, model creation, training, and checkpointing using QFI matrices directly.
"""
import argparse
import sys
import os
import torch
import numpy as np
import random
from pathlib import Path
from typing import Optional

# Import project modules
sys.path.append('.')
from configs.config import load_config
from data_utils.graph_dataloader import StreamingJetDataLoader
from src.gnn import create_qfi_jet_gnn, count_parameters
from src.trainer import QFIJetGNNTrainer, create_optimizer, create_scheduler
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


def setup_experiment_directory(config, config_path: str) -> str:
    """
    Create experiment directory with seed-based naming.
    
    Args:
        config: Configuration object
        config_path: Path to original config file
        
    Returns:
        Path to experiment directory
    """
    exp_name = f"{config.experiment.name}_{config.experiment.seed}"
    exp_dir = os.path.join(config.experiment.base_save_dir, exp_name)
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config copy to experiment directory
    config_copy_path = os.path.join(exp_dir, "config.yaml")
    import subprocess
    subprocess.run(["cp", config_path, config_copy_path])
    
    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir


def create_qfi_dataloaders(config):
    """
    Create QFI-based data loaders for training and validation.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating QFI graph data loaders...")
    
    # Training loader
    train_loader = StreamingJetDataLoader(
        h5_files=config.data.train_files,
        batch_size=config.data.batch_size,
        use_qfi_correlations=True
    )
    
    # Validation loader  
    val_loader = StreamingJetDataLoader(
        h5_files=config.data.val_files,
        batch_size=config.data.batch_size,
        use_qfi_correlations=True
    )
    
    logger.success(f"QFI data loaders created successfully")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    logger.info(f"QFI correlations enabled: {config.data.use_qfi_correlations}")
    
    return train_loader, val_loader


def create_qfi_model(config) -> torch.nn.Module:
    """
    Create QFI-based GNN model from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured QFI GNN model
    """
    logger.info(f"Creating QFI GNN model of type: {config.model.type}")
    
    # Handle backward compatibility for layer specifications
    if hasattr(config.model, 'mp_hidden_layers'):
        mp_layers = config.model.mp_hidden_layers
    elif hasattr(config.model, 'mp_mlp_layers'):
        # Convert old format [input, hidden1, hidden2, output] -> [hidden1, hidden2] 
        mp_layers = config.model.mp_mlp_layers[1:-1] if len(config.model.mp_mlp_layers) > 2 else [16, 8]
        logger.warning(f"Using backward compatibility: converted mp_mlp_layers {config.model.mp_mlp_layers} -> mp_hidden_layers {mp_layers}")
    else:
        mp_layers = [16, 8]  # Default
        logger.warning("No MP layer specification found, using default [16, 8]")
    
    # Determine classifier input dimension based on pooling type
    pooling_type = getattr(config.model, 'pooling', 'mean')
    if pooling_type == 'concat':
        classifier_input_dim = 2  # mean + max of 1D features
    elif pooling_type == 'matrix':
        classifier_input_dim = 100  # 10x10 matrix flattened
    else:
        classifier_input_dim = 1  # single pooling of 1D features
    
    # Handle classifier layers
    if hasattr(config.model, 'classifier_hidden_layers'):
        classifier_layers = config.model.classifier_hidden_layers.copy()
    else:
        classifier_layers = [32, 16, 8, 2]  # Default
        logger.warning("No classifier layer specification found, using default")
    
    # Create model
    if config.model.type.lower() == 'trainable':
        extra_kwargs = None
    else:
        extra_kwargs = config.model
    model = create_qfi_jet_gnn(
        message_type=config.model.type,
        num_layers=config.model.num_mp_layers,
        mp_mlp_layers=mp_layers,
        classifier_layers=classifier_layers,
        pooling=pooling_type,
        activation=getattr(config.model, 'activation', 'elu'),
        aggr=getattr(config.model, 'aggregation', 'add'), extra_kwargs=extra_kwargs
    )

    logger.success(f"QFI GNN model with message passing type: {config.model.type} created successfully")
    return model


def log_model_summary(model: torch.nn.Module, config):
    """Log QFI model architecture summary."""
    param_info = count_parameters(model)
    arch_info = model.get_architecture_info()
    
    logger.info("=" * 60)
    logger.info("QFI MODEL ARCHITECTURE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Message Passing Type: {arch_info['message_type']}")
    logger.info(f"Message Passing Layers: {arch_info['num_mp_layers']}")
    logger.info(f"MP Hidden Layers: {arch_info['mp_mlp_layers']}")
    logger.info(f"Classifier Layers: {arch_info['classifier_layers']}")
    logger.info(f"Pooling Type: {arch_info['pooling_type']}")
    logger.info(f"Activation: {arch_info['activation']}")
    logger.info(f"Graph Structure: 30 nodes (QFI diagonal), ~870 edges (QFI off-diagonal)")
    logger.info(f"Feature Dimensions: 1D nodes, 1D edges")
    logger.info("-" * 60)
    logger.info(f"Total Parameters: {param_info['trainable_parameters']:,}")
    logger.info(f"Trainable Parameters: {param_info['trainable_parameters']:,}")
    logger.info(f"Non-trainable Parameters: {param_info['non_trainable_parameters']:,}")
    logger.info(f"Model Size: {param_info['model_size_mb']:.2f} MB")
    logger.info("=" * 60)


def validate_configuration(config):
    """
    Validate configuration for QFI GNN training.
    
    Args:
        config: Configuration object
        
    Raises:
        ValueError: If configuration is invalid for QFI GNN
    """
    logger.info("Validating QFI GNN configuration...")
    
    # Check required sections
    required_sections = ['experiment', 'data', 'model', 'training']
    for section in required_sections:
        if not hasattr(config, section):
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data files
    if not config.data.train_files:
        raise ValueError("No training files specified in configuration")
    if not config.data.val_files:
        raise ValueError("No validation files specified in configuration")
    
    # Check if files exist
    missing_files = []
    for file_path in config.data.train_files[:3]:  # Check first few files
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"Some training files not found: {missing_files[:2]}...")
        logger.warning("Proceeding anyway - files might be created during training")
    
    # Validate model configuration
    if config.model.type.lower() not in ['trainable', 'gat']:
        raise ValueError(f"Invalid model type for QFI GNN: {config.model.type}. Use 'trainable' or 'gat'")
    
    # Validate batch size for QFI graphs (30 nodes each)
    if config.data.batch_size > 512:
        logger.warning(f"Large batch size ({config.data.batch_size}) for 30-node QFI graphs may cause memory issues")
    
    logger.success("Configuration validation passed")


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train QFI-based Jet GNN for binary classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="./configs/qfi_trainable.yaml",
        help="Path to QFI GNN configuration YAML file"
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
        logger.info("Loading QFI GNN configuration...")
        config = load_config(args.config)
        logger.success(f"Configuration loaded from {args.config}")
        
        # Validate configuration
        validate_configuration(config)
        
        # Setup experiment directory
        logger.info("Setting up experiment directory...")
        exp_dir = setup_experiment_directory(config, args.config)
        
        # Setup logging to file
        setup_logging(config.experiment.name + '_' + config.experiment.seed, exp_dir)
        logger.info(f"Logging setup complete for QFI GNN experiment")
        
        # Set random seeds for reproducibility
        logger.info("Setting up reproducibility...")
        set_seed(
            config.reproducibility.random_seed, 
            config.reproducibility.deterministic
        )
        
        # Setup device
        logger.info("Setting up compute device...")
        device = setup_device(config.hardware.device)
        
        # Create QFI data loaders
        train_loader, val_loader = create_qfi_dataloaders(config)
        
        # Create QFI model
        model = create_qfi_model(config)
        
        # Move model to device
        model = model.to(device)
        
        # Log model summary
        log_model_summary(model, config)
        
        # Create optimizer and scheduler
        logger.info("Setting up optimization...")
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        
        # Create QFI trainer
        logger.info("Creating QFI GNN trainer...")
        trainer = QFIJetGNNTrainer(
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
            logger.info(f"Resuming QFI GNN training from {args.resume}")
            resume_epoch = trainer.load_checkpoint(args.resume)
            trainer.current_epoch = resume_epoch
        elif config.resume.enabled and config.resume.checkpoint_path:
            logger.info(f"Resuming QFI GNN training from config: {config.resume.checkpoint_path}")
            resume_epoch = trainer.load_checkpoint(config.resume.checkpoint_path)
            trainer.current_epoch = resume_epoch
        
        # Start training
        logger.info("Starting QFI GNN training process...")
        logger.info("=" * 60)
        logger.info("🚀 QFI-BASED JET GNN TRAINING STARTED")
        logger.info("=" * 60)
        
        final_metrics = trainer.train()
        
        logger.info("=" * 60)
        logger.info("🎯 QFI GNN TRAINING COMPLETED")
        logger.info("=" * 60)
        
        # Log final results
        logger.success("QFI GNN training completed successfully!")
        logger.info("📊 FINAL RESULTS:")
        for metric_name, metric_value in final_metrics.items():
            if isinstance(metric_value, float):
                logger.info(f"   {metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"   {metric_name}: {metric_value}")
        
        # Log experiment info
        logger.info(f"📁 Results saved in: {exp_dir}")
        logger.info(f"🏆 Best model: {exp_dir}/checkpoints/best_model.pth")
        logger.info(f"📈 Training history: {exp_dir}/training_history.json")
        logger.info(f"⚙️  Model architecture: QFI-based with {model.architecture_info['num_mp_layers']} MP layers")
        
        logger.success("All QFI GNN training tasks completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("QFI GNN training interrupted by user (Ctrl+C)")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please check your file paths in the QFI GNN configuration")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your QFI GNN configuration parameters")
        return 1
    except Exception as e:
        logger.error(f"QFI GNN training failed with error: {e}")
        logger.exception("Full error traceback:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())