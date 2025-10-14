"""
Author: Aritra Bal, ETP
Date: ante diem octavum Idus Octobres anno ab urbe condita MMDCCLXXVIII

Simple 2-layer neural network for QFI matrix classification.
Trains on flattened 30x30 QFI matrices for top vs QCD jet classification.
"""

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple
import yaml
from loguru import logger
from sklearn.metrics import roc_auc_score
import sys
import argparse

class QFIDataset(Dataset):
    """
    PyTorch Dataset for loading QFI matrices from H5 files.
    """
    
    def __init__(self, h5_files: List[str]):
        """
        Initialize dataset by loading all data into memory.
        
        Args:
            h5_files: List of paths to H5 files
        """
        self.qfi_matrices = []
        self.labels = []
        
        logger.info(f"Loading data from {len(h5_files)} file(s)...")
        
        for file_path in h5_files:
            with h5py.File(file_path, 'r') as f:
                # Load QFI matrices and multiply by 4
                qfi = 4.0 * f['jetConstituentsQFI'][:]  # Shape: [n_jets, 30, 30]
                labels = f['truth_labels'][:]  # Shape: [n_jets]
                
                # Flatten QFI matrices to 900-dimensional vectors
                qfi_flat = qfi.reshape(-1, 900)  # Shape: [n_jets, 900]
                
                self.qfi_matrices.append(qfi_flat)
                self.labels.append(labels)
                
                logger.info(f"Loaded {len(labels)} jets from {Path(file_path).name}")
        
        # Concatenate all data
        self.qfi_matrices = np.concatenate(self.qfi_matrices, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        
        logger.success(f"Total dataset size: {len(self.labels)} jets")
        logger.info(f"QCD jets: {np.sum(self.labels == 0)}, Top jets: {np.sum(self.labels == 1)}")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (qfi_matrix, label)
        """
        qfi = torch.tensor(self.qfi_matrices[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return qfi, label


class SimpleQFINet(nn.Module):
    """
    Simple 2-layer neural network for QFI classification.
    Architecture: 900 -> 900 -> 900
    Output is the mean of the final 900 values.
    """
    
    def __init__(self, activation: str = 'identity'):
        """
        Initialize network.
        
        Args:
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'elu', 'identity')
        """
        super(SimpleQFINet, self).__init__()
        
        # Define layers
        self.layer1 = nn.Linear(900, 900)
        self.layer2 = nn.Linear(900, 900)
        
        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'identity': nn.Identity()
        }
        self.activation = activation_map.get(activation.lower(), nn.Identity())
        
        logger.info(f"Initialized SimpleQFINet with {activation} activation")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, 900]
            
        Returns:
            Output predictions of shape [batch_size]
        """
        # First layer: 900 -> 900
        x = self.layer1(x)
        x = self.activation(x)
        
        # Second layer: 900 -> 900
        x = self.layer2(x)
        x = self.activation(x)
        
        # Output: mean of 900 values
        output = torch.mean(x, dim=1)  # Shape: [batch_size]
        
        return output


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Tuple of (average_loss, accuracy, auc)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_probs = []
    
    for batch_idx, (qfi, labels) in enumerate(dataloader):
        qfi = qfi.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(qfi)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        probs = torch.sigmoid(outputs)
        predictions = (probs > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Store for AUC calculation
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    auc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, accuracy, auc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Validate the model.
    
    Args:
        model: Neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average_loss, accuracy, auc)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for qfi, labels in dataloader:
            qfi = qfi.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(qfi)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Store for AUC calculation
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    auc = roc_auc_score(all_labels, all_probs)
    
    return avg_loss, accuracy, auc


def create_scheduler(optimizer: optim.Optimizer, config: dict):
    """
    Create learning rate scheduler based on config.
    
    Args:
        optimizer: Optimizer instance
        config: Configuration dictionary
        
    Returns:
        Learning rate scheduler or None
    """
    if not config['training']['use_scheduler']:
        return None
    
    scheduler_type = config['training']['scheduler_type'].lower()
    
    if scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize AUC
            factor=config['training']['scheduler_factor'],
            patience=config['training']['scheduler_patience'],
            verbose=True
        )
        logger.info(f"Created ReduceLROnPlateau scheduler (patience={config['training']['scheduler_patience']}, factor={config['training']['scheduler_factor']})")
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['scheduler_patience'],
            gamma=config['training']['scheduler_factor']
        )
        logger.info(f"Created StepLR scheduler (step_size={config['training']['scheduler_patience']}, gamma={config['training']['scheduler_factor']})")
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
        logger.info("Created CosineAnnealingLR scheduler")
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, no scheduler will be used")
        return None
    
    return scheduler


class EarlyStopping:
    """
    Early stopping handler that monitors validation AUC.
    """
    
    def __init__(self, patience: int, min_delta: float):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_auc = 0.0
        self.early_stop = False
        
        logger.info(f"Early stopping enabled (patience={patience}, min_delta={min_delta})")
    
    def __call__(self, val_auc: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_auc: Current validation AUC
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc = val_auc
            self.counter = 0
        else:
            self.counter += 1
            logger.warning(f"Early stopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.warning(f"Early stopping triggered! Best AUC: {self.best_auc:.4f}")
                return True
        
        return False


def main():
    """Main training function."""
    # Accept path to yaml with argparse
    parser = argparse.ArgumentParser(description="Train SimpleQFINet on QFI matrices")
    parser.add_argument("--config", type=str, default="config_linearQFI.yaml", help="Path to configuration YAML file")
    args = parser.parse_args()
    # Load configuration
    logger.info("Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config['reproducibility']['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['reproducibility']['random_seed'])
    np.random.seed(config['reproducibility']['random_seed'])
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = QFIDataset(config['data']['train_files'])
    val_dataset = QFIDataset(config['data']['val_files'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=config['data']['pin_memory']
    )
    
    logger.success(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model
    logger.info("Creating model...")
    model = SimpleQFINet(activation=config['model']['activation'])
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Create early stopping
    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
    
    # Create output directory
    output_dir = Path(config['experiment']['base_save_dir']) / f"{config['experiment']['name']}_{config['experiment']['seed']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    # copy the config yaml file from the argument to the output dir as config.yaml
    import shutil
    shutil.copy(args.config, output_dir / "config.yaml")
    # Training loop
    logger.info("Starting training...")
    best_val_auc = 0.0
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        logger.info("=" * 60)
        
        # Train
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, AUC: {train_auc:.4f}")
        
        # Validate
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, AUC: {val_auc:.4f}")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.6f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_auc': val_auc,
            }, output_dir / 'best_model.pth')
            logger.success(f"Saved best model (val_auc: {val_auc:.4f})")
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_auc):
                logger.warning("Early stopping triggered, ending training")
                break
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_dir / 'final_model.pth')
    
    # Save weights and biases separately
    torch.save({
        'layer1_weight': model.layer1.weight.data.cpu(),
        'layer1_bias': model.layer1.bias.data.cpu(),
        'layer2_weight': model.layer2.weight.data.cpu(),
        'layer2_bias': model.layer2.bias.data.cpu(),
    }, output_dir / 'weights_and_biases.pth')
    
    logger.success(f"Training complete. Best validation AUC: {best_val_auc:.4f}")
    logger.success(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()