"""
Author: Aritra Bal, ETP
Date: ante diem octavum Idus Octobres anno ab urbe condita MMDCCLXXVIII

Simple 2-layer neural network for QFI matrix classification using prototype-based learning.
Trains on flattened 30x30 QFI matrices for top vs QCD jet classification.
Uses learnable class prototypes and hinge loss for maximum class separability.
"""

import argparse
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
import shutil
import sys


class QFIDataset(Dataset):
    """
    PyTorch Dataset for loading QFI matrices from H5 files.
    """
    
    def __init__(self, h5_files: List[str], symmetric: bool = False):
        """
        Initialize dataset by loading all data into memory.
        
        Args:
            h5_files: List of paths to H5 files
            symmetric: If True, use only upper triangle (465 elements); if False, use full matrix (900 elements)
        """
        self.qfi_matrices = []
        self.labels = []
        self.symmetric = symmetric
        
        # Calculate feature dimension
        if symmetric:
            self.feat_dim = 30 * 31 // 2  # Upper triangle including diagonal: 465
            logger.info("Using symmetric mode: extracting upper triangle (465 elements)")
        else:
            self.feat_dim = 900  # Full flattened matrix
            logger.info("Using full matrix mode: flattening entire matrix (900 elements)")
        
        logger.info(f"Loading data from {len(h5_files)} file(s)...")
        
        for file_path in h5_files:
            with h5py.File(file_path, 'r') as f:
                # Load QFI matrices and multiply by 4
                qfi = 4.0 * f['jetConstituentsQFI'][:]  # Shape: [n_jets, 30, 30]
                labels = f['truth_labels'][:]  # Shape: [n_jets]
                
                if symmetric:
                    # Extract upper triangle including diagonal
                    triu_indices = np.triu_indices(30, k=0)
                    qfi_flat = qfi[:, triu_indices[0], triu_indices[1]]  # Shape: [n_jets, 465]
                else:
                    # Flatten entire matrix
                    qfi_flat = qfi.reshape(-1, 900)  # Shape: [n_jets, 900]
                
                self.qfi_matrices.append(qfi_flat)
                self.labels.append(labels)
                
                logger.info(f"Loaded {len(labels)} jets from {Path(file_path).name}")
        
        # Concatenate all data
        self.qfi_matrices = np.concatenate(self.qfi_matrices, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        
        logger.success(f"Total dataset size: {len(self.labels)} jets")
        logger.info(f"Feature dimension: {self.feat_dim}")
        logger.info(f"QCD jets: {np.sum(self.labels == 0)}, Top jets: {np.sum(self.labels == 1)}")
    
    def __len__(self) -> int:
        return len(self.labels)
    def fetch_qfi_means(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch mean of QFI matrices in the dataset: for top and QCD jets.
        
        Returns:
            Tuple of (mean_qcd, mean_top) arrays of shape [feat_dim]
        """
        mean_qcd = np.mean(self.qfi_matrices[self.labels == 0], axis=0)
        mean_top = np.mean(self.qfi_matrices[self.labels == 1], axis=0)
        return mean_qcd, mean_top

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (qfi_features, label)
        """
        qfi = torch.tensor(self.qfi_matrices[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return qfi, label


class SimpleQFINet(nn.Module):
    """
    Simple 2-layer neural network for QFI embedding.
    Architecture: feat_dim -> feat_dim -> feat_dim
    Returns feat_dim-dimensional embedding for prototype comparison.
    """
    
    def __init__(self, feat_dim: int = 900, activation: str = 'identity'):
        """
        Initialize network.
        
        Args:
            feat_dim: Dimension of input/output features (465 for symmetric, 900 for full)
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'elu', 'identity')
        """
        super(SimpleQFINet, self).__init__()
        
        self.feat_dim = feat_dim
        
        # Define layers
        #self.layer1 = nn.Linear(feat_dim, feat_dim)
        #self.layer2 = nn.Linear(feat_dim, feat_dim)
        
        # Activation function
        activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'identity': nn.Identity()
        }
        self.activation = activation_map.get(activation.lower(), nn.Identity())
        
        logger.info(f"Initialized SimpleQFINet with {activation} activation, feat_dim={feat_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, feat_dim]
            
        Returns:
            feat_dim-dimensional embedding of shape [batch_size, feat_dim]
        """
        # First layer: feat_dim -> feat_dim
        #x = self.layer1(x)
        #x = self.activation(x)
        
        # Second layer: feat_dim -> feat_dim
        #x = self.layer2(x)
        #x = self.activation(x)
        
        return x  # Return full feat_dim-dimensional embedding


class PrototypeClassifier(nn.Module):
    """
    Prototype-based classifier with learnable class prototypes.
    Classifies based on distance to learned 900-dimensional prototypes.
    """
    
    def __init__(self, feat_dim: int = 900, num_classes: int = 2, means: Tuple[np.ndarray, np.ndarray] = None):
        """
        Initialize prototype classifier.
        
        Args:
            feat_dim: Dimension of feature embeddings
            num_classes: Number of classes (2 for binary)
        """
        super(PrototypeClassifier, self).__init__()
        
        # Learnable class prototypes: [num_classes, feat_dim] initialised to QCD and Top means if provided
        if means is not None:
            qcd_mean, top_mean = means
            assert qcd_mean.shape[0] == feat_dim and top_mean.shape[0] == feat_dim, "Mean shapes do not match feat_dim"
            initial_prototypes = np.stack([qcd_mean, top_mean], axis=0)  # Shape: [2, feat_dim]
            self.prototypes = nn.Parameter(torch.tensor(initial_prototypes, dtype=torch.float32))
            logger.info("Initialized prototypes with dataset means")
        else:
            self.prototypes = nn.Parameter(-1.+2*torch.rand(num_classes, feat_dim))
        
        logger.info(f"Initialized {num_classes} learnable prototypes of dimension {feat_dim}")
    
    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distances to class prototypes.
        
        Args:
            embeddings: [batch_size, feat_dim] embeddings
            
        Returns:
            Tuple of (dist_to_class0, dist_to_class1)
        """
        # Compute squared L2 distances to each prototype
        dist_to_0 = torch.sum((embeddings - self.prototypes[0]) ** 2, dim=1)  # [batch_size]
        dist_to_1 = torch.sum((embeddings - self.prototypes[1]) ** 2, dim=1)  # [batch_size]
        
        return dist_to_0, dist_to_1
    
    def get_score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get classification score.
        
        Args:
            embeddings: [batch_size, feat_dim] embeddings
            
        Returns:
            Classification scores [batch_size]. Higher score = more likely class 1 (Top)
        """
        dist_to_0, dist_to_1 = self.forward(embeddings)
        return dist_to_0 - dist_to_1  # Positive = closer to class 1
    
    def get_probability(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get probability of class 1 using softmax on negative distances.
        
        Args:
            embeddings: [batch_size, feat_dim] embeddings
            
        Returns:
            Probabilities of class 1 [batch_size]
        """
        score = self.get_score(embeddings)
        return torch.sigmoid(score)  # Convert score to probability
    

def train_epoch(
    model: nn.Module,
    prototype_classifier: PrototypeClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        prototype_classifier: Prototype classifier
        dataloader: Training data loader
        criterion: Loss function (MarginRankingLoss)
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Tuple of (average_loss, accuracy, auc)
    """
    model.train()
    prototype_classifier.train()
    
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
        
        # Forward pass: get embeddings
        embeddings = model(qfi)
        
        # Compute distances to prototypes
        dist_to_0, dist_to_1 = prototype_classifier(embeddings)
        
        # Margin ranking loss
        # For QCD (label=0): want dist_to_1 > dist_to_0 (target=1)
        # For Top (label=1): want dist_to_0 > dist_to_1 (target=-1)
        targets = 1 - 2 * labels.float()  # 0→1, 1→-1
        loss = criterion(dist_to_1, dist_to_0, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Predictions based on which prototype is closer
        predictions = (dist_to_0 > dist_to_1).long()  # 1 if closer to prototype 1
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Store for AUC calculation
        probs = prototype_classifier.get_probability(embeddings)
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
    prototype_classifier: PrototypeClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Validate the model.
    
    Args:
        model: Neural network model
        prototype_classifier: Prototype classifier
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average_loss, accuracy, auc)
    """
    model.eval()
    prototype_classifier.eval()
    
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
            embeddings = model(qfi)
            dist_to_0, dist_to_1 = prototype_classifier(embeddings)
            
            # Loss
            targets = 1 - 2 * labels.float()
            loss = criterion(dist_to_1, dist_to_0, targets)
            
            # Statistics
            total_loss += loss.item()
            predictions = (dist_to_0 > dist_to_1).long()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Store for AUC
            probs = prototype_classifier.get_probability(embeddings)
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
            verbose=True, threshold=config['training']['scheduler_delta']
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


def reconstruct_symmetric_matrix(upper_triangle: np.ndarray, matrix_size: int = 30) -> np.ndarray:
    """
    Reconstruct symmetric matrix from upper triangle including diagonal.
    
    Args:
        upper_triangle: 1D array of upper triangle values (465 elements for 30x30)
        matrix_size: Size of square matrix (default: 30)
        
    Returns:
        Reconstructed symmetric matrix of shape [matrix_size, matrix_size]
    """
    # Create empty matrix
    matrix = np.zeros((matrix_size, matrix_size))
    
    # Get indices for upper triangle including diagonal
    triu_indices = np.triu_indices(matrix_size, k=0)
    
    # Fill upper triangle
    matrix[triu_indices[0], triu_indices[1]] = upper_triangle
    
    # Mirror to lower triangle (make symmetric)
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    
    return matrix


def main():
    """Main training function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train prototype-based QFI classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_simple.yaml",
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get symmetric flag from config
    symmetric = config['model'].get('symmetric', False)
    feat_dim = 465 if symmetric else 900
    logger.info(f"Symmetric mode: {symmetric}, Feature dimension: {feat_dim}")
    
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
    train_dataset = QFIDataset(config['data']['train_files'], symmetric=symmetric)
    val_dataset = QFIDataset(config['data']['val_files'], symmetric=symmetric)
    
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
    
    # Create output directory
    output_dir = Path(config['experiment']['base_save_dir']) / f"{config['experiment']['name']}_{config['experiment']['seed']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Copy config file to experiment directory
    config_dest = output_dir / "config.yaml"
    shutil.copy2(args.config, config_dest)
    logger.info(f"Configuration copied to: {config_dest}")
    
    # Create model and prototype classifier
    logger.info("Creating model and prototype classifier...")
    model = SimpleQFINet(feat_dim=feat_dim, activation=config['model']['activation'])
    means = train_dataset.fetch_qfi_means() if config['model']['initialization'] == "qfi" else None

    prototype_classifier = PrototypeClassifier(feat_dim=feat_dim, num_classes=2, means=means)
    # Extract and save initial prototypes to an npz file
    initial_prototypes = prototype_classifier.prototypes.detach().cpu().numpy()  # [2, feat_dim]
    if symmetric:
        # Reconstruct full 30x30 matrices from upper triangle
        initial_qcd_prototype = reconstruct_symmetric_matrix(initial_prototypes[0], matrix_size=30)
        initial_top_prototype = reconstruct_symmetric_matrix(initial_prototypes[1], matrix_size=30)
    else:
        # Reshape full 900-D vectors to 30x30
        initial_qcd_prototype = initial_prototypes[0].reshape(30, 30)
        initial_top_prototype = initial_prototypes[1].reshape(30, 30)
    np.savez(
        output_dir / 'initial_prototypes.npz',
        qcd_prototype=initial_qcd_prototype,
        top_prototype=initial_top_prototype
    )
    logger.info(f"Saved initial prototypes to: {output_dir / 'initial_prototypes.npz'}")
    
    model = model.to(device)
    prototype_classifier = prototype_classifier.to(device)
    
    # Count parameters
    model_params = sum(p.numel() for p in model.parameters())
    proto_params = sum(p.numel() for p in prototype_classifier.parameters())
    total_params = model_params + proto_params
    
    logger.info(f"Model parameters: {model_params:,}")
    logger.info(f"Prototype parameters: {proto_params:,}")
    logger.info(f"Total parameters: {total_params:,}")
    
    # Loss function: Margin Ranking Loss for prototype comparison
    margin = config['training'].get('margin', 1.0)
    criterion = nn.MarginRankingLoss(margin=margin)
    logger.info(f"Using MarginRankingLoss with margin={margin}")
    
    # Optimizer: optimize both model and prototype parameters
    optimizer = optim.Adam(
        list(model.parameters()) + list(prototype_classifier.parameters()),
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
    
    # Training loop
    logger.info("Starting training...")
    logger.info("=" * 60)
    best_val_auc = 0.0
    val_losses = []
    val_aucs = []
    train_losses = []
    train_aucs = []
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        logger.info("=" * 60)
        
        # Train
        train_loss, train_acc, train_auc = train_epoch(
            model, prototype_classifier, train_loader, criterion, optimizer, device
        )
        logger.info(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%, AUC: {train_auc:.4f}")
        
        # Validate
        val_loss, val_acc, val_auc = validate(
            model, prototype_classifier, val_loader, criterion, device
        )
        logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, AUC: {val_auc:.4f}")

        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        train_losses.append(train_loss)
        train_aucs.append(train_auc)
        # Save validation metrics
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
                'prototype_classifier_state_dict': prototype_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_auc': val_auc,
            }, output_dir / 'best_model.pth')
            
            # Save prototypes as 30x30 matrices
            prototypes_np = prototype_classifier.prototypes.detach().cpu().numpy()  # [2, feat_dim]
            
            if symmetric:
                # Reconstruct full 30x30 matrices from upper triangle
                qcd_prototype = reconstruct_symmetric_matrix(prototypes_np[0], matrix_size=30)
                top_prototype = reconstruct_symmetric_matrix(prototypes_np[1], matrix_size=30)
            else:
                # Reshape full 900-D vectors to 30x30
                qcd_prototype = prototypes_np[0].reshape(30, 30)
                top_prototype = prototypes_np[1].reshape(30, 30)
            
            np.savez(
                output_dir / 'prototypes.npz',
                qcd_prototype=qcd_prototype,
                top_prototype=top_prototype
            )
            
            logger.success(f"Saved best model and prototypes (val_auc: {val_auc:.4f})")
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_auc):
                logger.warning("Early stopping triggered, ending training")
                break
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'prototype_classifier_state_dict': prototype_classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_dir / 'final_model.pth')
    # save train and val losses and aucs
    np.savez(
        output_dir / 'training_metrics.npz',
        train_losses=np.array(train_losses),
        train_aucs=np.array(train_aucs),
        val_losses=np.array(val_losses),
        val_aucs=np.array(val_aucs)
    )
    # Save final weights and biases
    # torch.save({
    #     'layer1_weight': model.layer1.weight.data.cpu(),
    #     'layer1_bias': model.layer1.bias.data.cpu(),
    #     'layer2_weight': model.layer2.weight.data.cpu(),
    #     'layer2_bias': model.layer2.bias.data.cpu(),
    # }, output_dir / 'weights_and_biases.pth')
    
    # Save final prototypes
    prototypes_np = prototype_classifier.prototypes.detach().cpu().numpy()
    
    if symmetric:
        # Reconstruct full 30x30 matrices from upper triangle
        qcd_prototype = reconstruct_symmetric_matrix(prototypes_np[0], matrix_size=30)
        top_prototype = reconstruct_symmetric_matrix(prototypes_np[1], matrix_size=30)
    else:
        # Reshape full 900-D vectors to 30x30
        qcd_prototype = prototypes_np[0].reshape(30, 30)
        top_prototype = prototypes_np[1].reshape(30, 30)
    
    np.savez(
        output_dir / 'final_prototypes.npz',
        qcd_prototype=qcd_prototype,
        top_prototype=top_prototype
    )
    
    logger.success(f"Training complete. Best validation AUC: {best_val_auc:.4f}")
    logger.success(f"Models and prototypes saved to: {output_dir}")
    logger.info(f"Best prototypes saved to: {output_dir / 'prototypes.npz'}")
    logger.info(f"Final prototypes saved to: {output_dir / 'final_prototypes.npz'}")


if __name__ == "__main__":
    main()