"""
Author: Aritra Bal, ETP
Date: ante diem octavum Idus Octobres anno ab urbe condita MMDCCLXXVIII

Test script for prototype-based QFI classification.
Evaluates trained model on test data and generates comprehensive analysis.
"""

import argparse
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple
import yaml
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_curve, roc_auc_score
from datetime import datetime
import sys

# Import classes from training script
from train_prototype import QFIDataset, SimpleQFINet, PrototypeClassifier


class QFIMatrixPlotter:
    """
    QFI matrix plotting utility for visualization and class-separated comparison.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize QFI plotter.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"QFI plotter initialized, saving to: {output_dir}")
    
    def plot_qfi_matrix(self, qfi_matrix: np.ndarray, plot_label: str, 
                       save_name: str, show_stats: bool = True, set_zero_diag: bool = False) -> None:
        """
        Plot QFI matrix with quantum circuit style formatting.
        
        Args:
            qfi_matrix: Array of shape (30, 30) - QFI matrix
            plot_label: Label for the plot title
            save_name: Filename for saving (without extension)
            show_stats: Whether to print matrix statistics
            set_zero_diag: Whether to set diagonal to zero before plotting
        """
        save_path = self.output_dir / save_name
        # normalise the matrix to -1 to 1 for better color scaling
        # Get matrix dimensions
        N_params = qfi_matrix.shape[0]  # Should be 30
        N_qubits = N_params // 3        # Should be 10
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Color scheme  
        colors = ['#0066FF', 'white', '#FF0066']  # Blue -> White -> Red
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue_white_red', colors, N=256)
        
        # Set up normalization
        plot_matrix = qfi_matrix.copy()
        if set_zero_diag:
            np.fill_diagonal(plot_matrix, 0.0)
        data_range = 3.
        
        if data_range == 0:
            data_range = 1  # Avoid division by zero
        if data_range < 0.5:
            data_range = 0.25 
        norm = matplotlib.colors.Normalize(vmin=-data_range, vmax=data_range)
        
        # Plot the matrix
        im = ax.matshow(plot_matrix, cmap=cmap, norm=norm)
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('QFI Value', fontsize=12)
        
        # Add dark grid lines to highlight 3x3 blocks
        for i in range(1, N_qubits):
            ax.axhline(y=3*i - 0.5, color='black', linewidth=2)
            ax.axvline(x=3*i - 0.5, color='black', linewidth=2)
        
        # Set tick positions and labels
        rotation_positions = [i+0.5 for i in range(3*N_qubits)]
        ax.set_xticks(rotation_positions)
        ax.set_yticks(rotation_positions)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Add rotation labels
        rotation_labels = ['$R_Z$', '$R_Y$', '$R_X$'] * N_qubits
        label_positions = [i for i in range(3*N_qubits)]
        
        for pos, label in zip(label_positions, rotation_labels):
            ax.text(pos, 3*N_qubits+0.1, label, ha='center', va='top', fontsize=10)
            ax.text(-0.7, pos, label, ha='right', va='center', fontsize=10)
        
        # Add qubit number labels
        qubit_positions = [1 + 3*i for i in range(N_qubits)]
        qubit_labels = [str(i) for i in range(N_qubits)]
        
        for pos, label in zip(qubit_positions, qubit_labels):
            ax.text(pos, -0.3, label, ha='center', va='top', fontsize=14, fontweight='bold')
            ax.text(-1.4, pos, label, ha='right', va='center', fontsize=14, fontweight='bold')
        
        # Set labels and title
        ax.set_xlabel('Qubit Number', labelpad=40, fontsize=14)
        ax.set_ylabel('Qubit Number', labelpad=40, fontsize=14)
        plt.title(plot_label, fontsize=18, pad=20)
        plt.tight_layout()
        
        # Save in both formats
        plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.close()
        
        # Print statistics
        if show_stats:
            logger.info(f"QFI Matrix Statistics for {plot_label}:")
            logger.info(f"  Range: [{qfi_matrix.min():.6f}, {qfi_matrix.max():.6f}]")
            logger.info(f"  Mean: {qfi_matrix.mean():.6f}")
            logger.info(f"  Std:  {qfi_matrix.std():.6f}")
            logger.info(f"  Saved to: {save_path}.png/.pdf")


def plot_log_roc_curves(metrics_list: List[Tuple[np.ndarray, np.ndarray, float]], 
                       labels: List[str], save_dir: str, timestamp: str):
    """
    Plot log ROC curves for multiple experiments.
    
    Args:
        metrics_list: List of (fpr, tpr, auc) tuples
        labels: List of legend labels for each experiment
        save_dir: Directory to save plots
        timestamp: Timestamp string for filename
    """
    plt.figure(figsize=(10, 8))
    
    logger.info("Creating log ROC curves...")
    
    for i, (fpr, tpr, auc_value) in enumerate(metrics_list):
        # Avoid division by zero in 1/fpr
        fpr_safe = np.maximum(fpr, 1e-6)
        background_rejection = 1.0 / fpr_safe
        
        label = f"{labels[i]} (AUC = {auc_value:.3f})"
        plt.plot(tpr, background_rejection, linewidth=2, label=label)
    
    # Baseline curve (random classifier: TPR = FPR)
    base_tpr = np.arange(0.001, 1.0, 0.01)
    base_fpr = base_tpr  # For random classifier
    plt.plot(base_tpr, 1.0/base_fpr, linewidth=2, label="Baseline (AUC = 0.5)", 
             linestyle='--', color='black', alpha=0.7)
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([1.0, 1.0e6])
    plt.yscale('log')
    plt.xlabel('Signal Efficiency (TPR)', fontsize=17)
    plt.ylabel('Background Rejection (FPR$^{-1}$)', fontsize=17)
    plt.title('Jet Classification Performance - Log ROC', fontsize=19, fontweight='bold')
    plt.legend(loc="upper right", fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save in both formats
    png_path = Path(save_dir) / f"ROC_curve.png"
    pdf_path = Path(save_dir) / f"ROC_curve.pdf"
    
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    logger.success(f"Log ROC curves saved to {png_path} and {pdf_path}")
    plt.close()


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


def load_models(checkpoint_path: Path, device: torch.device, activation: str = 'identity', feat_dim: int = 900) -> Tuple[nn.Module, nn.Module]:
    """
    Load trained model and prototype classifier from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load models on
        activation: Activation function name
        feat_dim: Feature dimension (465 for symmetric, 900 for full)
        
    Returns:
        Tuple of (model, prototype_classifier)
    """
    logger.info(f"Loading models from {checkpoint_path}")
    
    # Create model instances with correct feature dimension
    model = SimpleQFINet(feat_dim=feat_dim, activation=activation)
    prototype_classifier = PrototypeClassifier(feat_dim=feat_dim, num_classes=2)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    prototype_classifier.load_state_dict(checkpoint['prototype_classifier_state_dict'])
    
    # Move to device
    model = model.to(device)
    prototype_classifier = prototype_classifier.to(device)
    
    # Set to evaluation mode
    model.eval()
    prototype_classifier.eval()
    
    logger.success("Models loaded successfully")
    return model, prototype_classifier


def evaluate_model(
    model: nn.Module, 
    prototype_classifier: PrototypeClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    symmetric: bool = False
) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on test data.
    
    Args:
        model: Neural network model
        prototype_classifier: Prototype classifier
        dataloader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        symmetric: If True, embeddings are 465-D; if False, 900-D
        
    Returns:
        Tuple of (loss, accuracy, auc, fpr, tpr, thresholds, all_embeddings, all_labels)
    """
    model.eval()
    prototype_classifier.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_probs = []
    all_embeddings = []
    
    logger.info("Running inference on test data...")
    
    with torch.no_grad():
        for batch_idx, (qfi, labels) in enumerate(dataloader):
            qfi = qfi.to(device)
            labels = labels.to(device)
            
            # Forward pass: get feat_dim embeddings from layer 2
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
            
            # Store for metrics
            probs = prototype_classifier.get_probability(embeddings)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    auc_score = roc_auc_score(all_labels, all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    
    logger.success(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, AUC: {auc_score:.4f}")
    
    return avg_loss, accuracy, auc_score, fpr, tpr, thresholds, all_embeddings, all_labels


def save_inference_results(output_path: Path,
                          original_qfi: np.ndarray,
                          reconstructed_qfi: np.ndarray,
                          labels: np.ndarray):
    """
    Save inference results to H5 file.
    
    Args:
        output_path: Path to output H5 file
        original_qfi: Original QFI matrices [N, 30, 30]
        reconstructed_qfi: Reconstructed QFI matrices [N, 30, 30]
        labels: Truth labels [N]
    """
    logger.info(f"Saving inference results to {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('original_QFI', data=original_qfi)
        f.create_dataset('reconstructed_QFI', data=reconstructed_qfi)
        f.create_dataset('truth_labels', data=labels)

    logger.success(f"Inference results saved: {output_path}")


def main():
    """Main testing function."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Test prototype-based QFI classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to experiment directory containing trained model"
    )
    parser.add_argument(
        "--skip",
        action='store_true',
        help="Don't save inference results: takes time"
    )

    args = parser.parse_args()
    
    # Setup paths
    exp_dir = Path(args.experiment_dir)
    config_path = exp_dir / "config.yaml"
    checkpoint_path = exp_dir / "best_model.pth"
    prototypes_path = exp_dir / "prototypes.npz"
    results_dir = exp_dir / "results"
    plots_dir = results_dir / "plots"
    initial_prototypes_path = exp_dir / "initial_prototypes.npz"

    # Validate paths
    if not exp_dir.exists():
        logger.error(f"Experiment directory does not exist: {exp_dir}")
        sys.exit(1)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    if not prototypes_path.exists():
        logger.error(f"Prototypes file not found: {prototypes_path}")
        sys.exit(1)
    
    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PROTOTYPE-BASED QFI CLASSIFIER TESTING")
    logger.info("=" * 60)
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get symmetric flag and feature dimension
    symmetric = config['model'].get('symmetric', False)
    feat_dim = 465 if symmetric else 900
    logger.info(f"Symmetric mode: {symmetric}, Feature dimension: {feat_dim}")
    
    # Setup device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create test dataset
    logger.info("Creating test dataset...")
    test_dataset = QFIDataset(config['data']['test_files'], symmetric=symmetric)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['testing']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    logger.success(f"Test dataloader created with {len(test_loader)} batches")
    
    # Load models
    model, prototype_classifier = load_models(
        checkpoint_path, 
        device, 
        activation=config['model']['activation'],
        feat_dim=feat_dim
    )
    
    # Create loss function
    margin = config['training'].get('margin', 1.0)
    criterion = nn.MarginRankingLoss(margin=margin)
    
    # Evaluate model
    avg_loss, accuracy, auc_score, fpr, tpr, thresholds, all_embeddings, all_labels = evaluate_model(
        model, prototype_classifier, test_loader, criterion, device, symmetric=symmetric
    )
    
    # Get reconstructed QFI matrices
    if symmetric:
        # Reconstruct 30x30 matrices from 465-D upper triangle embeddings
        logger.info("Reconstructing 30x30 matrices from upper triangle embeddings...")
        reconstructed_qfi = np.zeros((len(all_embeddings), 30, 30))
        for i in range(len(all_embeddings)):
            reconstructed_qfi[i] = reconstruct_symmetric_matrix(all_embeddings[i], matrix_size=30)
    else:
        # Reshape 900-D embeddings to 30x30
        reconstructed_qfi = all_embeddings.reshape(-1, 30, 30)
    
    logger.info(f"Reconstructed QFI shape: {reconstructed_qfi.shape}")
    
    # Get original QFI matrices from dataset
    if symmetric:
        # Reconstruct original 30x30 matrices from upper triangle
        logger.info("Reconstructing original 30x30 matrices from upper triangle...")
        original_qfi = np.zeros((len(test_dataset.qfi_matrices), 30, 30))
        for i in range(len(test_dataset.qfi_matrices)):
            original_qfi[i] = reconstruct_symmetric_matrix(test_dataset.qfi_matrices[i], matrix_size=30)
    else:
        # Reshape from 900-D to 30x30
        original_qfi = test_dataset.qfi_matrices.reshape(-1, 30, 30)
    
    logger.info(f"Original QFI shape: {original_qfi.shape}")
    
    # Save inference results
    if not args.skip:
        output_h5_path = Path(config['testing']['output_dir']) / f"{config['experiment']['name']}_{config['experiment']['seed']}" / "inference.h5"
        save_inference_results(output_h5_path, original_qfi, reconstructed_qfi, all_labels)
    
    # Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = results_dir / "metrics.npz"
    
    np.savez(
        metrics_path,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        loss=avg_loss,
        accuracy=accuracy,
        auc=auc_score
    )
    logger.success(f"Metrics saved to {metrics_path}")
    
    # Plot ROC curve
    plot_log_roc_curves(
        [(fpr, tpr, auc_score)],
        ["Prototype Classifier"],
        plots_dir,
        timestamp
    )
    
    # Initialize QFI plotter
    plotter = QFIMatrixPlotter(plots_dir)
    
    # Separate QCD and Top jets
    qcd_mask = all_labels == 0
    top_mask = all_labels == 1
    
    logger.info(f"QCD jets: {qcd_mask.sum()}, Top jets: {top_mask.sum()}")
    
    # Plot mean original QFI matrices
    logger.info("Plotting mean original QFI matrices...")
    mean_qcd_original = original_qfi[qcd_mask].mean(axis=0)
    mean_top_original = original_qfi[top_mask].mean(axis=0)
    
    plotter.plot_qfi_matrix(
        mean_qcd_original,
        "Mean Original QCD QFI Matrix",
        "mean_original_qcd_qfi",
        show_stats=True,
        set_zero_diag=False
    )
    
    plotter.plot_qfi_matrix(
        mean_top_original,
        "Mean Original Top QFI Matrix",
        "mean_original_top_qfi",
        show_stats=True,
        set_zero_diag=False
    )
    
    # Plot prototype matrices
    logger.info("Plotting learned prototype matrices...")
    prototypes_data = np.load(prototypes_path)
    qcd_prototype = prototypes_data['qcd_prototype']
    top_prototype = prototypes_data['top_prototype']
    initial_prototypes_data = np.load(initial_prototypes_path)
    initial_qcd_prototype = initial_prototypes_data['qcd_prototype']
    initial_top_prototype = initial_prototypes_data['top_prototype']

    plotter.plot_qfi_matrix(
        qcd_prototype,
        "Learned QCD Prototype",
        "learned_qcd_prototype",
        show_stats=True,
        set_zero_diag=False
    )
    
    plotter.plot_qfi_matrix(
        top_prototype,
        "Learned Top Prototype",
        "learned_top_prototype",
        show_stats=True,
        set_zero_diag=False
    )
    # Plot initial prototypes for comparison
    plotter.plot_qfi_matrix(
        initial_qcd_prototype,
        "Initial QCD Prototype",
        "initial_qcd_prototype",
        show_stats=True,
        set_zero_diag=False
    )
    plotter.plot_qfi_matrix(
        initial_top_prototype,
        "Initial Top Prototype",
        "initial_top_prototype",
        show_stats=True,
        set_zero_diag=False
    )
    # Compute and plot difference between learned and initial prototypes
    diff_qcd = qcd_prototype - initial_qcd_prototype
    diff_top = top_prototype - initial_top_prototype
    plotter.plot_qfi_matrix(
        diff_qcd,
        "Difference: Learned - Initial QCD Prototype",
        "diff_learned_minus_initial_qcd_prototype",
        show_stats=True,
        set_zero_diag=False
    )
    plotter.plot_qfi_matrix(
        diff_top,
        "Difference: Learned - Initial Top Prototype",
        "diff_learned_minus_initial_top_prototype",
        show_stats=True,
        set_zero_diag=False
    )
    # Plot mean reconstructed QFI matrices
    logger.info("Plotting mean reconstructed QFI matrices...")
    mean_qcd_reconstructed = reconstructed_qfi[qcd_mask].mean(axis=0)
    mean_top_reconstructed = reconstructed_qfi[top_mask].mean(axis=0)
    
    plotter.plot_qfi_matrix(
        mean_qcd_reconstructed,
        "Mean Reconstructed QCD QFI Matrix",
        "mean_reconstructed_qcd_qfi",
        show_stats=True,
        set_zero_diag=False
    )
    
    plotter.plot_qfi_matrix(
        mean_top_reconstructed,
        "Mean Reconstructed Top QFI Matrix",
        "mean_reconstructed_top_qfi",
        show_stats=True,
        set_zero_diag=False
    )
    # plot differences between top and qcd for all 3 cases
    logger.info("Plotting difference between Top and QCD mean QFI matrices...")
    diff_original = mean_top_original - mean_qcd_original
    diff_reconstructed = mean_top_reconstructed - mean_qcd_reconstructed
    diff_prototype = top_prototype - qcd_prototype
    plotter.plot_qfi_matrix(
        diff_original,
        "Difference: Mean Original Top - QCD QFI Matrix",
        "diff_mean_original_top_minus_qcd_qfi",
        show_stats=True,
        set_zero_diag=False
    )
    plotter.plot_qfi_matrix(
        diff_reconstructed,
        "Difference: Mean Reconstructed Top - QCD QFI Matrix",
        "diff_mean_reconstructed_top_minus_qcd_qfi",
        show_stats=True,
        set_zero_diag=False
    )
    plotter.plot_qfi_matrix(
        diff_prototype,
        "Difference: Learned Top - QCD Prototype",
        "diff_learned_top_minus_qcd_prototype",
        show_stats=True,
        set_zero_diag=False
    )

    # Final summary
    logger.info("=" * 60)
    logger.info("TESTING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Test Loss: {avg_loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    logger.info(f"Test AUC: {auc_score:.4f}")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Plots saved to: {plots_dir}")
    if not args.skip:
        logger.info(f"Inference data saved to: {output_h5_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()