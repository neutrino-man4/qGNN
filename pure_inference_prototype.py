"""
Author: Aritra Bal, ETP
Date: ante diem octavum Idus Octobres anno ab urbe condita MMDCCLXXVIII

Inference-only script using class mean QFI matrices as prototypes.
No training - pure geometric classification based on distances to class centroids.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Tuple
import yaml
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import roc_curve, roc_auc_score
from datetime import datetime
import sys

# Import dataset from training script
from train_prototype import QFIDataset


class QFIMatrixPlotter:
    """QFI matrix plotting utility."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"QFI plotter initialized, saving to: {output_dir}")
    
    def plot_qfi_matrix(self, qfi_matrix: np.ndarray, plot_label: str, 
                       save_name: str, show_stats: bool = True, set_zero_diag: bool = False) -> None:
        """Plot QFI matrix with quantum circuit style formatting."""
        save_path = self.output_dir / save_name
        
        N_params = qfi_matrix.shape[0]
        N_qubits = N_params // 3
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        colors = ['#0066FF', 'white', '#FF0066']
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue_white_red', colors, N=256)
        
        plot_matrix = qfi_matrix.copy()
        if set_zero_diag:
            np.fill_diagonal(plot_matrix, 0.0)
        data_range = 1.5
        
        if data_range == 0:
            data_range = 1
        if data_range < 0.5:
            data_range = 0.25
        norm = matplotlib.colors.Normalize(vmin=-data_range, vmax=data_range)
        
        im = ax.matshow(plot_matrix, cmap=cmap, norm=norm)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('QFI Value', fontsize=12)
        
        for i in range(1, N_qubits):
            ax.axhline(y=3*i - 0.5, color='black', linewidth=2)
            ax.axvline(x=3*i - 0.5, color='black', linewidth=2)
        
        rotation_positions = [i+0.5 for i in range(3*N_qubits)]
        ax.set_xticks(rotation_positions)
        ax.set_yticks(rotation_positions)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        rotation_labels = ['$R_Z$', '$R_Y$', '$R_X$'] * N_qubits
        label_positions = [i for i in range(3*N_qubits)]
        
        for pos, label in zip(label_positions, rotation_labels):
            ax.text(pos, 3*N_qubits+0.1, label, ha='center', va='top', fontsize=10)
            ax.text(-0.7, pos, label, ha='right', va='center', fontsize=10)
        
        qubit_positions = [1 + 3*i for i in range(N_qubits)]
        qubit_labels = [str(i) for i in range(N_qubits)]
        
        for pos, label in zip(qubit_positions, qubit_labels):
            ax.text(pos, -0.3, label, ha='center', va='top', fontsize=14, fontweight='bold')
            ax.text(-1.4, pos, label, ha='right', va='center', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Qubit Number', labelpad=40, fontsize=14)
        ax.set_ylabel('Qubit Number', labelpad=40, fontsize=14)
        plt.title(plot_label, fontsize=18, pad=20)
        plt.tight_layout()
        
        plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
        plt.close()
        
        if show_stats:
            logger.info(f"QFI Matrix Statistics for {plot_label}:")
            logger.info(f"  Range: [{qfi_matrix.min():.6f}, {qfi_matrix.max():.6f}]")
            logger.info(f"  Mean: {qfi_matrix.mean():.6f}")
            logger.info(f"  Std:  {qfi_matrix.std():.6f}")
            logger.info(f"  Saved to: {save_path}.png/.pdf")


def plot_log_roc_curves(fpr: np.ndarray, tpr: np.ndarray, auc_value: float, 
                       save_dir: str, timestamp: str):
    """Plot log ROC curve."""
    plt.figure(figsize=(10, 8))
    
    logger.info("Creating log ROC curve...")
    
    fpr_safe = np.maximum(fpr, 1e-6)
    background_rejection = 1.0 / fpr_safe
    
    label = f"Mean Prototype Classifier (AUC = {auc_value:.3f})"
    plt.plot(tpr, background_rejection, linewidth=2, label=label)
    
    base_tpr = np.arange(0.001, 1.0, 0.01)
    base_fpr = base_tpr
    plt.plot(base_tpr, 1.0/base_fpr, linewidth=2, label="Baseline (AUC = 0.5)", 
             linestyle='--', color='black', alpha=0.7)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([1.0, 1.0e6])
    plt.yscale('log')
    plt.xlabel('Signal Efficiency (TPR)', fontsize=17)
    plt.ylabel('Background Rejection (FPR$^{-1}$)', fontsize=17)
    plt.title('Jet Classification Performance - Mean Prototypes', fontsize=19, fontweight='bold')
    plt.legend(loc="upper right", fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    png_path = Path(save_dir) / f"AUC_mean_prototypes_{timestamp}.png"
    pdf_path = Path(save_dir) / f"AUC_mean_prototypes_{timestamp}.pdf"
    
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    logger.success(f"Log ROC curve saved to {png_path} and {pdf_path}")
    plt.close()


def reconstruct_symmetric_matrix(upper_triangle: np.ndarray, matrix_size: int = 30) -> np.ndarray:
    """Reconstruct symmetric matrix from upper triangle including diagonal."""
    matrix = np.zeros((matrix_size, matrix_size))
    triu_indices = np.triu_indices(matrix_size, k=0)
    matrix[triu_indices[0], triu_indices[1]] = upper_triangle
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    return matrix


def compute_class_means(dataset: QFIDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean QFI for each class.
    
    Args:
        dataset: QFI dataset
        
    Returns:
        Tuple of (mean_qcd, mean_top) arrays of shape [feat_dim]
    """
    logger.info("Computing class mean prototypes...")
    
    qcd_mask = dataset.labels == 0
    top_mask = dataset.labels == 1
    
    mean_qcd = np.mean(dataset.qfi_matrices[qcd_mask], axis=0)
    mean_top = np.mean(dataset.qfi_matrices[top_mask], axis=0)
    
    logger.info(f"QCD samples: {qcd_mask.sum()}, Top samples: {top_mask.sum()}")
    logger.success("Class means computed")
    
    return mean_qcd, mean_top


def classify_with_prototypes(test_data: np.ndarray, 
                            prototype_qcd: np.ndarray, 
                            prototype_top: np.ndarray,
                            test_labels: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify test data based on distance to prototypes.
    
    Args:
        test_data: Test QFI data [N, feat_dim]
        prototype_qcd: QCD prototype [feat_dim]
        prototype_top: Top prototype [feat_dim]
        test_labels: True labels [N]
        
    Returns:
        Tuple of (accuracy, auc, fpr, tpr, thresholds)
    """
    logger.info("Computing distances to prototypes...")
    
    # Compute squared L2 distances
    dist_to_qcd = np.sum((test_data - prototype_qcd) ** 2, axis=1)
    dist_to_top = np.sum((test_data - prototype_top) ** 2, axis=1)
    
    # Classify: label=1 (Top) if closer to Top prototype
    predictions = (dist_to_qcd > dist_to_top).astype(int)
    
    # Accuracy
    correct = np.sum(predictions == test_labels)
    accuracy = 100.0 * correct / len(test_labels)
    
    # Score for AUC: higher score = more likely Top
    # Convert distance difference to probability-like score
    scores = dist_to_qcd - dist_to_top
    probs = 1.0 / (1.0 + np.exp(-scores))  # Sigmoid
    
    # AUC and ROC curve
    auc_score = roc_auc_score(test_labels, probs)
    fpr, tpr, thresholds = roc_curve(test_labels, probs)
    
    logger.success(f"Accuracy: {accuracy:.2f}%, AUC: {auc_score:.4f}")
    
    return accuracy, auc_score, fpr, tpr, thresholds


def main():
    """Main inference function."""
    
    parser = argparse.ArgumentParser(
        description="Pure inference using mean QFI prototypes (no training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get symmetric flag
    symmetric = config['model'].get('symmetric', False)
    feat_dim = 465 if symmetric else 900
    logger.info(f"Symmetric mode: {symmetric}, Feature dimension: {feat_dim}")
    
    # Setup output directory
    output_dir = Path(config['experiment']['base_save_dir'])
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    # copy config.yaml
    import shutil
    shutil.copy(args.config, output_dir / "config.yaml")
    logger.info("=" * 60)
    logger.info("MEAN PROTOTYPE INFERENCE (NO TRAINING)")
    logger.info("=" * 60)
    
    # Load training data to compute prototypes
    logger.info("Loading training data to compute class means...")
    train_dataset = QFIDataset(config['data']['train_files'], symmetric=symmetric)
    
    # Compute class mean prototypes
    mean_qcd, mean_top = compute_class_means(train_dataset)
    
    # Load test data
    logger.info("Loading test data...")
    test_dataset = QFIDataset(config['data']['test_files'], symmetric=symmetric)
    
    # Classify test data using mean prototypes
    accuracy, auc_score, fpr, tpr, thresholds = classify_with_prototypes(
        test_dataset.qfi_matrices,
        mean_qcd,
        mean_top,
        test_dataset.labels
    )
    
    # Save metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = output_dir / "metrics.npz"
    
    np.savez(
        metrics_path,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        accuracy=accuracy,
        auc=auc_score
    )
    logger.success(f"Metrics saved to {metrics_path}")
    
    # Plot ROC curve
    plot_log_roc_curves(fpr, tpr, auc_score, plots_dir, timestamp)
    
    # Reconstruct 30x30 matrices for plotting
    if symmetric:
        logger.info("Reconstructing 30x30 matrices from upper triangles...")
        mean_qcd_30x30 = reconstruct_symmetric_matrix(mean_qcd, matrix_size=30)
        mean_top_30x30 = reconstruct_symmetric_matrix(mean_top, matrix_size=30)
        
        # Reconstruct original test QFI
        original_qfi = np.zeros((len(test_dataset.qfi_matrices), 30, 30))
        for i in range(len(test_dataset.qfi_matrices)):
            original_qfi[i] = reconstruct_symmetric_matrix(test_dataset.qfi_matrices[i], matrix_size=30)
    else:
        mean_qcd_30x30 = mean_qcd.reshape(30, 30)
        mean_top_30x30 = mean_top.reshape(30, 30)
        original_qfi = test_dataset.qfi_matrices.reshape(-1, 30, 30)
    
    # Save mean prototypes
    np.savez(
        output_dir / 'mean_prototypes.npz',
        qcd_prototype=mean_qcd_30x30,
        top_prototype=mean_top_30x30
    )
    logger.info(f"Mean prototypes saved to {output_dir / 'mean_prototypes.npz'}")
    
    # Initialize plotter
    plotter = QFIMatrixPlotter(plots_dir)
    
    # Separate test QCD and Top jets
    qcd_mask = test_dataset.labels == 0
    top_mask = test_dataset.labels == 1
    
    logger.info(f"Test - QCD jets: {qcd_mask.sum()}, Top jets: {top_mask.sum()}")
    
    # Plot mean original test QFI matrices
    logger.info("Plotting mean original test QFI matrices...")
    mean_qcd_test = original_qfi[qcd_mask].mean(axis=0)
    mean_top_test = original_qfi[top_mask].mean(axis=0)
    
    plotter.plot_qfi_matrix(
        mean_qcd_test,
        "Mean Test QCD QFI Matrix",
        "mean_test_qcd_qfi",
        show_stats=True
    )
    
    plotter.plot_qfi_matrix(
        mean_top_test,
        "Mean Test Top QFI Matrix",
        "mean_test_top_qfi",
        show_stats=True
    )
    
    # Plot mean prototype matrices (computed from training data)
    logger.info("Plotting mean prototype matrices from training data...")
    
    plotter.plot_qfi_matrix(
        mean_qcd_30x30,
        "Mean QCD Prototype (from training)",
        "mean_qcd_prototype",
        show_stats=True
    )
    
    plotter.plot_qfi_matrix(
        mean_top_30x30,
        "Mean Top Prototype (from training)",
        "mean_top_prototype",
        show_stats=True
    )
    
    # Final summary
    logger.info("=" * 60)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    logger.info(f"Test AUC: {auc_score:.4f}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Plots saved to: {plots_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()