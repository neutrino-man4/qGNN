"""
Author: Aritra Bal, ETP
Date: ante diem sextum Nonas Ianuarias anno ab urbe condita MMDCCLXXVIII

Test script for QFI-based Jet GNN classification with class-separated QFI matrix analysis.
Loads trained model, evaluates on test data, and visualizes QFI matrices by jet class.
"""
import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.metrics import roc_curve, auc, confusion_matrix
from loguru import logger
import tqdm
import time

# Import project modules
sys.path.append('.')
from configs.config import load_config
from data_utils.graph_dataloader import StreamingJetDataLoader
from src.gnn import create_qfi_jet_gnn, count_parameters


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
                       save_name: str, show_stats: bool = True, set_zero_diag:bool = False) -> None:
        """
        Plot QFI matrix with quantum circuit style formatting.
        
        Args:
            qfi_matrix: Array of shape (30, 30) - QFI matrix
            plot_label: Label for the plot title
            save_name: Filename for saving (without extension)
            show_stats: Whether to print matrix statistics
        """
        save_path = self.output_dir / save_name
        
        # Get matrix dimensions
        N_params = qfi_matrix.shape[0]  # Should be 30
        N_qubits = N_params // 3        # Should be 10
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Color scheme  
        colors = ['#0066FF', 'white', '#FF0066']  # Blue -> White -> Red
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue_white_red', colors, N=256)
        # diag_vals= np.diag(qfi_matrix)
        # diag_max = max(abs(diag_vals.min()), abs(diag_vals.max()))
        # diag_vals = diag_vals/diag_max if diag_max>0 else diag_vals
        # np.fill_diagonal(qfi_matrix, diag_vals)
        # Set up normalization
        if set_zero_diag:
            np.fill_diagonal(qfi_matrix, 0.0)
        data_range = max(abs(qfi_matrix.min()), abs(qfi_matrix.max()))
        
        if data_range == 0:
            data_range = 1  # Avoid division by zero
        if data_range < 0.5:
            data_range = 0.25 
        norm = matplotlib.colors.Normalize(vmin=-data_range, vmax=data_range)
        
        # Plot the matrix
        im = ax.matshow(qfi_matrix, cmap=cmap, norm=norm)
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
    
    def plot_class_separated_matrices(self, original_qfi: np.ndarray, reconstructed_qfi: np.ndarray,
                                    labels: np.ndarray) -> None:
        """
        Plot class-separated average QFI matrices and differences.
        
        Args:
            original_qfi: Original QFI matrices [N, 30, 30]
            reconstructed_qfi: Reconstructed QFI matrices [N, 30, 30]
            labels: Class labels [N] (0=QCD, 1=TTbar)
        """
        # Separate by class
        qcd_mask = labels == 0
        ttbar_mask = labels == 1
        
        logger.info(f"Class separation: {np.sum(qcd_mask)} QCD, {np.sum(ttbar_mask)} TTbar jets")
        
        if not np.any(qcd_mask) or not np.any(ttbar_mask):
            logger.warning("Missing one or both jet classes, skipping class-separated analysis")
            return
        
        # Compute class averages
        qcd_original_avg = np.mean(original_qfi[qcd_mask], axis=0)
        ttbar_original_avg = np.mean(original_qfi[ttbar_mask], axis=0)
        qcd_recon_avg = np.mean(reconstructed_qfi[qcd_mask], axis=0)
        ttbar_recon_avg = np.mean(reconstructed_qfi[ttbar_mask], axis=0)
        # Plot 4 average matrices
        #import pdb;pdb.set_trace()
        self.plot_qfi_matrix(
            qcd_original_avg,
            "Average Original QFI Matrix (QCD Jets)",
            "avg_original_qfi_qcd",
            show_stats=True
        )
        
        self.plot_qfi_matrix(
            ttbar_original_avg,
            "Average Original QFI Matrix (TTbar Jets)", 
            "avg_original_qfi_ttbar",
            show_stats=True
        )
        
        self.plot_qfi_matrix(
            qcd_recon_avg,
            "Average Reconstructed QFI Matrix (QCD Jets)",
            "avg_reconstructed_qfi_qcd",
            show_stats=True, set_zero_diag=True
        )
        
        self.plot_qfi_matrix(
            ttbar_recon_avg,
            "Average Reconstructed QFI Matrix (TTbar Jets)",
            "avg_reconstructed_qfi_ttbar", 
            show_stats=True, set_zero_diag=True
        )
        
        # Compute and plot class differences
        original_diff = ttbar_original_avg - qcd_original_avg
        recon_diff = ttbar_recon_avg - qcd_recon_avg
        
        self.plot_qfi_matrix(
            original_diff,
            "Class Difference in Original QFI (TTbar - QCD)",
            "class_difference_original_qfi",
            show_stats=True, set_zero_diag=True
        )
        
        self.plot_qfi_matrix(
            recon_diff,
            "Class Difference in Reconstructed QFI (TTbar - QCD)",
            "class_difference_reconstructed_qfi",
            show_stats=True, set_zero_diag=True
        )
        
        logger.success("Class-separated QFI analysis completed")
    
    def plot_log_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc_value: float,
                          model_label: str = "QFI GNN") -> None:
        """
        Plot log-scale ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc_value: AUC value
            model_label: Label for the model
        """
        plt.figure(figsize=(10, 8))
        
        # Avoid division by zero in 1/fpr
        fpr_safe = np.maximum(fpr, 1e-6)
        background_rejection = 1.0 / fpr_safe
        
        label = f"{model_label} (AUC = {auc_value:.3f})"
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
        png_path = self.output_dir / f"roc_curve.png"
        pdf_path = self.output_dir / f"roc_curve.pdf"
        
        plt.savefig(png_path, dpi=600, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        
        logger.success(f"Log ROC curve saved to {png_path} and {pdf_path}")


def create_qfi_dataloader(config: Any, max_samples: Optional[int] = None) -> StreamingJetDataLoader:
    """
    Create QFI test dataloader.
    
    Args:
        config: Configuration object
        max_samples: Maximum number of samples to process (None for all)
        
    Returns:
        QFI test dataloader
    """
    logger.info("Creating QFI dataloader...")
    
    files = config.data.test_files
    if max_samples:
        logger.info(f"Limiting to approximately {max_samples} samples")
    
    loader = StreamingJetDataLoader(
        h5_files=files,
        batch_size=config.testing.batch_size,
        use_qfi_correlations=config.data.use_qfi_correlations
    )
    
    logger.success(f"QFI dataloader created with {len(loader)} batches")
    return loader


def run_qfi_inference_and_analysis(model: torch.nn.Module, loader: StreamingJetDataLoader,
                                 device: torch.device, max_batches: Optional[int] = None) -> Dict[str, Any]:
    """
    Run inference and collect QFI reconstruction analysis.
    
    Args:
        model: Trained QFI GNN model
        loader: Data loader
        device: Device to run on
        max_batches: Maximum batches to process (None for all)
        
    Returns:
        Dictionary with inference results and QFI data
    """
    model.eval()
    
    # Storage for results
    all_predictions = []
    all_probabilities = []
    all_labels = []
    original_qfi_matrices = []
    reconstructed_qfi_matrices = []
    
    logger.info("Running QFI inference and reconstruction analysis...")
    
    batch_count = 0
    max_batches = max_batches or len(loader)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="QFI Inference", total=min(max_batches, len(loader))):
            if batch_count >= max_batches:
                break
                
            batch = batch.to(device)
            
            # Forward pass for classification
            logits = model(batch)
            predictions = torch.argmax(logits, dim=1)
            probabilities = F.softmax(logits, dim=1)[:, 1]
            
            # Store classification results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            # QFI reconstruction
            try:
                reconstructed_qfi = model.reconstruct_qfi_matrices(batch)
                original_qfi = reconstruct_original_qfi_from_batch(batch)
                # Store QFI matrices (convert to CPU numpy)
                original_qfi_matrices.append(original_qfi.cpu().numpy())
                reconstructed_qfi_matrices.append(reconstructed_qfi.cpu().numpy())
                
            except Exception as e:
                logger.warning(f"QFI reconstruction failed for batch {batch_count}: {e}")
            
            batch_count += 1
    
    # Concatenate QFI matrices
    if original_qfi_matrices:
        original_qfi_matrices = np.concatenate(original_qfi_matrices, axis=0)
        reconstructed_qfi_matrices = np.concatenate(reconstructed_qfi_matrices, axis=0)
        logger.info(f"Collected QFI matrices: {original_qfi_matrices.shape}")
    else:
        logger.error("No QFI matrices were successfully reconstructed!")
    
    return {
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'labels': np.array(all_labels),
        'original_qfi': original_qfi_matrices,
        'reconstructed_qfi': reconstructed_qfi_matrices,
        'batches_processed': batch_count
    }


def reconstruct_original_qfi_from_batch(batch) -> torch.Tensor:
    """
    Reconstruct original QFI matrices from batch input data.
    
    Args:
        batch: PyG batch object
        
    Returns:
        Original QFI matrices [batch_size, 30, 30]
    """
    batch_size = int(batch.batch.max().item()) + 1
    device = batch.x.device
    num_nodes = 30
    
    # Vectorized diagonal reconstruction
    x_reshaped = batch.x.view(batch_size, num_nodes).squeeze(-1)  # [batch_size, 30]
    qfi_matrices = torch.diag_embed(x_reshaped)  # [batch_size, 30, 30]
    
    # Vectorized off-diagonal reconstruction
    edge_batch = batch.batch[batch.edge_index[0]]  # [num_edges]
    src_local = batch.edge_index[0] % num_nodes  # [num_edges]
    tgt_local = batch.edge_index[1] % num_nodes  # [num_edges]
    #import pdb;pdb.set_trace()
    # Set all off-diagonal elements at once using advanced indexing
    qfi_matrices[edge_batch, src_local, tgt_local] = batch.edge_attr.squeeze(-1)
    return qfi_matrices


def calculate_classification_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate standard classification metrics."""
    y_true = results['labels']
    y_pred = results['predictions'] 
    y_prob = results['probabilities']
    
    # Basic metrics
    accuracy = np.mean(y_true == y_pred)
    
    # ROC AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'precision': precision, 
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'confusion_matrix': cm
    }


def calculate_qfi_reconstruction_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate QFI reconstruction quality metrics."""
    if 'original_qfi' not in results or 'reconstructed_qfi' not in results:
        logger.warning("No QFI matrices available for reconstruction analysis")
        return {}
    
    original = results['original_qfi']
    reconstructed = results['reconstructed_qfi']
    
    # Flatten matrices for comparison
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()
    
    # Compute metrics
    mse = np.mean((orig_flat - recon_flat) ** 2)
    mae = np.mean(np.abs(orig_flat - recon_flat))
    max_abs_error = np.max(np.abs(orig_flat - recon_flat))
    
    # Correlation coefficient
    corr_coef = np.corrcoef(orig_flat, recon_flat)[0, 1]
    
    return {
        'qfi_mse': mse,
        'qfi_mae': mae, 
        'qfi_max_abs_error': max_abs_error,
        'qfi_correlation': corr_coef
    }


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(
        description="Analyze QFI-based Jet GNN with class-separated QFI reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--experiment-dir", 
        type=str, 
        required=True,
        help="Path to experiment directory containing trained QFI GNN model"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (None for all)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info("Loading QFI GNN configuration...")
        config_path = os.path.join(args.experiment_dir, "config.yaml")
        config = load_config(config_path)
        logger.success(f"Configuration loaded from {config_path}")
        
        # Setup device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        
        # Create results directory
        results_dir = os.path.join(args.experiment_dir, 'qfi_analysis')
        Path(results_dir).mkdir(exist_ok=True)
        logger.info(f"QFI analysis results will be saved to: {results_dir}")
        
        # Create dataloader
        loader = create_qfi_dataloader(config)
        
        # Create QFI GNN model
        logger.info("Creating QFI GNN model...")
        model = create_qfi_jet_gnn(
            message_type=config.model.type,
            num_layers=config.model.num_mp_layers,
            mp_mlp_layers=getattr(config.model, 'mp_hidden_layers', [16, 8]),
            classifier_layers=config.model.classifier_hidden_layers,
            pooling=config.model.pooling,
            aggr=getattr(config.model, 'aggregation', 'add'),
            activation=getattr(config.model, 'activation', 'elu'),
            extra_kwargs=config.model if config.model.type.lower()=='gat' else None
        )
        model = model.to(device)
        
        # Load trained model
        best_model_path = os.path.join(args.experiment_dir, "checkpoints", "best_model.pth")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model not found at: {best_model_path}")
        
        logger.info(f"Loading trained QFI GNN from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.success("QFI GNN model loaded successfully")
        
        # Log model info
        param_info = count_parameters(model)
        logger.info(f"Model parameters: {param_info['trainable_parameters']:,}")
        
        # Run inference and QFI analysis
        results = run_qfi_inference_and_analysis(
            model, loader, device, args.max_batches
        )
        
        # Calculate classification metrics
        logger.info("Calculating classification metrics...")
        classification_metrics = calculate_classification_metrics(results)
        
        # Calculate QFI reconstruction metrics
        logger.info("Calculating QFI reconstruction metrics...")
        qfi_metrics = calculate_qfi_reconstruction_metrics(results)
        
        # Combine all metrics
        all_metrics = {**classification_metrics, **qfi_metrics}
        
        # Log results
        logger.info("=" * 60)
        logger.info("QFI GNN ANALYSIS RESULTS")
        logger.info("=" * 60)
        logger.info("Classification Performance:")
        logger.info(f"  Accuracy: {classification_metrics['accuracy']:.4f}")
        logger.info(f"  AUC: {classification_metrics['auc']:.4f}")
        logger.info(f"  Precision: {classification_metrics['precision']:.4f}")
        logger.info(f"  Recall: {classification_metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {classification_metrics['f1_score']:.4f}")
        
        if qfi_metrics:
            logger.info("QFI Reconstruction Quality:")
            logger.info(f"  MSE: {qfi_metrics['qfi_mse']:.6f}")
            logger.info(f"  MAE: {qfi_metrics['qfi_mae']:.6f}")
            logger.info(f"  Max Abs Error: {qfi_metrics['qfi_max_abs_error']:.6f}")
            logger.info(f"  Correlation: {qfi_metrics['qfi_correlation']:.4f}")
        logger.info("=" * 60)
        
        # Create QFI visualizations
        if 'original_qfi' in results and len(results['original_qfi']) > 0:
            logger.info("Creating class-separated QFI matrix visualizations...")
            plotter = QFIMatrixPlotter(results_dir)
            
            # Plot class-separated average matrices and differences
            plotter.plot_class_separated_matrices(
                results['original_qfi'], 
                results['reconstructed_qfi'],
                results['labels']
            )
            
            # Plot log ROC curve
            logger.info("Creating log-scale ROC curve...")
            plotter.plot_log_roc_curve(
                classification_metrics['fpr'],
                classification_metrics['tpr'],
                classification_metrics['auc'],
                "QFI GNN"
            )
            
            logger.success("QFI visualizations completed")
        
        # Save metrics to files
        metrics_path = os.path.join(results_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_metrics = {}
            for key, value in all_metrics.items():
                if isinstance(value, np.ndarray):
                    json_metrics[key] = value.tolist()
                elif isinstance(value, np.floating):
                    json_metrics[key] = float(value)
                else:
                    json_metrics[key] = value
            json.dump(json_metrics, f, indent=2)
        
        # Save metrics to NPZ for plotting
        metrics_npz_path = os.path.join(results_dir, "metrics.npz")
        np.savez(metrics_npz_path, **{k: v for k, v in all_metrics.items() if isinstance(v, np.ndarray)})
        
        logger.success("QFI GNN analysis completed successfully!")
        logger.info(f"All results saved in: {results_dir}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"QFI GNN analysis failed: {e}")
        logger.exception("Full traceback:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())