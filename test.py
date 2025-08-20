#!/usr/bin/env python3
"""
Author: Aritra Bal, ETP
Date: XVIII Sextilis anno ab urbe condita MMDCCLXXVIII

Test script for Jet GNN classification.
Loads trained model and evaluates on test data with ROC and SIC analysis.
Memory-efficient implementation with vectorized processing.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
import h5py
import matplotlib.pyplot as plt
#import mplhep as hep
from sklearn.metrics import roc_curve, auc, confusion_matrix
from torch_geometric.data import Data, Batch
from loguru import logger
import tqdm

# Import project modules
sys.path.append('.')
from configs.config import load_config
from src.gnn import create_jet_gnn


def load_test_data_vectorized(test_files: List[str], test_n: int, use_qfi_correlations: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load exactly test_n jets from test files using vectorized operations.
    
    Args:
        test_files: List of test file paths
        test_n: Number of jets to load
        use_qfi_correlations: Whether to use QFI correlations
        
    Returns:
        Tuple of (node_features, edge_features, labels, file_indices, jet_indices)
    """
    logger.info(f"Loading {test_n} test jets from {len(test_files)} files...")
    
    # Create edge index (fully connected graph)
    max_particles = 10
    sources = torch.arange(max_particles).repeat(max_particles)
    targets = torch.arange(max_particles).repeat_interleave(max_particles)
    edge_index = torch.stack([sources, targets], dim=0)  # [2, 100]
    
    all_node_features = []
    all_edge_features = []
    all_labels = []
    all_file_indices = []
    all_jet_indices = []
    jets_loaded = 0
    
    for file_idx, file_path in enumerate(test_files):
        if jets_loaded >= test_n:
            break
            
        logger.info(f"Reading file {file_idx + 1}/{len(test_files)}: {os.path.basename(file_path)}")
        
        with h5py.File(file_path, 'r') as f:
            file_size = f['truth_labels'].shape[0]
            jets_to_read = min(test_n - jets_loaded, file_size)
            
            # Read data in bulk
            node_features = f['jetConstituentsList'][:jets_to_read]  # [jets_to_read, 10, 3]
            qfi_matrices = f['jetConstituentsQFI'][:jets_to_read]    # [jets_to_read, 30, 30]
            file_labels = f['truth_labels'][:jets_to_read]          # [jets_to_read]
            jet_pts = f['jetFeatures'][:jets_to_read, 0]            # [jets_to_read]
            
            # Normalize particle pt by jet pt (vectorized)
            node_features = node_features.copy()
            node_features[..., 0] = node_features[..., 0] / jet_pts[:, None]
            
            # Extract edge features in vectorized way
            if use_qfi_correlations:
                edge_features = _extract_edge_features_vectorized(qfi_matrices, edge_index)  # [jets_to_read, 100, 9]
            else:
                # Identity baseline for all jets
                identity_flat = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
                edge_features = np.tile(identity_flat, (jets_to_read, 100, 1))  # [jets_to_read, 100, 9]
            
            # Store data
            all_node_features.append(node_features)
            all_edge_features.append(edge_features)
            all_labels.append(file_labels)
            all_file_indices.extend([file_idx] * jets_to_read)
            all_jet_indices.extend(list(range(jets_to_read)))
            
            jets_loaded += jets_to_read
            
            if jets_loaded >= test_n:
                break
    
    # Concatenate all data
    node_features = np.concatenate(all_node_features, axis=0)  # [test_n, 10, 3]
    edge_features = np.concatenate(all_edge_features, axis=0)  # [test_n, 100, 9]
    labels = np.concatenate(all_labels, axis=0)               # [test_n]
    file_indices = np.array(all_file_indices)                 # [test_n]
    jet_indices = np.array(all_jet_indices)                   # [test_n]
    
    logger.success(f"Loaded {jets_loaded} test jets")
    logger.info(f"Label distribution: {np.sum(labels == 0)} QCD (0), {np.sum(labels == 1)} TTbar (1)")
    logger.info(f"Data shapes: nodes {node_features.shape}, edges {edge_features.shape}")
    
    return node_features, edge_features, labels, file_indices, jet_indices


def _extract_edge_features_vectorized(qfi_matrices: np.ndarray, edge_index: torch.Tensor) -> np.ndarray:
    """Extract 3x3 submatrices from QFI correlation matrices for all jets at once."""
    n_jets = qfi_matrices.shape[0]
    src_nodes = edge_index[0].numpy()  # [100]
    dst_nodes = edge_index[1].numpy()  # [100]
    
    # Vectorized extraction for all jets simultaneously
    src_base = src_nodes * 3  # [100]
    dst_base = dst_nodes * 3  # [100]
    
    # Create indices for 3x3 submatrix extraction
    row_offsets = np.tile([0, 0, 0, 1, 1, 1, 2, 2, 2], len(src_nodes))  # [900]
    col_offsets = np.tile([0, 1, 2, 0, 1, 2, 0, 1, 2], len(dst_nodes))  # [900]
    
    row_indices = np.repeat(src_base, 9) + row_offsets  # [900]
    col_indices = np.repeat(dst_base, 9) + col_offsets  # [900]
    
    # Extract elements for all jets at once using advanced indexing
    # qfi_matrices[:, row_indices, col_indices] gives [n_jets, 900]
    extracted = qfi_matrices[:, row_indices, col_indices]  # [n_jets, 900]
    
    # Reshape to [n_jets, 100, 9]
    edge_features = extracted.reshape(n_jets, 100, 9)
    
    return edge_features


def create_data_loader_from_arrays(node_features: np.ndarray, edge_features: np.ndarray, 
                                 labels: np.ndarray, batch_size: int) -> List[Batch]:
    """Create batches from numpy arrays."""
    n_jets = len(labels)
    
    # Create edge index (same for all graphs)
    max_particles = 10
    sources = torch.arange(max_particles).repeat(max_particles)
    targets = torch.arange(max_particles).repeat_interleave(max_particles)
    edge_index = torch.stack([sources, targets], dim=0)  # [2, 100]
    
    batches = []
    for i in range(0, n_jets, batch_size):
        end_idx = min(i + batch_size, n_jets)
        batch_size_actual = end_idx - i
        
        data_list = []
        for j in range(i, end_idx):
            x = torch.tensor(node_features[j], dtype=torch.float32)      # [10, 3]
            edge_attr = torch.tensor(edge_features[j], dtype=torch.float32)  # [100, 9]
            y = torch.tensor(labels[j], dtype=torch.long)
            
            data_list.append(Data(
                x=x,
                edge_index=edge_index.clone(),
                edge_attr=edge_attr,
                y=y
            ))
        
        batches.append(Batch.from_data_list(data_list))
    
    return batches


def run_inference_vectorized(model: torch.nn.Module, batches: List[Batch], 
                           device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on pre-created batches."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    logger.info(f"Running inference on {len(batches)} batches...")
    
    with torch.no_grad():
        for batch in tqdm.tqdm(batches, desc="Inference"):
            batch = batch.to(device)
            
            # Forward pass
            logits = model(batch)
            predictions = torch.argmax(logits, dim=1)
            probabilities = F.softmax(logits, dim=1)[:, 1]  # Probability of TTbar class
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)


def save_predictions_h5(test_files: List[str], file_indices: np.ndarray, jet_indices: np.ndarray,
                       true_labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray,
                       results_dir: Path):
    """Save predictions to H5 files with naming convention."""
    logger.info("Saving predictions to H5 files...")
    
    # Group predictions by file
    unique_file_indices = np.unique(file_indices)
    
    for file_idx in unique_file_indices:
        # Get mask for this file
        file_mask = file_indices == file_idx
        
        # Extract data for this file
        file_true_labels = true_labels[file_mask]
        file_predictions = predictions[file_mask]
        file_probabilities = probabilities[file_mask]
        file_jet_indices = jet_indices[file_mask]
        
        # Generate output filename
        input_path = test_files[file_idx]
        input_dir = os.path.dirname(input_path)
        input_filename = os.path.basename(input_path)
        
        # Replace directory: test -> inferred
        output_dir = input_dir.replace('/test', '/inferred')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Replace filename pattern
        if 'TTBar+ZJets' in input_filename:
            # TTBar+ZJets_124.h5 -> TTBar+ZJets_infer_124.h5
            output_filename = input_filename.replace('TTBar+ZJets_', 'TTBar+ZJets_infer_')
        else:
            # Fallback: just add _infer before .h5
            output_filename = input_filename.replace('.h5', '_infer.h5')
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Save to H5 file
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('jet_indices', data=file_jet_indices, compression='gzip')
            f.create_dataset('true_labels', data=file_true_labels, compression='gzip')
            f.create_dataset('predicted_labels', data=file_predictions, compression='gzip')
            f.create_dataset('ttbar_probabilities', data=file_probabilities, compression='gzip')
            
            # Add metadata
            f.attrs['input_file'] = input_path
            f.attrs['n_jets'] = len(file_true_labels)
            f.attrs['n_ttbar'] = np.sum(file_true_labels == 1)
            f.attrs['n_qcd'] = np.sum(file_true_labels == 0)
        
        logger.info(f"Saved {len(file_true_labels)} predictions to: {output_path}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate classification metrics."""
    # Basic metrics
    accuracy = np.mean(y_true == y_pred)
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str, metrics: Dict[str, float]):
    """Plot ROC curve with CMS style."""
    #plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = metrics['auc']
    
    # Plot ROC curve
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    # Formatting
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Jet Classification')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add metrics text
    textstr = f'Accuracy: {metrics["accuracy"]:.3f}\nPrecision: {metrics["precision"]:.3f}\nRecall: {metrics["recall"]:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved to: {save_path}")


def plot_sic_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str, metrics: Dict[str, float]):
    """Plot SIC (Significance Improvement Characteristic) curve with CMS style."""
    #plt.style.use(hep.style.CMS)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate SIC curve (Signal efficiency vs Background rejection)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    background_rejection = 1 - fpr  # Background rejection = 1 - False Positive Rate
    signal_efficiency = tpr         # Signal efficiency = True Positive Rate
    
    # Plot SIC curve
    ax.plot(signal_efficiency, background_rejection, linewidth=2, 
            label=f'SIC Curve (AUC = {metrics["auc"]:.3f})')
    
    # Reference line (random classifier)
    ax.plot([0, 1], [1, 0], 'k--', alpha=0.5, label='Random Classifier')
    
    # Formatting
    ax.set_xlabel('Signal Efficiency (TTbar)')
    ax.set_ylabel('Background Rejection (1 - QCD FPR)')
    ax.set_title('SIC Curve - Jet Classification')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add working point markers for common signal efficiencies
    for eff in [0.5, 0.7, 0.9]:
        # Find closest point
        idx = np.argmin(np.abs(signal_efficiency - eff))
        if idx < len(background_rejection):
            ax.plot(signal_efficiency[idx], background_rejection[idx], 'ro', markersize=6)
            ax.annotate(f'ε_s={eff:.1f}\nR_b={background_rejection[idx]:.3f}', 
                       xy=(signal_efficiency[idx], background_rejection[idx]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"SIC curve saved to: {save_path}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(
        description="Test Jet GNN classification model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="./configs/base.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--experiment-dir", 
        type=str, 
        required=True,
        help="Path to experiment directory containing the trained model"
    )
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
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
        results_dir = os.path.join(args.experiment_dir,'results')
        Path(results_dir).mkdir(exist_ok=True)
        logger.info(f"Results will be saved to: {results_dir}")
        
        # Load test data (vectorized)
        node_features, edge_features, true_labels, file_indices, jet_indices = load_test_data_vectorized(
            config.data.test_files,
            config.data.test_n,
            config.data.use_qfi_correlations
        )
        
        # Create model
        logger.info("Creating model...")
        model = create_jet_gnn(
            model_type=config.model.type,
            num_layers=config.model.num_mp_layers,
            mp_hidden_layers=config.model.mp_hidden_layers,
            classifier_hidden_layers=config.model.classifier_hidden_layers,
            pooling=config.model.pooling,
            activation=config.model.activation,
            residual=config.model.residual_connections
        )
        model = model.to(device)
        
        # Load best model checkpoint
        best_model_path = os.path.join(args.experiment_dir, "checkpoints", "best_model.pth")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model not found at: {best_model_path}")
        
        logger.info(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.success("Model loaded successfully")
        
        # Create batches from arrays
        logger.info("Creating data batches...")
        batches = create_data_loader_from_arrays(
            node_features, edge_features, true_labels, config.testing.batch_size
        )
        
        # Run inference
        predictions, probabilities = run_inference_vectorized(model, batches, device)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = calculate_metrics(true_labels, predictions, probabilities)
        
        # Log results
        logger.info("=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                logger.info(f"{metric_name:20s}: {metric_value:.4f}")
            else:
                logger.info(f"{metric_name:20s}: {metric_value}")
        logger.info("=" * 60)
        
        # Save metrics to JSON
        metrics_path = os.path.join(results_dir, "test_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # Plot and save ROC curve
        roc_path = os.path.join(results_dir, "roc_curve.png")
        plot_roc_curve(true_labels, probabilities, roc_path, metrics)
        
        # Plot and save SIC curve
        sic_path = os.path.join(results_dir, "sic_curve.png")
        plot_sic_curve(true_labels, probabilities, sic_path, metrics)
        
        # Save predictions to H5 files
        if config.testing.save_predictions:
            save_predictions_h5(
                config.data.test_files, file_indices, jet_indices,
                true_labels, predictions, probabilities, results_dir
            )
        
        logger.success("Testing completed successfully!")
        logger.info(f"All results saved in: {results_dir}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        logger.exception("Full traceback:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())