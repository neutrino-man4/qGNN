#!/usr/bin/env python3
"""
Author: Aritra Bal, ETP
Date: die Martis ante diem tertium Nonas Septembres anno ab urbe condita MMDCCLXXVIII

Inference script for fixed correlation jet classifier using pre-computed class statistics.
Performs Fisher Linear Discriminant or Mahalanobis distance classification on test data.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, Tuple, Any
import torch
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
import time
import tqdm
import mplhep as hep
# Import project modules
sys.path.append('.')
from configs.config import load_config
from data_utils.graph_dataloader import JetGraphDataloader
from src.graph_classifier import create_jet_graph

hep.style.use(hep.style.CMS)

def setup_device() -> torch.device:
    """Setup and return the appropriate device for computation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def load_and_verify_statistics(experiment_dir: str, config: Any) -> Dict[str, torch.Tensor]:
    """
    Load class statistics and verify compatibility with current config.
    
    Args:
        experiment_dir: Path to experiment directory
        config: Configuration object
        
    Returns:
        Dictionary containing class statistics
        
    Raises:
        FileNotFoundError: If statistics files are not found
        ValueError: If configuration mismatch detected
    """
    # Load statistics file
    stats_path = os.path.join(experiment_dir, "sample_statistics.pth")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")
    
    statistics = torch.load(stats_path, map_location='cpu')
    logger.info(f"Loaded statistics from: {stats_path}")
    
    # Load and verify human-readable summary
    summary_path = os.path.join(experiment_dir, "statistics_summary.json")
    if not os.path.exists(summary_path):
        logger.warning(f"Summary file not found: {summary_path}")
        logger.warning("Skipping configuration verification")
        return statistics
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Verify configuration compatibility
    saved_pooling = summary.get('pooling_method')
    saved_layers = summary.get('num_mp_layers')
    
    config_pooling = config.model.pooling
    config_layers = config.model.num_mp_layers
    
    if saved_pooling != config_pooling:
        raise ValueError(
            f"Pooling method mismatch! "
            f"Config: {config_pooling}, Saved statistics: {saved_pooling}"
        )
    
    if saved_layers != config_layers:
        raise ValueError(
            f"Number of message passing layers mismatch! "
            f"Config: {config_layers}, Saved statistics: {saved_layers}"
        )
    
    logger.success("Configuration verification passed")
    logger.info(f"Using: {saved_layers} layers, {saved_pooling} pooling")
    
    return statistics


def create_test_dataloader(config: Any) -> JetGraphDataloader:
    """
    Create JetGraphDataloader for test data with averaged QFI matrices.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured JetGraphDataloader for test files
    """
    qfi_means_path = os.path.join(config.data.stat_path, "qfi_means.npz")
    
    if not os.path.exists(qfi_means_path):
        raise FileNotFoundError(f"QFI means file not found: {qfi_means_path}")
    
    logger.info(f"Creating test dataloader with QFI means from: {qfi_means_path}")
    
    dataloader = JetGraphDataloader(
        h5_files=config.data.test_files,  # Use test files, not train files
        batch_size=config.testing.batch_size,
        use_qfi_correlations=config.data.use_qfi_correlations,
        mean_qfi_path=qfi_means_path, mode='inference'
    )
    logger.success(f"Test dataloader created with {len(dataloader)} batches")
    return dataloader


def extract_graph_features(model: torch.nn.Module, dataloader: JetGraphDataloader, 
                         device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract graph-level features and true labels from test dataset.
    
    Args:
        model: Fixed correlation model in default mode
        dataloader: Test data loader
        device: Device to run inference on
        
    Returns:
        Tuple of (features, true_labels)
    """
    model.eval()
    all_features = []
    all_labels = []
    
    logger.info("Extracting graph-level features from test data...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader,total=len(dataloader))):
            batch = batch.to(device)
            
            # Forward pass to get graph-level features
            features = model(batch)  # [batch_size, feature_dim]
            
            all_features.append(features.cpu())
            all_labels.append(batch.y.cpu())
    
    # Concatenate all batches
    features = torch.cat(all_features, dim=0)  # [total_samples, feature_dim]
    labels = torch.cat(all_labels, dim=0)      # [total_samples]
    
    logger.success(f"Extracted features: {features.shape}, labels: {labels.shape}")
    logger.info(f"Label distribution: {torch.sum(labels == 0)} QCD, {torch.sum(labels == 1)} TTbar")
    
    return features, labels


def fisher_linear_discriminant(features: torch.Tensor, ttbar_mean: torch.Tensor, 
                             qcd_mean: torch.Tensor, pooled_cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Fisher Linear Discriminant classification.
    
    Args:
        features: [N, d] - graph-level features
        ttbar_mean: [d] - TTbar class mean
        qcd_mean: [d] - QCD class mean  
        pooled_cov: [d, d] - pooled covariance matrix
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    logger.info("Applying Fisher Linear Discriminant...")
    
    # Compute inverse covariance
    inv_cov = torch.linalg.inv(pooled_cov)
    
    # Fisher weight vector: w = Σ^(-1) * (μ_ttbar - μ_qcd)
    mean_diff = ttbar_mean - qcd_mean  # [d]
    fisher_weights = inv_cov @ mean_diff  # [d]
    
    # Overall mean for centering
    overall_mean = (ttbar_mean + qcd_mean) / 2.0  # [d]
    
    # Fisher discriminant scores: w^T * (x - μ_overall)
    centered_features = features - overall_mean.unsqueeze(0)  # [N, d]
    fisher_scores = centered_features @ fisher_weights  # [N]
    
    # Convert to probabilities (positive score favors TTbar)
    probabilities = torch.sigmoid(fisher_scores)  # [N]
    predictions = (fisher_scores > 0).long()      # [N]
    
    logger.info(f"Fisher scores range: [{fisher_scores.min():.3f}, {fisher_scores.max():.3f}]")
    return predictions, probabilities


def mahalanobis_distance_classification(features: torch.Tensor, ttbar_mean: torch.Tensor,
                                      qcd_mean: torch.Tensor, pooled_cov: torch.Tensor, classical:bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Mahalanobis distance classification.
    
    Args:
        features: [N, d] - graph-level features
        ttbar_mean: [d] - TTbar class mean
        qcd_mean: [d] - QCD class mean
        pooled_cov: [d, d] - pooled covariance matrix
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    logger.info("Applying Mahalanobis distance classification...")
    
    # Compute inverse covariance
    inv_cov = torch.linalg.inv(pooled_cov)
    # Calculate squared Mahalanobis distances to each class mean
    diff_ttbar = features - ttbar_mean.unsqueeze(0)  # [N, d]
    diff_qcd = features - qcd_mean.unsqueeze(0)      # [N, d]
    if classical:
        inv_cov=inv_cov[:4,:4]
        diff_ttbar = diff_ttbar[:,:4]
        diff_qcd = diff_qcd[:,:4]
        logger.info("Using classical (non-QFI) features only")
        logger.info("We use the following QCD Means: ", qcd_mean[:4].cpu().numpy())
        logger.info("We use the following TTbar Means: ", ttbar_mean[:4].cpu().numpy())
        logger.info("We use the following Pooled Covariance: ", pooled_cov[:4,:4].cpu().numpy())
        logger.info("Rest of the features (probably features 5-7 are ignored, even if they were printed above in the dataloader.\n Take a break to process")
        time.sleep(3)
    # Vectorized Mahalanobis distance calculation
    dist_ttbar_sq = torch.sum(diff_ttbar * (diff_ttbar @ inv_cov), dim=-1)  # [N]
    dist_qcd_sq = torch.sum(diff_qcd * (diff_qcd @ inv_cov), dim=-1)        # [N]
    
    # Classify to closer mean (smaller distance)
    predictions = (dist_ttbar_sq < dist_qcd_sq).long()  # [N], 1=TTbar, 0=QCD
    
    # Convert distances to probabilities
    # Closer to TTbar mean = higher TTbar probability
    total_dist = dist_ttbar_sq + dist_qcd_sq + 1e-8  # Small epsilon for stability
    probabilities = dist_qcd_sq / total_dist          # [N]
    logger.info(f"Distance ranges - TTbar: [{dist_ttbar_sq.min():.3f}, {dist_ttbar_sq.max():.3f}]")
    logger.info(f"Distance ranges - QCD: [{dist_qcd_sq.min():.3f}, {dist_qcd_sq.max():.3f}]")
    
    return predictions, probabilities


def compute_metrics(true_labels: torch.Tensor, predictions: torch.Tensor, 
                   probabilities: torch.Tensor) -> Dict[str, Any]:
    """
    Compute classification metrics.
    
    Args:
        true_labels: [N] - true labels (0=QCD, 1=TTbar)
        predictions: [N] - predicted labels (0=QCD, 1=TTbar)  
        probabilities: [N] - TTbar class probabilities
        
    Returns:
        Dictionary containing all metrics
    """
    logger.info("Computing classification metrics...")
    
    # Convert to numpy for sklearn
    true_labels_np = true_labels.numpy()
    predictions_np = predictions.numpy()
    probabilities_np = probabilities.numpy()
    
    # Basic metrics
    accuracy = accuracy_score(true_labels_np, predictions_np)
    
    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels_np, probabilities_np)
    roc_auc = auc(fpr, tpr)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"AUC: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'true_labels': true_labels_np,
        'predictions': predictions_np,
        'probabilities': probabilities_np
    }


def plot_roc_curve(metrics: Dict[str, Any], save_path: str):
    """Plot and save ROC curve."""
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot ROC curve
    ax.plot(metrics['fpr'], metrics['tpr'], linewidth=2, 
            label=f'AUC = {metrics["auc"]:.3f} | Accuracy = {metrics["accuracy"]:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    # Formatting
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Fixed Correlation Jet Classifier')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add metrics text
    textstr = f'Accuracy: {metrics["accuracy"]:.3f}\nAUC: {metrics["auc"]:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
    #         verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved to: {save_path}")


def plot_sic_curve(metrics: Dict[str, Any], save_path: str):
    """Plot and save SIC (Significance Improvement Characteristic) curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # SIC curve: Signal efficiency vs Background rejection
    fpr, tpr = metrics['fpr'], metrics['tpr']
    background_rejection = 1 - fpr
    signal_efficiency = tpr
    
    # Plot SIC curve
    ax.plot(signal_efficiency, background_rejection, linewidth=2,
            label=f'SIC Curve (AUC = {metrics["auc"]:.3f})')
    ax.plot([0, 1], [1, 0], 'k--', alpha=0.5, label='Random Classifier')
    
    # Formatting
    ax.set_xlabel('Signal Efficiency (TTbar)')
    ax.set_ylabel('Background Rejection (1 - QCD FPR)')
    ax.set_title('SIC Curve - Fixed Correlation Jet Classifier')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Add working point markers
    for eff in [0.5, 0.7, 0.9]:
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
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference with fixed correlation jet classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--classical",
        action="store_true",
        help="Use classical (non-QFI) features only"
    )
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        logger.success(f"Configuration loaded from {args.config}")
        
        # Setup device
        device = setup_device()
        
        # Load and verify statistics
        logger.info("Loading class statistics...")
        experiment_dir = os.path.join(config.experiment.base_save_dir, config.experiment.name + '_' + config.experiment.seed)
        statistics = load_and_verify_statistics(experiment_dir, config)
        
        # Extract statistics
        ttbar_mean = statistics['ttbar_class_mean'].to(device)
        qcd_mean = statistics['qcd_class_mean'].to(device)
        pooled_cov = statistics['pooled_covariance'].to(device)
        
        logger.info(f"TTbar class mean: {ttbar_mean.cpu().numpy()}")
        logger.info(f"QCD class mean: {qcd_mean.cpu().numpy()}")
        logger.info(f"Pooled covariance determinant: {torch.linalg.det(pooled_cov):.6f}")
        
        # Create test dataloader
        logger.info("Creating test dataloader...")
        test_dataloader = create_test_dataloader(config)
        
        # Create model
        logger.info("Creating fixed correlation model...")
        if config.model.num_mp_layers > 1:
            logger.warning("Only single-layer message passing is supported. This will fail in the next step.")
        
        model = create_jet_graph(
            message_type='fixed_correlation',
            num_layers=config.model.num_mp_layers,
            pooling=config.model.pooling,
            output_mode='default'
        )
        model = model.to(device)
        
        # Extract graph-level features
        features, true_labels = extract_graph_features(model, test_dataloader, device)
        features = features.to(device)
        
        logger.info(f"Running inference on {len(features)} test samples")
        
        # Apply classification method
        distance_method = config.model.distance.lower()
        if distance_method == 'fisher':
            predictions, probabilities = fisher_linear_discriminant(
                features, ttbar_mean, qcd_mean, pooled_cov
            )
        elif distance_method == 'mahalanobis':
            predictions, probabilities = mahalanobis_distance_classification(
                features, ttbar_mean, qcd_mean, pooled_cov, classical=args.classical
            )
        else:
            raise ValueError(f"Unknown distance method: {distance_method}. Use 'fisher' or 'mahalanobis'")
        
        # Compute metrics
        metrics = compute_metrics(true_labels, predictions.cpu(), probabilities.cpu())
        
        # Create results directory
        results_dir = os.path.join(experiment_dir, "results")
        if args.classical:
            results_dir = os.path.join(results_dir, "classical")
        Path(results_dir).mkdir(exist_ok=True)
        
        # Save metrics to NPZ file
        metrics_path = os.path.join(results_dir, "metrics.npz")
        np.savez(
            metrics_path,
            fpr=metrics['fpr'],
            tpr=metrics['tpr'], 
            thresholds=metrics['thresholds'],
            auc=metrics['auc'],
            accuracy=metrics['accuracy'],
            true_labels=metrics['true_labels'],
            predictions=metrics['predictions'],
            probabilities=metrics['probabilities']
        )
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # Create plots
        roc_path = os.path.join(results_dir, "roc_curve.png")
        plot_roc_curve(metrics, roc_path)
        
        sic_path = os.path.join(results_dir, "sic_curve.png")
        plot_sic_curve(metrics, sic_path)
        #import pdb;pdb.set_trace()
        # Log final results
        logger.info("=" * 60)
        logger.info("FINAL INFERENCE RESULTS")
        logger.info("=" * 60)
        logger.info(f"Classification method: {distance_method.upper()}")
        logger.info(f"Test samples: {len(features)}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"Results saved in: {results_dir}")
        logger.info("=" * 60)
        
        logger.success("Inference completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.exception("Full traceback:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())