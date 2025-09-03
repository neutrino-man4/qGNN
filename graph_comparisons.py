"""
Author: Aritra Bal, ETP
Date: die Martis ante diem tertium Nonas Septembres anno ab urbe condita MMDCCLXXVIII

Script to create comparative ROC and SIC plots from multiple experiment metrics.
Generates log ROC curves and Significance Improvement Characteristic curves.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import time
from loguru import logger


def load_metrics_from_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load TPR, FPR, and AUC from NPZ file.
    
    Args:
        npz_path: Path to NPZ file containing metrics
        
    Returns:
        Tuple of (fpr, tpr, auc)
    """
    try:
        data = np.load(npz_path)
        fpr = data['fpr']
        tpr = data['tpr']
        auc_value = float(data['auc'])
        
        logger.info(f"Loaded metrics from {npz_path}: AUC = {auc_value:.3f}")
        return fpr, tpr, auc_value
        
    except Exception as e:
        logger.error(f"Failed to load metrics from {npz_path}: {e}")
        raise


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
    plt.ylim([1.0, 1.0e4])
    plt.yscale('log')
    plt.xlabel('Signal Efficiency (TPR)', fontsize=17)
    plt.ylabel('Background Rejection (FPR$^{-1}$)', fontsize=17)
    plt.title('Jet Classification Performance - Log ROC', fontsize=19, fontweight='bold')
    plt.legend(loc="upper right", fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save in both formats
    png_path = Path(save_dir) / f"AUC_{timestamp}.png"
    pdf_path = Path(save_dir) / f"AUC_{timestamp}.pdf"
    
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    logger.success(f"Log ROC curves saved to {png_path} and {pdf_path}")
    plt.close()


def plot_sic_curves(metrics_list: List[Tuple[np.ndarray, np.ndarray, float]], 
                   labels: List[str], save_dir: str, timestamp: str):
    """
    Plot SIC (Significance Improvement Characteristic) curves for multiple experiments.
    
    Args:
        metrics_list: List of (fpr, tpr, auc) tuples
        labels: List of legend labels for each experiment
        save_dir: Directory to save plots
        timestamp: Timestamp string for filename
    """
    plt.figure(figsize=(10, 8))
    
    logger.info("Creating SIC curves...")
    
    for i, (fpr, tpr, auc_value) in enumerate(metrics_list):
        # Calculate Significance Improvement: TPR/sqrt(FPR)
        # Avoid division by zero
        fpr_safe = np.maximum(fpr, 1e-6)
        significance_improvement = tpr / np.sqrt(fpr_safe)
        
        label = f"{labels[i]}"
        plt.plot(tpr, significance_improvement, linewidth=2, label=label)
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 4.0])
    plt.xlabel('Signal Efficiency (TPR)', fontsize=17)
    plt.ylabel('Significance Improvement (TPR/$\sqrt{FPR}$)', fontsize=17)
    plt.title('SIC Curves Comparison', fontsize=19, fontweight='bold')
    plt.legend(loc="lower right", fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save in both formats
    png_path = Path(save_dir) / f"SIC_{timestamp}.png"
    pdf_path = Path(save_dir) / f"SIC_{timestamp}.pdf"
    
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    logger.success(f"SIC curves saved to {png_path} and {pdf_path}")
    plt.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create comparative ROC and SIC plots from multiple experiment metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--npz-files",
        type=str,
        nargs='+',
        required=True,
        help="Paths to NPZ files containing metrics (fpr, tpr, auc)"
    )
    
    parser.add_argument(
        "--labels",
        type=str,
        nargs='+', 
        required=True,
        help="Legend labels for each experiment (same order as NPZ files)"
    )
    
    return parser.parse_args()


def validate_inputs(npz_files: List[str], labels: List[str]) -> None:
    """
    Validate input arguments.
    
    Args:
        npz_files: List of NPZ file paths
        labels: List of legend labels
        
    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If NPZ files don't exist
    """
    if len(npz_files) != len(labels):
        raise ValueError(
            f"Number of NPZ files ({len(npz_files)}) must match "
            f"number of labels ({len(labels)})"
        )
    
    if len(npz_files) == 0:
        raise ValueError("At least one NPZ file must be provided")
    
    # Check if all NPZ files exist
    for npz_file in npz_files:
        if not Path(npz_file).exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_file}")
    
    logger.info(f"Validated {len(npz_files)} NPZ files and {len(labels)} labels")


def main():
    """Main function to create comparison plots."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        logger.info("Starting comparison plot generation...")
        logger.info(f"NPZ files: {args.npz_files}")
        logger.info(f"Labels: {args.labels}")
        
        # Validate inputs
        validate_inputs(args.npz_files, args.labels)
        
        # Create output directory
        save_dir = "./experiments/graph_comparisons"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {save_dir}")
        
        # Generate timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Load metrics from all NPZ files
        logger.info("Loading metrics from NPZ files...")
        metrics_list = []
        for npz_file in args.npz_files:
            fpr, tpr, auc_value = load_metrics_from_npz(npz_file)
            metrics_list.append((fpr, tpr, auc_value))
        
        logger.success(f"Loaded metrics from {len(metrics_list)} experiments")
        
        # Create log ROC curves
        plot_log_roc_curves(metrics_list, args.labels, save_dir, timestamp)
        
        # Create SIC curves
        plot_sic_curves(metrics_list, args.labels, save_dir, timestamp)
        
        # Summary
        logger.info("=" * 60)
        logger.info("COMPARISON PLOTS SUMMARY")
        logger.info("=" * 60)
        for i, (_, _, auc_value) in enumerate(metrics_list):
            logger.info(f"{args.labels[i]:30s}: AUC = {auc_value:.3f}")
        logger.info(f"Plots saved in: {save_dir}")
        logger.info("=" * 60)
        
        logger.success("Comparison plots created successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
        logger.exception("Full traceback:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())