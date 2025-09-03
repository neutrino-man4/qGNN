"""
Author: Aritra Bal, ETP
Date: die Martis ante diem tertium Nonas Septembres anno ab urbe condita MMDCCLXXVIII

Script to compute class statistics (means and covariance matrices and determinants and stuff) from training data
Please don't mix up and use validation/test data instead. Check your config YAML before using this
using FixedCorrelationMessage passing for statistical jet classification as of 2nd September 2025, maybe we implement more advanced techniques in the future. MAYBEEE
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any
import torch
import numpy as np
from loguru import logger
import time

# Import project modules
sys.path.append('.')
from configs.config import load_config
from data_utils.graph_dataloader import JetGraphDataloader
from src.graph_classifier import create_jet_graph


def setup_device() -> torch.device:
    """Setup and return the appropriate device for computation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def create_statistics_dataloader(config: Any, stat_path: str) -> JetGraphDataloader:
    """
    Create JetGraphDataloader with averaged QFI matrices for computing statistics.
    
    Args:
        config: Configuration object
        stat_path: Path to directory containing QFI means
        
    Returns:
        Configured JetGraphDataloader
    """
    qfi_means_path = os.path.join(stat_path, "qfi_means.npz")
    
    if not os.path.exists(qfi_means_path):
        raise FileNotFoundError(f"QFI means file not found: {qfi_means_path}")
    
    #logger.info(f"Creating dataloader with QFI means from: {qfi_means_path}")
    
    dataloader = JetGraphDataloader(
        h5_files=config.data.train_files,
        batch_size=config.data.batch_size,
        use_qfi_correlations=config.data.use_qfi_correlations,
        mean_qfi_path=qfi_means_path
    )
    
    logger.success(f"Dataloader created with {len(dataloader)} batches")
    return dataloader


def main():
    """Main function to compute and save class statistics."""
    parser = argparse.ArgumentParser(
        description="Compute class statistics for fixed correlation jet classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="./configs/base.yaml",
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        logger.success(f"Configuration loaded from {args.config}")
        
        # Get paths and parameters from config
        stat_path = config.data.stat_path
        pooling_method = config.model.pooling
        num_layers = config.model.num_mp_layers
        
        Path(stat_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Statistics will be saved to: {stat_path}")
        
        # Setup device
        logger.info("Setting up compute device...")
        device = setup_device()
        
        # Create dataloader with QFI means
        #logger.info("Creating dataloader with class-averaged QFI matrices...")
        dataloader = create_statistics_dataloader(config, stat_path)
        
        # Create fixed correlation model in default mode
        logger.info("Creating JetGraph model for statistics computation...")
        model = create_jet_graph(
            message_type='fixed_correlation',
            num_layers=num_layers,
            pooling=pooling_method,
            output_mode='default'  # Extract raw features for computing statistics
        )
        model = model.to(device)
        
        # Log model info
        arch_info = model.get_architecture_info()
        logger.info(f"Model architecture: {arch_info['num_mp_layers']} layers, "
                   f"{arch_info['pooling_type']} pooling, {arch_info['feature_dim']}D features")
        logger.info("Model has no trainable parameters (fixed operations only)")
        
        # Compute class statistics
        logger.info("Computing class statistics from training data...")
        logger.info("This may take several minutes depending on dataset size...")
        
        start_time = time.time()
        statistics = model.compute_class_statistics(dataloader, device)
        computation_time = time.time() - start_time
        
        logger.success(f"Statistics computation completed in {computation_time:.2f} seconds")
        
        # Log computed statistics
        logger.info("=" * 60)
        logger.info("COMPUTED CLASS STATISTICS")
        logger.info("=" * 60)
        logger.info(f"TTbar jets: {statistics['n_ttbar']}")
        logger.info(f"QCD jets: {statistics['n_qcd']}")
        logger.info(f"Feature dimension: {statistics['ttbar_mean'].shape[0]}D")
        logger.info(f"Pooling method: {pooling_method}")
        logger.info(f"Message passing layers: {num_layers}")
        
        # Log mean vectors
        ttbar_mean_np = statistics['ttbar_mean'].cpu().numpy()
        qcd_mean_np = statistics['qcd_mean'].cpu().numpy()
        logger.info(f"TTbar mean: {ttbar_mean_np}")
        logger.info(f"QCD mean: {qcd_mean_np}")
        
        # Log covariance info
        cov_det = torch.linalg.det(statistics['pooled_covariance']).item()
        cov_trace = torch.trace(statistics['pooled_covariance']).item()
        logger.info(f"Pooled covariance determinant: {cov_det:.6f}")
        logger.info(f"Pooled covariance trace: {cov_trace:.6f}")
        logger.info("=" * 60)
        
        # Save statistics to file
        stats_filename = "sample_statistics.pth"
        experiment_dir = os.path.join(config.experiment.base_save_dir, config.experiment.name + '_' + config.experiment.seed)
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)
        stats_path = os.path.join(experiment_dir, stats_filename)

        # Prepare statistics with specific naming convention
        statistics_to_save = {
            'ttbar_class_mean': statistics['ttbar_mean'],
            'qcd_class_mean': statistics['qcd_mean'],
            'pooled_covariance': statistics['pooled_covariance'],
            'n_ttbar': statistics['n_ttbar'],
            'n_qcd': statistics['n_qcd'],
            'pooling_method': pooling_method,
            'num_mp_layers': num_layers,
            'feature_dimension': statistics['ttbar_mean'].shape[0]
        }
        
        torch.save(statistics_to_save, stats_path)
        logger.success(f"Statistics saved to: {stats_path}")
        
        # Also save human-readable summary
        summary = {
            'computation_time_seconds': computation_time,
            'pooling_method': pooling_method,
            'num_mp_layers': num_layers,
            'feature_dimension': int(statistics['ttbar_mean'].shape[0]),
            'n_ttbar_jets': int(statistics['n_ttbar']),
            'n_qcd_jets': int(statistics['n_qcd']),
            'ttbar_mean': ttbar_mean_np.tolist(),
            'qcd_mean': qcd_mean_np.tolist(),
            'covariance_determinant': float(cov_det),
            'covariance_trace': float(cov_trace),
            'config_file': args.config,
            'qfi_means_file': os.path.join(stat_path, "qfi_means.npz"),
            'statistics_file': stats_path
        }

        summary_path = os.path.join(experiment_dir, "statistics_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Human-readable summary saved to: {summary_path}")
        
        # Final instructions
        logger.info("\n" + "=" * 60)
        logger.info("NEXT STEPS FOR INFERENCE")
        logger.info("=" * 60)
        logger.info("1. Use the computed statistics for Fisher/Mahalanobis classification:")
        logger.info(f"   Statistics file: {stats_path}")
        logger.info("2. Load statistics in your inference script using:")
        logger.info("   stats = torch.load('sample_statistics.pth')")
        logger.info("3. Access statistics with keys:")
        logger.info("   stats['ttbar_class_mean'], stats['qcd_class_mean'], stats['pooled_covariance']")
        logger.info("4. Set model to 'fisher' or 'mahalanobis' mode and calibrate:")
        logger.info("   model.set_class_means(stats['ttbar_class_mean'], stats['qcd_class_mean'], stats['pooled_covariance'])")
        logger.success("Class statistics computation completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please check your file paths and ensure QFI means file exists")
        return 1
    except Exception as e:
        logger.error(f"Statistics computation failed: {e}")
        logger.exception("Full traceback:")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())