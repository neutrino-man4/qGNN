# Author: Aritra Bal, ETP
# Date: die Mercurii ante diem quartum Idus Ianuarias anno ab urbe condita MMDCCLXXVIII

"""
Multi-experiment comparison script for Jet GNN classification results.
Loads multiple experiments using the config system and creates comparative plots.
"""

import argparse
import sys
import os
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from typing import List, Dict, Any, Tuple
import logging
import mplhep
# Import project modules
sys.path.append('.')
from configs.config import load_config, Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for CMS publication-quality plots
mplhep.style.use("CMS")

class ExperimentLoader:
    """Load experiment results and metadata using the config system."""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.config = self._load_config()
        self.y_true, self.y_pred_proba = self._load_predictions()
        self.training_history = self._load_training_history()

    def _load_config(self) -> Config:
        """Load experiment configuration using the project's config system."""
        config_file = self.experiment_dir / "config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
            
        logger.info(f"Loading config from: {config_file}")
        
        try:
            return load_config(config_file)
        except Exception as e:
            logger.error(f"Error loading config from {config_file}: {e}")
            raise
    
    def _find_prediction_files(self) -> List[Path]:
        """Find prediction h5 files based on config output directory."""
        # Get the expected output directory from config
        output_dir = Path(self.config.testing.output_dir) / f"{self.config.experiment.name}_{self.config.experiment.seed}"
        
        logger.info(f"Looking for prediction files in: {output_dir}")
        
        if not output_dir.exists():
            logger.warning(f"Output directory does not exist: {output_dir}")
            return []
        
        # Find h5 files with 'infer' in the name (as per test.py naming convention)
        h5_files = list(output_dir.glob("*infer*.h5"))
        
        if not h5_files:
            logger.warning(f"No inference h5 files found in {output_dir}")
            
        return h5_files
    
    def _load_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load predictions from h5 files."""
        prediction_files = self._find_prediction_files()
        
        if not prediction_files:
            logger.warning(f"No prediction files found for experiment {self.experiment_dir}")
            return np.array([]), np.array([])
        
        all_y_true = []
        all_y_pred_proba = []
        
        for pred_file in prediction_files:
            logger.info(f"Loading predictions from: {pred_file}")
            
            try:
                with h5py.File(pred_file, 'r') as f:
                    # Keys based on test.py: 'true_labels', 'ttbar_probabilities'
                    if 'true_labels' not in f.keys() or 'ttbar_probabilities' not in f.keys():
                        logger.error(f"Required keys not found in {pred_file}")
                        logger.error(f"Available keys: {list(f.keys())}")
                        continue
                    
                    y_true = f['true_labels'][:]
                    y_pred_proba = f['ttbar_probabilities'][:]
                    
                    all_y_true.append(y_true)
                    all_y_pred_proba.append(y_pred_proba)
                    
                    logger.info(f"Loaded {len(y_true)} predictions from {pred_file.name}")
                    
            except Exception as e:
                logger.error(f"Error loading predictions from {pred_file}: {e}")
                continue
        
        if not all_y_true:
            logger.warning("No predictions could be loaded")
            return np.array([]), np.array([])
        
        # Concatenate all predictions
        y_true_combined = np.concatenate(all_y_true)
        y_pred_proba_combined = np.concatenate(all_y_pred_proba)
        
        logger.info(f"Combined {len(y_true_combined)} total predictions")
        logger.info(f"Label distribution: {np.sum(y_true_combined == 0)} QCD, {np.sum(y_true_combined == 1)} TTbar")
        
        return y_true_combined, y_pred_proba_combined
    
    def get_legend_label(self) -> str:
        """Get legend label from config."""
        return self.config.testing.desc
    
    def get_experiment_name(self) -> str:
        """Get experiment name from config."""
        return self.config.experiment.name
    
    def get_experiment_seed(self) -> str:
        """Get experiment seed from config."""
        return self.config.experiment.seed
    
    def get_num_jets(self) -> int:
        """Get number of jets in predictions."""
        return len(self.y_true)
    
    def has_valid_predictions(self) -> bool:
        """Check if predictions are available and valid."""
        return len(self.y_true) > 0 and len(self.y_pred_proba) > 0 and len(self.y_true) == len(self.y_pred_proba)
    
    def _load_training_history(self) -> Dict[str, List[float]]:
        """Load training history from JSON file."""
        history_file = self.experiment_dir / "training_history.json"
    
        if not history_file.exists():
            logger.warning(f"Training history file not found: {history_file}")
            return {}
        
        try:
            import json
            with open(history_file, 'r') as f:
                history = json.load(f)
            logger.info(f"Loaded training history from: {history_file}")
            return history
        except Exception as e:
            logger.error(f"Error loading training history from {history_file}: {e}")
            return {}

    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history data."""
        return getattr(self, 'training_history', {})

class ExperimentComparison:
    """Compare multiple experiments with visualizations."""
    
    def __init__(self, experiment_dirs: List[Path]):
        self.experiments = []
        
        # Load all experiments
        for exp_dir in experiment_dirs:
            try:
                exp = ExperimentLoader(exp_dir)
                if exp.has_valid_predictions():
                    self.experiments.append(exp)
                    logger.info(f"Loaded experiment: {exp.get_legend_label()} ({exp.get_num_jets()} jets)")
                else:
                    logger.warning(f"Skipping experiment {exp_dir}: no valid predictions found")
            except Exception as e:
                logger.error(f"Failed to load experiment from {exp_dir}: {e}")
        
        if not self.experiments:
            raise ValueError("No valid experiments loaded!")
        
        # Check if all experiments have same number of jets
        jet_counts = [exp.get_num_jets() for exp in self.experiments]
        if len(set(jet_counts)) > 1:
            logger.warning(f"Warning: Experiments have different numbers of jets: {dict(zip([exp.get_legend_label() for exp in self.experiments], jet_counts))}")
        
        # Determine output directory
        self.output_dir = self._determine_output_dir(experiment_dirs)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename prefix
        self.filename_prefix = self._generate_filename_prefix()
        
        logger.info(f"Successfully loaded {len(self.experiments)} experiments")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _determine_output_dir(self, experiment_dirs: List[Path]) -> Path:
        """Determine output directory based on common base path."""
        # Find common parent directory
        if len(experiment_dirs) == 1:
            base_dir = experiment_dirs[0].parent
        else:
            # Find common path
            common_parts = []
            paths = [list(exp_dir.parts) for exp_dir in experiment_dirs]
            
            for i in range(min(len(p) for p in paths)):
                if all(p[i] == paths[0][i] for p in paths):
                    common_parts.append(paths[0][i])
                else:
                    break
            
            if common_parts:
                base_dir = Path(*common_parts)
            else:
                base_dir = Path('.')
        
        return base_dir / "comparisons"
    
    def _generate_filename_prefix(self) -> str:
        """Generate filename prefix based on experiment names and seeds."""
        name_seed_pairs = []
        for exp in self.experiments:
            name = exp.get_experiment_name()
            seed = exp.get_experiment_seed()
            name_seed_pairs.append(f"{name}_{seed}")
        
        return "_VS_".join(name_seed_pairs)
    
    def _save_plot(self, filename_suffix: str, fig=None):
        """Save plot in both PNG (DPI=600) and PDF formats."""
        if fig is None:
            fig = plt.gcf()
            
        base_filename = f"{self.filename_prefix}_{filename_suffix}"
        
        # Save as PNG with high DPI
        png_path = self.output_dir / f"{base_filename}.png"
        fig.savefig(png_path, dpi=600, bbox_inches='tight', format='png')
        logger.info(f"Saved plot to {png_path}")
        
        # Save as PDF
        pdf_path = self.output_dir / f"{base_filename}.pdf"
        fig.savefig(pdf_path, bbox_inches='tight', format='pdf')
        logger.info(f"Saved plot to {pdf_path}")
    
    def plot_roc_curves(self):
        """Plot ROC curves for all experiments."""
        plt.figure(figsize=(10, 8))
        
        auc_values = []
        for exp in self.experiments:
            fpr, tpr, _ = roc_curve(exp.y_true, exp.y_pred_proba)
            roc_auc = auc(fpr, tpr)
            auc_values.append(roc_auc)
            
            label = f"{exp.get_legend_label()} (AUC = {roc_auc:.3f})"
            plt.plot(fpr, tpr, linewidth=2, label=label)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Baseline (AUC = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=17)
        plt.ylabel('True Positive Rate', fontsize=17)
        plt.title('Top tagging performance', fontsize=19, fontweight='bold')
        plt.legend(loc="lower right", fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save in both formats
        self._save_plot("ROC")
        plt.clf()
        
        return max(auc_values)
    
    def plot_log_roc_curves(self):
        """Plot log ROC curves for all experiments."""
        plt.figure(figsize=(10, 8))
        
        auc_values = []
        for exp in self.experiments:
            fpr, tpr, _ = roc_curve(exp.y_true, exp.y_pred_proba)
            roc_auc = auc(fpr, tpr)
            auc_values.append(roc_auc)
            
            label = f"{exp.get_legend_label()} (AUC = {roc_auc:.3f})"
            plt.plot(tpr, 1.0/fpr, linewidth=2, label=label)

        base_tpr=np.arange(0.001, 1.0, 0.01)
        base_fpr=np.arange(0.001, 1.0, 0.01)
        plt.plot(base_tpr, 1.0/base_fpr, linewidth=2, label="Baseline (AUC = 0.5)", linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([1.0, 1.0e4])
        plt.yscale('log')
        plt.xlabel('Signal Efficiency (TPR)', fontsize=17)
        plt.ylabel('Background Rejection (FPR$^{-1}$)', fontsize=17)
        plt.title('Top tagging performance', fontsize=19, fontweight='bold')
        plt.legend(loc="upper right", fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save in both formats
        self._save_plot("LOG_ROC")
        plt.clf()
        
        return 1
    
    def plot_precision_recall_curves(self):
        """Plot Precision-Recall curves for all experiments."""
        plt.figure(figsize=(10, 8))
        
        for exp in self.experiments:
            precision, recall, _ = precision_recall_curve(exp.y_true, exp.y_pred_proba)
            pr_auc = auc(recall, precision)
            
            label = f"{exp.get_legend_label()} (AUC = {pr_auc:.3f})"
            plt.plot(recall, precision, linewidth=2, label=label)
        
        # Random baseline
        positive_rate = np.mean([exp.y_true.mean() for exp in self.experiments])
        plt.axhline(y=positive_rate, color='k', linestyle='--', alpha=0.5, 
                   label=f'Random (AUC = {positive_rate:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=17)
        plt.ylabel('Precision', fontsize=17)
        plt.title('Precision-Recall Curves Comparison', fontsize=19, fontweight='bold')
        plt.legend(loc="lower left", fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save in both formats
        self._save_plot("PR")
        plt.clf()
    
    def plot_sic_curves(self):
        """Plot SIC (Significance Improvement Characteristic) curves for all experiments."""
        plt.figure(figsize=(10, 8))
        
        for exp in self.experiments:
            fpr, tpr, _ = roc_curve(exp.y_true, exp.y_pred_proba)
            #background_rejection = 1 - fpr
            #signal_efficiency = tpr
            
            label = f"{exp.get_legend_label()}"
            plt.plot(tpr, tpr/np.sqrt(fpr), linewidth=2, label=label)
        #sic = tpr/np.sqrt(fpr)
        #plt.plot([0, 1], [1, 0], 'k--', linewidth=1, alpha=0.5, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 4.0])
        plt.xlabel('QCD Efficiency', fontsize=17)
        plt.ylabel('Significance Improvement', fontsize=17)
        plt.title('SIC Curves Comparison', fontsize=19, fontweight='bold')
        plt.legend(loc="lower left", fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save in both formats
        self._save_plot("SIC")
        plt.clf()
    
    def plot_training_curves(self):
        """Plot training evolution curves for all experiments."""
        # Check if any experiment has training history
        has_history = any(exp.get_training_history() for exp in self.experiments)
        if not has_history:
            logger.warning("No training history found for any experiment. Skipping training curves.")
            return
        
        colors = ['#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # yellow-green / chartreuse
            '#17becf'  # cyan
        ]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss Evolution
        for idx, exp in enumerate(self.experiments):
            history = exp.get_training_history()
            if 'train_loss' in history and 'val_loss' in history:
                epochs = range(1, len(history['train_loss']) + 1)
                label = exp.get_legend_label()
                color = colors[idx % len(colors)]
                axes[0].plot(epochs, history['train_loss'], linewidth=2,
                            label=f'{label} (Train)', linestyle='-', marker='o', markersize=3, color=color)
                axes[0].plot(epochs, history['val_loss'], linewidth=2,
                            label=f'{label} (Val)', linestyle='--', marker='s', markersize=3, color=color)
        
        axes[0].set_xlabel('Epoch', fontsize=19)
        axes[0].set_ylabel('Loss', fontsize=19)
        axes[0].set_title('Loss Evolution', fontsize=21, fontweight='bold')
        axes[0].legend(fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # AUC and Accuracy Evolution on same plot
        # First plot all AUC curves
        for idx, exp in enumerate(self.experiments):
            history = exp.get_training_history()
            if 'val_auc' in history and history['val_auc']:
                epochs = range(1, len(history['val_auc']) + 1)
                color = colors[idx % len(colors)]
                axes[1].plot(epochs, history['val_auc'], linewidth=2,
                            label=f'{exp.get_legend_label()} (AUC)', linestyle='-', 
                            marker='o', markersize=4, color=color)
        
        # Then plot all accuracy curves
        for idx, exp in enumerate(self.experiments):
            history = exp.get_training_history()
            if 'val_accuracy' in history and history['val_accuracy']:
                epochs = range(1, len(history['val_accuracy']) + 1)
                color = colors[idx % len(colors)]
                axes[1].plot(epochs, history['val_accuracy'], linewidth=2,
                            label=f'{exp.get_legend_label()} (Accuracy)', linestyle='--', 
                            marker='s', markersize=4, color=color)
        
        axes[1].set_xlabel('Epoch', fontsize=19)
        axes[1].set_ylabel('Metric', fontsize=19)
        axes[1].set_title('Evolution of Performance Metrics', fontsize=21, fontweight='bold')
        axes[1].legend(fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0.8, 1.0])
        
        plt.tight_layout()
        
        # Save in both formats
        self._save_plot("training_curves", fig)
        plt.clf()

    def plot_metrics_comparison(self):
        """Plot bar chart comparing key metrics."""
        labels = [exp.get_legend_label() for exp in self.experiments]
        auc_scores = []
        accuracies = []
        
        for exp in self.experiments:
            # Calculate AUC
            fpr, tpr, _ = roc_curve(exp.y_true, exp.y_pred_proba)
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            
            # Calculate accuracy (using 0.5 threshold)
            y_pred = (exp.y_pred_proba > 0.5).astype(int)
            accuracy = (y_pred == exp.y_true).mean()
            accuracies.append(accuracy)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC comparison
        bars1 = axes[0].bar(range(len(labels)), auc_scores, alpha=0.8)
        axes[0].set_xlabel('Experiments', fontsize=17)
        axes[0].set_ylabel('AUC Score', fontsize=17)
        axes[0].set_title('AUC Comparison', fontsize=19, fontweight='bold')
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=15)
        
        # Accuracy comparison
        bars2 = axes[1].bar(range(len(labels)), accuracies, alpha=0.8, color='orange')
        axes[1].set_xlabel('Experiments', fontsize=17)
        axes[1].set_ylabel('Accuracy', fontsize=17)
        axes[1].set_title('Accuracy Comparison', fontsize=19, fontweight='bold')
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        # Add value labels on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=15)
        
        plt.tight_layout()
        
        # Save in both formats
        self._save_plot("metrics", fig)
        plt.clf()
    
    def generate_report(self):
        """Generate complete comparison report."""
        logger.info("Generating comparison plots...")
        
        # Create all plots
        best_auc = self.plot_roc_curves()
        self.plot_log_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_sic_curves()
        self.plot_metrics_comparison()
        self.plot_training_curves() 
        # Generate text report
        report_filename = f"{self.filename_prefix}_report.txt"
        report_path = self.output_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write("Jet GNN Experiments Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Number of experiments compared: {len(self.experiments)}\n")
            f.write(f"Generated on: {np.datetime64('now')}\n\n")
            
            # Calculate performance for each experiment
            exp_results = []
            for exp in self.experiments:
                fpr, tpr, _ = roc_curve(exp.y_true, exp.y_pred_proba)
                roc_auc = auc(fpr, tpr)
                y_pred = (exp.y_pred_proba > 0.5).astype(int)
                accuracy = (y_pred == exp.y_true).mean()
                
                exp_results.append((exp.get_legend_label(), roc_auc, accuracy, exp.get_num_jets()))
            
            # Sort by AUC
            exp_results.sort(key=lambda x: x[1], reverse=True)
            
            f.write("Experiment Rankings by AUC:\n")
            f.write("-" * 30 + "\n")
            
            for i, (label, auc_score, accuracy, num_jets) in enumerate(exp_results, 1):
                f.write(f"{i:2d}. {label:30s} - AUC: {auc_score:.4f}, Acc: {accuracy:.4f}, Jets: {num_jets}\n")
            
            f.write(f"\nAll plots saved to: {self.output_dir}\n")
        
        logger.info(f"Report generated: {report_path}")
        logger.info(f"Best performing experiment: {exp_results[0][0]} (AUC: {exp_results[0][1]:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple Jet GNN experiments using config system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comparisons.py exp1/ exp2/ exp3/
  python comparisons.py ./experiments/convGNN_0001/ ./experiments/quantum_convGNN_0002/
  python comparisons.py ./experiments/*/
        """
    )
    
    parser.add_argument(
        'experiment_dirs',
        nargs='+',
        type=Path,
        help='Directories containing experiment results with config.yaml files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate experiment directories
    valid_dirs = []
    for exp_dir in args.experiment_dirs:
        if exp_dir.is_dir():
            config_file = exp_dir / "config.yaml"
            if config_file.exists():
                valid_dirs.append(exp_dir)
                logger.info(f"Found experiment directory: {exp_dir}")
            else:
                logger.warning(f"Config file not found in: {exp_dir}")
        else:
            logger.warning(f"Directory not found: {exp_dir}")
    
    if not valid_dirs:
        logger.error("No valid experiment directories found!")
        sys.exit(1)
    
    try:
        # Create comparison object and generate report
        comparison = ExperimentComparison(valid_dirs)
        comparison.generate_report()
        
        logger.info("Comparison completed successfully!")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()