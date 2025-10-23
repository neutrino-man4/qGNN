"""
Script to compare validation metrics across multiple training runs.

Author: Aritra Bal, ETP
Date: ante diem VIII Idus Octobris anno ab urbe condita MMDCCLXXVIII
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)


def load_metrics(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load validation metrics from an npz file.
    
    Args:
        npz_path: Path to the npz file containing training metrics
        
    Returns:
        Tuple of (val_losses, val_aucs) arrays
    """
    data = np.load(npz_path)
    val_losses = data['val_losses']
    val_aucs = data['val_aucs']
    return val_losses, val_aucs


def find_common_base(paths: List[Path]) -> Path:
    """
    Find the lowest common base directory for all provided paths.
    
    Args:
        paths: List of Path objects
        
    Returns:
        Common base directory path
    """
    # Convert to absolute paths and get parts
    abs_paths = [p.resolve() for p in paths]
    parts_list = [p.parts for p in abs_paths]
    
    # Find common prefix
    common_parts = []
    for parts in zip(*parts_list):
        if len(set(parts)) == 1:
            common_parts.append(parts[0])
        else:
            break
    
    # Construct common base path
    if common_parts:
        return Path(*common_parts)
    else:
        return Path('.')


def extract_run_names(paths: List[Path], base_dir: Path) -> List[str]:
    """
    Extract run names from paths relative to base directory.
    
    Args:
        paths: List of metric file paths
        base_dir: Common base directory
        
    Returns:
        List of run names (directory names containing the metrics)
    """
    run_names = []
    for path in paths:
        try:
            relative = path.resolve().relative_to(base_dir.resolve())
            # Get the parent directory name (the run directory)
            run_name = relative.parent.name if relative.parent.name else relative.stem
            run_names.append(run_name)
        except ValueError:
            # If path is not relative to base_dir, use parent dir name
            run_names.append(path.parent.name)
    return run_names


def plot_metrics(
    npz_paths: List[Path],
    labels: List[str],
    output_dir: Path
) -> None:
    """
    Plot validation loss and AUC curves for multiple runs.
    
    Args:
        npz_paths: List of paths to npz metric files
        labels: List of labels for each run
        output_dir: Directory to save output plots
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all metrics
    all_val_losses = []
    all_val_aucs = []
    
    for path in npz_paths:
        val_losses, val_aucs = load_metrics(path)
        all_val_losses.append(val_losses)
        all_val_aucs.append(val_aucs)
    
    # Plot validation loss
    fig, ax = plt.subplots(figsize=(10, 6))
    for val_losses, label in zip(all_val_losses, labels):
        epochs = np.arange(1, len(val_losses) + 1)
        ax.plot(epochs, val_losses, marker='o', markersize=4, label=label)
    
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Validation Loss', fontsize=16)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'val_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'val_loss_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved validation loss plots to {output_dir}")
    
    # Plot validation AUC
    fig, ax = plt.subplots(figsize=(10, 6))
    for val_aucs, label in zip(all_val_aucs, labels):
        epochs = np.arange(1, len(val_aucs) + 1)
        ax.plot(epochs, val_aucs, marker='o', markersize=4, label=label)

    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Validation AUC', fontsize=16)
    ax.legend(loc='best', fontsize=15)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'val_auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'val_auc_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved validation AUC plots to {output_dir}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Compare validation metrics across multiple training runs'
    )
    parser.add_argument(
        '--npz-files',
        nargs='+',
        type=Path,
        help='Paths to npz metric files'
    )
    parser.add_argument(
        '--labels',
        nargs='+',
        type=str,
        help='Labels for each run (same order as npz_files)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.npz_files) < 2:
        parser.error('At least two npz files are required for comparison')
    
    if args.labels and len(args.labels) != len(args.npz_files):
        parser.error(f'Number of labels ({len(args.labels)}) must match number of files ({len(args.npz_files)})')
    
    # Check that all files exist
    for npz_path in args.npz_files:
        if not npz_path.exists():
            parser.error(f'File not found: {npz_path}')
    
    # Use provided labels or generate from paths
    if args.labels:
        labels = args.labels
    else:
        base_dir = find_common_base(args.npz_files)
        labels = extract_run_names(args.npz_files, base_dir)
    
    # Find common base and create output directory name
    base_dir = find_common_base(args.npz_files)
    run_names = extract_run_names(args.npz_files, base_dir)
    comparison_name = '_vs_'.join(run_names)
    
    output_dir = base_dir / 'comparisons' / comparison_name
    
    # Plot metrics
    plot_metrics(args.npz_files, labels, output_dir)
    
    print(f"Comparison complete. Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()