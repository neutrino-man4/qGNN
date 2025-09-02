"""
Author: Aritra Bal, ETP
Date: ante diem quartum Nonas Septembres anno ab urbe condita MMDCCLXXVIII

Script to plot mean QFI matrices for QCD and top jets.
Loads saved QFI data and creates publication-quality visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import os
import pathlib
from typing import Tuple
import argparse


def plot_qfi_matrix(qfi_matrix: np.ndarray, plot_label: str, save_path: str) -> None:
    """
    Plot QFI matrix with quantum circuit style formatting.
    
    Args:
        qfi_matrix: Array of shape (30, 30) - QFI matrix where 30 = 3*10 qubits
        plot_label: Label for the plot title
        save_path: Path to save the plot (without extension)
    """
    # Ensure output directory exists
    pathlib.Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    
    # Get matrix dimensions
    N_params = qfi_matrix.shape[0]  # Should be 30
    N_qubits = N_params // 3        # Should be 10
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Color scheme  
    colors = ['#0066FF', 'white', '#FF0066']  # Blue -> White -> Red
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue_white_red', colors, N=256)
    
    # Set up normalization (adjust range based on actual data)
    data_range = max(abs(qfi_matrix.min()), abs(qfi_matrix.max()))
    norm = matplotlib.colors.Normalize(vmin=-data_range, vmax=data_range)
    
    # Plot the matrix
    im = ax.matshow(qfi_matrix, cmap=cmap, norm=norm)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('QFI Value', fontsize=12)
    
    # Add dark grid lines to highlight 3x3 blocks (between qubit blocks)
    for i in range(1, N_qubits):
        ax.axhline(y=3*i - 0.5, color='black', linewidth=2)
        ax.axvline(x=3*i - 0.5, color='black', linewidth=2)
    
    # Set tick positions
    rotation_positions = [i+0.5 for i in range(3*N_qubits)]
    ax.set_xticks(rotation_positions)
    ax.set_yticks(rotation_positions)
    
    # Remove tick labels (we'll add them manually between ticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Turn off minor ticks and top/right axis labels
    ax.tick_params(which='minor', length=0)
    ax.tick_params(top=False, labeltop=False, right=False, labelright=False)
    
    # Make tick marks smaller
    ax.tick_params(axis='both', length=3, width=0.5, which='major')
    
    # Add rotation labels between ticks
    rotation_labels = ['$R_Z$', '$R_Y$', '$R_X$'] * N_qubits
    label_positions = [i for i in range(3*N_qubits)]  # Between ticks
    
    for pos, label in zip(label_positions, rotation_labels):
        ax.text(pos, 3*N_qubits+0.1, label, ha='center', va='top', fontsize=10, 
                transform=ax.transData)
        ax.text(-0.7, pos, label, ha='right', va='center', fontsize=10, 
                transform=ax.transData)
    
    # Add qubit number labels (away from axis, at center of each 3x3 block)
    qubit_positions = [1 + 3*i for i in range(N_qubits)]
    qubit_labels = [str(i) for i in range(N_qubits)]
    
    # Add qubit numbers manually as text (away from axis)
    for pos, label in zip(qubit_positions, qubit_labels):
        ax.text(pos, -0.3, label, ha='center', va='top', fontsize=14, 
                fontweight='bold', transform=ax.transData)
        ax.text(-1.4, pos, label, ha='right', va='center', fontsize=14,
                fontweight='bold', transform=ax.transData)
    
    # Set axis labels
    ax.set_xlabel('Qubit Number', labelpad=40, fontsize=14)
    ax.set_ylabel('Qubit Number', labelpad=40, fontsize=14)
    plt.title(plot_label, fontsize=18, pad=20)
    plt.tight_layout()
    
    # Save in both formats
    plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"QFI Matrix Statistics for {plot_label}:")
    print(f"  Range: [{qfi_matrix.min():.6f}, {qfi_matrix.max():.6f}]")
    print(f"  Mean: {qfi_matrix.mean():.6f}")
    print(f"  Std:  {qfi_matrix.std():.6f}")
    print(f"  Matrix shape: {qfi_matrix.shape} ({N_qubits} qubits)")
    print(f"  Saved to: {save_path}.png and {save_path}.pdf")
    print()


def load_qfi_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load QFI matrices from saved NPZ file.
    
    Args:
        data_path: Path to the QFI data NPZ file
        
    Returns:
        Tuple containing (qcd_qfi, top_qfi, metadata)
    """
    print(f"Loading QFI data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"QFI data file not found: {data_path}")
    
    # Load data
    data = np.load(data_path)
    
    qcd_qfi = data['qcd_qfi']
    top_qfi = data['top_qfi']
    
    # Extract metadata
    metadata = {
        'n_qcd': int(data['n_qcd']),
        'n_top': int(data['n_top'])
    }
    
    print(f"Loaded QFI data:")
    print(f"  QCD QFI shape: {qcd_qfi.shape}")
    print(f"  Top QFI shape: {top_qfi.shape}")
    print(f"  QCD jets used: {metadata['n_qcd']}")
    print(f"  Top jets used: {metadata['n_top']}")
    print()
    
    return qcd_qfi, top_qfi, metadata


def main():
    """Main function to load QFI data and create plots."""
    parser = argparse.ArgumentParser(
        description="Plot mean QFI matrices for QCD and top jets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/ceph/abal/QML/qGNN/merged/train/qfi/qfi_means.npz",
        help="Path to QFI data NPZ file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/work/abal/qGNN/plotters/plots/",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    try:
        print("Starting QFI matrix plotting...")
        print(f"Data path: {args.data_path}")
        print(f"Output directory: {args.output_dir}")
        print()
        
        # Load QFI data
        qcd_qfi, top_qfi, metadata = load_qfi_data(args.data_path)
        
        # Create output directory
        output_dir = pathlib.Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot QCD QFI matrix
        qcd_label = f"Average QCD Jet QFI Matrix (N={metadata['n_qcd']})"
        qcd_save_path = output_dir / "qcd_avg_qfi"
        plot_qfi_matrix(qcd_qfi, qcd_label, str(qcd_save_path))
        
        # Plot top QFI matrix
        top_label = f"Average Top Jet QFI Matrix (N={metadata['n_top']})"
        top_save_path = output_dir / "top_avg_qfi"
        plot_qfi_matrix(top_qfi, top_label, str(top_save_path))
        
        # Create difference plot
        qfi_diff = top_qfi - qcd_qfi
        diff_label = f"QFI Difference (Top - QCD)"
        diff_save_path = output_dir / "qfi_difference"
        plot_qfi_matrix(qfi_diff, diff_label, str(diff_save_path))
        
        print("All QFI matrix plots created successfully!")
        print(f"Files saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during plotting: {e}")
        raise


if __name__ == "__main__":
    main()