"""
QFI Matrix Analysis and Visualization for Top and QCD Jets
Author: Aritra Bal, ETP
Date: XVI Kalendas Novembres MMDCCLXXVIII

This script analyzes QFI matrices from jet data, identifies representative
top and QCD jets based on n-subjettiness ratios and pT range, and visualizes 
their QFI matrices and constituent distributions.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as patches
from pathlib import Path
from typing import Tuple, Optional
import logging
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_jet_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load jet data from HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Tuple of (QFI matrices, extra features, truth labels, jet pt, jet constituents)
    """
    logger.info(f"Loading data from: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Extract QFI matrices (N, 30, 30)
        qfi_matrices = 4*f['jetConstituentsQFI'][:]
        
        # Extract extra features (N, 6)
        # Order: jet_nparticles, jet_sdmass, jet_tau1, jet_tau2, jet_tau3, jet_tau4
        extra_features = f['jetExtraFeatures'][:]
        
        # Extract truth labels (N,)
        truth_labels = f['truth_labels'][:]
        
        # Extract jet pT (N,)
        jet_pt = f['jetFeatures'][:, 0]
        
        # Extract jet constituents (N, 100, 3) - pt, eta, phi
        jet_constituents = f['jetConstituentsList'][:]
    
    # Normalize constituent pT by jet pT
    jet_constituents_norm = jet_constituents.copy()
    jet_constituents_norm[:, :, 0] = jet_constituents[:, :, 0] / jet_pt[:, np.newaxis]
    
    logger.info(f"Loaded {len(qfi_matrices)} jets")
    logger.info(f"QFI matrices shape: {qfi_matrices.shape}")
    logger.info(f"Extra features shape: {extra_features.shape}")
    logger.info(f"Truth labels shape: {truth_labels.shape}")
    logger.info(f"Jet pT shape: {jet_pt.shape}")
    logger.info(f"Jet constituents shape: {jet_constituents.shape}")
    logger.info(f"Jet pT range: [{jet_pt.min():.1f}, {jet_pt.max():.1f}] GeV")
    
    return qfi_matrices, extra_features, truth_labels, jet_pt, jet_constituents_norm


def calculate_tau32_ratios(extra_features: np.ndarray) -> np.ndarray:
    """
    Calculate tau3/tau2 n-subjettiness ratios.
    
    Args:
        extra_features: Array of shape (N, 6) with jet features
        
    Returns:
        Array of tau3/tau2 ratios
    """
    tau2 = extra_features[:, 3]  # tau2 is at index 3
    tau3 = extra_features[:, 4]  # tau3 is at index 4
    
    # Avoid division by zero
    tau32_ratios = np.divide(tau3, tau2, out=np.zeros_like(tau3), where=(tau2 != 0))
    
    logger.info(f"Calculated tau3/tau2 ratios: min={tau32_ratios.min():.3f}, max={tau32_ratios.max():.3f}")
    
    return tau32_ratios


def find_representative_jets(tau32_ratios: np.ndarray, truth_labels: np.ndarray, 
                            jet_pt: np.ndarray, pt_range: Tuple[float, float] = (700, 750), tau1: np.ndarray = None) -> Tuple[Optional[int], Optional[int]]:
    """
    Find representative top and QCD jets based on tau32 criteria and pT range.
    
    Args:
        tau32_ratios: Array of tau3/tau2 ratios
        truth_labels: Array of truth labels (1=top, 0=QCD)
        jet_pt: Array of jet transverse momenta
        pt_range: Tuple of (min_pt, max_pt) in GeV
        
    Returns:
        Tuple of (top jet index, QCD jet index)
    """
    # Create pT selection mask
    pt_mask = (jet_pt >= pt_range[0]) & (jet_pt <= pt_range[1])
    logger.info(f"Jets in pT range [{pt_range[0]}, {pt_range[1]}] GeV: {pt_mask.sum()}")
    
    # Find top jets with tau32 < 0.2 and in pT range
    top_mask = (truth_labels == 1) & (tau32_ratios < 0.2) & pt_mask
    top_indices = np.where(top_mask)[0]
    
    # Find QCD jets with tau32 > 0.75 and in pT range
    qcd_mask = (truth_labels == 0) & (tau1 < 0.25) & pt_mask
    qcd_indices = np.where(qcd_mask)[0]
    
    logger.info(f"Found {len(top_indices)} top jets with tau32 < 0.2 in pT range")
    logger.info(f"Found {len(qcd_indices)} QCD jets with tau32 > 0.75 in pT range")
    
    # Select the first qualifying jet from each category
    top_idx = top_indices[0] if len(top_indices) > 0 else None
    qcd_idx = qcd_indices[0] if len(qcd_indices) > 0 else None
    
    if top_idx is not None:
        logger.info(f"Selected top jet at index {top_idx} with tau32 = {tau32_ratios[top_idx]:.4f}, pT = {jet_pt[top_idx]:.1f} GeV")
    if qcd_idx is not None:
        logger.info(f"Selected QCD jet at index {qcd_idx} with tau32 = {tau32_ratios[qcd_idx]:.4f}, pT = {jet_pt[qcd_idx]:.1f} GeV")
    
    return top_idx, qcd_idx, top_indices, qcd_indices


def plot_qfi_matrix(qfi_matrix: np.ndarray, jet_type: str, tau32_value: float, 
                   output_dir: str, save_name: str, set_zero_diag: bool = False) -> None:
    """
    Plot a single QFI matrix with quantum circuit style formatting.
    
    Args:
        qfi_matrix: QFI matrix of shape (30, 30)
        jet_type: Type of jet ('Top' or 'QCD')
        tau32_value: tau3/tau2 ratio value
        output_dir: Directory to save plots
        save_name: Filename for saving (without extension)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_name+= "_QFI"
    save_path = output_path / save_name
    # matrix is of shape NxN. Set all elements in diagonal, diagonal-1 and diagonal+1 to zero for better visualization
    
    # Get matrix dimensions
    N_params = qfi_matrix.shape[0]  # Should be 30
    N_qubits = N_params // 3        # Should be 10
    # set diagonal to zero for better visualization
    #np.fill_diagonal(qfi_matrix, 0)
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Color scheme
    colors = ['#0066FF', 'white', '#FF0066']  # Blue -> White -> Red
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue_white_red', colors, N=256)
    
    # Set up normalization
    plot_matrix = qfi_matrix.copy()
    # set self correlations to zero for better visualisation
    if set_zero_diag:
        for i in range(N_qubits):
            plot_matrix[3*i:3*i+3, 3*i:3*i+3] = 0
    # Determine data range for better visualization
    max_abs_value = np.max(np.abs(plot_matrix))
    data_range = 0.4#max_abs_value if max_abs_value > 0.5 else 0.5
    norm = matplotlib.colors.Normalize(vmin=-data_range, vmax=data_range)
    
    # Plot the matrix
    im = ax.matshow(plot_matrix, cmap=cmap, norm=norm)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('QFI Value', fontsize=16)
    
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
        ax.text(pos, 3*N_qubits+0.1, label, ha='center', va='top', fontsize=12)
        ax.text(-0.7, pos, label, ha='right', va='center', fontsize=12)
    
    # Add qubit number labels
    qubit_positions = [1 + 3*i for i in range(N_qubits)]
    qubit_labels = [str(i) for i in range(N_qubits)]
    
    for pos, label in zip(qubit_positions, qubit_labels):
        ax.text(pos, -0.3, label, ha='center', va='top', fontsize=14, fontweight='bold')
        ax.text(-1.4, pos, label, ha='right', va='center', fontsize=14, fontweight='bold')
    
    # Set labels and title with tau32 value
    ax.set_xlabel('Qubit Number', labelpad=40, fontsize=19)
    ax.set_ylabel('Qubit Number', labelpad=40, fontsize=19)
    plt.title(f'{jet_type} Jet QFI Matrix (τ₃₂ = {tau32_value:.4f})', fontsize=22, pad=20)
    plt.tight_layout()
    
    # Save in both formats
    plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()
    
    # Print statistics
    logger.info(f"QFI Matrix Statistics for {jet_type} jet:")
    logger.info(f"  Range: [{plot_matrix.min():.6f}, {plot_matrix.max():.6f}]")
    logger.info(f"  Mean: {plot_matrix.mean():.6f}")
    logger.info(f"  Std:  {plot_matrix.std():.6f}")
    logger.info(f"  Saved to: {save_path}.png/.pdf")


def plot_jet_constituents_2d(jet_constituents: np.ndarray, jet_type: str, 
                             jet_pt: float, tau32_value: float,
                             output_dir: str, save_name: str, n_particles: int = 10) -> None:
    """
    Plot 2D eta-phi distribution of top N jet constituents.
    
    Args:
        jet_constituents: Array of shape (100, 3) with normalized pt, eta, phi
        jet_type: Type of jet ('Top' or 'QCD')
        jet_pt: Jet transverse momentum in GeV
        tau32_value: tau3/tau2 ratio value
        output_dir: Directory to save plots
        save_name: Filename for saving (without extension)
        n_particles: Number of highest pT particles to plot
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f"{save_name}_constituents"
    
    # Extract top N particles
    top_particles = jet_constituents[:n_particles] # already sorted
    top_pts = top_particles[:, 0]
    top_etas = top_particles[:, 1]
    top_phis = top_particles[:, 2]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Set up the plot
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot particles as circles
    # Color map from highest pT (red) to lowest pT (blue)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_particles))
    
    for i, (pt, eta, phi) in enumerate(zip(top_pts, top_etas, top_phis)):
        # Skip particles with zero pT
        if pt > 0:
            circle = patches.Circle((phi, eta), radius=pt, 
                                   color=colors[i], alpha=0.4, 
                                   edgecolor=colors[i], linewidth=2)
            ax.add_patch(circle)
            
            # Add text label for particle rank
            #ax.text(phi, eta, str(i+1), ha='center', va='center', fontsize=10, fontweight='bold')
            ax.text(phi + pt*np.sqrt(np.random.uniform(0, 0.7))*np.cos(np.random.uniform(0, 2*np.pi)), \
                eta + pt*np.sqrt(np.random.uniform(0, 0.7))*np.sin(np.random.uniform(0, 2*np.pi)),\
                     str(i+1), ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add axis lines at origin
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel(r'$\phi - \phi_\mathrm{jet}$', fontsize=19)
    ax.set_ylabel(r'$\eta - \eta_\mathrm{jet}$', fontsize=19)
    plt.title(f'{jet_type} Jet Constituents (Top {n_particles})\n' + 
         f'$p_{{\mathrm{{T}}}}^{{\mathrm{{jet}}}}$ = {jet_pt:.1f} GeV, $\\tau_{{32}}$ = {tau32_value:.4f}', 
         fontsize=22, pad=15)
    
    # Add colorbar for pT
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                               norm=plt.Normalize(vmin=min(top_pts[top_pts > 0]), 
                                                 vmax=max(top_pts)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    # add space between colorbar and label
    cbar.set_label(r'$p_{{\mathrm{{T}}}}^\mathrm{particle} / p_{{\mathrm{{T}}}}^\mathrm{jet}$', fontsize=16, labelpad=10)
    plt.tight_layout()
    
    # Save in both formats
    plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved constituent plot for {jet_type} jet to: {save_path}.png/.pdf")
    logger.info(f"  Top {n_particles} particle pT/jet_pT range: [{top_pts[top_pts > 0].min():.4f}, {top_pts.max():.4f}]")

def QFI_reducer(qfi_matrices: np.ndarray) -> np.ndarray:
    """
    Reduce N,30,30 QFI matrices to N,10,10 by computing Frobenius norms of 3x3 blocks.
    
    Each 3x3 submatrix block is reduced to its Frobenius norm, normalized by sqrt(9)=3
    to ensure values are bounded in [0,1].
    
    Args:
        qfi_matrices: Array of shape (N, 30, 30) containing QFI matrices
        
    Returns:
        reduced_qfi: Array of shape (N, 10, 10) with Frobenius norms of 3x3 blocks
    """
    N = qfi_matrices.shape[0]
    reduced_qfi = np.zeros((N, 10, 10), dtype=np.float32)
    
    # Iterate over 10x10 grid of 3x3 blocks
    for i in range(10):
        for j in range(10):
            # Extract 3x3 block for all N matrices at once
            block = qfi_matrices[:, 3*i:3*(i+1), 3*j:3*(j+1)]  # [N, 3, 3]
            
            # Compute Frobenius norm: sqrt(sum of squared elements)
            frobenius_norm = np.sqrt(np.sum(block**2, axis=(1, 2)))  # [N]
            
            # Normalize by sqrt(9) = 3 to bound in [0, 1]
            reduced_qfi[:, i, j] = frobenius_norm / 3.0
    
    return reduced_qfi


def plot_2d_jet_lines(jet_constituents: np.ndarray, reduced_qfi: np.ndarray,
                      jet_type: str, jet_pt: float, tau32_value: float,
                      output_dir: str, save_name: str, n_particles: int = 10) -> None:
    """
    Plot 2D eta-phi distribution of top N jet constituents with QFI-weighted lines.
    
    Args:
        jet_constituents: Array of shape (100, 3) with normalized pt, eta, phi
        reduced_qfi: Reduced QFI matrix of shape (10, 10)
        jet_type: Type of jet ('Top' or 'QCD')
        jet_pt: Jet transverse momentum in GeV
        tau32_value: tau3/tau2 ratio value
        output_dir: Directory to save plots
        save_name: Filename for saving (without extension)
        n_particles: Number of highest pT particles to plot
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / f"{save_name}_qfi_lines"
    
    # Extract top N particles
    top_particles = jet_constituents[:n_particles]  # already sorted
    top_pts = top_particles[:, 0]
    top_etas = top_particles[:, 1]
    top_phis = top_particles[:, 2]
    
    # Filter out zero-pT particles
    valid_mask = top_pts > 0
    valid_pts = top_pts[valid_mask]
    valid_etas = top_etas[valid_mask]
    valid_phis = top_phis[valid_mask]
    n_valid = len(valid_pts)
    pt_max = 0.25#valid_pts.max()

    if n_valid == 0:
        logger.warning(f"No valid particles with pt > 0 for {save_name}")
        return
    
    # Compute zoomed-in limits with wiggle factor
    eta_min, eta_max = valid_etas.min() - 0.05, valid_etas.max() + 0.05
    phi_min, phi_max = valid_phis.min() - 0.05, valid_phis.max() + 0.05
    
    # Create figure with extra space for colorbars
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Set up the plot
    ax.set_xlim(phi_min, phi_max)
    ax.set_ylim(eta_min, eta_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    #divider = make_axes_locatable(ax)

# Create colorbar axes with same size
    #cax_qfi = divider.append_axes("right", size="3%", pad=0.1)
    #cax_pt = divider.append_axes("right", size="3%", pad=0.35)

    # Extract upper triangle of reduced QFI (excluding diagonal)
    qfi_upper_indices = np.triu_indices(n_particles, k=1)
    qfi_values = reduced_qfi[qfi_upper_indices]
    
    qfi_min = 0.0#qfi_values.min()
    qfi_max = 0.275#qfi_values.max()
    
    # Plot QFI-weighted lines between particle pairs
    # Use reds colormap for QFI values
    qfi_cmap = plt.cm.Reds # use reds
    qfi_norm = plt.Normalize(vmin=qfi_min, vmax=qfi_max)

    for idx in range(len(qfi_upper_indices[0])):
        i = qfi_upper_indices[0][idx]
        j = qfi_upper_indices[1][idx]
        
        # Only plot if both particles are valid
        if i < n_valid and j < n_valid:
            qfi_val = qfi_values[idx]
            line_color = qfi_cmap(qfi_norm(qfi_val))
            
            ax.plot([valid_phis[i], valid_phis[j]], 
                   [valid_etas[i], valid_etas[j]],
                   color=line_color, linewidth=1.25, alpha=0.8, zorder=1)
    
    # Plot particles as fixed-radius circles with viridis colormap
    # Create custom viridis with white at pt=0
    viridis_colors = plt.cm.viridis(np.linspace(0, 1, 256))
    viridis_colors[0] = [1, 1, 1, 1]  # Set lowest value to white
    pt_cmap = LinearSegmentedColormap.from_list('viridis_white', viridis_colors)
    
    pt_norm = plt.Normalize(vmin=0, vmax=pt_max)
    radius_plot = 0.03*(phi_max - phi_min)
    for i, (pt, eta, phi) in enumerate(zip(valid_pts, valid_etas, valid_phis)):
        circle_color = pt_cmap(pt_norm(pt))
        circle = patches.Circle((phi, eta), radius=radius_plot,
                               color=circle_color, alpha=0.9,
                               edgecolor='black', linewidth=1.5, zorder=2)
        ax.add_patch(circle)
        
        # Add particle rank label with slight offset
        offset_r = 1.15*radius_plot
        offset_angle = np.random.uniform(0, 2*np.pi)
        ax.text(phi + offset_r * np.cos(offset_angle),
               eta + offset_r * np.sin(offset_angle),
               str(i+1), ha='center', va='center', 
               fontsize=9, fontweight='bold', zorder=3)
    
    # Add axis lines at origin (if in range)
    if eta_min <= 0 <= eta_max:
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
    if phi_min <= 0 <= phi_max:
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel(r'$\phi - \phi_\mathrm{jet}$', fontsize=19)
    ax.set_ylabel(r'$\eta - \eta_\mathrm{jet}$', fontsize=19)
    plt.title(f'{jet_type} Jet Constituents with QFI Correlations\n' +
         f'$p_{{\mathrm{{T}}}}^{{\mathrm{{jet}}}}$ = {jet_pt:.1f} GeV, $\\tau_{{32}}$ = {tau32_value:.4f}',
         fontsize=20, pad=15)
    # fig.suptitle(f'{jet_type} Jet Constituents with QFI Correlations\n' +
    #          f'$p_{{\mathrm{{T}}}}^{{\mathrm{{jet}}}}$ = {jet_pt:.1f} GeV, $\\tau_{{32}}$ = {tau32_value:.4f}',
    #          fontsize=20, y=0.98, pad=15)

    # Add colorbar for pT with sufficient spacing
    pt_sm = plt.cm.ScalarMappable(cmap=pt_cmap, norm=pt_norm)
    pt_sm.set_array([])
    cbar_pt = plt.colorbar(pt_sm, ax=ax, fraction=0.036, pad=0.12, aspect=20, shrink=0.8)
    cbar_pt.set_label(r'$p_{\mathrm{T}}^\mathrm{particle} / p_{\mathrm{T}}^\mathrm{jet}$',
                      fontsize=14, labelpad=12)
    #ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)
    # Add colorbar for QFI with sufficient spacing
    qfi_sm = plt.cm.ScalarMappable(cmap=qfi_cmap, norm=qfi_norm)
    qfi_sm.set_array([])
    cbar_qfi = plt.colorbar(qfi_sm, ax=ax, fraction=0.036, pad=0.04, aspect=20, shrink=0.8)
    cbar_qfi.set_label(r'Reduced QFI ($||\mathcal{Q}_{kl}||_F$)',
                       fontsize=14, labelpad=12)
    
    plt.tight_layout()
    
    # Save in both formats
    plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved QFI line plot for {jet_type} jet to: {save_path}.png/.pdf")
    logger.info(f"  Valid particles: {n_valid}/{n_particles}")
    logger.info(f"  QFI range: [{qfi_values.min():.4f}, {qfi_values.max():.4f}]")

def main():
    """Main execution function."""
    # File path and parameters
    file_path = "/ceph/abal/QML/qGNN/merged/test/TTBar+ZJets_070.h5"
    output_dir = "QFI_plots"
    pt_range = (950, 1000)  # GeV
    
    # Load data
    qfi_matrices, extra_features, truth_labels, jet_pt, jet_constituents_norm = load_jet_data(file_path)
    
    # Calculate tau32 ratios
    tau32_ratios = calculate_tau32_ratios(extra_features)
    tau1 = extra_features[:, 2]  # tau1 is at index 2
    # Find representative jets
    top_idx, qcd_idx, top_indices, qcd_indices = find_representative_jets(tau32_ratios, truth_labels, jet_pt, pt_range, tau1)
    
    reduced_qfi_matrix = QFI_reducer(qfi_matrices)  # Returns (N, 10, 10)

    # select random different indices
    top_idx = np.random.choice(top_indices, size=1)[0] if len(top_indices)>0 else None
    qcd_idx = np.random.choice(qcd_indices, size=1)[0] if len(qcd_indices)>0 else None
    print("Selected top jet index:", top_idx)
    print("Selected QCD jet index:", qcd_idx)
    # Call the plotting function with QFI lines
    plot_2d_jet_lines(
        jet_constituents=jet_constituents_norm[top_idx],  # Shape (100, 3)
        reduced_qfi=reduced_qfi_matrix[top_idx],         # Shape (10, 10)
        jet_type='Top',                  # 'Top' or 'QCD'
        jet_pt=jet_pt[top_idx],                      # Jet pT in GeV
        tau32_value=tau32_ratios[top_idx],            # tau3/tau2 ratio
        output_dir=output_dir,              # Directory path
        save_name=f"top_example_{top_idx}",                # Filename prefix
        n_particles=10                      # Number of particles to plot
    )
    plot_2d_jet_lines(
        jet_constituents=jet_constituents_norm[qcd_idx],  # Shape (100, 3)
        reduced_qfi=reduced_qfi_matrix[qcd_idx],         # Shape (10, 10)
        jet_type='QCD',                  # 'Top' or 'QCD'
        jet_pt=jet_pt[qcd_idx],                      # Jet pT in GeV
        tau32_value=tau32_ratios[qcd_idx],            # tau3/tau2 ratio
        output_dir=output_dir,              # Directory path
        save_name=f"qcd_example_{qcd_idx}",                # Filename prefix
        n_particles=10                      # Number of particles to plot
    )
    # Plot top jet QFI matrix and constituents
    if top_idx is not None:
        # Plot QFI matrix
        plot_qfi_matrix(
            qfi_matrices[top_idx],
            jet_type="Top",
            tau32_value=tau32_ratios[top_idx],
            output_dir=output_dir,
            save_name=f"top_example_{top_idx}"
        )
        
        # Plot jet constituents
        # plot_jet_constituents_2d(
        #     jet_constituents_norm[top_idx],
        #     jet_type="Top",
        #     jet_pt=jet_pt[top_idx],
        #     tau32_value=tau32_ratios[top_idx],
        #     output_dir=output_dir,
        #     save_name=f"top_example_{top_idx}"
        # )
    else:
        logger.warning(f"No top jet found with tau32 < 0.2 in pT range {pt_range} GeV")
    
    # Plot QCD jet QFI matrix and constituents
    if qcd_idx is not None:
        # Plot QFI matrix
        plot_qfi_matrix(
            qfi_matrices[qcd_idx],
            jet_type="QCD",
            tau32_value=tau32_ratios[qcd_idx],
            output_dir=output_dir,
            save_name=f"qcd_example_{qcd_idx}"
        )
        
        # Plot jet constituents
        # plot_jet_constituents_2d(
        #     jet_constituents_norm[qcd_idx],
        #     jet_type="QCD",
        #     jet_pt=jet_pt[qcd_idx],
        #     tau32_value=tau32_ratios[qcd_idx],
        #     output_dir=output_dir,
        #     save_name=f"qcd_example_{qcd_idx}"
        # )
    else:
        logger.warning(f"No QCD jet found with tau32 > 0.75 in pT range {pt_range} GeV")
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()