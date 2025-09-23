"""
Author: Aritra Bal, ETP
Date: ante diem quartum Nonas Ianuarias anno ab urbe condita MMDCCLXXVIII

Fully vectorized streaming PyTorch Geometric DataLoader for jet classification using QFI matrices.
Eliminates all nested loops for maximum performance.
"""
import h5py
import torch
import numpy as np
from typing import List, Iterator
from torch_geometric.data import Data, Batch
import time
from loguru import logger


class StreamingJetDataLoader:
    """
    Vectorized streaming DataLoader that creates graphs directly from QFI matrices.
    Each QFI matrix becomes a 30-node undirected graph where:
    - Node features: QFI diagonal elements (30 nodes, 1D features)
    - Edge features: QFI off-diagonal elements (870 edges, 1D features)
    - Graph is undirected since QFI is symmetric
    """
    
    def __init__(
        self,
        h5_files: List[str],
        batch_size: int = 32,
        use_qfi_correlations: bool = True
    ):
        """
        Initialize QFI graph dataloader.
        
        Args:
            h5_files: List of paths to H5 files
            batch_size: Batch size for yielding
            use_qfi_correlations: If True, use QFI values; if False, use identity matrix
        """
        self.h5_files = h5_files
        self.batch_size = batch_size
        self.use_qfi_correlations = use_qfi_correlations
        
        if not self.use_qfi_correlations:
            logger.warning("Identity baseline mode: QFI values will be replaced with identity matrix")
            time.sleep(3)
        
        # Create edge connectivity for 30-node undirected graph
        self.num_nodes = 30
        self.edge_index, self.src_indices, self.tgt_indices = self._create_edge_connectivity()
        
        # Pre-compute identity baseline edge features if needed
        if not self.use_qfi_correlations:
            self._precompute_identity_edges()
        
        # Reset state
        self.current_file_idx = 0
        self.current_jet_idx = 0
        self.current_file = None
        self.current_file_size = 0
        
        logger.info(f"QFI Graph structure: {self.num_nodes} nodes, {self.edge_index.shape[1]} edges")
        
    def _create_edge_connectivity(self) -> tuple:
        """
        Create edge connectivity and index mappings for vectorized edge extraction.
        
        Returns:
            Tuple of (edge_index, src_indices, tgt_indices)
        """
        sources = []
        targets = []
        
        # Create all pairs (i,j) where i != j for fully connected graph
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:  # Exclude self-loops
                    sources.append(i)
                    targets.append(j)
        
        edge_index = torch.stack([torch.tensor(sources, dtype=torch.long), 
                                 torch.tensor(targets, dtype=torch.long)], dim=0)
        
        # Store indices for vectorized extraction
        src_indices = torch.tensor(sources, dtype=torch.long)
        tgt_indices = torch.tensor(targets, dtype=torch.long)
        logger.debug(f"Created edge connectivity: {edge_index.shape[1]} edges for {self.num_nodes}-node graph")
        return edge_index, src_indices, tgt_indices
    
    def _precompute_identity_edges(self) -> None:
        """Pre-compute edge features for identity matrix baseline."""
        identity_matrix = torch.eye(self.num_nodes, dtype=torch.float32)
        self.identity_edge_features = identity_matrix[self.src_indices, self.tgt_indices].unsqueeze(-1)
        logger.debug("Pre-computed identity edge features")
    
    def __len__(self) -> int:
        """Return number of batches."""
        if not hasattr(self, '_total_jets'):
            self._total_jets = 0
            for file_path in self.h5_files:
                with h5py.File(file_path, 'r') as f:
                    self._total_jets += f['truth_labels'].shape[0]
        return (self._total_jets + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Batch]:
        """Initialize iterator."""
        self.current_file_idx = 0
        self.current_jet_idx = 0
        self._open_current_file()
        return self
    
    def __next__(self) -> Batch:
        """Get next batch."""
        # Move to next file if current is exhausted
        while self.current_jet_idx >= self.current_file_size:
            self._close_current_file()
            self.current_file_idx += 1
            if self.current_file_idx >= len(self.h5_files):
                raise StopIteration
            self._open_current_file()
            self.current_jet_idx = 0
        
        # Read batch of jets
        batch_end = min(self.current_jet_idx + self.batch_size, self.current_file_size)
        batch_data = self._read_batch(self.current_jet_idx, batch_end)
        self.current_jet_idx = batch_end
        return batch_data
    
    def _open_current_file(self) -> None:
        """Open current file."""
        file_path = self.h5_files[self.current_file_idx]
        self.current_file = h5py.File(file_path, 'r')
        self.current_file_size = self.current_file['truth_labels'].shape[0]
        logger.debug(f"Opened file: {file_path} with {self.current_file_size} jets")
    
    def _close_current_file(self) -> None:
        """Close current file."""
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
    
    def _read_batch(self, start_idx: int, end_idx: int) -> Batch:
        """
        Fully vectorized batch reading that eliminates all loops.
        
        Args:
            start_idx: Starting jet index in current file
            end_idx: Ending jet index in current file
            
        Returns:
            PyG Batch object with QFI graphs
        """
        # Read batch data at once
        qfi_matrices = self.current_file['jetConstituentsQFI'][start_idx:end_idx]  # [batch_size, 30, 30]
        labels = self.current_file['truth_labels'][start_idx:end_idx]              # [batch_size]
        
        # Scale QFI matrices by 4.0
        qfi_matrices = 4.0 * qfi_matrices
        batch_size = len(labels)
        
        # Convert to tensors once
        if self.use_qfi_correlations:
            qfi_tensors = torch.from_numpy(qfi_matrices).float()  # [batch_size, 30, 30]
        else:
            # Use identity matrices for all graphs
            qfi_tensors = torch.eye(self.num_nodes, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Extract node features (diagonal elements) vectorially
        x_batched = torch.diagonal(qfi_tensors, dim1=-2, dim2=-1).unsqueeze(-1)  # [batch_size, 30, 1]
        x_flat = x_batched.flatten(0, 1)  # [batch_size * 30, 1]
        
        # Create batched edge features vectorially
        if self.use_qfi_correlations:
            # Extract all edge features at once using advanced indexing
            edge_attr_batched = qfi_tensors[:, self.src_indices, self.tgt_indices]  # [batch_size, num_edges]
            edge_attr_batched = edge_attr_batched.unsqueeze(-1)  # [batch_size, num_edges, 1]
            edge_attr_flat = edge_attr_batched.flatten(0, 1)  # [batch_size * num_edges, 1]
        else:
            # Use pre-computed identity edge features
            edge_attr_flat = self.identity_edge_features.repeat(batch_size, 1)  # [batch_size * num_edges, 1]
        
        # Create batched edge index
        num_edges = self.edge_index.shape[1]
        edge_index_batched = self.edge_index.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        # Add node offsets for each graph in batch vectorially
        offsets = torch.arange(batch_size, dtype=torch.long).unsqueeze(-1).unsqueeze(-1) * self.num_nodes
        edge_index_batched += offsets  # Broadcasting handles the addition
        
        edge_index_flat = edge_index_batched.permute(1, 0, 2).reshape(2, -1)  # [2, batch_size * num_edges]
        
        # Create batch assignment tensor
        batch_tensor = torch.repeat_interleave(torch.arange(batch_size, dtype=torch.long), self.num_nodes)
        
        # Convert labels to tensor
        y_tensor = torch.from_numpy(labels).long()
        # Create single Batch object directly (no individual Data objects!)
        return Batch(
            x=x_flat,
            edge_index=edge_index_flat,
            edge_attr=edge_attr_flat,
            y=y_tensor,
            batch=batch_tensor
        )
    
    def __del__(self):
        """Cleanup file handle."""
        self._close_current_file()


def get_total_jets(h5_files: List[str]) -> int:
    """Get total number of jets across all files."""
    total = 0
    for file_path in h5_files:
        with h5py.File(file_path, 'r') as f:
            total += f['truth_labels'].shape[0]
    return total


def analyze_qfi_statistics_vectorized(h5_files: List[str], max_jets: int = 1000) -> dict:
    """
    Vectorized QFI matrix statistics analysis.
    
    Args:
        h5_files: List of H5 file paths
        max_jets: Maximum number of jets to analyze
        
    Returns:
        Dictionary with QFI statistics
    """
    logger.info(f"Analyzing QFI statistics from up to {max_jets} jets (vectorized)...")
    
    all_diagonal = []
    all_off_diagonal = []
    jets_processed = 0
    
    for file_path in h5_files:
        if jets_processed >= max_jets:
            break
            
        with h5py.File(file_path, 'r') as f:
            file_size = f['truth_labels'].shape[0]
            jets_to_read = min(max_jets - jets_processed, file_size)
            
            # Read batch of QFI matrices at once
            qfi_matrices = 4.0 * f['jetConstituentsQFI'][:jets_to_read]  # [jets, 30, 30]
            
            # Extract diagonal elements vectorially
            diagonal_elements = np.diagonal(qfi_matrices, axis1=-2, axis2=-1)  # [jets, 30]
            all_diagonal.append(diagonal_elements.flatten())
            
            # Extract upper triangular off-diagonal elements vectorially
            triu_indices = np.triu_indices(30, k=1)  # Upper triangular indices
            off_diagonal_elements = qfi_matrices[:, triu_indices[0], triu_indices[1]]  # [jets, 435]
            all_off_diagonal.append(off_diagonal_elements.flatten())
            
            jets_processed += jets_to_read
    
    # Concatenate all data
    diagonal_elements = np.concatenate(all_diagonal)
    off_diagonal_elements = np.concatenate(all_off_diagonal)
    
    statistics = {
        'jets_analyzed': jets_processed,
        'diagonal_stats': {
            'mean': float(np.mean(diagonal_elements)),
            'std': float(np.std(diagonal_elements)),
            'min': float(np.min(diagonal_elements)),
            'max': float(np.max(diagonal_elements)),
            'count': len(diagonal_elements)
        },
        'off_diagonal_stats': {
            'mean': float(np.mean(off_diagonal_elements)),
            'std': float(np.std(off_diagonal_elements)),
            'min': float(np.min(off_diagonal_elements)),
            'max': float(np.max(off_diagonal_elements)),
            'count': len(off_diagonal_elements)
        }
    }
    
    logger.info("Vectorized QFI Statistics:")
    logger.info(f"  Diagonal elements - Mean: {statistics['diagonal_stats']['mean']:.4f}, "
               f"Std: {statistics['diagonal_stats']['std']:.4f}")
    logger.info(f"  Off-diagonal elements - Mean: {statistics['off_diagonal_stats']['mean']:.4f}, "
               f"Std: {statistics['off_diagonal_stats']['std']:.4f}")
    
    return statistics


if __name__ == "__main__":
    import tqdm
    
    # Example file paths
    train_files = [
        "/ceph/abal/QML/qGNN/merged/train/TTBar+ZJets_001.h5",
    ]
    
    # Test vectorized QFI graph dataloader
    logger.info("Testing Vectorized QFI Graph DataLoader")
    dataloader = StreamingJetDataLoader(
        h5_files=train_files,
        batch_size=64,
        use_qfi_correlations=True
    )
    
    print(f"Total batches: {len(dataloader)}")
    print(f"Total jets: {get_total_jets(train_files)}")
    
    # Time the iteration
    start_time = time.time()
    total_jets_processed = 0
    
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        total_jets_processed += batch.num_graphs
        
        # Print stats for first batch
        if batch_idx == 0:
            print(f"Batch shape - x: {batch.x.shape}, edge_index: {batch.edge_index.shape}")
            print(f"Edge attr shape: {batch.edge_attr.shape}, y shape: {batch.y.shape}")
            print(f"First batch size: {batch.num_graphs}")
            print(f"Node features range: [{batch.x.min():.4f}, {batch.x.max():.4f}]")
            print(f"Edge features range: [{batch.edge_attr.min():.4f}, {batch.edge_attr.max():.4f}]")
            
        # Process only first few batches for testing
        if batch_idx >= 10:
            break
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processed {total_jets_processed} jets in {processing_time:.2f} seconds")
    print(f"Rate: {total_jets_processed/processing_time:.0f} jets/second")
    
    # Compare with vectorized statistics analysis
    print("\n" + "="*50)
    print("VECTORIZED QFI STATISTICS ANALYSIS")
    print("="*50)
    stats = analyze_qfi_statistics_vectorized(train_files, max_jets=1000)
    
    print(f"\nAnalyzed {stats['jets_analyzed']} jets")
    print("Diagonal Elements (Node Features):")
    print(f"  Range: [{stats['diagonal_stats']['min']:.4f}, {stats['diagonal_stats']['max']:.4f}]")
    print(f"  Mean ± Std: {stats['diagonal_stats']['mean']:.4f} ± {stats['diagonal_stats']['std']:.4f}")
    
    print("Off-diagonal Elements (Edge Features):")
    print(f"  Range: [{stats['off_diagonal_stats']['min']:.4f}, {stats['off_diagonal_stats']['max']:.4f}]")
    print(f"  Mean ± Std: {stats['off_diagonal_stats']['mean']:.4f} ± {stats['off_diagonal_stats']['std']:.4f}")