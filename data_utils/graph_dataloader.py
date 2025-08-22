"""
Author: Aritra Bal, ETP
Date: XVIII Sextilis anno ab urbe condita MMDCCLXXVIII

Simplified streaming PyTorch Geometric DataLoader for jet classification.
Reads batches of jets at once for maximum efficiency.
"""

import h5py
import torch
import numpy as np
from typing import List, Iterator
from torch_geometric.data import Data, Batch
import time

class StreamingJetDataLoader:
    """
    Simple streaming DataLoader that reads batches of jets efficiently.
    """
    
    def __init__(
        self,
        h5_files: List[str],
        batch_size: int = 32,
        use_qfi_correlations: bool = True
    ):
        """
        Initialize streaming dataloader.
        
        Args:
            h5_files: List of paths to H5 files
            batch_size: Batch size for yielding
            use_qfi_correlations: If True, use QFI correlations; if False, use identity baseline
        """
        self.h5_files = h5_files
        self.batch_size = batch_size
        self.use_qfi_correlations = use_qfi_correlations
        if not self.use_qfi_correlations:
            print('#'*50)
            print("Using identity baseline (no QFI correlations)")
            print('#'*50)
            
            time.sleep(5)
        
        # Create static edge connectivity (fully connected graph)
        max_particles = 10
        sources = torch.arange(max_particles).repeat(max_particles)
        targets = torch.arange(max_particles).repeat_interleave(max_particles)
        self.edge_index = torch.stack([sources, targets], dim=0)  # [2, 100]
        
        # Reset state
        self.current_file_idx = 0
        self.current_jet_idx = 0
        self.current_file = None
        self.current_file_size = 0
    
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
    
    def _close_current_file(self) -> None:
        """Close current file."""
        if self.current_file is not None:
            self.current_file.close()
            self.current_file = None
    
    def _read_batch(self, start_idx: int, end_idx: int) -> Batch:
        """Read a batch of jets from current file."""
        # Read batch data at once
        node_features = self.current_file['jetConstituentsList'][start_idx:end_idx]  # [batch, 10, 3]
        qfi_matrices = 4*self.current_file['jetConstituentsQFI'][start_idx:end_idx]    # [batch, 30, 30], the multiplication by 4 is to bring the bounds to [-1,1]
        labels = self.current_file['truth_labels'][start_idx:end_idx]               # [batch]
        jet_pts = self.current_file['jetFeatures'][start_idx:end_idx, 0]           # [batch]
        
        # Normalize particle pt by jet pt
        node_features = node_features.copy()
        node_features[..., 0] = node_features[..., 0] / jet_pts[:, None]
        
        # Create list of PyG Data objects
        data_list = []
        for i in range(len(labels)):
            # Node features
            x = torch.tensor(node_features[i], dtype=torch.float32)  # [10, 3]
            
            # Edge features from QFI matrix
            if self.use_qfi_correlations:
                edge_attr = self._extract_edge_features(qfi_matrices[i])  # [100, 9]
            else:
                # Identity baseline
                identity_flat = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32) # zero matrix for baseline, use only x_i and x_j for the message
                edge_attr = identity_flat.repeat(100, 1)  # [100, 9]
            
            # Label
            y = torch.tensor(labels[i], dtype=torch.long)
            
            data_list.append(Data(
                x=x,
                edge_index=self.edge_index.clone(),
                edge_attr=edge_attr,
                y=y
            ))
        
        return Batch.from_data_list(data_list)
    
    def _extract_edge_features(self, qfi_matrix: np.ndarray) -> torch.Tensor:
        """Extract 3x3 submatrices from QFI correlation matrix for each edge."""
        # Get source and destination indices for all edges
        src_nodes = self.edge_index[0].numpy()  # [100]
        dst_nodes = self.edge_index[1].numpy()  # [100]
        
        # Vectorized extraction of 3x3 submatrices
        src_base = src_nodes * 3
        dst_base = dst_nodes * 3
        
        # Create indices for 3x3 submatrix extraction
        row_offsets = np.tile([0, 0, 0, 1, 1, 1, 2, 2, 2], len(src_nodes))
        col_offsets = np.tile([0, 1, 2, 0, 1, 2, 0, 1, 2], len(dst_nodes))
        
        row_indices = np.repeat(src_base, 9) + row_offsets
        col_indices = np.repeat(dst_base, 9) + col_offsets
        
        # Extract all elements and reshape to [100, 9]
        extracted = qfi_matrix[row_indices, col_indices].reshape(-1, 9)
        
        return torch.tensor(extracted, dtype=torch.float32)
    
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

if __name__ == "__main__":
    import tqdm
    
    # Example file paths
    train_files = [
        "/ceph/abal/QML/qGNN/train/ZJetsToNuNu_008.h5",
        #"/ceph/abal/QML/qGNN/val/ZJetsToNuNu_120.h5",
    ]
    
    # Create dataloader
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
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Processed {total_jets_processed} jets in {processing_time:.2f} seconds")
    print(f"Rate: {total_jets_processed/processing_time:.0f} jets/second")