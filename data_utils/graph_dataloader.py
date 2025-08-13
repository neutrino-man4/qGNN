"""
Author: Aritra Bal, ETP
Date: XIII Idibus Sextilibus anno ab urbe condita MMDCCLXXVIII

PyTorch Geometric DataLoader for jet classification using graph neural networks.
Constructs fully connected graphs with correlation-based edge features.
"""

import h5py
import torch
import numpy as np
from typing import List, Optional, Tuple
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import os
from pathlib import Path


class JetGraphDataset(Dataset):
    """
    Dataset for loading jet data as graphs for GNN-based classification.
    
    Each jet becomes a fully connected directed graph with:
    - 10 nodes (particles) with 3 features each (pt, deta, dphi)
    - 100 directed edges with 9 features each (from QFI correlation matrix)
    """
    
    def __init__(
        self, 
        h5_files: List[str],
        max_particles: int = 10,
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            h5_files: List of paths to H5 files
            max_particles: Number of particles per jet (nodes in graph)
            transform: Optional transform to apply to each Data object
            pre_transform: Optional pre-transform to apply during processing
        """
        self.h5_files = h5_files
        self.max_particles = max_particles
        self.transform = transform
        self.pre_transform = pre_transform
        
        # Cache static edge connectivity (same for all graphs)
        self.edge_index = self._create_edge_index()
        
        # Build index mapping: global_idx -> (file_idx, local_jet_idx)
        self.index_map = self._build_index_map()
        
        super().__init__()
    
    def _create_edge_index(self) -> torch.Tensor:
        """Create fully connected directed graph edge index."""
        sources = torch.arange(self.max_particles).repeat(self.max_particles)
        targets = torch.arange(self.max_particles).repeat_interleave(self.max_particles)
        return torch.stack([sources, targets], dim=0)  # [2, 100]
    
    def _build_index_map(self) -> List[Tuple[int, int]]:
        """Build mapping from global index to (file_index, local_jet_index)."""
        index_map = []
        
        for file_idx, h5_file in enumerate(self.h5_files):
            with h5py.File(h5_file, 'r') as f:
                n_jets = f['truth_labels'].shape[0]
                for local_idx in range(n_jets):
                    index_map.append((file_idx, local_idx))
        
        return index_map
    
    def len(self) -> int:
        """Return total number of jets across all files."""
        return len(self.index_map)
    
    def get(self, idx: int) -> Data:
        """
        Get a single jet as a PyG Data object.
        
        Args:
            idx: Global jet index
            
        Returns:
            PyG Data object representing the jet as a graph
        """
        file_idx, local_idx = self.index_map[idx]
        h5_file = self.h5_files[file_idx]
        
        with h5py.File(h5_file, 'r') as f:
            # Load raw data for this jet
            node_features = f['jetConstituentsList'][local_idx]  # [10, 3]
            qfi_matrix = f['jetConstituentsQFI'][local_idx]      # [30, 30]
            label = f['truth_labels'][local_idx]                # scalar
            jet_pt = f['jetFeatures'][local_idx,0]                  # [4]
        # Convert to tensors
        node_features[...,0]=node_features[...,0]/jet_pt # Normalise particle pt by jet pt. 
        
        x = torch.tensor(node_features, dtype=torch.float32)  # [10, 3]
        y = torch.tensor(label, dtype=torch.long)             # [1]
        jet_pt = torch.tensor(jet_pt, dtype=torch.float32)    # [4]
        # Extract edge features from QFI matrix
        edge_attr = self._extract_edge_features(qfi_matrix)   # [100, 9]
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=self.edge_index.clone(),  # [2, 100]
            edge_attr=edge_attr,                 # [100, 9]
            y=y
        )
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        return data
    
    def _extract_edge_features(self, qfi_matrix: np.ndarray) -> torch.Tensor:
        """
        Extract edge features from QFI correlation matrix.
        
        Args:
            qfi_matrix: QFI matrix of shape [30, 30]
            
        Returns:
            Edge features tensor of shape [100, 9]
        """
        edge_features = []
        
        # Iterate through all edges (src, dst)
        for edge_idx in range(self.edge_index.size(1)):
            src, dst = self.edge_index[:, edge_idx]
            
            # Extract 3x3 correlation submatrix between particles src and dst
            start_src, end_src = 3 * src, 3 * src + 3
            start_dst, end_dst = 3 * dst, 3 * dst + 3
            
            submatrix = qfi_matrix[start_src:end_src, start_dst:end_dst]  # [3, 3]
            edge_features.append(submatrix.flatten())  # [9]
        
        return torch.tensor(np.stack(edge_features), dtype=torch.float32)  # [100, 9]


def create_dataloaders(
    train_files: List[str],
    val_files: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders.
    
    Args:
        train_files: List of training H5 files
        val_files: List of validation H5 files
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = JetGraphDataset(train_files)
    val_dataset = JetGraphDataset(val_files)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Data]) -> Data:
    """
    Custom collate function for batching graphs (optional, PyG handles this automatically).
    
    Args:
        batch: List of Data objects
        
    Returns:
        Batched Data object
    """
    # PyG's default Batch.from_data_list handles this efficiently
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


# Example usage
if __name__ == "__main__":
    # Example file paths
    train_files = [
        "/ceph/abal/train/ZJetsToNuNu_008.h5",
    ]
    
    val_files = [
        "/ceph/abal/train/ZJetsToNuNu_027.h5",
    ]
    
    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_files=train_files,
        val_files=val_files,
        batch_size=64,
        num_workers=8
    )
    
    # Test loading a batch
    for batch in train_loader:
        print(f"Batch info:")
        print(f"  Number of graphs: {batch.num_graphs}")
        print(f"  Node features: {batch.x.shape}")      # [batch_size*10, 3]
        print(f"  Edge features: {batch.edge_attr.shape}") # [batch_size*100, 9]
        print(f"  Edge index: {batch.edge_index.shape}")   # [2, batch_size*100]
        print(f"  Labels: {batch.y.shape}")               # [batch_size]
        break