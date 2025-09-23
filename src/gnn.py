"""
Author: Aritra Bal, ETP
Date: ante diem quartum Nonas Ianuarias anno ab urbe condita MMDCCLXXVIII

QFI-based Graph Neural Network architecture for jet classification.
Processes 30-node QFI graphs with reconstruction capability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import List, Optional, Union, Dict, Any, Tuple
from loguru import logger
import src.layers as layers


class QFIJetGNN(nn.Module):
    """
    QFI-based Graph Neural Network for jet classification.
    
    Architecture:
    1. Multiple message passing layers (updates both nodes and edges)
    2. Global pooling to create graph-level representation
    3. Classification MLP for binary prediction
    4. QFI matrix reconstruction capability
    
    Args:
        message_type: Type of message passing ('trainable', 'fixed', etc.)
        num_mp_layers: Number of message passing layers
        mp_mlp_layers: Hidden layer sizes for message passing MLPs
        classifier_layers: MLP architecture for final classifier
        pooling_type: Global pooling method ('mean', 'max', 'add', 'concat')
        activation: Activation function
    """
    
    def __init__(
        self,
        message_type: str = 'trainable',
        num_mp_layers: int = 3,
        mp_mlp_layers: List[int] = [16, 8],
        classifier_layers: List[int] = [32, 16, 8, 2],
        pooling_type: str = 'mean',
        activation: Union[str, nn.Module] = 'elu',
        aggr: str = 'add', extra_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        logger.info(f"Initializing QFIJetGNN with message_type={message_type}, num_layers={num_mp_layers}")
        
        self.message_type = message_type.lower()
        self.num_mp_layers = num_mp_layers
        self.num_nodes = 30  # QFI matrix dimension
        
        # Handle activation function
        if isinstance(activation, str):
            activation_map = {
                'relu': nn.ReLU(),
                'elu': nn.ELU(),
                'leaky_relu': nn.LeakyReLU(),
                'gelu': nn.GELU(),
                'silu': nn.SiLU(),
                'tanh': nn.Tanh()
            }
            if activation.lower() not in activation_map:
                raise ValueError(f"Unsupported activation: {activation}")
            self.activation = activation_map[activation.lower()]
        else:
            self.activation = activation
        use_activation = activation
        # Create message passing layers
        self.mp_layers = nn.ModuleList()
        for i in range(num_mp_layers):
            if self.message_type == 'trainable':
                # if i==num_mp_layers-1:
                #    use_activation='identity'  # No activation on last layer
                mp_layer = layers.create_qfi_message_passing(
                    message_type='trainable',
                    mlp_layers=mp_mlp_layers.copy(),
                    activation=use_activation,
                    aggr=aggr
                )
            elif self.message_type.lower() == "gat":
                if extra_kwargs is None:
                    extra_kwargs = {}
                    logger.warning("No extra_kwargs provided for GAT layer; using defaults")
                    import time;time.sleep(3)
                from src.layers import create_quantum_gat_layer
                mp_layer = create_quantum_gat_layer(extra_kwargs)
            elif self.message_type == 'fixed':
                mp_layer = layers.create_qfi_message_passing(
                    message_type='fixed'
                )
            else:
                raise ValueError(f"Unsupported message type: {self.message_type}")
            
            self.mp_layers.append(mp_layer)
            logger.debug(f"Created MP layer {i+1}/{num_mp_layers}")
        
        # Global pooling
        self.pooling = layers.GlobalPooling(pooling_type)
        
        # Determine classifier input dimension
        if pooling_type == 'concat':
            classifier_input_dim = 2  # 1D node features, concat gives 2D
        elif pooling_type == 'matrix':
            classifier_input_dim = 100  # Full matrix flattening
        else:
            classifier_input_dim = 1  # Single pooling gives 1D
        
        # Adjust classifier input dimension if needed
        if classifier_layers[0] != classifier_input_dim:
            logger.warning(f"Modified classifier input dimension: add {classifier_input_dim} at the very beginning")
            classifier_layers = [classifier_input_dim] + classifier_layers
        
        # Final classifier
        self.classifier = layers.ConfigurableMLP(
            layer_sizes=classifier_layers,
            activation=self.activation,
            dropout=0.1
        )
        
        # Store current edge features and indices for QFI reconstruction
        self.current_edge_features = None
        self.current_edge_index = None
        self.current_node_features = None
        # Store architecture info
        self.architecture_info = {
            'message_type': message_type,
            'num_mp_layers': num_mp_layers,
            'mp_mlp_layers': mp_mlp_layers,
            'classifier_layers': classifier_layers,
            'pooling_type': pooling_type,
            'activation': str(self.activation),
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
        
        logger.info(f"QFIJetGNN created with {self.architecture_info['total_parameters']:,} parameters")
        logger.info(f"Architecture: {num_mp_layers} MP layers, {pooling_type} pooling")
    
    
    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Forward pass through the QFI GNN.
        
        Args:
            data: PyG Data or Batch object containing:
                - x: Node features [N, 1] - QFI diagonal elements
                - edge_index: Edge connectivity [2, E] 
                - edge_attr: Edge features [E, 1] - QFI off-diagonal elements
                - batch: Batch assignment (for Batch objects)
                
        Returns:
            Classification logits [batch_size, 2]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch_tensor = getattr(data, 'batch', None)
        
        # If single graph, create batch tensor
        if batch_tensor is None:
            batch_tensor = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Message passing layers
        current_edge_features = edge_attr
        for i, mp_layer in enumerate(self.mp_layers):
            if self.message_type == 'trainable':
                # Trainable layers return (nodes, edges)
                x, current_edge_features = mp_layer(x, edge_index, current_edge_features)
                
            #logger.debug(f"MP layer {i+1}: node range [{x.min():.3f}, {x.max():.3f}], "
            #            f"edge range [{current_edge_features.min():.3f}, {current_edge_features.max():.3f}]")
        current_node_features = x
        self.current_node_features = current_node_features.detach()
        # Store current features for QFI reconstruction
        self.current_edge_features = current_edge_features.detach()
        self.current_edge_index = edge_index
        
        # Global pooling to get graph-level representation
        if self.pooling.pooling_type == 'matrix':
            # For matrix pooling, pass edge_attr and edge_index
            graph_features = self.pooling(x, batch_tensor, edge_attr=current_edge_features, edge_index=edge_index)
        else:
            graph_features = self.pooling(x, batch_tensor)
        # Final classification
        logits = self.classifier(graph_features)
        
        return logits
    
    def reconstruct_qfi_matrices(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Reconstruct complete QFI matrices from current node and edge features.
        
        Args:
            data: PyG Data or Batch object
            
        Returns:
            Reconstructed QFI matrices [batch_size, 30, 30]
            
        Raises:
            RuntimeError: If forward pass hasn't been called yet
        """
        if self.current_edge_features is None or self.current_edge_index is None or self.current_node_features is None:
            raise RuntimeError("Must call forward() before QFI reconstruction")
        
        batch_tensor = getattr(data, 'batch', None)
        if batch_tensor is None:
            batch_tensor = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        
        # Determine batch size
        batch_size = int(batch_tensor.max().item()) + 1
        device = data.x.device
        
        # Vectorized diagonal reconstruction
        x_reshaped = self.current_node_features.reshape(batch_size, self.num_nodes)  # [batch_size, 30]
        qfi_matrices = torch.diag_embed(x_reshaped)  # [batch_size, 30, 30]
        
        # Vectorized off-diagonal reconstruction
        edge_batch = batch_tensor[self.current_edge_index[0]]  # [num_edges]
        src_local = self.current_edge_index[0] % self.num_nodes  # [num_edges]
        tgt_local = self.current_edge_index[1] % self.num_nodes  # [num_edges]
        
        # Set all off-diagonal elements at once using advanced indexing
        qfi_matrices[edge_batch, src_local, tgt_local] = self.current_edge_features.squeeze(-1)
        return qfi_matrices
    
    def get_qfi_reconstruction_stats(self, original_qfi: torch.Tensor, 
                                   reconstructed_qfi: torch.Tensor) -> Dict[str, float]:
        """
        Compare original and reconstructed QFI matrices.
        
        Args:
            original_qfi: Original QFI matrices [batch_size, 30, 30]
            reconstructed_qfi: Reconstructed QFI matrices [batch_size, 30, 30]
            
        Returns:
            Dictionary with comparison statistics
        """
        # Flatten matrices for comparison
        orig_flat = original_qfi.flatten()
        recon_flat = reconstructed_qfi.flatten()
        
        # Compute statistics
        mse = F.mse_loss(recon_flat, orig_flat).item()
        mae = F.l1_loss(recon_flat, orig_flat).item()
        
        # Correlation coefficient
        orig_centered = orig_flat - orig_flat.mean()
        recon_centered = recon_flat - recon_flat.mean()
        correlation = (orig_centered * recon_centered).sum() / (
            torch.sqrt((orig_centered ** 2).sum()) * torch.sqrt((recon_centered ** 2).sum())
        )
        
        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation.item(),
            'max_absolute_error': (orig_flat - recon_flat).abs().max().item()
        }
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Return architecture configuration."""
        return self.architecture_info.copy()


def create_qfi_jet_gnn(
    message_type: str = 'trainable',
    num_layers: int = 3,
    mp_mlp_layers: List[int] = [16, 8],
    classifier_layers: List[int] = [32, 16, 8, 2],
    pooling: str = 'mean',
    activation: str = 'elu',
    aggr: str = 'add', extra_kwargs: Optional[Dict[str, Any]] = None
) -> QFIJetGNN:
    """
    Factory function to create QFI-based GNN models.
    
    Args:
        message_type: 'trainable' or 'fixed'
        num_layers: Number of message passing layers
        mp_mlp_layers: Hidden layer sizes for MP MLPs
        classifier_layers: Hidden layer sizes for classifier
        pooling: Global pooling type ('mean', 'max', 'add', 'concat')
        activation: Activation function name
        
    Returns:
        Configured QFIJetGNN model
    """
    logger.info(f"Creating {message_type} QFI GNN with factory function")
    
    model = QFIJetGNN(
        message_type=message_type,
        num_mp_layers=num_layers,
        mp_mlp_layers=mp_mlp_layers,
        classifier_layers=classifier_layers,
        pooling_type=pooling,
        activation=activation,
        aggr=aggr, extra_kwargs=extra_kwargs
    )
    
    logger.success(f"QFI GNN model created successfully!")
    logger.info(f"Model architecture: {model.get_architecture_info()}")
    logger.info(f"Activation: {activation}")
    logger.info(f"Aggregation: {aggr}")
    import time;time.sleep(2) # Pause for readability in logs
    return model


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Count trainable and total parameters in the model.
    
    Returns:
        Dictionary with parameter counts and model size info
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (trainable_params * 4) / (1024 * 1024)
    
    return {
        'trainable_parameters': trainable_params,
        'total_parameters': total_params,
        'model_size_mb': round(model_size_mb, 2),
        'non_trainable_parameters': total_params - trainable_params
    }


if __name__ == "__main__":
    # Test QFI GNN models
    logger.info("Testing QFI GNN Models")
    
    # Create sample QFI graph data (30 nodes)
    batch_size = 2
    num_nodes = 30
    
    # Node features: QFI diagonal elements
    x = torch.randn(batch_size * num_nodes, 1) * 0.5
    
    # Create fully connected edge index (excluding self-loops)
    sources, targets = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                sources.append(i)
                targets.append(j)
    
    edge_index_single = torch.tensor([sources, targets], dtype=torch.long)
    
    # Replicate for batch
    edge_index = edge_index_single.clone()
    for b in range(1, batch_size):
        offset_edges = edge_index_single + b * num_nodes
        edge_index = torch.cat([edge_index, offset_edges], dim=1)
    
    # Edge features: QFI off-diagonal elements
    edge_attr = torch.randn(edge_index.shape[1], 1) * 0.3
    
    # Labels
    y = torch.randint(0, 2, (batch_size,))
    batch_tensor = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    # Create Data object
    from torch_geometric.data import Batch as PyGBatch
    data = PyGBatch(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch_tensor)
    
    # Original QFI matrices for comparison
    original_qfi = torch.randn(batch_size, num_nodes, num_nodes) * 0.3
    original_qfi = (original_qfi + original_qfi.transpose(-2, -1)) / 2.0  # Make symmetric only because the QFI actually is symmetric
    
    print(f"Test data created:")
    print(f"  Nodes: {x.shape}")
    print(f"  Edges: {edge_index.shape}")
    print(f"  Edge attr: {edge_attr.shape}")
    
    # Test trainable model
    print("\n" + "="*60)
    print("TESTING TRAINABLE QFI GNN")
    print("="*60)
    
    model = create_qfi_jet_gnn(
        message_type='trainable',
        num_layers=2,
        mp_mlp_layers=[16, 8],
        pooling='mean'
    )
    
    # Forward pass
    with torch.no_grad():
        output = model(data)
        print(f"Model output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        # Test QFI reconstruction
        reconstructed_qfi = model.reconstruct_qfi_matrices(data)
        print(f"Reconstructed QFI shape: {reconstructed_qfi.shape}")
        print(f"QFI reconstruction range: [{reconstructed_qfi.min():.4f}, {reconstructed_qfi.max():.4f}]")

    
    # Model info
    param_info = count_parameters(model)
    arch_info = model.get_architecture_info()
    
    print(f"\nModel Statistics:")
    print(f"  Parameters: {param_info['trainable_parameters']:,}")
    print(f"  Model size: {param_info['model_size_mb']} MB")
    print(f"  MP layers: {arch_info['num_mp_layers']}")
    print(f"  Pooling: {arch_info['pooling_type']}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")  
    print("="*60)