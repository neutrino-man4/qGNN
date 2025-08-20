import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import List, Optional, Union, Callable
from loguru import logger


class ConfigurableMLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron with variable depth and width.
    
    Args:
        layer_sizes: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
        activation: Activation function (default: ELU)
        dropout: Dropout probability (default: 0.0)
        batch_norm: Whether to use batch normalization (default: False)
    """
    
    def __init__(
        self, 
        layer_sizes: List[int],
        activation: nn.Module = nn.ELU(),
        dropout: float = 0.0,
        batch_norm: bool = False
    ):
        super().__init__()
        
        if len(layer_sizes) < 2:
            logger.error(f"layer_sizes must have at least 2 elements, got {len(layer_sizes)}")
            raise ValueError("layer_sizes must have at least 2 elements (input and output)")
        
        logger.debug(f"Creating MLP with architecture: {layer_sizes}")
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            # Add batch norm for all layers except the last one
            if batch_norm and i < len(layer_sizes) - 2:
                self.batch_norms.append(nn.BatchNorm1d(layer_sizes[i + 1]))
        
        total_params = sum(p.numel() for p in self.parameters())
        logger.debug(f"MLP created with {total_params:,} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):  # All layers except last
            x = layer(x)
            
            # Apply batch norm if enabled
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Apply activation
            x = self.activation(x)
            
            # Apply dropout
            if self.dropout is not None:
                x = self.dropout(x)
        
        # Final layer (no activation/dropout)
        x = self.layers[-1](x)
        return x

class GlobalPooling(nn.Module):
    """
    Global graph pooling layer with multiple pooling strategies.
    
    Args:
        pooling_type: Type of pooling ('mean', 'max', 'add', 'concat')
        If 'concat', combines mean and max pooling
    """
    
    def __init__(self, pooling_type: str = 'mean'):
        super().__init__()
        self.pooling_type = pooling_type.lower()
        
        if self.pooling_type not in ['mean', 'max', 'add', 'concat']:
            logger.error(f"Unsupported pooling type: {pooling_type}")
            raise ValueError(f"Unsupported pooling type: {pooling_type}")
        
        logger.debug(f"GlobalPooling initialized with type: {self.pooling_type}")
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, features]
            batch: Batch assignment vector [N]
        
        Returns:
            Graph-level features [batch_size, features] or [batch_size, 2*features] for concat
        """
        if self.pooling_type == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling_type == 'max':
            return global_max_pool(x, batch)
        elif self.pooling_type == 'add':
            return global_add_pool(x, batch)
        elif self.pooling_type == 'concat':
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            return torch.cat([mean_pool, max_pool], dim=-1)

class CorrelationMessagePassing(MessagePassing):
    """
    Message passing layer that applies QFI correlation matrices to node features.
    
    Basic idea: we want MLP([x_i, x_j, C_ji @ x_j]) where C_ji is the correlation matrix, in this case the QFI.
    Take a look at the graph_dataloader where we extract the corresponding QFI block for every two-particle interaction
    
    Args:
        mlp_layers: List defining MLP architecture [input_dim, hidden1, ..., output_dim]
        activation: Activation function for MLP
        aggr: Aggregation method ('add', 'mean', 'max')
    """
    
    def __init__(
        self, 
        mlp_layers: List[int],
        activation: nn.Module = nn.ELU(),
        aggr: str = 'add'
    ):
        super().__init__(aggr=aggr)
        
        # Input dimension should be: x_i (3) + x_j (3) + C_ji @ x_j (3) = 9
        if mlp_layers[0] != 9:
            logger.warning(f"CorrelationMP: Expected input dimension 9, got {mlp_layers[0]}. Adjusting automatically.")
            mlp_layers[0] = 9
        
        logger.info(f"Creating CorrelationMessagePassing with MLP: {mlp_layers}, aggregation: {aggr}")
        self.mlp = ConfigurableMLP(mlp_layers, activation=activation)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, 3]
            edge_index: Edge connectivity [2, E] 
            edge_attr: Edge features [E, 9] (flattened 3x3 correlation matrices)
        
        Returns:
            Updated node features [N, output_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute messages using correlation-weighted features.
        
        Args:
            x_i: Target node features [E, 3]
            x_j: Source node features [E, 3]
            edge_attr: Edge features [E, 9] (flattened correlation matrices)
        
        Returns:
            Messages [E, output_dim]
        """
        # Reshape edge_attr from [E, 9] to [E, 3, 3] correlation matrices
        batch_size = edge_attr.size(0)
        C_ji = edge_attr.view(batch_size, 3, 3)  # [E, 3, 3]
        
        # Apply correlation matrix to source features: C_ji @ x_j
        x_j_corr = torch.bmm(C_ji, x_j.unsqueeze(-1)).squeeze(-1)  # [E, 3]
        
        # Concatenate features: [x_i, x_j, x_j_corr]
        message_input = torch.cat([x_i, x_j, x_j_corr], dim=-1)  # [E, 9]
        
        # Apply MLP to generate message
        return self.mlp(message_input)


class BilinearMessagePassing(MessagePassing):
    """
    Message passing layer using bilinear interactions with QFI correlation matrices.
    
    Uses bilinear form: x_i^T @ C_ji @ x_j for correlation-aware similarity.
    
    Args:
        mlp_layers: List defining MLP architecture [input_dim, hidden1, ..., output_dim]
        activation: Activation function for MLP
        aggr: Aggregation method ('add', 'mean', 'max')
    """
    
    def __init__(
        self, 
        mlp_layers: List[int],
        activation: nn.Module = nn.ELU(),
        aggr: str = 'add'
    ):
        super().__init__(aggr=aggr)
        
        # Input dimension: x_i (3) + x_j (3) + bilinear_term (1) = 7
        if mlp_layers[0] != 7:
            logger.warning(f"BilinearMP: Expected input dimension 7, got {mlp_layers[0]}. Adjusting automatically.")
            mlp_layers[0] = 7
        
        logger.info(f"Creating BilinearMessagePassing with MLP: {mlp_layers}, aggregation: {aggr}")
        self.mlp = ConfigurableMLP(mlp_layers, activation=activation)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, 3]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, 9] (flattened 3x3 correlation matrices)
        
        Returns:
            Updated node features [N, output_dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute messages using bilinear correlation interactions.
        
        Args:
            x_i: Target node features [E, 3]
            x_j: Source node features [E, 3] 
            edge_attr: Edge features [E, 9] (flattened correlation matrices)
        
        Returns:
            Messages [E, output_dim]
        """
        # Reshape edge_attr from [E, 9] to [E, 3, 3] correlation matrices
        batch_size = edge_attr.size(0)
        C_ji = edge_attr.view(batch_size, 3, 3)  # [E, 3, 3]
        
        # Compute bilinear interaction: x_i^T @ C_ji @ x_j
        # Using einsum for efficient batch computation
        bilinear_term = torch.einsum('ei,eij,ej->e', x_i, C_ji, x_j).unsqueeze(-1)  # [E, 1]
        
        # Concatenate features: [x_i, x_j, bilinear_term]
        message_input = torch.cat([x_i, x_j, bilinear_term], dim=-1)  # [E, 7]
        
        # Apply MLP to generate message
        return self.mlp(message_input)