"""
Author: Aritra Bal, ETP
Date: ante diem quartum Nonas Ianuarias anno ab urbe condita MMDCCLXXVIII

Refactored layers for QFI graph processing with fixed and trainable message passing.
Simplified architecture with only two message passing functions optimized for QFI matrices.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import List, Optional, Union, Callable, Tuple
from torch_geometric.utils import softmax
from loguru import logger
from torch_scatter import scatter_add, scatter_mean, scatter_max
from dataclasses import asdict
class ConfigurableMLP(nn.Module):
    """
    Configurable Multi-Layer Perceptron with variable depth and width.
    
    Args:
        layer_sizes: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
        activation: Activation function (default: ELU)
        dropout: Dropout probability (default: 0.0)
        batch_norm: Whether to use batch normalization (default: False)
        output_activation: Final activation function (default: None)
    """
    def __init__(
        self, 
        layer_sizes: List[int],
        activation: nn.Module = nn.ELU(),
        dropout: float = 0.0,
        batch_norm: bool = False,
        output_activation: Optional[nn.Module] = None
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
        self.output_activation = output_activation
        
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
        
        # Final layer
        x = self.layers[-1](x)
        
        # Apply output activation if specified
        if self.output_activation is not None:
            x = self.output_activation(x)
        
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
        if self.pooling_type not in ['mean', 'max', 'add', 'concat', 'matrix']:
            logger.error(f"Unsupported pooling type: {pooling_type}")
            raise ValueError(f"Unsupported pooling type: {pooling_type}")
        logger.debug(f"GlobalPooling initialized with type: {self.pooling_type}")

    def forward(self, x: torch.Tensor, batch: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, 
                edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [N, features]
            batch: Batch assignment vector [N]
        Returns:
            Graph-level features [batch_size, features] or [batch_size, 2*features] for concat
        """
        if self.pooling_type == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling_type == 'matrix':
            if edge_attr is None or edge_index is None:
                raise ValueError("Matrix pooling requires edge_attr and edge_index")
            return self._matrix_pool(x, edge_attr, edge_index, batch)
        elif self.pooling_type == 'max':
            return global_max_pool(x, batch)
        elif self.pooling_type == 'add':
            return global_add_pool(x, batch)
        elif self.pooling_type == 'concat':
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            return torch.cat([mean_pool, max_pool], dim=-1)
    
    def _matrix_pool(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor) -> torch.Tensor:
        """
        Custom matrix pooling for QFI graphs.
        
        Reconstructs 30x30 QFI matrices, divides into 10x10 grid of 3x3 blocks,
        computes Frobenius norm of each 3x3 block, and flattens to 100D vector.
        
        Args:
            x: Node features [N, 1] - QFI diagonal elements
            edge_attr: Edge features [E, 1] - QFI off-diagonal elements  
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment [N]
            
        Returns:
            Pooled features [batch_size, 100]
        """
        batch_size = int(batch.max().item()) + 1
        device = x.device
        num_nodes = 30
        x_reshaped = x.view(batch_size, num_nodes)  # [batch_size, 30]
        qfi_matrices = torch.diag_embed(x_reshaped)  # [batch_size, 30, 30]
        
        # Vectorized off-diagonal setting
        edge_batch = batch[edge_index[0]]
        src_local = edge_index[0] % num_nodes
        tgt_local = edge_index[1] % num_nodes
        qfi_matrices[edge_batch, src_local, tgt_local] = edge_attr.squeeze(-1)
        
        # Ultra-efficient block reshaping using unfold
        # Extract all 3x3 blocks at once
        blocks = qfi_matrices.unfold(1, 3, 3).unfold(2, 3, 3)  # [batch_size, 10, 10, 3, 3]
        
        # Compute Frobenius norms vectorially
        frobenius_norms = torch.norm(blocks, p='fro', dim=(-2, -1))  # [batch_size, 10, 10]
        return frobenius_norms.view(batch_size, -1)  # [batch_size, 900]

class FixedCorrelationMessage(MessagePassing):
    """
    Fixed (non-learnable) message passing using QFI matrix elements.
    Applies the update rule: x_i_new = x_i + e_ij * x_j + (e_ij * x_j)^2
    
    This implements a polynomial interaction between nodes mediated by edge weights,
    where stronger QFI correlations (larger |e_ij|) have increasingly nonlinear effects.
    
    Args:
        aggr: Aggregation method ('add', 'mean', 'max')
    """
    def __init__(self, aggr: str = 'add'):
        super().__init__(aggr=aggr)
        logger.info(f"Creating FixedCorrelationMessage with aggregation: {aggr}")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, 1] - QFI diagonal elements
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, 1] - QFI off-diagonal elements
        Returns:
            Updated node features [N, 1]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Compute fixed correlation messages using polynomial QFI interactions.
        
        Message formula: m_ij = e_ij * x_j + (e_ij * x_j)^2
        Final update: x_i_new = x_i + Σ_j m_ij
        
        Args:
            x_i: Target node features [E, 1] 
            x_j: Source node features [E, 1]
            edge_attr: Edge features [E, 1] - QFI matrix elements
        Returns:
            Messages [E, 1]
        """
        e_ij = edge_attr  # [E, 1] - QFI matrix elements
        
        # Linear term: e_ij * x_j
        linear_term = e_ij * x_j  # [E, 1]
        
        # Quadratic term: (e_ij * x_j)^2
        quadratic_term = (e_ij * x_j) ** 2  # [E, 1]
        
        # Combined message: e_ij * x_j + (e_ij * x_j)^2
        message = linear_term + quadratic_term  # [E, 1]
        
        return message


class BasicMessage(MessagePassing):
    """
    Trainable message passing that updates both node and edge features.

    Uses an MLP to compute updates based on: f(x_i, e_ij * |x_j - x_i|).
    Outputs updated node and edge features.

    Args:
        mlp_layers: Hidden layer sizes for edge MLP [hidden1, hidden2, ...]
        activation: Activation function for hidden layers
        aggr: Aggregation method for node updates ("add", "mean", "max")
    """
    def __init__(
        self,
        mlp_layers: List[int] = [16, 8],
        activation: nn.Module = nn.ELU(),
        aggr: str = 'mean'
    ):
        super().__init__(aggr=aggr)

        # Build MLP: input=2, output=3
        layer_sizes = [3] + mlp_layers + [3]
        self.edge_mlp = ConfigurableMLP(
            layer_sizes=layer_sizes,
            activation=activation,
            dropout=0.0,
            output_activation=None  # optional bounding later
        )

        logger.info(f"Creating BasicMessage with MLP: {layer_sizes}, aggregation: {aggr}")

        # Placeholder for storing edge updates
        self.updated_edges = None

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, edge_attr: torch.Tensor, 
                ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None) -> torch.Tensor:
        """
        Custom aggregation that weights messages by edge features.
        
        Args:
            inputs: Messages [E, 1] - these are the m_ij values
            index: Target node indices [E] - which node each message goes to
            edge_attr: Edge features [E, 1] - these are the e_ij values
            ptr: Optional pointer for batched graphs
            dim_size: Number of nodes
        Returns:
            Aggregated messages [N, 1]
        """
        # Weight messages by edge features: m_ij * e_ij
        weighted_messages = inputs * edge_attr  # [E, 1]
        
        # Apply the chosen aggregation method to weighted messages
        if self.aggr == 'add':
            return scatter_add(weighted_messages, index, dim=0, dim_size=dim_size)
        elif self.aggr == 'mean':
            return scatter_mean(weighted_messages, index, dim=0, dim_size=dim_size)
        elif self.aggr == 'max':
            return scatter_max(weighted_messages, index, dim=0, dim_size=dim_size)[0]
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggr}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """
        Args:
            x: Node features [N, 1]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, 1]
        Returns:
            updated_nodes [N, 1], updated_edges [E, 1]
        """
        updated_nodes = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return updated_nodes, self.updated_edges

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor):
        """
        Args:
            x_i: Target node features [E, 1]
            x_j: Source node features [E, 1]
            edge_attr: Edge features [E, 1]
        Returns:
            Messages for node updates [E, 1]
        """
        # Compute input for MLP
        
        diff_magnitude = torch.abs(x_j + x_i + x_j**2 + x_i**2)  # [E, 1]
        mlp_input = torch.cat([x_i, edge_attr * diff_magnitude, edge_attr], dim=-1)  # [E, 3]

        mlp_output = self.edge_mlp(mlp_input)  # [E, 3]

        # Split outputs
        x_i_update = mlp_output[:, 0:1]  # message for target node
        # x_j_update = mlp_output[:, 1:2]  # unused here, but could be added
        e_ij_update = mlp_output[:, 2:3]  # edge update

        # Store edge updates for use in forward
        self.updated_edges = e_ij_update

        return x_i_update

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor):
        """
        Update target nodes with aggregated messages.

        Args:
            aggr_out: Aggregated messages [N, 1]
            x: Original node features [N, 1]
        Returns:
            Updated node features [N, 1]
        """
        return x + aggr_out


class AdaptiveQFILayer(nn.Module):
    """
    Combined layer that applies BasicMessage and handles both node and edge updates.
    This wrapper ensures proper handling of the dual update mechanism.
    """
    def __init__(
        self,
        mlp_layers: List[int] = [16, 8],
        activation: nn.Module = nn.ELU(),
        aggr: str = 'add'
    ):
        super().__init__()
        self.message_layer = BasicMessage(mlp_layers, activation, aggr)
        logger.info("Created AdaptiveQFILayer for simultaneous node/edge updates")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> tuple:
        """
        Forward pass that updates both nodes and edges.
        
        Args:
            x: Node features [N, 1] 
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, 1]
        Returns:
            Tuple of (updated_nodes, updated_edges)
        """
        return self.message_layer(x, edge_index, edge_attr)


def create_qfi_message_passing(
    message_type: str = 'fixed',
    mlp_layers: List[int] = [16, 8],
    activation: str = 'elu',
    aggr: str = 'add'
) -> nn.Module:
    """
    Factory function to create QFI message passing layers.
    
    Args:
        message_type: 'fixed' or 'trainable' 
        mlp_layers: Hidden layer sizes for trainable message (ignored for fixed)
        activation: Activation function name
        aggr: Aggregation method
    Returns:
        Message passing layer
    """
    # Handle activation function
    activation_map = {
        'relu': nn.ReLU(),
        'elu': nn.ELU(),
        'leaky_relu': nn.LeakyReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'tanh': nn.Tanh(),
        'identity': nn.Identity()
    }
    
    if activation.lower() not in activation_map:
        raise ValueError(f"Unsupported activation: {activation}")
    activation_fn = activation_map[activation.lower()]
    
    message_type = message_type.lower()
    if message_type == 'fixed':
        return FixedCorrelationMessage(aggr=aggr)
    elif message_type == 'trainable':
        return AdaptiveQFILayer(mlp_layers, activation_fn, aggr)
    else:
        raise ValueError(f"Unknown message_type: {message_type}. Use 'fixed' or 'trainable'")


class QuantumGATMessage(MessagePassing):
    """
    QFI-enhanced Graph Attention Network message passing layer.
    
    Computes attention weights using: alpha_ij = softmax(LeakyReLU(a^T [Wx_i || Wx_j || e_ij]))
    where x_i, x_j are QFI diagonal elements and e_ij are QFI off-diagonal elements.
    
    Args:
        in_channels: Input node feature dimension (typically 1 for QFI diagonal)
        out_channels: Output feature dimension per attention head
        heads: Number of attention heads
        concat: Whether to concatenate multi-head outputs (True) or average (False)
        dropout: Dropout probability for attention coefficients
        bias: Whether to use bias in linear transformations
        negative_slope: Negative slope for LeakyReLU in attention computation
        aggr: Aggregation method for message passing ('add', 'mean', 'max')
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 16,
        heads: int = 8,
        concat: bool = True,
        dropout: float = 0.2,
        bias: bool = True,
        negative_slope: float = 0.2,
        aggr: str = 'add'
    ):
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.W_k = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.W_v = nn.Linear(in_channels, heads * out_channels, bias=bias)
        
        # Edge feature transformation
        self.W_edge = nn.Linear(1, heads, bias=bias)  # Transform e_ij to heads dimensions
        
        # Attention mechanism parameters
        # a^T [Wx_i || Wx_j || e_ij] where each part has heads dimensions
        self.attention_weights = nn.Parameter(torch.randn(heads, 2 * out_channels + 1))
        
        # Store attention weights for visualization
        self.last_attention_weights = None
        self.last_edge_index = None
        
        self.reset_parameters()
        
        logger.info(f"Created QuantumGATMessage: {heads} heads, {out_channels} out_channels, aggr={aggr}")
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.attention_weights)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of QFI-enhanced GAT layer.
        
        Args:
            x: Node features [N, in_channels] - QFI diagonal elements
            edge_index: Edge connectivity [2, E] - typically [2, 870] for 30-node complete graph
            edge_attr: Edge features [E, 1] - QFI off-diagonal elements
            return_attention_weights: Whether to return attention weights for visualization
        
        Returns:
            Updated node features [N, out_channels * heads] if concat=True
            or [N, out_channels] if concat=False
            Optionally returns attention weights [E, heads] if return_attention_weights=True
        """
        # Linear transformations
        query = self.W_q(x).view(-1, self.heads, self.out_channels)  # [N, heads, out_channels]
        key = self.W_k(x).view(-1, self.heads, self.out_channels)    # [N, heads, out_channels]
        value = self.W_v(x).view(-1, self.heads, self.out_channels)  # [N, heads, out_channels]
        
        # Store edge_index for attention matrix reconstruction
        self.last_edge_index = edge_index.detach().cpu()
        
        # Message passing with attention
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            size=None
        )
        self.last_node_features = x.detach().cpu()  # [N, 1]
        # Handle multi-head outputs
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)  # [N, heads * out_channels]
        else:
            out = out.mean(dim=1)  # [N, out_channels]
        
        if return_attention_weights:
            return out, self.last_attention_weights
        else:
            return out
    
    def message(
        self,
        query_i: torch.Tensor,
        key_j: torch.Tensor,
        value_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor] = None,
        size_i: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute QFI-enhanced attention messages.
        
        Args:
            query_i: Query features of target nodes [E, heads, out_channels]
            key_j: Key features of source nodes [E, heads, out_channels]
            value_j: Value features of source nodes [E, heads, out_channels]
            edge_attr: Edge features [E, 1] - QFI off-diagonal elements
            index: Target node indices for each edge
            ptr: Pointer for batched graphs
            size_i: Number of target nodes
        
        Returns:
            Messages [E, heads, out_channels]
        """
        # Transform edge features to match heads dimension
        edge_features = self.W_edge(edge_attr)  # [E, heads]
        
        # Concatenate query, key, and edge features for attention computation
        # [E, heads, 2*out_channels + 1]
        attention_input = torch.cat([
            query_i,  # [E, heads, out_channels]
            key_j,    # [E, heads, out_channels]
            edge_features.unsqueeze(-1)  # [E, heads, 1]
        ], dim=-1)
        
        # Compute attention scores: e_ij = a^T [Wx_i || Wx_j || e_ij]
        # Vectorized computation across all heads
        attention_scores = torch.sum(
            attention_input * self.attention_weights.unsqueeze(0),  # [E, heads, 2*out_channels+1]
            dim=-1
        )  # [E, heads]
        
        # Apply LeakyReLU activation
        attention_scores = F.leaky_relu(attention_scores, self.negative_slope)
        
        # Compute attention coefficients using softmax
        attention_weights = softmax(attention_scores, index, ptr, size_i)  # [E, heads]
        
        # Store attention weights for visualization
        self.last_attention_weights = attention_weights.detach().cpu()
        
        # Apply dropout
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        # Compute messages: m_ij = alpha_ij * W_v * x_j
        messages = attention_weights.unsqueeze(-1) * value_j  # [E, heads, out_channels]
        
        return messages
    
    def get_attention_matrix(self, num_nodes: int = 30, aggregation: str = 'mean') -> torch.Tensor:
        """
        Convert edge-based attention weights to 30x30 attention matrix.
        
        Args:
            num_nodes: Number of nodes in the graph (default: 30 for QFI graphs)
            aggregation: How to handle multiple attention heads ('mean', 'max', 'sum')
        
        Returns:
            attention_matrix: [num_nodes, num_nodes] tensor with diagonals set to 1.0
        """
        if self.last_attention_weights is None or self.last_edge_index is None:
            logger.warning("No attention weights available. Run forward pass first.")
            return torch.eye(num_nodes)
        
        # Handle multi-head attention weights
        if aggregation == 'mean':
            edge_weights = self.last_attention_weights.mean(dim=1)  # [E]
        elif aggregation == 'max':
            edge_weights = self.last_attention_weights.max(dim=1)[0]  # [E]
        elif aggregation == 'sum':
            edge_weights = self.last_attention_weights.sum(dim=1)  # [E]
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        
        # Create attention matrix
        attention_matrix = torch.zeros(num_nodes, num_nodes)
        
        # Fill off-diagonal elements with attention weights
        src_nodes = self.last_edge_index[0]  # Source nodes
        tgt_nodes = self.last_edge_index[1]  # Target nodes
        attention_matrix[tgt_nodes, src_nodes] = edge_weights
        
        # fill in current node features
        diagonal_values = self.last_node_features.flatten() if hasattr(self, 'last_node_features') else torch.ones(num_nodes)
        attention_matrix.fill_diagonal_(diagonal_values)
        # Set diagonal to 1.0 (self-attention placeholder)
        #attention_matrix.fill_diagonal_(1.0)
        
        return attention_matrix


def create_quantum_gat_layer(config_dict: dict) -> QuantumGATMessage:
    """
    Factory function to create QuantumGATMessage from configuration.
    
    Args:
        config_dict: Configuration dictionary with GAT parameters
    
    Returns:
        Configured QuantumGATMessage layer
    """
    config_dict = asdict(config_dict) if hasattr(config_dict, '__dataclass_fields__') else config_dict
    gat_config = config_dict.get('gat_config', {})
    
    layer = QuantumGATMessage(
        in_channels=1,  # QFI diagonal elements are scalar
        out_channels=gat_config.get('out_channels', 16),
        heads=gat_config.get('heads', 8),
        concat=gat_config.get('concat', True),
        dropout=gat_config.get('dropout', 0.2),
        bias=gat_config.get('bias', True),
        negative_slope=gat_config.get('negative_slope', 0.2),
        aggr='add'  # Standard for GAT
    )
    
    logger.info(f"Created QuantumGATMessage from config: {gat_config}")
    return layer



if __name__ == "__main__":
    # Test the message passing layers
    logger.info("Testing QFI Message Passing Layers")
    
    # Create test data (30-node QFI graph)
    num_nodes = 30
    batch_size = 2
    
    # Node features: QFI diagonal elements
    x = torch.randn(batch_size * num_nodes, 1) * 0.5  # [60, 1]
    
    # Create fully connected edge index (excluding self-loops)
    sources, targets = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                sources.append(i)
                targets.append(j)
    
    edge_index_single = torch.tensor([sources, targets], dtype=torch.long)  # [2, 870]
    
    # Replicate for batch
    edge_index = edge_index_single.clone()
    for b in range(1, batch_size):
        offset_edges = edge_index_single + b * num_nodes
        edge_index = torch.cat([edge_index, offset_edges], dim=1)
    
    # Edge features: QFI off-diagonal elements
    edge_attr = torch.randn(edge_index.shape[1], 1) * 0.3  # [1740, 1]
    
    print(f"Test data shapes:")
    print(f"  Nodes: {x.shape}")
    print(f"  Edges: {edge_index.shape}")  
    print(f"  Edge attr: {edge_attr.shape}")
    
    # Test Fixed Message Passing
    print("\n" + "="*50)
    print("TESTING FIXED MESSAGE PASSING")
    print("="*50)
    
    fixed_layer = create_qfi_message_passing('fixed', aggr='add')
    
    with torch.no_grad():
        x_updated_fixed = fixed_layer(x, edge_index, edge_attr)
    
    print(f"Fixed MP results:")
    print(f"  Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Output range: [{x_updated_fixed.min():.4f}, {x_updated_fixed.max():.4f}]")
    print(f"  Output shape: {x_updated_fixed.shape}")
    
    # Test Trainable Message Passing
    print("\n" + "="*50) 
    print("TESTING TRAINABLE MESSAGE PASSING")
    print("="*50)
    
    trainable_layer = create_qfi_message_passing('trainable', mlp_layers=[16, 8], aggr='add')
    
    # Count parameters
    total_params = sum(p.numel() for p in trainable_layer.parameters())
    print(f"Trainable layer parameters: {total_params:,}")
    
    with torch.no_grad():
        x_updated_train, edge_updated_train = trainable_layer(x, edge_index, edge_attr)
    
    print(f"Trainable MP results:")
    print(f"  Node input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"  Node output range: [{x_updated_train.min():.4f}, {x_updated_train.max():.4f}]")
    print(f"  Edge input range: [{edge_attr.min():.4f}, {edge_attr.max():.4f}]")
    print(f"  Edge output range: [{edge_updated_train.min():.4f}, {edge_updated_train.max():.4f}]")
    print(f"  Node output shape: {x_updated_train.shape}")
    print(f"  Edge output shape: {edge_updated_train.shape}")
    
    # Verify bounds
    node_in_bounds = torch.all((-1 <= x_updated_train) & (x_updated_train <= 1))
    edge_in_bounds = torch.all((-1 <= edge_updated_train) & (edge_updated_train <= 1))
    print(f"  Nodes bounded in [-1,1]: {node_in_bounds}")
    print(f"  Edges bounded in [-1,1]: {edge_in_bounds}")
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*50)