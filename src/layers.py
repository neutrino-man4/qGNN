import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax
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

class UniCorrelationMessagePassing(MessagePassing):
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
        if mlp_layers[0] != 15:
            logger.warning(f"UniCorrelationMP: Expected input dimension 15, got {mlp_layers[0]}. Adjusting automatically.")
            mlp_layers[0] = 15

        logger.info(f"Creating UniCorrelationMP with MLP: {mlp_layers}, aggregation: {aggr}")
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
        Compute messages using the entire correlation matrix as a flattened input
        
        Args:
            x_i: Target node features [E, 3]
            x_j: Source node features [E, 3]
            edge_attr: Edge features [E, 9] (flattened correlation matrices)
        
        Returns:
            Messages [E, output_dim]
        """
        # Reshape edge_attr from [E, 9] to [E, 3, 3] correlation matrices
        batch_size = edge_attr.size(0)
        C_ji = edge_attr#.view(batch_size, 3, 3)  # [E, 3, 3]
        
        # Apply correlation matrix to source features: C_ji @ x_j
        #x_j_corr = torch.bmm(C_ji, x_j.unsqueeze(-1)).squeeze(-1)  # [E, 3]
        
        # Concatenate features: [x_i, x_j, x_j_corr]
        message_input = torch.cat([x_i, x_j, C_ji], dim=-1)  # [E, 6]

        # Apply MLP to generate message
        return self.mlp(message_input)

class Conv1DMessagePassing(MessagePassing):
    """
    Message passing layer that applies 1D convolution to concatenated node and correlation features.
    
    Basic idea: Conv1D([x_i, x_j, C_ji]) where C_ji is the flattened QFI correlation matrix.
    
    Args:
        out_channels: Number of output channels for Conv1d
        kernel_size: Kernel size for Conv1d
        mlp_layers: dimensions of the MLP to use to cast the Conv1D output to a (3,1) tensor
        activation: Activation function
        aggr: Aggregation method ('add', 'mean', 'max')
    """
    
    def __init__(
        self, 
        out_channels: int = 4,
        kernel_size: int = 5,
        mlp_layers: List[int] = [3, 16, 8, 4, 2],
        activation: nn.Module = nn.ELU(),
        aggr: str = 'add'
    ):
        super().__init__(aggr=aggr)
        
        # Input dimension: x_i (3) + x_j (3) + C_ji (9) = 15
        self.input_dim = 15
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.mlp_layers = mlp_layers
        # 1D Convolution: treats 15 features as sequence
        self.conv1d = nn.Conv1d(
            in_channels=1, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            bias=True,
            padding=(kernel_size - 1) // 2  # Keep same sequence length
        )
        
        # Calculate conv output dimension
        conv_output_length = self.input_dim  # With padding='same'
        conv_output_dim = out_channels * conv_output_length
        
        # Optional final MLP for dimension reduction
        self.mlp_layers = [conv_output_dim] + mlp_layers + [3]
        self.output_dim = 3
        self.activation = activation
        
        self.final_mlp = ConfigurableMLP(
            layer_sizes=self.mlp_layers,
            activation=self.activation
        )
        
        logger.info(f"Conv1DMessagePassing: out_channels={out_channels}, kernel_size={kernel_size}, "
                   f"conv_output_dim={conv_output_dim}, final_dim=3, mlp_layers={mlp_layers}")
    
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
        Compute messages using 1D convolution on concatenated features.
        
        Args:
            x_i: Target node features [E, 3]
            x_j: Source node features [E, 3]
            edge_attr: Edge features [E, 9] (flattened correlation matrices)
        
        Returns:
            Messages [E, output_dim]
        """
        # Concatenate all features: [x_i, x_j, C_ji]
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)  # [E, 15]
        
        # Reshape for Conv1d: [batch, channels, sequence]
        conv_input = message_input.unsqueeze(1)  # [E, 1, 15]
        
        # Apply 1D convolution
        conv_output = self.conv1d(conv_input)  # [E, out_channels, 15]
        conv_output = self.activation(conv_output)
        
        # Flatten conv output
        conv_output = conv_output.view(conv_output.size(0), -1)  # [E, out_channels * 15]
        
        # Optional final MLP
        return self.final_mlp(conv_output)

class CorrelationModulatedGAT(MessagePassing):
    """
    Graph Attention Network layer modulated by correlation matrices.
    
    Implements attention mechanism: alpha_ij = exp(e_ij + lambda C_ij) / SUM_k exp(e_ik + lambda*C_ik)
    where e_ij = LeakyReLU(a^T[Wx_i || Wx_j]) is the standard GAT
    C_ij is the QFI present in the edge attributes - not exactly a correlation matrix but can be treated as such
    and lambda is a learnable temperature parameter.

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension per head
        heads: Number of attention heads
        concat: Whether to concatenate or average multi-head outputs
        dropout: Dropout probability for attention coefficients
        bias: Whether to use bias in linear transformations
        negative_slope: Negative slope for LeakyReLU in attention
        correlation_mode: How to use correlation matrix ('scalar', 'trace', 'frobenius') --> stick to Frobenius for now
        mlp_layers: Hidden dimensions for output MLP (final output will be 3D)
        activation: Activation function for MLP
        **kwargs: Additional arguments for MessagePassing
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int, 
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        negative_slope: float = 0.2,
        correlation_mode: str = 'frobenius',
        mlp_layers: List[int] = [32,16,12,8],
        activation: nn.Module = nn.ELU(),
        **kwargs
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.correlation_mode = correlation_mode
        
        # Linear transformations for query, key, value
        self.W_q = nn.Linear(in_channels, self.heads * out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, self.heads * out_channels, bias=False)  
        self.W_v = nn.Linear(in_channels, self.heads * out_channels, bias=bias)
        self.edge_mlp = ConfigurableMLP(
            layer_sizes=[3*in_channels,12,16,self.heads],
            activation=activation,
            dropout=dropout
        )
        # Attention parameter vector
        self.att = nn.Parameter(torch.Tensor(1, self.heads, 2 * out_channels))

        # Learnable temperature parameter λ for correlation modulation
        self.lambda_param = nn.Parameter(torch.Tensor(self.heads, 1))
        self.edge_lambda_param = nn.Parameter(torch.Tensor(self.heads, 1))
        # Determine MLP input dimension
        if concat:
            mlp_input_dim = self.heads * out_channels
        else:
            mlp_input_dim = out_channels
        
        # Output MLP to project to 3D (always outputs 3 features)
        mlp_layer_sizes = [mlp_input_dim] + mlp_layers
        if mlp_layer_sizes[-1] != 3:
            mlp_layer_sizes.append(3)  # final check: should not be triggered but anyways
        self.output_mlp = ConfigurableMLP(
            layer_sizes=mlp_layer_sizes,
            activation=activation,
            dropout=0.0  # No dropout in output MLP for stability
        )
        
        # Final bias for 3D output
        if bias:
            self.final_bias = nn.Parameter(torch.Tensor(3))
        else:
            self.register_parameter('final_bias', None)
            
        self.reset_parameters()
        
        logger.info(f"CorrelationModulatedGAT: in_channels={in_channels}, out_channels={out_channels}, "
                   f"heads={heads}, concat={concat}, correlation_mode={correlation_mode}, "
                   f"mlp_layers={mlp_layer_sizes}")
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.constant_(self.edge_lambda_param, 1.0)  # Initialize edge λ to 1.0
        nn.init.constant_(self.lambda_param, 1.0)  # Initialize λ to 1.0
        if self.final_bias is not None:
            nn.init.constant_(self.final_bias, 0.0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: torch.Tensor,
        size: Optional[tuple] = None,
        return_attention_weights: Optional[bool] = None
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass of correlation-modulated GAT layer.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, 9] (flattened 3x3 QFI matrices)
            size: Size of source and target nodes
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node features [N, 3] (always 3D output due to MLP)
            Optionally returns attention weights if return_attention_weights=True
        """
        # Linear transformations
        query = self.W_q(x).view(-1, self.heads, self.out_channels)  # [N, heads, out_channels]
        key = self.W_k(x).view(-1, self.heads, self.out_channels)    # [N, heads, out_channels]  
        value = self.W_v(x).view(-1, self.heads, self.out_channels)  # [N, heads, out_channels]
        
        # Propagate messages
        out = self.propagate(
            edge_index, 
            query=query, 
            key=key, 
            value=value, 
            edge_attr=edge_attr,
            size=size
        ) 
        
        
        # Handle multi-head outputs
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)  # [N, heads * out_channels]
        else:
            out = out.mean(dim=1)  # [N, out_channels] - Average across heads
        
        # Apply output MLP to get 3D features
        out = self.output_mlp(out)  # [N, 3]
        
        # Add final bias if specified
        if self.final_bias is not None:
            out += self.final_bias
            
        if return_attention_weights:
            # Note: For simplicity, we're not returning attention weights in this implementation
            # This would require modifications to store and return the computed attention coefficients
            return out, None
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
        Compute messages with correlation-modulated attention.
        
        Args:
            query_i: Query features of target nodes [E, heads, out_channels]
            key_j: Key features of source nodes [E, heads, out_channels]  
            value_j: Value features of source nodes [E, heads, out_channels]
            edge_attr: Edge attributes [E, 9] (flattened QFI matrices)
            index: Target node indices for each edge
            ptr: Pointer for batched graphs
            size_i: Number of target nodes
            
        Returns:
            Messages [E, heads, out_channels]
        """
        # Compute standard GAT attention scores
        # e_ij = LeakyReLU(a^T [Wx_i || Wx_j])
        qk_concat = torch.cat([query_i, key_j], dim=-1)  # [E, heads, 2 * out_channels]
        e_ij = (qk_concat * self.att).sum(dim=-1)  # [E, heads]
        e_ij = F.leaky_relu(e_ij, self.negative_slope)
        
        # Extract correlation coefficient from edge attributes
        C_ij = self._extract_correlation(edge_attr).squeeze()  # [E,] --> get rid of extra dimensions: the correlation feature here is the norm (whether frobenius or other)
        edge_lambda_broadcast = self.edge_lambda_param.transpose(0, 1)  # [1, heads]
        edge_value = edge_lambda_broadcast * self.edge_mlp(edge_attr)  # [E, heads]

        # Modulate attention with correlation: e_ij + λ * C_ij
        # Broadcast λ and C_ij to match attention dimensions
        lambda_broadcast = self.lambda_param.transpose(0, 1)  # [1, heads]
        C_ij_broadcast = C_ij.unsqueeze(1)  # [E, 1]

        modulated_attention = e_ij + C_ij_broadcast*lambda_broadcast*10  # [E, heads]

        # Compute attention coefficients using softmax
        alpha = softmax(modulated_attention, index, ptr, size_i)  # [E, heads]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # Compute messages: m_ij = α_ij * W_v * x_j
        alpha_expanded = alpha.unsqueeze(-1)  # [E, heads, 1]
        messages = alpha_expanded * value_j  # [E, heads, out_channels]
        
        return messages
    
    def _extract_correlation(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Extract correlation coefficient from edge attributes.
        
        Args:
            edge_attr: Edge attributes [E, 9] (flattened 3x3 QFI matrices)
            
        Returns:
            Correlation coefficients [E, 1]
        """
        # Reshape flattened QFI matrices to 3x3
        correlation_matrices = edge_attr.view(-1, 3, 3)  # [E, 3, 3]
        
        if self.correlation_mode == 'scalar':
            # Use the (0,0) element of the correlation matrix (top-left)
            return correlation_matrices[:, 0, 0:1]  # [E, 1]
        elif self.correlation_mode == 'trace':
            # Compute trace of correlation matrices
            trace = torch.diagonal(correlation_matrices, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)  # [E, 1]
            return trace / 3.0  # Normalize by matrix size
        elif self.correlation_mode == 'frobenius':
            # Compute Frobenius norm of correlation matrices
            frobenius_norm = torch.norm(correlation_matrices, p='fro', dim=(-2, -1), keepdim=True)  # [E, 1]
            return frobenius_norm / 3.0  # Normalize
        else:
            raise ValueError(f"Unknown correlation_mode: {self.correlation_mode}")
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'correlation_mode={self.correlation_mode})')