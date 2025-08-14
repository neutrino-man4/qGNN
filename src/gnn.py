"""
Author: Aritra Bal, ETP
Date: XIII Idibus Sextilibus anno ab urbe condita MMDCCLXXVIII

Graph Neural Network architectures for jet classification with correlation-aware message passing.
Compatible with the jet graph dataloader for particle physics applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import List, Optional, Union, Callable
import warnings
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


class JetGNN(nn.Module):
    """
    Complete Graph Neural Network for jet classification.
    
    Architecture:
    1. Multiple message passing layers (Correlation or Bilinear)
    2. Global pooling to create graph-level representation  
    3. Classification MLP for final prediction
    
    Args:
        message_type: Type of message passing ('correlation' or 'bilinear')
        num_mp_layers: Number of message passing layers
        mp_mlp_layers: MLP architecture for message passing [input, hidden1, ..., output]
        classifier_layers: MLP architecture for final classifier [input, hidden1, ..., 2]
        pooling_type: Global pooling method ('mean', 'max', 'add', 'concat')
        activation: Activation function
        residual_connections: Whether to use residual connections in MP layers
    """
    
    def __init__(
        self,
        message_type: str = 'correlation',
        num_mp_layers: int = 3,
        mp_mlp_layers: List[int] = [9, 16, 8, 3],  # For correlation: [9, ..., 3]
        classifier_layers: List[int] = [3, 16, 8, 4, 2],  # [input, hidden, ..., 2]
        pooling_type: str = 'mean',
        activation: Union[str, nn.Module] = 'elu',
        residual_connections: bool = True
    ):
        super().__init__()
        
        logger.info(f"Initializing JetGNN with message_type={message_type}, num_layers={num_mp_layers}")
        logger.debug(f"MP MLP layers: {mp_mlp_layers}")
        logger.debug(f"Classifier layers: {classifier_layers}")
        
        self.message_type = message_type.lower()
        self.num_mp_layers = num_mp_layers
        self.residual_connections = residual_connections
        
        # Handle activation function
        if isinstance(activation, str):
            activation_map = {
                'relu': nn.ReLU(),
                'elu': nn.ELU(),
                'leaky_relu': nn.LeakyReLU(),
                'gelu': nn.GELU(),
                'silu': nn.SiLU()
            }
            if activation.lower() not in activation_map:
                logger.error(f"Unsupported activation: {activation}")
                raise ValueError(f"Unsupported activation: {activation}")
            self.activation = activation_map[activation.lower()]
            logger.debug(f"Using activation function: {activation}")
        else:
            self.activation = activation
            logger.debug(f"Using custom activation function: {type(activation).__name__}")
        
        # Validate message passing architecture
        if self.message_type == 'correlation' and mp_mlp_layers[0] != 9:
            logger.warning(f"Adjusting MP input dimension from {mp_mlp_layers[0]} to 9 for correlation message passing")
            mp_mlp_layers[0] = 9
        elif self.message_type == 'bilinear' and mp_mlp_layers[0] != 7:
            logger.warning(f"Adjusting MP input dimension from {mp_mlp_layers[0]} to 7 for bilinear message passing")
            mp_mlp_layers[0] = 7
        
        # Ensure MP layers output 3D features (same as input) for residual connections
        if self.residual_connections and mp_mlp_layers[-1] != 3:
            logger.warning(f"For residual connections, adjusting MP output from {mp_mlp_layers[-1]} to 3")
            mp_mlp_layers[-1] = 3
        
        # Create message passing layers
        self.mp_layers = nn.ModuleList()
        for i in range(num_mp_layers):
            logger.debug(f"Creating message passing layer {i+1}/{num_mp_layers}")
            if self.message_type == 'correlation':
                mp_layer = CorrelationMessagePassing(
                    mlp_layers=mp_mlp_layers.copy(),
                    activation=self.activation
                )
            elif self.message_type == 'bilinear':
                mp_layer = BilinearMessagePassing(
                    mlp_layers=mp_mlp_layers.copy(), 
                    activation=self.activation
                )
            else:
                logger.error(f"Unsupported message type: {self.message_type}")
                raise ValueError(f"Unsupported message type: {self.message_type}")
            
            self.mp_layers.append(mp_layer)
        
        # Global pooling
        logger.debug(f"Creating global pooling with type: {pooling_type}")
        self.pooling = GlobalPooling(pooling_type)
        
        # Adjust classifier input dimension based on pooling
        if pooling_type == 'concat':
            classifier_input_dim = mp_mlp_layers[-1] * 2
            logger.debug(f"Using concat pooling, classifier input dim: {classifier_input_dim}")
        else:
            classifier_input_dim = mp_mlp_layers[-1]
        
        if classifier_layers[0] != classifier_input_dim:
            logger.warning(f"Adjusting classifier input dimension from {classifier_layers[0]} to {classifier_input_dim}")
            classifier_layers[0] = classifier_input_dim
        
        # Final classifier
        logger.debug(f"Creating classifier with layers: {classifier_layers}")
        self.classifier = ConfigurableMLP(
            layer_sizes=classifier_layers,
            activation=self.activation,
            dropout=0.1  # Add some regularization in classifier
        )
        
        # Store architecture info
        self.architecture_info = {
            'message_type': message_type,
            'num_mp_layers': num_mp_layers,
            'mp_mlp_layers': mp_mlp_layers,
            'classifier_layers': classifier_layers,
            'pooling_type': pooling_type,
            'activation': str(self.activation),
            'residual_connections': residual_connections
        }
        
        # Log model summary
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"JetGNN created successfully with {total_params:,} total parameters")
        logger.info(f"Architecture: {num_mp_layers} MP layers, {pooling_type} pooling, residual={residual_connections}")
    
    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            data: PyG Data or Batch object containing:
                - x: Node features [N, 3]
                - edge_index: Edge connectivity [2, E]
                - edge_attr: Edge features [E, 9]
                - batch: Batch assignment (for Batch objects)
        
        Returns:
            Classification logits [batch_size, 2]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', None)
        
        # If single graph, create batch tensor
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Message passing layers
        for mp_layer in self.mp_layers:
            x_new = mp_layer(x, edge_index, edge_attr)
            
            # Apply residual connection if enabled and dimensions match
            if self.residual_connections and x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new
        
        # Global pooling to get graph-level representation
        graph_features = self.pooling(x, batch)  # [batch_size, feature_dim]
        
        # Final classification
        logits = self.classifier(graph_features)  # [batch_size, 2]
        
        return logits
    
    def get_architecture_info(self) -> dict:
        """Return architecture configuration for logging/saving."""
        return self.architecture_info.copy()


def create_jet_gnn(
    model_type: str = 'correlation',
    num_layers: int = 3,
    mp_hidden_layers: List[int] = [16, 8],
    classifier_hidden_layers: List[int] = [16, 8, 4], 
    pooling: str = 'mean',
    activation: str = 'elu',
    residual: bool = True
) -> JetGNN:
    """
    Factory function to create JetGNN models with simplified configuration.
    Just for abstraction, nothing too complex to see here
    
    Args:
        model_type: 'correlation' or 'bilinear'
        num_layers: Number of message passing layers
        mp_hidden_layers: Hidden layer sizes for MP MLPs [hidden1, hidden2, ...]
        classifier_hidden_layers: Hidden layer sizes for classifier [hidden1, ..., hidden_final]
        pooling: Global pooling type ('mean', 'max', 'add', 'concat')
        activation: Activation function name
        residual: Whether to use residual connections
    
    Returns:
        Configured JetGNN model
    
    Examples:
        # Simple correlation-based model
        model = create_jet_gnn('correlation', num_layers=3, mp_hidden_layers=[16, 8])

        # Bilinear model with different architecture
        model = create_jet_gnn('bilinear', mp_hidden_layers=[32, 16, 8], classifier_hidden_layers=[32, 16, 8])
    """
    
    logger.info(f"🏗️ Creating {model_type} JetGNN with factory function")
    logger.debug(f"Config: layers={num_layers}, mp_hidden={mp_hidden_layers}, pooling={pooling}")
    
    # Determine input dimensions based on model type
    if model_type.lower() == 'correlation':
        mp_input_dim = 9  # [x_i, x_j, C_ji @ x_j] = [3 + 3 + 3]
        logger.debug("Using correlation message passing (input_dim=9)")
    elif model_type.lower() == 'bilinear':
        mp_input_dim = 7  # [x_i, x_j, bilinear_term] = [3 + 3 + 1]
        logger.debug("Using bilinear message passing (input_dim=7)")
    else:
        logger.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Build MP MLP layers: [input, hidden1, hidden2, ..., 3]
    mp_mlp_layers = [mp_input_dim] + mp_hidden_layers + [3]
    logger.debug(f"MP MLP architecture: {mp_mlp_layers}")
    
    # Determine classifier input dimension
    if pooling == 'concat':
        classifier_input_dim = 6  # mean + max pooling
        logger.debug("Using concat pooling (classifier_input_dim=6)")
    else:
        classifier_input_dim = 3
        logger.debug(f"Using {pooling} pooling (classifier_input_dim=3)")
    
    # Build classifier layers: [input, hidden1, hidden2, ..., 2]
    classifier_layers = [classifier_input_dim] + classifier_hidden_layers + [2]
    logger.debug(f"Classifier architecture: {classifier_layers}")
    
    # Create and return model
    model = JetGNN(
        message_type=model_type,
        num_mp_layers=num_layers,
        mp_mlp_layers=mp_mlp_layers,
        classifier_layers=classifier_layers,
        pooling_type=pooling,
        activation=activation,
        residual_connections=residual
    )
    
    logger.success(f"✅ JetGNN model created successfully!")
    return model


# Model parameter counting utility
def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable and total parameters in the model.
    
    Returns:
        Dictionary with parameter counts and model size info
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Estimate model size in MB (assuming float32)
    model_size_mb = (trainable_params * 4) / (1024 * 1024)
    
    return {
        'trainable_parameters': trainable_params,
        'total_parameters': total_params, 
        'model_size_mb': round(model_size_mb, 2),
        'non_trainable_parameters': total_params - trainable_params
    }


if __name__ == "__main__":
    # Example usage and testing
    logger.info("🧪 Testing JetGNN Models")
    
    # Create sample data (compatible with dataloader output)
    batch_size = 32
    num_nodes = 10
    
    logger.debug(f"Creating test data: batch_size={batch_size}, num_nodes={num_nodes}")
    
    # Create fake batch data
    x = torch.randn(batch_size * num_nodes, 3)  # Node features
    edge_index = torch.randint(0, num_nodes, (2, batch_size * 100))  # Edge connectivity
    edge_attr = torch.randn(batch_size * 100, 9)  # Edge features
    y = torch.randint(0, 2, (batch_size,))  # Labels
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)  # Batch assignment
    
    # Create Data object
    from torch_geometric.data import Batch as PyGBatch
    data = PyGBatch(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch)
    
    logger.debug(f"Test data created: x.shape={x.shape}, edge_attr.shape={edge_attr.shape}")
    
    # Test different model configurations
    configs = [
        {
            'name': 'Correlation GNN (Default)',
            'model': create_jet_gnn('correlation')
        },
        {
            'name': 'Bilinear GNN (Default)', 
            'model': create_jet_gnn('bilinear')
        },
        {
            'name': 'Deep Correlation GNN',
            'model': create_jet_gnn('correlation', num_layers=5, mp_hidden_layers=[32, 16, 8])
        },
        {
            'name': 'Bilinear with Concat Pooling',
            'model': create_jet_gnn('bilinear', pooling='concat', classifier_hidden_layers=[32, 16])
        }
    ]
    
    for config in configs:
        logger.info(f"🔬 Testing: {config['name']}")
        model = config['model']
        
        # Forward pass
        with torch.no_grad():
            output = model(data)
        
        # Model info
        param_info = count_parameters(model)
        arch_info = model.get_architecture_info()
        
        logger.info(f"   ✅ Output shape: {output.shape}")
        logger.info(f"   📊 Parameters: {param_info['trainable_parameters']:,}")
        logger.info(f"   💾 Model size: {param_info['model_size_mb']} MB")
        logger.info(f"   🏗️ Architecture: {arch_info['num_mp_layers']} MP layers, {arch_info['pooling_type']} pooling")
    
    logger.success("🎉 All model tests completed successfully!")