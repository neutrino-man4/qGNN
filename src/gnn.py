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
from typing import List, Optional, Union, Callable, Dict, Any
from loguru import logger
import src.layers as layers
class JetGNN(nn.Module):
    """
    Complete Graph Neural Network for jet classification.
    
    Architecture:
    1. Multiple message passing layers (Correlation, Bilinear, Conv1D, or GAT)
    2. Global pooling to create graph-level representation  
    3. Classification MLP for final prediction
    
    Args:
        message_type: Type of message passing ('correlation', 'uni-correlation', 'bilinear', 'conv1d', 'GAT')
        num_mp_layers: Number of message passing layers
        mp_mlp_layers: MLP architecture for message passing [input, hidden1, ..., output]
        classifier_layers: MLP architecture for final classifier [input, hidden1, ..., 2]
        pooling_type: Global pooling method ('mean', 'max', 'add', 'concat')
        activation: Activation function
        residual_connections: Whether to use residual connections in MP layers
        extra_params: Dictionary containing type-specific parameters (for GAT, Conv1D)
    """
    
    def __init__(
        self,
        message_type: str = 'correlation',
        num_mp_layers: int = 3,
        mp_mlp_layers: List[int] = [9, 16, 8, 3],  # For correlation: [9, ..., 3]
        classifier_layers: List[int] = [3, 16, 8, 4, 2],  # [input, hidden, ..., 2]
        pooling_type: str = 'mean',
        activation: Union[str, nn.Module] = 'elu',
        residual_connections: bool = True,
        conv_out_channels: int = None,  # Deprecated - use extra_params instead
        extra_params: Dict[str, Any] = None
    ):
        super().__init__()
        
        logger.info(f"Initializing JetGNN with message_type={message_type}, num_layers={num_mp_layers}")
        logger.debug(f"MP MLP layers: {mp_mlp_layers}")
        logger.debug(f"Classifier layers: {classifier_layers}")
        
        self.message_type = message_type.lower()
        self.num_mp_layers = num_mp_layers
        self.residual_connections = residual_connections
        
        # Handle extra parameters
        if extra_params is None:
            extra_params = {}
        self.extra_params = extra_params
        
        # Backward compatibility for conv_out_channels
        if conv_out_channels is not None and 'out_channels' not in extra_params:
            extra_params['out_channels'] = conv_out_channels
            logger.warning("conv_out_channels is deprecated, use extra_params instead")
        
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
        
        # Validate message passing architecture for traditional types
        if self.message_type.lower() == 'correlation' and mp_mlp_layers[0] != 9:
            logger.warning(f"Adjusting MP input dimension from {mp_mlp_layers[0]} to 9 for correlation message passing")
            mp_mlp_layers[0] = 9
        elif self.message_type.lower() == 'uni-correlation' and mp_mlp_layers[0] != 15:
            logger.warning(f"Adjusting MP input dimension from {mp_mlp_layers[0]} to 15 for uni-correlation message passing")
            mp_mlp_layers[0] = 15
        elif self.message_type.lower() == 'bilinear' and mp_mlp_layers[0] != 7:
            logger.warning(f"Adjusting MP input dimension from {mp_mlp_layers[0]} to 7 for bilinear message passing")
            mp_mlp_layers[0] = 7
        elif self.message_type.lower() == 'conv1d' and 'out_channels' not in extra_params:
            logger.warning(f"Setting out_channels to 4 for conv1d message passing")
            extra_params['out_channels'] = 4

        # Ensure MP layers output 3D features (same as input) for residual connections
        if self.residual_connections and mp_mlp_layers[-1] != 3:
            logger.warning(f"For residual connections, adjusting MP output from {mp_mlp_layers[-1]} to 3")
            mp_mlp_layers[-1] = 3
        
        # GAT always outputs 3D due to internal MLP
        if self.message_type.lower() == 'gat':
            logger.info("GAT layers always output 3D features due to internal MLP projection")
        
        # Create message passing layers
        self.mp_layers = nn.ModuleList()
        for i in range(num_mp_layers):
            logger.debug(f"Creating message passing layer {i+1}/{num_mp_layers}")
            
            if self.message_type.lower() == 'correlation':
                mp_layer = layers.CorrelationMessagePassing(
                    mlp_layers=mp_mlp_layers.copy(),
                    activation=self.activation
                )
            elif self.message_type.lower() == 'uni-correlation':
                mp_layer = layers.UniCorrelationMessagePassing(
                    mlp_layers=mp_mlp_layers.copy(),
                    activation=self.activation
                )
            elif self.message_type.lower() == 'bilinear':
                mp_layer = layers.BilinearMessagePassing(
                    mlp_layers=mp_mlp_layers.copy(), 
                    activation=self.activation
                )
            elif self.message_type.lower() == 'conv1d':
                mp_layer = layers.Conv1DMessagePassing(
                    #mlp_layers=mp_mlp_layers.copy(),
                    activation=self.activation,
                    **extra_params  # Pass Conv1D-specific parameters
                )
            elif self.message_type.lower() == 'gat':
                # For GAT, we need in_channels and out_channels
                in_channels = 3  # Always 3D particle features
                out_channels = extra_params.get('out_channels', 4)  # Default per-head dimension

                gat_mlp_layers = mp_mlp_layers.copy()
                # If residual connections = True, the last layer should have 3 nodes. 
                # Already taken care of before, just a reminder to anyone reading this here :)

                # Extract GAT-specific parameters
                
                gat_params = {
                    'heads': extra_params.get('heads', 8),
                    'concat': extra_params.get('concat', True),
                    'dropout': extra_params.get('dropout', 0.1),
                    'bias': extra_params.get('bias', True),
                    'negative_slope': extra_params.get('negative_slope', 0.2),
                    'correlation_mode': extra_params.get('correlation_mode', 'frobenius'),
                    'mlp_layers': gat_mlp_layers,  # Use derived MLP layers
                    'activation': self.activation
                }

                mp_layer = layers.CorrelationModulatedGAT(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    **gat_params
                )
                logger.info(f"Created GAT layer with in_channels={in_channels}, out_channels={out_channels}, heads={gat_params['heads']}, Correlation Mode: {gat_params['correlation_mode']}")
            else:
                logger.error(f"Unsupported message type: {self.message_type}")
                raise ValueError(f"Unsupported message type: {self.message_type}")
            
            self.mp_layers.append(mp_layer)
        
        # Global pooling
        logger.debug(f"Creating global pooling with type: {pooling_type}")
        self.pooling = layers.GlobalPooling(pooling_type)

        # Determine output dimension from last MP layer
        if self.message_type in ['gat']:
            mp_output_dim = 3  # GAT always outputs 3D
        else:
            mp_output_dim = mp_mlp_layers[-1]
        
        # Adjust classifier input dimension based on pooling
        if pooling_type == 'concat':
            classifier_input_dim = mp_output_dim * 2
            logger.debug(f"Using concat pooling, classifier input dim: {classifier_input_dim}")
        else:
            classifier_input_dim = mp_output_dim
        
        if classifier_layers[0] != classifier_input_dim:
            logger.warning(f"Adjusting classifier input dimension from {classifier_layers[0]} to {classifier_input_dim}")
            classifier_layers[0] = classifier_input_dim
        
        # Final classifier
        logger.debug(f"Creating classifier with layers: {classifier_layers}")
        self.classifier = layers.ConfigurableMLP(
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
            'residual_connections': residual_connections,
            'extra_params': extra_params.copy()
        }
        
        # Log model summary
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"JetGNN created successfully with {total_params:,} total parameters")
        logger.info(f"Architecture: {num_mp_layers} MP layers, {pooling_type} pooling, residual={residual_connections}")
        if extra_params:
            logger.debug(f"Extra parameters: {extra_params}")
    
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
        # Fetch the embeddings from the last layer if it exists
        # if len(self.mp_layers) > 0 and hasattr(self.mp_layers[0], 'fetch_embedding'):
        #     self.last_layer_embeddings = self.mp_layers[0].fetch_embedding()
        # else:
        #     self.last_layer_embeddings = None
        
        # Global pooling to get graph-level representation
        graph_features = self.pooling(x, batch)  # [batch_size, feature_dim]
        
        # Final classification
        logits = self.classifier(graph_features)  # [batch_size, 2]
        
        return logits
    
    def get_reconstructed_qfi_batch_vectorized(self, edge_index, batch, num_graphs):
        """
        Reconstruct 30x30 QFI matrices from learned edge embeddings for batched data.
        
        This method takes the 9-dimensional embeddings learned by the final Conv1D message 
        passing layer, reshapes them to 3x3 matrices, and places them in the correct 
        positions within 30x30 QFI matrices (one for each graph in the batch).
        
        Args:
            edge_index: Edge connectivity tensor [2, N_edges] with global node indices
            batch: Batch assignment tensor [N_nodes] indicating which graph each node belongs to
            num_graphs: Number of graphs in the batch
            
        Returns:
            torch.Tensor: Reconstructed QFI matrices [batch_size, 30, 30]
            
        Raises:
            ValueError: If embeddings haven't been computed (forward pass not run)
        """
        if not hasattr(self, 'last_layer_embeddings') or self.last_layer_embeddings is None:
            raise ValueError("No embeddings available. Run forward pass first.")
        
        # Step 1: Reshape embeddings from [N_edges, 9] to [N_edges, 3, 3]
        # Each edge has a 9-dim embedding representing a flattened 3x3 correlation matrix
        edge_matrices = self.last_layer_embeddings.view(-1, 3, 3)  # [N_edges, 3, 3]
        
        # Step 2: Initialize output tensor for all graphs in batch
        # Each graph gets a 30x30 QFI matrix (10 particles × 3 features each = 30)
        qfi_batch = torch.zeros(
            num_graphs, 30, 30, 
            device=edge_matrices.device, 
            dtype=edge_matrices.dtype
        )
        
        # Step 3: Determine which graph (AKA batch) each edge belongs to
        # One graph = one batch
        # Use the source node's batch assignment to assign edges to graphs
        # This works because all edges in a graph connect nodes from the same graph
        edge_batch = batch[edge_index[0]]  # [N_edges] - graph ID for each edge
        
        # Step 4: Process each graph separately
        # We need to loop over graphs because we need to convert global node indices 
        # to local indices (0-9) for proper placement in the 30x30 matrix
        for graph_idx in range(num_graphs):
            # Step 4a: Find all edges belonging to this graph
            graph_edge_mask = (edge_batch == graph_idx)  # [N_edges] boolean mask
            
            if not graph_edge_mask.any():
                # Skip if no edges in this graph (shouldn't happen with fully connected graphs)
                continue
            
            # Step 4b: Extract edge information for this graph
            graph_src = edge_index[0][graph_edge_mask]  # [N_graph_edges] global source indices
            graph_dst = edge_index[1][graph_edge_mask]  # [N_graph_edges] global dest indices  
            graph_embeddings = edge_matrices[graph_edge_mask]  # [N_graph_edges, 3, 3]
            
            # Step 4c: Convert global node indices to local indices (0-9)
            # In batched PyG data, nodes are globally indexed across all graphs
            # Graph 0: nodes 0-9, Graph 1: nodes 10-19, etc.
            # We need local indices 0-9 for each graph to place in 30x30 matrix
            graph_nodes = torch.where(batch == graph_idx)[0]  # All nodes in this graph
            node_offset = graph_nodes[0]  # First node index for this graph
            
            local_src = graph_src - node_offset  # Convert to local indices [0-9]
            local_dst = graph_dst - node_offset  # Convert to local indices [0-9]
            
            # Step 4d: Vectorized placement of 3x3 embeddings in 30x30 matrix
            # For each edge i->j, place the 3x3 embedding at position:
            # Rows: [3*i : 3*(i+1)], Cols: [3*j : 3*(j+1)]
            # This is because each particle has 3 features (pt, eta, phi)
            
            n_graph_edges = len(local_src)
            
            # Create row and column start positions for each edge
            row_starts = local_src * 3  # [N_graph_edges] - starting row for each edge
            col_starts = local_dst * 3  # [N_graph_edges] - starting col for each edge
            
            # Create offset arrays for 3x3 placement
            offsets = torch.arange(3, device=edge_matrices.device)  # [3] = [0, 1, 2]
            
            # Expand row and column indices to cover full 3x3 blocks
            # row_starts: [N_edges] -> [N_edges, 1] -> [N_edges, 3] 
            all_rows = row_starts.unsqueeze(1) + offsets.unsqueeze(0)  # [N_edges, 3]
            all_cols = col_starts.unsqueeze(1) + offsets.unsqueeze(0)  # [N_edges, 3]
            
            # Create 2D meshgrid for full indexing
            # We need [N_edges, 3, 3] indices for both rows and cols
            all_rows_2d = all_rows.unsqueeze(2).expand(-1, -1, 3)  # [N_edges, 3, 3]
            all_cols_2d = all_cols.unsqueeze(1).expand(-1, 3, -1)  # [N_edges, 3, 3]
            
            # Flatten for advanced indexing
            flat_rows = all_rows_2d.flatten()  # [N_edges * 9]
            flat_cols = all_cols_2d.flatten()  # [N_edges * 9]
            flat_values = graph_embeddings.flatten()  # [N_edges * 9]
            
            # Step 4e: Place all values using advanced indexing
            # This is equivalent to multiple qfi_matrix[3*i:3*(i+1), 3*j:3*(j+1)] = embedding[k]
            # but done in a vectorized way for all edges at once
            qfi_batch[graph_idx, flat_rows, flat_cols] = flat_values
        #import pdb;pdb.set_trace()
        return qfi_batch


    def get_reconstructed_qfi_single(self, edge_index):
        """
        Reconstruct 30x30 QFI matrix for a single graph (non-batched).
        
        Simplified version for single graph inference where batching is not needed.
        
        Args:
            edge_index: Edge connectivity tensor [2, N_edges] with local node indices
            
        Returns:
            torch.Tensor: Reconstructed QFI matrix [30, 30]
        """
        if not hasattr(self, 'last_layer_embeddings') or self.last_layer_embeddings is None:
            raise ValueError("No embeddings available. Run forward pass first.")
        
        # Reshape embeddings to 3x3 matrices
        edge_matrices = self.last_layer_embeddings.view(-1, 3, 3)  # [N_edges, 3, 3]
        
        # Initialize 30x30 QFI matrix
        qfi_matrix = torch.zeros(30, 30, device=edge_matrices.device, dtype=edge_matrices.dtype)
        
        # Extract source and destination nodes (already local indices 0-9)
        src_nodes = edge_index[0]  # [N_edges]
        dst_nodes = edge_index[1]  # [N_edges]
        
        # Vectorized placement (same logic as batch version but simpler)
        row_starts = src_nodes * 3
        col_starts = dst_nodes * 3
        offsets = torch.arange(3, device=edge_matrices.device)
        
        all_rows = row_starts.unsqueeze(1) + offsets.unsqueeze(0)  # [N_edges, 3]
        all_cols = col_starts.unsqueeze(1) + offsets.unsqueeze(0)  # [N_edges, 3]
        
        all_rows_2d = all_rows.unsqueeze(2).expand(-1, -1, 3)  # [N_edges, 3, 3]
        all_cols_2d = all_cols.unsqueeze(1).expand(-1, 3, -1)  # [N_edges, 3, 3]
        
        flat_rows = all_rows_2d.flatten()
        flat_cols = all_cols_2d.flatten()
        flat_values = edge_matrices.flatten()
        
        qfi_matrix[flat_rows, flat_cols] = flat_values
        
        return qfi_matrix

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
    residual: bool = True,
    conv_out_channels: int = None,  # Deprecated
    **kwargs  # This captures extra parameters from config
) -> JetGNN:
    """
    Factory function to create JetGNN models with simplified configuration.
    
    Args:
        model_type: 'correlation', 'uni-correlation', 'bilinear', 'conv1d', 'GAT'
        num_layers: Number of message passing layers
        mp_hidden_layers: Hidden layer sizes for MP MLPs [hidden1, hidden2, ...]
        classifier_hidden_layers: Hidden layer sizes for classifier [hidden1, ..., hidden_final]
        pooling: Global pooling type ('mean', 'max', 'add', 'concat')
        activation: Activation function name
        residual: Whether to use residual connections
        conv_out_channels: Deprecated - use kwargs instead
        **kwargs: Extra parameters for specific message passing types (GAT, Conv1D)
    
    Returns:
        Configured JetGNN model
    
    Examples:
        # Simple correlation-based model
        model = create_jet_gnn('correlation', num_layers=3, mp_hidden_layers=[16, 8])

        # GAT model with specific parameters
        model = create_jet_gnn('GAT', heads=8, correlation_mode='frobenius', mlp_layers=[64, 32])
        
        # Conv1D model with specific parameters
        model = create_jet_gnn('conv1d', out_channels=4, kernel_size=5)
    """
    
    logger.info(f"Creating {model_type} JetGNN with factory function")
    logger.debug(f"Config: layers={num_layers}, mp_hidden={mp_hidden_layers}, pooling={pooling}")
    if kwargs:
        logger.debug(f"Extra parameters: {kwargs}")
    
    model_type_lower = model_type.lower()
    
    # Determine input dimensions and build MP layers for traditional message passing
    if model_type_lower == 'correlation':
        mp_input_dim = 9  # [x_i, x_j, C_ji @ x_j] = [3 + 3 + 3]
        mp_mlp_layers = [mp_input_dim] + mp_hidden_layers + [3]
        logger.debug("Using correlation message passing (input_dim=9)")
        
    elif model_type_lower == 'uni-correlation':
        mp_input_dim = 15  # [x_i, x_j, C_ji ] = [3 + 3 + 9]
        mp_mlp_layers = [mp_input_dim] + mp_hidden_layers + [3]
        logger.debug("Using uni-correlation message passing (input_dim=15)")
        
    elif model_type_lower == 'bilinear':
        mp_input_dim = 7  # [x_i, x_j, bilinear_term] = [3 + 3 + 1]
        mp_mlp_layers = [mp_input_dim] + mp_hidden_layers + [3]
        logger.debug("Using bilinear message passing (input_dim=7)")
        
    elif model_type_lower == 'conv1d':
        # For conv1d, MLP layers are handled internally
        mp_mlp_layers = mp_hidden_layers
        logger.debug("Using 1D convolutional message passing")
        
    elif model_type_lower == 'gat':
        mp_mlp_layers = mp_hidden_layers  
        logger.debug("Using GAT message passing")
        
    else:
        logger.error(f"Unsupported model type: {model_type}")
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.debug(f"MP MLP architecture: {mp_mlp_layers}")
    
    # Determine classifier input dimension
    if pooling == 'concat':
        classifier_input_dim = 6  # mean + max pooling (always 3D output)
        logger.debug("Using concat pooling (classifier_input_dim=6)")
    else:
        classifier_input_dim = 3  # All message passing types output 3D
        logger.debug(f"Using {pooling} pooling (classifier_input_dim=3)")
    
    # Build classifier layers: [input, hidden1, hidden2, ..., 2]
    classifier_layers = [classifier_input_dim] + classifier_hidden_layers + [2]
    logger.debug(f"Classifier architecture: {classifier_layers}")
    
    # Handle backward compatibility
    extra_params = kwargs.copy()
    if conv_out_channels is not None:
        extra_params['out_channels'] = conv_out_channels
        logger.warning("conv_out_channels is deprecated, use **kwargs instead")
    
    # Create and return model
    model = JetGNN(
        message_type=model_type,
        num_mp_layers=num_layers,
        mp_mlp_layers=mp_mlp_layers,
        classifier_layers=classifier_layers,
        pooling_type=pooling,
        activation=activation,
        residual_connections=residual,
        extra_params=extra_params
    )
    
    logger.success(f"JetGNN model created successfully!")
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
    logger.info("Testing JetGNN Models")
    
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
            'name': 'DeepCorrGNN',
            'model': create_jet_gnn('correlation', num_layers=5, mp_hidden_layers=[32, 16, 8])
        },
        {
            'name': 'Bilinear with Concat Pooling',
            'model': create_jet_gnn('bilinear', pooling='concat', classifier_hidden_layers=[32, 16])
        }
    ]
    
    for config in configs:
        logger.info(f"Testing: {config['name']}")
        model = config['model']
        
        # Forward pass
        with torch.no_grad():
            output = model(data)
        
        # Model info
        param_info = count_parameters(model)
        arch_info = model.get_architecture_info()
        
        logger.info(f"Output shape: {output.shape}")
        logger.info(f"Parameters: {param_info['trainable_parameters']:,}")
        logger.info(f"Model size: {param_info['model_size_mb']} MB")
        logger.info(f"Architecture: {arch_info['num_mp_layers']} MP layers, {arch_info['pooling_type']} pooling")
    
    logger.success("All model tests completed successfully!")