"""
Author: Aritra Bal, ETP
Date: ante diem quintum Kalendas Septembres anno ab urbe condita MMDCCLXXVIII

Fixed-operation graph classifiers for jet classification using statistical discrimination.
Implements non-learnable message passing with Fisher and Mahalanobis discrimination methods.
Compatible with the jet graph dataloader for particle physics applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Optional, Union, Tuple, Dict, Any
from loguru import logger

# Import project modules - FixedCorrelationMessage will be implemented in src.layers
from src.layers import GlobalPooling
# from src.layers import FixedCorrelationMessage  # To be implemented later


class JetGraph(nn.Module):
    """
    Fixed-operation Graph Neural Network for jet classification using statistical discrimination.
    
    Unlike JetGNN, this class uses completely fixed (non-learnable) message passing operations
    and applies statistical classification methods (Fisher discriminant, Mahalanobis distance)
    based on pre-computed class statistics from training data.
    
    Architecture:
    1. Fixed message passing layers (no learnable parameters)
    2. Global pooling to create graph-level representation
    3. Statistical discrimination (Fisher/Mahalanobis) or raw feature output
    
    Args:
        message_type: Type of fixed message passing ('fixed_correlation' only at the moment)
        num_mp_layers: Number of message passing layers
        pooling_type: Global pooling method ('mean', 'max', 'add', 'concat')
        output_mode: Classification method ('default', 'fisher', 'mahalanobis')
        feature_dim: Expected dimension of graph-level features (default: 6 for concat pooling)
        extra_params: Additional parameters for fixed message passing operations
    """
    
    def __init__(
        self,
        message_type: str = 'fixed_correlation',
        num_mp_layers: int = 3,
        pooling_type: str = 'concat',  # Default to concat for 6D features
        output_mode: str = 'default',
        feature_dim: int = 6,  # 3D mean + 3D max = 6D for concat pooling
        extra_params: Dict[str, Any] = None
    ):
        super().__init__()
        
        logger.info(f"Initializing JetGraph with message_type={message_type}, "
                   f"num_layers={num_mp_layers}, output_mode={output_mode}")
        
        self.message_type = message_type.lower()
        self.num_mp_layers = num_mp_layers
        self.output_mode = output_mode.lower()
        self.feature_dim = feature_dim
        
        if extra_params is None:
            extra_params = {}
        self.extra_params = extra_params
        
        # Validate output mode
        if self.output_mode not in ['default', 'fisher', 'mahalanobis']:
            raise ValueError(f"Invalid output_mode: {output_mode}. Must be 'default', 'fisher', or 'mahalanobis'")
        
        # Create fixed message passing layers
        self.mp_layers = nn.ModuleList()
        for i in range(num_mp_layers):
            logger.debug(f"Creating fixed message passing layer {i+1}/{num_mp_layers}")
            
            if self.message_type == 'fixed_correlation':
                # Will use FixedCorrelationMessage from src.layers when implemented
                # For now, create placeholder that will be replaced
                mp_layer = self._create_placeholder_fixed_mp()
            else:
                raise ValueError(f"Unsupported fixed message type: {self.message_type}")
            
            self.mp_layers.append(mp_layer)
        
        # Global pooling
        logger.debug(f"Creating global pooling with type: {pooling_type}")
        self.pooling = GlobalPooling(pooling_type)
        
        # Set expected feature dimension based on pooling
        if pooling_type == 'concat':
            expected_dim = 6  # 3D mean + 3D max
        else:
            expected_dim = 3  # Single pooling output
        
        if self.feature_dim != expected_dim:
            logger.warning(f"Feature dimension mismatch: expected {expected_dim} for {pooling_type} pooling, got {feature_dim}")
            self.feature_dim = expected_dim
        
        # Statistical classification parameters (registered as buffers, not parameters)
        self.register_buffer('ttbar_mean', torch.zeros(self.feature_dim))
        self.register_buffer('qcd_mean', torch.zeros(self.feature_dim))
        self.register_buffer('pooled_covariance', torch.eye(self.feature_dim))
        self.register_buffer('inv_pooled_covariance', torch.eye(self.feature_dim))
        self.register_buffer('class_means_set', torch.tensor(False))
        
        # Store architecture info (no learnable parameters to count)
        self.architecture_info = {
            'message_type': message_type,
            'num_mp_layers': num_mp_layers,
            'pooling_type': pooling_type,
            'output_mode': output_mode,
            'feature_dim': self.feature_dim,
            'extra_params': extra_params.copy(),
            'total_parameters': 0,  # No learnable parameters
            'is_fixed_operations': True
        }
        
        logger.info(f"JetGraph created successfully with {self.feature_dim}D features")
        logger.info(f"Fixed operations: {num_mp_layers} MP layers, {pooling_type} pooling, {output_mode} discrimination")
        logger.warning("Class means not set - use set_class_means() before classification")
    
    def _create_placeholder_fixed_mp(self):
        """Create placeholder for fixed message passing - will be replaced with actual implementation."""
        class PlaceholderFixedMP(nn.Module):
            def __init__(self):
                super().__init__()
                logger.warning("Using placeholder fixed message passing - implement FixedCorrelationMessage in src.layers")
            
            def forward(self, x, edge_index, edge_attr):
                # Placeholder: return input unchanged
                return x
        
        return PlaceholderFixedMP()
    
    def set_class_means(
        self, 
        ttbar_mean: Union[torch.Tensor, np.ndarray, List[float]], 
        qcd_mean: Union[torch.Tensor, np.ndarray, List[float]],
        pooled_covariance: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> None:
        """
        Set the class means and covariance matrix for statistical discrimination.
        
        Args:
            ttbar_mean: 6D mean vector for TTbar jets (label 1)
            qcd_mean: 6D mean vector for QCD jets (label 0)  
            pooled_covariance: Pooled covariance matrix (6x6) for Mahalanobis distance.
                              If None, uses identity matrix.
        
        Note:
            These should be computed from your training dataset and saved in config
            or computed in a separate calibration step.
        """
        # Convert to tensors and ensure correct dimensions
        ttbar_mean = torch.as_tensor(ttbar_mean, dtype=torch.float32)
        qcd_mean = torch.as_tensor(qcd_mean, dtype=torch.float32)
        
        if ttbar_mean.shape != (self.feature_dim,):
            raise ValueError(f"TTbar mean shape {ttbar_mean.shape} doesn't match feature_dim {self.feature_dim}")
        if qcd_mean.shape != (self.feature_dim,):
            raise ValueError(f"QCD mean shape {qcd_mean.shape} doesn't match feature_dim {self.feature_dim}")
        
        # Set means
        self.ttbar_mean.copy_(ttbar_mean)
        self.qcd_mean.copy_(qcd_mean)
        
        # Set covariance matrix
        if pooled_covariance is not None:
            pooled_cov = torch.as_tensor(pooled_covariance, dtype=torch.float32)
            if pooled_cov.shape != (self.feature_dim, self.feature_dim):
                raise ValueError(f"Covariance shape {pooled_cov.shape} doesn't match ({self.feature_dim}, {self.feature_dim})")
            
            self.pooled_covariance.copy_(pooled_cov)
            
            # Compute and store inverse for efficiency
            try:
                self.inv_pooled_covariance.copy_(torch.linalg.inv(pooled_cov))
            except torch.linalg.LinAlgError:
                logger.warning("Covariance matrix is singular, adding regularization")
                regularized = pooled_cov + 1e-6 * torch.eye(self.feature_dim)
                self.inv_pooled_covariance.copy_(torch.linalg.inv(regularized))
        
        # Mark as set
        self.class_means_set.copy_(torch.tensor(True))
        
        logger.success(f"Class means set successfully")
        logger.info(f"TTbar mean: {ttbar_mean.numpy()}")
        logger.info(f"QCD mean: {qcd_mean.numpy()}")
        if pooled_covariance is not None:
            logger.info(f"Pooled covariance determinant: {torch.linalg.det(self.pooled_covariance):.6f}")
    
    def get_class_means(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the stored class means.
        
        Returns:
            Tuple of (ttbar_mean, qcd_mean) tensors
        """
        if not self.class_means_set:
            logger.warning("Class means have not been set yet")
        return self.ttbar_mean.clone(), self.qcd_mean.clone()
    
    def fisher_discriminant(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply Fisher Linear Discriminant Analysis for classification.
        
        The Fisher discriminant is: w^T * x + b where w = Σ^(-1) * (μ₁ - μ₀)
        and the decision boundary is at w^T * x + b = 0
        
        Args:
            features: Graph-level features [batch_size, feature_dim]
            
        Returns:
            Fisher discriminant scores [batch_size, 2] (QCD score, TTbar score)
        """
        if not self.class_means_set:
            raise RuntimeError("Class means must be set before using Fisher discriminant")
        
        # Fisher weight vector: w = Σ^(-1) * (μ₁ - μ₀)
        mean_diff = self.ttbar_mean - self.qcd_mean  # [feature_dim]
        fisher_weights = torch.matmul(self.inv_pooled_covariance, mean_diff)  # [feature_dim]
        
        # Fisher scores: w^T * (x - μ_pooled) where μ_pooled = (μ₁ + μ₀) / 2
        pooled_mean = (self.ttbar_mean + self.qcd_mean) / 2.0
        centered_features = features - pooled_mean.unsqueeze(0)  # [batch_size, feature_dim]
        
        # Compute discriminant scores
        fisher_scores = torch.matmul(centered_features, fisher_weights)  # [batch_size]
        
        # Convert to class probabilities: positive scores favor TTbar, negative favor QCD
        qcd_scores = -fisher_scores  # Negative of Fisher score for QCD
        ttbar_scores = fisher_scores  # Positive Fisher score for TTbar
        
        return torch.stack([qcd_scores, ttbar_scores], dim=-1)  # [batch_size, 2]
    
    def mahalanobis_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply Mahalanobis distance-based classification.
        
        Computes Mahalanobis distance to each class mean:
        d²(x, μᵢ) = (x - μᵢ)^T * Σ^(-1) * (x - μᵢ)
        
        Classification based on minimum distance (or negative log-likelihood).
        
        Args:
            features: Graph-level features [batch_size, feature_dim]
            
        Returns:
            Negative Mahalanobis distances [batch_size, 2] (QCD, TTbar)
            Higher values indicate higher likelihood of belonging to that class
        """
        if not self.class_means_set:
            raise RuntimeError("Class means must be set before using Mahalanobis distance")
        
        batch_size = features.size(0)
        
        # Compute squared Mahalanobis distances to each class mean
        # Distance to QCD mean
        diff_qcd = features - self.qcd_mean.unsqueeze(0)  # [batch_size, feature_dim]
        dist_qcd_sq = torch.sum(diff_qcd * torch.matmul(diff_qcd, self.inv_pooled_covariance), dim=-1)  # [batch_size]
        
        # Distance to TTbar mean  
        diff_ttbar = features - self.ttbar_mean.unsqueeze(0)  # [batch_size, feature_dim]
        dist_ttbar_sq = torch.sum(diff_ttbar * torch.matmul(diff_ttbar, self.inv_pooled_covariance), dim=-1)  # [batch_size]
        
        # Return negative distances (higher = more likely)
        # This makes it compatible with standard classification where higher scores = higher probability
        qcd_scores = -dist_qcd_sq
        ttbar_scores = -dist_ttbar_sq
        
        return torch.stack([qcd_scores, ttbar_scores], dim=-1)  # [batch_size, 2]
    
    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Forward pass through the fixed graph operations.
        
        Args:
            data: PyG Data or Batch object containing:
                - x: Node features [N, 3]
                - edge_index: Edge connectivity [2, E]
                - edge_attr: Edge features [E, 9] 
                - batch: Batch assignment (for Batch objects)
        
        Returns:
            Output depends on output_mode:
            - 'default': Raw graph-level features [batch_size, feature_dim]
            - 'fisher': Fisher discriminant scores [batch_size, 2]
            - 'mahalanobis': Mahalanobis distance scores [batch_size, 2]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', None)
        
        # If single graph, create batch tensor
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply fixed message passing layers
        for mp_layer in self.mp_layers:
            # Note: Fixed message passing doesn't support residual connections
            # since the goal is to use predetermined operations only
            x = mp_layer(x, edge_index, edge_attr)
        
        # Global pooling to get graph-level representation
        graph_features = self.pooling(x, batch)  # [batch_size, feature_dim]
        
        # Apply discrimination method based on output_mode
        if self.output_mode == 'default':
            return graph_features  # Raw features for computing class means
        elif self.output_mode == 'fisher':
            return self.fisher_discriminant(graph_features)
        elif self.output_mode == 'mahalanobis':
            return self.mahalanobis_distance(graph_features)
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
    
    def compute_class_statistics(self, dataloader, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Compute class means and pooled covariance from a training dataloader.
        
        This method runs the model in 'default' mode to extract features,
        then computes the sample statistics needed for Fisher/Mahalanobis classification.
        
        Args:
            dataloader: Training data loader
            device: Device to run computation on
            
        Returns:
            Dictionary containing computed statistics:
            - 'ttbar_mean': Mean of TTbar class features [feature_dim]
            - 'qcd_mean': Mean of QCD class features [feature_dim] 
            - 'pooled_covariance': Pooled covariance matrix [feature_dim, feature_dim]
            - 'n_ttbar': Number of TTbar samples
            - 'n_qcd': Number of QCD samples
        """
        logger.info("Computing class statistics from training data...")
        
        # Temporarily set to default mode
        original_mode = self.output_mode
        self.output_mode = 'default'
        
        self.eval()
        ttbar_features = []
        qcd_features = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                features = self.forward(batch)  # [batch_size, feature_dim]
                
                # Separate by class
                ttbar_mask = batch.y == 1
                qcd_mask = batch.y == 0
                
                if ttbar_mask.any():
                    ttbar_features.append(features[ttbar_mask])
                if qcd_mask.any():
                    qcd_features.append(features[qcd_mask])
        
        # Concatenate all features
        if ttbar_features:
            ttbar_features = torch.cat(ttbar_features, dim=0)  # [n_ttbar, feature_dim]
        else:
            raise ValueError("No TTbar samples found in training data")
            
        if qcd_features:
            qcd_features = torch.cat(qcd_features, dim=0)  # [n_qcd, feature_dim]
        else:
            raise ValueError("No QCD samples found in training data")
        
        # Compute class means
        ttbar_mean = torch.mean(ttbar_features, dim=0)  # [feature_dim]
        qcd_mean = torch.mean(qcd_features, dim=0)      # [feature_dim]
        
        # Compute pooled covariance matrix
        n_ttbar, n_qcd = ttbar_features.size(0), qcd_features.size(0)
        
        # Center the features
        ttbar_centered = ttbar_features - ttbar_mean.unsqueeze(0)
        qcd_centered = qcd_features - qcd_mean.unsqueeze(0)
        
        # Compute within-class covariance matrices
        cov_ttbar = torch.matmul(ttbar_centered.T, ttbar_centered) / (n_ttbar - 1)
        cov_qcd = torch.matmul(qcd_centered.T, qcd_centered) / (n_qcd - 1)
        
        # Pooled covariance: weighted average of class covariances
        pooled_cov = ((n_ttbar - 1) * cov_ttbar + (n_qcd - 1) * cov_qcd) / (n_ttbar + n_qcd - 2)
        
        # Store statistics
        statistics = {
            'ttbar_mean': ttbar_mean,
            'qcd_mean': qcd_mean,
            'pooled_covariance': pooled_cov,
            'n_ttbar': n_ttbar,
            'n_qcd': n_qcd
        }
        
        # Restore original mode
        self.output_mode = original_mode
        
        logger.success(f"Class statistics computed: {n_ttbar} TTbar, {n_qcd} QCD samples")
        logger.info(f"TTbar mean: {ttbar_mean.cpu().numpy()}")
        logger.info(f"QCD mean: {qcd_mean.cpu().numpy()}")
        logger.info(f"Covariance determinant: {torch.linalg.det(pooled_cov):.6f}")
        
        return statistics
    
    def calibrate_from_statistics(self, statistics: Dict[str, torch.Tensor]) -> None:
        """
        Set class means and covariance from pre-computed statistics.
        
        Args:
            statistics: Dictionary with 'ttbar_mean', 'qcd_mean', 'pooled_covariance'
        """
        self.set_class_means(
            statistics['ttbar_mean'],
            statistics['qcd_mean'], 
            statistics['pooled_covariance']
        )
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Return architecture configuration for logging/saving."""
        return self.architecture_info.copy()
    
    def get_fixed_operations_info(self) -> Dict[str, Any]:
        """Return information about the fixed operations used."""
        return {
            'message_type': self.message_type,
            'num_mp_layers': self.num_mp_layers,
            'pooling_type': self.pooling.pooling_type,
            'output_mode': self.output_mode,
            'class_means_set': bool(self.class_means_set),
            'extra_params': self.extra_params
        }
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Override state_dict to include statistical parameters as buffers.
        Fixed operations have no learnable parameters, only statistical calibration data.
        """
        return super().state_dict(destination, prefix, keep_vars)


def create_jet_graph(
    message_type: str = 'fixed_correlation',
    num_layers: int = 3,
    pooling: str = 'concat',
    output_mode: str = 'default',
    **kwargs
) -> JetGraph:
    """
    Factory function to create JetGraph models with fixed operations.
    
    Args:
        message_type: Type of fixed message passing ('fixed_correlation', etc.)
        num_layers: Number of message passing layers
        pooling: Global pooling type ('mean', 'max', 'add', 'concat')
        output_mode: Classification method ('default', 'fisher', 'mahalanobis')
        **kwargs: Extra parameters for fixed message passing
    
    Returns:
        Configured JetGraph model
    
    Examples:
        # Create model for computing class statistics
        model = create_jet_graph('fixed_correlation', output_mode='default')
        
        # Create model for Fisher discrimination
        model = create_jet_graph('fixed_correlation', output_mode='fisher')
        
        # Create model for Mahalanobis distance classification
        model = create_jet_graph('fixed_correlation', output_mode='mahalanobis')
    """
    logger.info(f"Creating {message_type} JetGraph with factory function")
    logger.debug(f"Config: layers={num_layers}, pooling={pooling}, mode={output_mode}")
    
    # Determine feature dimension based on pooling
    if pooling == 'concat':
        feature_dim = 6  # 3D mean + 3D max
    else:
        feature_dim = 3  # Single pooling method
    
    model = JetGraph(
        message_type=message_type,
        num_mp_layers=num_layers,
        pooling_type=pooling,
        output_mode=output_mode,
        feature_dim=feature_dim,
        extra_params=kwargs
    )
    
    logger.success(f"JetGraph model created successfully!")
    logger.info(f"Fixed operations: {num_layers} layers, {pooling} pooling, {feature_dim}D features")
    
    return model


def save_class_statistics(statistics: Dict[str, torch.Tensor], save_path: str) -> None:
    """
    Save computed class statistics to file for later use.
    
    Args:
        statistics: Dictionary containing class means and covariance
        save_path: Path to save statistics (will be saved as .pth file)
    """
    # Convert tensors to numpy for JSON compatibility
    numpy_stats = {}
    for key, value in statistics.items():
        if isinstance(value, torch.Tensor):
            numpy_stats[key] = value.cpu().numpy()
        else:
            numpy_stats[key] = value
    
    torch.save(numpy_stats, save_path)
    logger.info(f"Class statistics saved to: {save_path}")


def load_class_statistics(load_path: str) -> Dict[str, torch.Tensor]:
    """
    Load class statistics from file.
    
    Args:
        load_path: Path to load statistics from
        
    Returns:
        Dictionary containing class statistics as tensors
    """
    stats = torch.load(load_path, map_location='cpu')
    
    # Convert back to tensors if needed
    tensor_stats = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            tensor_stats[key] = torch.from_numpy(value).float()
        else:
            tensor_stats[key] = value
    
    logger.info(f"Class statistics loaded from: {load_path}")
    return tensor_stats


if __name__ == "__main__":
    # Example usage and testing
    logger.info("Testing JetGraph Fixed Classifier")
    
    # Create test data (compatible with dataloader output)
    batch_size = 16
    num_nodes = 10
    
    # Create fake batch data
    x = torch.randn(batch_size * num_nodes, 3)  # Node features
    edge_index = torch.randint(0, num_nodes, (2, batch_size * 100))  # Edge connectivity  
    edge_attr = torch.randn(batch_size * 100, 9)  # Edge features (QFI correlations)
    y = torch.randint(0, 2, (batch_size,))  # Labels (0=QCD, 1=TTbar)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)  # Batch assignment
    
    # Create Data object
    from torch_geometric.data import Batch as PyGBatch
    data = PyGBatch(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch)
    
    # Test different configurations
    logger.info("Testing default mode (for computing statistics)...")
    model_default = create_jet_graph(output_mode='default', pooling='concat')
    
    with torch.no_grad():
        features = model_default(data)  # Should return [batch_size, 6] raw features
    
    logger.info(f"Raw features shape: {features.shape}")
    
    # Set dummy class means for testing discrimination
    dummy_ttbar_mean = torch.tensor([1.0, 0.5, 0.2, 0.8, 0.3, 0.6])
    dummy_qcd_mean = torch.tensor([0.2, 0.8, 0.5, 0.3, 0.7, 0.1])
    dummy_cov = torch.eye(6) * 0.1  # Simple diagonal covariance
    
    # Test Fisher discriminant
    logger.info("Testing Fisher discriminant...")
    model_fisher = create_jet_graph(output_mode='fisher', pooling='concat')
    model_fisher.set_class_means(dummy_ttbar_mean, dummy_qcd_mean, dummy_cov)
    
    with torch.no_grad():
        fisher_scores = model_fisher(data)
    
    logger.info(f"Fisher scores shape: {fisher_scores.shape}")
    logger.info(f"Fisher score range: [{fisher_scores.min():.3f}, {fisher_scores.max():.3f}]")
    
    # Test Mahalanobis distance
    logger.info("Testing Mahalanobis distance...")
    model_mahal = create_jet_graph(output_mode='mahalanobis', pooling='concat')
    model_mahal.set_class_means(dummy_ttbar_mean, dummy_qcd_mean, dummy_cov)
    
    with torch.no_grad():
        mahal_scores = model_mahal(data)
    
    logger.info(f"Mahalanobis scores shape: {mahal_scores.shape}")
    logger.info(f"Mahalanobis score range: [{mahal_scores.min():.3f}, {mahal_scores.max():.3f}]")
    
    # Test statistics computation and saving
    logger.info("Testing statistics computation...")
    fake_stats = {
        'ttbar_mean': dummy_ttbar_mean,
        'qcd_mean': dummy_qcd_mean,
        'pooled_covariance': dummy_cov,
        'n_ttbar': 1000,
        'n_qcd': 1000
    }
    
    save_class_statistics(fake_stats, 'test_statistics.pth')
    loaded_stats = load_class_statistics('test_statistics.pth')
    
    logger.success("All JetGraph tests completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Implement FixedCorrelationMessage in src.layers")
    logger.info("2. Run model in 'default' mode on training data to compute real class statistics")
    logger.info("3. Save statistics and use for Fisher/Mahalanobis classification")