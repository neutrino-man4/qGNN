"""
Author: Aritra Bal, ETP
Date: die Iovis ante diem duodecimum Kalendas Septembres anno ab urbe condita MMDCCLXXVIII

Configuration loader for Jet GNN training with dataclass-based config management.
Loads YAML configuration files and converts them to structured Python objects.
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any, Dict
from pathlib import Path
from loguru import logger


@dataclass
class ExperimentConfig:
    """Experiment identification and output configuration."""
    seed: str = "0001"
    name: str = "jet_gnn_experiment"
    base_save_dir: str = "./experiments"


@dataclass
class ReproducibilityConfig:
    """Reproducibility settings for deterministic training."""
    random_seed: int = 42
    deterministic: bool = True


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    train_files: List[str] = field(default_factory=list)
    val_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    batch_size: int = 64
    use_qfi_correlations: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    test_n: int = 1000


@dataclass
class GATConfig:
    """GAT-specific configuration."""
    out_channels: int = 4
    heads: int = 8
    concat: bool = True
    dropout: float = 0.1
    negative_slope: float = 0.2
    correlation_mode: str = "trace"  # "scalar", "trace", "frobenius"
    add_self_loops: bool = True
    bias: bool = True


@dataclass  
class Conv1DConfig:
    """Conv1D message passing configuration."""
    out_channels: int = 4
    kernel_size: int = 5
    mlp_layers: List[int] = field(default_factory=lambda: [3, 16, 8, 4, 2])


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    type: str = "correlation"  # "correlation", "uni-correlation", "bilinear", "conv1d", "GAT"
    num_mp_layers: int = 3
    mp_hidden_layers: List[int] = field(default_factory=lambda: [16, 8])
    classifier_hidden_layers: List[int] = field(default_factory=lambda: [16, 8])
    pooling: str = "mean"  # "mean", "max", "add", "concat"
    activation: str = "elu"
    residual_connections: bool = True
    
    # Nested configurations for specific model types
    gat_config: GATConfig = field(default_factory=GATConfig)
    conv1d_config: Conv1DConfig = field(default_factory=Conv1DConfig)
    def get_extra_config(self) -> Dict[str, Any]:
        """Get the extra configuration for the current model type."""
        if self.type.lower() == "gat":
            return self.gat_config.__dict__
        elif self.type.lower() == "conv1d":
            return self.conv1d_config.__dict__
        else:
            return {}  # No extra config needed for correlation, uni-correlation, bilinear


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool = True
    patience: int = 10
    monitor: str = "val_auc"
    min_delta: float = 1e-4
    mode: str = "max"  # "min" for loss, "max" for accuracy/auc


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    num_epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    use_scheduler: bool = True
    scheduler_type: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    gradient_clip_val: float = 1.0
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass
class ValidationConfig:
    """Validation configuration."""
    val_metrics: List[str] = field(default_factory=lambda: ["accuracy", "auc"])


@dataclass
class CheckpointingConfig:
    """Model checkpointing configuration."""
    save_frequency: int = 10
    save_best_only: bool = False
    best_metric: str = "val_auc"
    best_mode: str = "max"
    save_optimizer: bool = True
    save_scheduler: bool = True


@dataclass
class HardwareConfig:
    """Hardware and performance configuration."""
    device: str = "auto"
    mixed_precision: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    log_filename: str = "training.log"
    log_frequency: int = 10


@dataclass
class TestingConfig:
    """Testing and inference configuration."""
    batch_size: int = 128
    save_predictions: bool = True
    predictions_file: str = "test_predictions.csv"
    metrics_file: str = "test_metrics.json"


@dataclass
class ResumeConfig:
    """Resume training configuration."""
    enabled: bool = False
    checkpoint_path: Optional[str] = None


@dataclass
class Config:
    """Complete configuration object containing all settings."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file and convert to Config object.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config object with all settings loaded from YAML
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML file is malformed
        ValueError: If required configuration values are missing or invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    
    if yaml_data is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    try:
        # Create config objects from YAML data with defaults for missing sections
        config = Config(
            experiment=_create_experiment_config(yaml_data.get('experiment', {})),
            reproducibility=_create_reproducibility_config(yaml_data.get('reproducibility', {})),
            data=_create_data_config(yaml_data.get('data', {})),
            model=_create_model_config(yaml_data.get('model', {})),
            training=_create_training_config(yaml_data.get('training', {})),
            validation=_create_validation_config(yaml_data.get('validation', {})),
            checkpointing=_create_checkpointing_config(yaml_data.get('checkpointing', {})),
            hardware=_create_hardware_config(yaml_data.get('hardware', {})),
            logging=_create_logging_config(yaml_data.get('logging', {})),
            testing=_create_testing_config(yaml_data.get('testing', {})),
            resume=_create_resume_config(yaml_data.get('resume', {}))
        )
        
        # Validate configuration
        _validate_config(config)
        
        logger.success(f"Configuration loaded successfully: {config.experiment.name}")
        return config
        
    except Exception as e:
        raise ValueError(f"Error creating configuration object: {e}")


def _create_experiment_config(data: dict) -> ExperimentConfig:
    """Create ExperimentConfig from YAML data."""
    return ExperimentConfig(
        seed=data.get('seed', '0001'),
        name=data.get('name', 'jet_gnn_experiment'),
        base_save_dir=data.get('base_save_dir', './experiments')
    )


def _create_reproducibility_config(data: dict) -> ReproducibilityConfig:
    """Create ReproducibilityConfig from YAML data."""
    return ReproducibilityConfig(
        random_seed=data.get('random_seed', 42),
        deterministic=data.get('deterministic', True)
    )


def _create_data_config(data: dict) -> DataConfig:
    """Create DataConfig from YAML data."""
    return DataConfig(
        train_files=data.get('train_files', []),
        val_files=data.get('val_files', []),
        test_files=data.get('test_files', []),
        batch_size=data.get('batch_size', 64),
        use_qfi_correlations=data.get('use_qfi_correlations', True),
        num_workers=data.get('num_workers', 8),
        pin_memory=data.get('pin_memory', True),
        test_n=data.get('test_n', 1000)
    )


def _create_gat_config(data: dict) -> GATConfig:
    """Create GATConfig from YAML data."""
    return GATConfig(
        out_channels=data.get('out_channels', 4),
        heads=data.get('heads', 8),
        concat=data.get('concat', True),
        dropout=data.get('dropout', 0.1),
        negative_slope=data.get('negative_slope', 0.2),
        correlation_mode=data.get('correlation_mode', 'trace'),
        bias=data.get('bias', True)
    )


def _create_conv1d_config(data: dict) -> Conv1DConfig:
    """Create Conv1DConfig from YAML data."""
    return Conv1DConfig(
        out_channels=data.get('out_channels', 4),
        kernel_size=data.get('kernel_size', 5),
    )


def _create_model_config(data: dict) -> ModelConfig:
    """Create ModelConfig from YAML data."""
    return ModelConfig(
        type=data.get('type', 'correlation'),
        num_mp_layers=data.get('num_mp_layers', 3),
        mp_hidden_layers=data.get('mp_hidden_layers', [16, 8]),
        classifier_hidden_layers=data.get('classifier_hidden_layers', [16, 8]),
        pooling=data.get('pooling', 'mean'),
        activation=data.get('activation', 'elu'),
        residual_connections=data.get('residual_connections', True),
        gat_config=_create_gat_config(data.get('gat_config', {})),
        conv1d_config=_create_conv1d_config(data.get('conv1d_config', {}))
    )


def _create_training_config(data: dict) -> TrainingConfig:
    """Create TrainingConfig from YAML data."""
    early_stopping_data = data.get('early_stopping', {})
    early_stopping = EarlyStoppingConfig(
        enabled=early_stopping_data.get('enabled', True),
        patience=early_stopping_data.get('patience', 10),
        monitor=early_stopping_data.get('monitor', 'val_auc'),
        min_delta=early_stopping_data.get('min_delta', 1e-4),
        mode=early_stopping_data.get('mode', 'max'),
    )
    
    return TrainingConfig(
        num_epochs=data.get('num_epochs', 100),
        learning_rate=data.get('learning_rate', 0.001),
        optimizer=data.get('optimizer', 'adam'),
        weight_decay=data.get('weight_decay', 1e-4),
        use_scheduler=data.get('use_scheduler', True),
        scheduler_type=data.get('scheduler_type', 'reduce_on_plateau'),
        scheduler_patience=data.get('scheduler_patience', 5),
        scheduler_factor=data.get('scheduler_factor', 0.5),
        gradient_clip_val=data.get('gradient_clip_val', 1.0),
        early_stopping=early_stopping
    )


def _create_validation_config(data: dict) -> ValidationConfig:
    """Create ValidationConfig from YAML data."""
    return ValidationConfig(
        val_metrics=data.get('val_metrics', ['accuracy', 'auc'])
    )


def _create_checkpointing_config(data: dict) -> CheckpointingConfig:
    """Create CheckpointingConfig from YAML data."""
    return CheckpointingConfig(
        save_frequency=data.get('save_frequency', 10),
        save_best_only=data.get('save_best_only', False),
        best_metric=data.get('best_metric', 'val_auc'),
        best_mode=data.get('best_mode', 'max'),
        save_optimizer=data.get('save_optimizer', True),
        save_scheduler=data.get('save_scheduler', True)
    )


def _create_hardware_config(data: dict) -> HardwareConfig:
    """Create HardwareConfig from YAML data."""
    return HardwareConfig(
        device=data.get('device', 'auto'),
        mixed_precision=data.get('mixed_precision', False)
    )


def _create_logging_config(data: dict) -> LoggingConfig:
    """Create LoggingConfig from YAML data."""
    return LoggingConfig(
        level=data.get('level', 'INFO'),
        console_output=data.get('console_output', True),
        file_output=data.get('file_output', True),
        log_filename=data.get('log_filename', 'training.log'),
        log_frequency=data.get('log_frequency', 10)
    )


def _create_testing_config(data: dict) -> TestingConfig:
    """Create TestingConfig from YAML data."""
    return TestingConfig(
        batch_size=data.get('batch_size', 128),
        save_predictions=data.get('save_predictions', True),
        predictions_file=data.get('predictions_file', 'test_predictions.csv'),
        metrics_file=data.get('metrics_file', 'test_metrics.json')
    )


def _create_resume_config(data: dict) -> ResumeConfig:
    """Create ResumeConfig from YAML data."""
    return ResumeConfig(
        enabled=data.get('enabled', False),
        checkpoint_path=data.get('checkpoint_path', None)
    )


def _validate_config(config: Config) -> None:
    """
    Validate configuration values for consistency and correctness.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration values are invalid
    """
    # Validate model type
    if config.model.type.lower() not in ['correlation','uni-correlation', 'bilinear', 'conv1d', 'gat']:
        raise ValueError(f"Invalid model type: {config.model.type}. Must be 'correlation', 'uni-correlation', 'bilinear', 'conv1d', or 'GAT' (case-insensitive)")

    # Validate type-specific configurations
    if config.model.type.lower() == "gat":
        if config.model.gat_config.correlation_mode not in ['scalar', 'trace', 'frobenius']:
            raise ValueError(f"Invalid GAT correlation mode: {config.model.gat_config.correlation_mode}")
        if config.model.gat_config.heads <= 0:
            raise ValueError(f"GAT heads must be positive: {config.model.gat_config.heads}")
    
    if config.model.type.lower() == "conv1d":
        if config.model.conv1d_config.out_channels <= 0:
            raise ValueError(f"Conv1D out_channels must be positive: {config.model.conv1d_config.out_channels}")
        if config.model.conv1d_config.kernel_size <= 0:
            raise ValueError(f"Conv1D kernel_size must be positive: {config.model.conv1d_config.kernel_size}")

    # Validate pooling type
    if config.model.pooling not in ['mean', 'max', 'add', 'concat']:
        raise ValueError(f"Invalid pooling type: {config.model.pooling}")
    
    # Validate optimizer
    if config.training.optimizer not in ['adam', 'adamw', 'sgd']:
        raise ValueError(f"Invalid optimizer: {config.training.optimizer}")
    
    # Validate scheduler type
    if config.training.scheduler_type not in ['reduce_on_plateau', 'step', 'cosine']:
        raise ValueError(f"Invalid scheduler type: {config.training.scheduler_type}")
    
    # Validate early stopping mode
    if config.training.early_stopping.mode not in ['min', 'max']:
        raise ValueError(f"Invalid early stopping mode: {config.training.early_stopping.mode}")
    
    # Validate batch sizes
    if config.data.batch_size <= 0:
        raise ValueError(f"Batch size must be positive: {config.data.batch_size}")
    
    if config.testing.batch_size <= 0:
        raise ValueError(f"Test batch size must be positive: {config.testing.batch_size}")
    
    # Validate file paths exist (only check if files are specified)
    if config.data.train_files:
        for file_path in config.data.train_files:
            if not Path(file_path).exists():
                logger.warning(f"Training file not found: {file_path}")
    
    if config.data.val_files:
        for file_path in config.data.val_files:
            if not Path(file_path).exists():
                logger.warning(f"Validation file not found: {file_path}")
    
    logger.debug("Configuration validation completed successfully")


def save_config(config: Config, save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dictionary recursively
    def dataclass_to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {field: dataclass_to_dict(getattr(obj, field)) for field in obj.__dataclass_fields__}
        elif isinstance(obj, list):
            return [dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: dataclass_to_dict(value) for key, value in obj.items()}
        else:
            return obj
    
    config_dict = dataclass_to_dict(config)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)
    
    logger.info(f"Configuration saved to: {save_path}")


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config("./base.yaml")
        print("Configuration loaded successfully!")
        print(f"Experiment: {config.experiment.name}_{config.experiment.seed}")
        print(f"Model: {config.model.type} with {config.model.num_mp_layers} layers")
        print(f"Training: {config.training.num_epochs} epochs with {config.training.optimizer}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")