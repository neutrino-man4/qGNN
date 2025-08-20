# Author: Aritra Bal, ETP
# Date: Idibus Sextilibus anno ab urbe condita MMDCCLXXVIII

"""
Basic logging setup for the qGNN based on loguru, slightly better than the default logging scheiße
"""

from loguru import logger
import sys
import pathlib,os 
from typing import Optional


def setup_logging(
    experiment_name: str = "jet_gnn", 
    log_dir: Optional[str] = '/work/abal/qGNN/logs/tmp/',
    log_level: str = "INFO",
) -> None:
    """
    Configure loguru logging for the entire project.
    
    Args:
        experiment_name: Name for the log file (e.g., 'baseline_gnn', 'attention_model')
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_dir: Directory for log files (default: './logs')
    
    Usage:
        # In main.py (before importing other modules):
        from logging_setup import setup_logging
        setup_logging("my_experiment", "INFO")
        
        # In any other script:
        from loguru import logger
        logger.info("Your message here")
    """
    
    # Remove default loguru handler to avoid duplicates
    logger.remove()
    
    # Console logging with colors and clean format
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[script]}</cyan> | {message}",
        colorize=True
    )
    
    # File logging with detailed format
    if log_dir is None:
        log_dir = "/tmp/abal/logs"
    # dont use paths, use os to join paths, and then use pathlib to make directories/files
    pathlib.Path(log_dir).mkdir(exist_ok=True, parents=True)

    logger.add(
        os.path.join(log_dir, f"{experiment_name}.log"),
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",      # Rotate when file reaches 10MB
        retention="1 week",    # Keep logs for 1 week
        compression="zip"      # Compress rotated logs
    )
    
    # Add script name context for better tracking
    logger.configure(extra={"script": "setup"})
    
    logger.info(f"Logging initialized for experiment: {experiment_name}")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {os.path.join(log_dir, f'{experiment_name}.log')}")


def get_script_logger(script_name: str):
    """
    Get a logger with script name context.
    
    Args:
        script_name: Name of the script (e.g., 'dataloader', 'training', 'inference')
    
    Returns:
        Configured logger with script context
        
    Usage:
        # In dataloader.py:
        from logging_setup import get_script_logger
        logger = get_script_logger("dataloader")
        logger.info("Loading data...")
    """
    return logger.bind(script=script_name)


# Example usage and testing
if __name__ == "__main__":
    # Test the logging setup
    setup_logging("test_experiment", "DEBUG")
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test script-specific logger
    test_logger = get_script_logger("test_script")
    test_logger.info("This message shows which script it came from")
    
    print("\n✅ Logging setup complete! Check the 'logs/' directory for log files.")