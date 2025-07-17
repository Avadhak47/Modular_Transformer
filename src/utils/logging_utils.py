"""Logging utilities for mathematical reasoning training."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
):
    """Setup comprehensive logging configuration."""
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)


def log_model_info(model_info: Dict[str, Any], logger: logging.Logger):
    """Log comprehensive model information."""
    logger.info("Model Information:")
    logger.info(f"  Name: {model_info.get('model_name', 'Unknown')}")
    logger.info(f"  PE Type: {model_info.get('pe_type', 'Unknown')}")
    logger.info(f"  Total Parameters: {model_info.get('total_parameters', 0):,}")
    logger.info(f"  Trainable Parameters: {model_info.get('trainable_parameters', 0):,}")
    logger.info(f"  Trainable %: {model_info.get('trainable_percentage', 0):.2f}%")