"""
Logging Configuration Module

This module provides centralized logging configuration for the Customer Churn MLOps Pipeline.
It implements rotating file handlers, console handlers, and structured logging formats.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logging(
    component_name: str,
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration for a component with both console and file handlers.
    
    Args:
        component_name: Name of the component (used for logger name and log file)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(component_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Console handler - INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation - DEBUG level
    log_file = os.path.join(log_dir, f'{component_name}.log')
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(component_name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger for a component.
    
    Args:
        component_name: Name of the component
        log_level: Optional log level override
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(component_name)
    
    # If logger not configured yet, set it up
    if not logger.handlers:
        level = log_level or os.getenv('LOG_LEVEL', 'INFO')
        log_dir = os.getenv('LOG_PATH', 'logs')
        setup_logging(component_name, level, log_dir)
    
    return logger
