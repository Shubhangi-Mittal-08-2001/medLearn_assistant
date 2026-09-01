"""Centralized logging configuration for MedLearn Assistant backend."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Module-level logger
_root_logger = None


def configure_logging() -> logging.Logger:
    """
    Configure root logging to write to both console and a rotating file.
    
    Returns:
        logging.Logger: The configured root logger instance
    """
    global _root_logger
    
    backend_dir = Path(__file__).resolve().parents[2]
    log_dir = backend_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "medlearn_assistant.log"

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    _root_logger = logging.getLogger()
    _root_logger.setLevel(level)
    _root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    _root_logger.addHandler(stream_handler)

    # Rotating File Handler (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    _root_logger.addHandler(file_handler)

    return _root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module/name.
    
    Args:
        name: The name of the logger (typically __name__)
        
    Returns:
        logging.Logger: A logger instance for the given name
    """
    return logging.getLogger(name)
