"""
Logging configuration module for FinDocAI.

This module sets up structured logging using structlog for both FastAPI and Celery.
"""

import logging
import structlog
import json
from datetime import datetime


def setup_logging():
    """
    Set up structured logging using structlog.
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(serializer=json.dumps)
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure the root logger to use structlog
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    return structlog.get_logger()


def get_logger(name=None):
    """
    Get a configured logger instance.
    
    Args:
        name (str, optional): Name of the logger. If None, returns the root logger.
        
    Returns:
        structlog.BoundLogger: Configured logger instance
    """
    return structlog.get_logger(name)


# Create a global logger instance
logger = setup_logging()