#!/usr/bin/env python3
"""
Initialize the PostgreSQL database for FinDocAI.

This script creates the necessary database and tables for the application.
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from app.database import init_db
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def main():
    """Initialize the PostgreSQL database."""
    logger.info("Initializing PostgreSQL database for FinDocAI...")
    
    try:
        init_db()
        logger.info("Database initialized successfully!")
        logger.info("Tables created:")
        logger.info("- documents: stores document metadata, status, and processed results")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


if __name__ == "__main__":
    main()