#!/usr/bin/env python3
"""
Initialize the PostgreSQL database for FinDocAI.

This script creates the necessary database and tables for the application.
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from app.database import init_db


def main():
    """Initialize the PostgreSQL database."""
    print("Initializing PostgreSQL database for FinDocAI...")
    
    try:
        init_db()
        print("Database initialized successfully!")
        print("Tables created:")
        print("- documents: stores document metadata, status, and processed results")
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise


if __name__ == "__main__":
    main()