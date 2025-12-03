"""
PostgreSQL Database Module for FinDocAI

This module provides functions for interacting with a PostgreSQL database
to store document metadata, status, and processed results.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime
from typing import Optional
import json

# Import structured logging
from app.utils.logging_config import get_logger

# Database connection parameters
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5434')  # Changed to 5434 to match docker-compose
DB_NAME = os.getenv('DB_NAME', 'findocai')
DB_USER = os.getenv('DB_USER', 'findocai_user')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'findocai_password')

logger = get_logger(__name__)


def get_db_connection():
    """Get a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except psycopg2.Error as e:
        logger.error("Failed to connect to PostgreSQL database", error=str(e))
        raise


def init_db():
    """Initialize the database and create the documents table if it doesn't exist."""
    try:
        # Connect to PostgreSQL with autocommit to create database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database='postgres',  # Connect to default postgres db to create new db
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_NAME}'")
        exists = cursor.fetchone()
        if not exists:
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            logger.info(f"Created database {DB_NAME}")
        
        cursor.close()
        conn.close()
        
        # Now connect to the specific database to create tables
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create the documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                doc_id VARCHAR(255) UNIQUE NOT NULL,
                filename VARCHAR(500) NOT NULL,
                status VARCHAR(100) NOT NULL DEFAULT 'uploaded',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary TEXT,
                entities TEXT
            )
        ''')

        # Create an index on doc_id for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_id ON documents (doc_id)')

        conn.commit()
        conn.close()
        logger.info("Database initialized", db_host=DB_HOST, db_name=DB_NAME)
    except psycopg2.Error as e:
        logger.error("Failed to initialize database", error=str(e))
        raise


def create_document_record(doc_id: str, filename: str) -> bool:
    """Create a new document record in the database with 'uploaded' status."""
    db_logger = logger.bind(doc_id=doc_id, filename=filename)
    db_logger.info("Creating document record")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO documents (doc_id, filename, status)
            VALUES (%s, %s, %s)
        ''', (doc_id, filename, 'uploaded'))

        conn.commit()
        conn.close()
        db_logger.info("Document record created successfully")
        return True
    except psycopg2.Error as e:
        db_logger.error("Failed to create document record", error=str(e))
        return False


def update_document_status(doc_id: str, status: str) -> bool:
    """Update the status of a document in the database."""
    db_logger = logger.bind(doc_id=doc_id, status=status)
    db_logger.info("Updating document status")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE documents
            SET status = %s, updated_at = CURRENT_TIMESTAMP
            WHERE doc_id = %s
        ''', (status, doc_id))

        conn.commit()
        conn.close()
        rows_affected = cursor.rowcount
        db_logger.info("Document status updated", rows_affected=rows_affected)
        return rows_affected > 0
    except psycopg2.Error as e:
        db_logger.error("Failed to update document status", error=str(e))
        return False


def get_document_status(doc_id: str) -> Optional[dict]:
    """Get the status of a document from the database."""
    db_logger = logger.bind(doc_id=doc_id)
    db_logger.info("Retrieving document status")

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('''
            SELECT doc_id, filename, status, created_at, updated_at
            FROM documents
            WHERE doc_id = %s
        ''', (doc_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            # Convert RealDictRow to regular dict
            result = dict(row)
            db_logger.info("Document status retrieved", status=result['status'])
            return result
        db_logger.info("Document not found")
        return None
    except psycopg2.Error as e:
        db_logger.error("Failed to retrieve document status", error=str(e))
        return None


def get_all_documents() -> list:
    """Get all documents from the database."""
    logger.info("Retrieving all documents")
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('SELECT doc_id, filename, status, created_at, updated_at FROM documents')
        rows = cursor.fetchall()
        conn.close()

        documents = [dict(row) for row in rows]
        logger.info("Retrieved all documents", count=len(documents))
        return documents
    except psycopg2.Error as e:
        logger.error("Failed to retrieve all documents", error=str(e))
        return []


def update_document_summary(doc_id: str, summary_data: dict) -> bool:
    """Update the summary information for a document."""
    db_logger = logger.bind(doc_id=doc_id)
    db_logger.info("Updating document summary")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Convert summary_data to JSON string for storage
        summary_json = json.dumps(summary_data)

        cursor.execute('''
            UPDATE documents
            SET summary = %s, updated_at = CURRENT_TIMESTAMP
            WHERE doc_id = %s
        ''', (summary_json, doc_id))

        conn.commit()
        conn.close()
        rows_affected = cursor.rowcount
        db_logger.info("Document summary updated", rows_affected=rows_affected)
        return rows_affected > 0
    except psycopg2.Error as e:
        db_logger.error("Failed to update document summary", error=str(e))
        return False


def get_document_summary(doc_id: str) -> Optional[dict]:
    """Get the summary of a document from the database."""
    db_logger = logger.bind(doc_id=doc_id)
    db_logger.info("Retrieving document summary")

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('SELECT summary FROM documents WHERE doc_id = %s', (doc_id,))
        row = cursor.fetchone()
        conn.close()

        if row and row['summary']:
            summary = json.loads(row['summary'])
            db_logger.info("Document summary retrieved")
            return summary
        db_logger.info("Document summary not found")
        return None
    except (psycopg2.Error, json.JSONDecodeError) as e:
        db_logger.error("Failed to retrieve document summary", error=str(e))
        return None


def update_document_entities(doc_id: str, entities_data: dict) -> bool:
    """Update the entities information for a document."""
    db_logger = logger.bind(doc_id=doc_id)
    db_logger.info("Updating document entities")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Convert entities_data to JSON string for storage
        entities_json = json.dumps(entities_data)

        cursor.execute('''
            UPDATE documents
            SET entities = %s, updated_at = CURRENT_TIMESTAMP
            WHERE doc_id = %s
        ''', (entities_json, doc_id))

        conn.commit()
        conn.close()
        rows_affected = cursor.rowcount
        db_logger.info("Document entities updated", rows_affected=rows_affected)
        return rows_affected > 0
    except psycopg2.Error as e:
        db_logger.error("Failed to update document entities", error=str(e))
        return False


def get_document_entities(doc_id: str) -> Optional[dict]:
    """Get the entities of a document from the database."""
    db_logger = logger.bind(doc_id=doc_id)
    db_logger.info("Retrieving document entities")

    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute('SELECT entities FROM documents WHERE doc_id = %s', (doc_id,))
        row = cursor.fetchone()
        conn.close()

        if row and row['entities']:
            entities = json.loads(row['entities'])
            db_logger.info("Document entities retrieved")
            return entities
        db_logger.info("Document entities not found")
        return None
    except (psycopg2.Error, json.JSONDecodeError) as e:
        db_logger.error("Failed to retrieve document entities", error=str(e))
        return None