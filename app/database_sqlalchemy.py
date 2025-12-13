"""
SQLAlchemy-based PostgreSQL Database Module for FinDocAI

This module provides functions for interacting with a PostgreSQL database
using SQLAlchemy ORM for storing document metadata, status, and processed results.
"""

import os
import json
from typing import Optional, Dict, Any, List
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

# Import centralized settings
from app.config import settings

# Import structured logging
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# SQLAlchemy setup
DATABASE_URL = f"postgresql://{settings.db_user}:{settings.db_password.get_secret_value()}@{settings.db_host}:{settings.db_port}/{settings.db_name}"

engine = create_engine(DATABASE_URL, echo=False)  # Set echo=True for SQL debugging
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String(255), unique=True, index=True, nullable=False)
    filename = Column(String(500), nullable=False)
    status = Column(String(100), nullable=False, default='uploaded')
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    summary = Column(Text)
    entities = Column(Text)


def get_db_session() -> Session:
    """
    Get a database session from the sessionmaker.

    Yields:
        SQLAlchemy session object
    """
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        logger.error("Failed to create database session", error=str(e))
        raise


def init_db() -> None:
    """
    Initialize the database and create the documents table if it doesn't exist.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized", db_host=settings.db_host, db_name=settings.db_name)
    except SQLAlchemyError as e:
        logger.error("Failed to initialize database", error=str(e))
        raise


def create_document_record(doc_id: str, filename: str) -> bool:
    """
    Create a new document record in the database with 'uploaded' status.

    Args:
        doc_id: Unique document identifier
        filename: Name of the document file

    Returns:
        True if record was created successfully, False otherwise
    """
    db_logger = logger.bind(doc_id=doc_id, filename=filename)
    db_logger.info("Creating document record")

    db = None
    try:
        db = get_db_session()
        document = Document(
            doc_id=doc_id,
            filename=filename,
            status='uploaded'
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        db_logger.info("Document record created successfully")
        return True
    except SQLAlchemyError as e:
        if db:
            db.rollback()
        db_logger.error("Failed to create document record", error=str(e))
        return False
    finally:
        if db:
            db.close()


def update_document_status(doc_id: str, status: str) -> bool:
    """
    Update the status of a document in the database.

    Args:
        doc_id: Unique document identifier
        status: New status value

    Returns:
        True if status was updated successfully, False otherwise
    """
    db_logger = logger.bind(doc_id=doc_id, status=status)
    db_logger.info("Updating document status")

    db = None
    try:
        db = get_db_session()
        result = db.query(Document).filter(Document.doc_id == doc_id).update({
            Document.status: status,
            Document.updated_at: func.current_timestamp()
        })
        db.commit()
        success = result > 0
        db_logger.info("Document status updated", rows_affected=result)
        return success
    except SQLAlchemyError as e:
        if db:
            db.rollback()
        db_logger.error("Failed to update document status", error=str(e))
        return False
    finally:
        if db:
            db.close()


def get_document_status(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the status of a document from the database.

    Args:
        doc_id: Unique document identifier

    Returns:
        Dictionary with document status info, or None if not found
    """
    db_logger = logger.bind(doc_id=doc_id)
    db_logger.info("Retrieving document status")

    db = None
    try:
        db = get_db_session()
        document = db.query(Document).filter(Document.doc_id == doc_id).first()
        
        if document:
            result = {
                'doc_id': document.doc_id,
                'filename': document.filename,
                'status': document.status,
                'created_at': document.created_at,
                'updated_at': document.updated_at
            }
            db_logger.info("Document status retrieved", status=result['status'])
            return result
        db_logger.info("Document not found")
        return None
    except SQLAlchemyError as e:
        db_logger.error("Failed to retrieve document status", error=str(e))
        return None
    finally:
        if db:
            db.close()


def get_all_documents() -> List[Dict[str, Any]]:
    """
    Get all documents from the database.

    Returns:
        List of dictionaries with document status info
    """
    logger.info("Retrieving all documents")
    db = None
    try:
        db = get_db_session()
        documents = db.query(Document).all()
        
        result = []
        for doc in documents:
            result.append({
                'doc_id': doc.doc_id,
                'filename': doc.filename,
                'status': doc.status,
                'created_at': doc.created_at,
                'updated_at': doc.updated_at
            })
        
        logger.info("Retrieved all documents", count=len(result))
        return result
    except SQLAlchemyError as e:
        logger.error("Failed to retrieve all documents", error=str(e))
        return []
    finally:
        if db:
            db.close()


def update_document_summary(doc_id: str, summary_data: Dict[str, Any]) -> bool:
    """
    Update the summary information for a document.

    Args:
        doc_id: Unique document identifier
        summary_data: Dictionary with summary information

    Returns:
        True if summary was updated successfully, False otherwise
    """
    db_logger = logger.bind(doc_id=doc_id)
    db_logger.info("Updating document summary")

    db = None
    try:
        db = get_db_session()
        # Convert summary_data to JSON string for storage
        summary_json = json.dumps(summary_data)
        
        result = db.query(Document).filter(Document.doc_id == doc_id).update({
            Document.summary: summary_json,
            Document.updated_at: func.current_timestamp()
        })
        db.commit()
        success = result > 0
        db_logger.info("Document summary updated", rows_affected=result)
        return success
    except (SQLAlchemyError, json.JSONDecodeError) as e:
        if db:
            db.rollback()
        db_logger.error("Failed to update document summary", error=str(e))
        return False
    finally:
        if db:
            db.close()


def get_document_summary(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the summary of a document from the database.

    Args:
        doc_id: Unique document identifier

    Returns:
        Dictionary with document summary, or None if not found
    """
    db_logger = logger.bind(doc_id=doc_id)
    db_logger.info("Retrieving document summary")

    db = None
    try:
        db = get_db_session()
        document = db.query(Document).filter(Document.doc_id == doc_id).first()
        
        if document and document.summary:
            summary = json.loads(document.summary)
            db_logger.info("Document summary retrieved")
            return summary
        db_logger.info("Document summary not found")
        return None
    except (SQLAlchemyError, json.JSONDecodeError) as e:
        db_logger.error("Failed to retrieve document summary", error=str(e))
        return None
    finally:
        if db:
            db.close()


def update_document_entities(doc_id: str, entities_data: Dict[str, Any]) -> bool:
    """
    Update the entities information for a document.

    Args:
        doc_id: Unique document identifier
        entities_data: Dictionary with entity information

    Returns:
        True if entities were updated successfully, False otherwise
    """
    db_logger = logger.bind(doc_id=doc_id)
    db_logger.info("Updating document entities")

    db = None
    try:
        db = get_db_session()
        # Convert entities_data to JSON string for storage
        entities_json = json.dumps(entities_data)
        
        result = db.query(Document).filter(Document.doc_id == doc_id).update({
            Document.entities: entities_json,
            Document.updated_at: func.current_timestamp()
        })
        db.commit()
        success = result > 0
        db_logger.info("Document entities updated", rows_affected=result)
        return success
    except (SQLAlchemyError, json.JSONDecodeError) as e:
        if db:
            db.rollback()
        db_logger.error("Failed to update document entities", error=str(e))
        return False
    finally:
        if db:
            db.close()


def get_document_entities(doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the entities of a document from the database.

    Args:
        doc_id: Unique document identifier

    Returns:
        Dictionary with document entities, or None if not found
    """
    db_logger = logger.bind(doc_id=doc_id)
    db_logger.info("Retrieving document entities")

    db = None
    try:
        db = get_db_session()
        document = db.query(Document).filter(Document.doc_id == doc_id).first()
        
        if document and document.entities:
            entities = json.loads(document.entities)
            db_logger.info("Document entities retrieved")
            return entities
        db_logger.info("Document entities not found")
        return None
    except (SQLAlchemyError, json.JSONDecodeError) as e:
        db_logger.error("Failed to retrieve document entities", error=str(e))
        return None
    finally:
        if db:
            db.close()


def delete_document_record(doc_id: str) -> bool:
    """
    Delete a document record from the database.

    Args:
        doc_id: Unique document identifier

    Returns:
        True if record was deleted successfully, False otherwise
    """
    db_logger = logger.bind(doc_id=doc_id)
    db_logger.info("Deleting document record")

    db = None
    try:
        db = get_db_session()
        result = db.query(Document).filter(Document.doc_id == doc_id).delete()
        db.commit()
        success = result > 0
        db_logger.info("Document record deletion completed", rows_affected=result)
        return success
    except SQLAlchemyError as e:
        if db:
            db.rollback()
        db_logger.error("Failed to delete document record", error=str(e))
        return False
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    # Initialize the database when run directly
    init_db()