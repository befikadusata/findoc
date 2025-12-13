"""
SQLAlchemy-based PostgreSQL Database Module for FinDocAI

This module provides the core database setup and session management for the application.
It follows the FastAPI recommendation of a dependency-injected, session-per-request pattern.
"""

import json
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.models import Base, Document
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- Database Engine and Session Configuration ---
DATABASE_URL = f"postgresql://{settings.db_user}:{settings.db_password.get_secret_value()}@{settings.db_host}:{settings.db_port}/{settings.db_name}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Dependency for FastAPI ---

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency to get a managed database session per request.
    Ensures the session is always closed after the request is finished.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Context Manager for Non-API Scopes (e.g., Celery Workers) ---

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager to provide a database session for background tasks or scripts.
    Ensures the session is properly handled and closed.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        logger.error("Database error occurred in session, rolling back.", error=str(e))
        db.rollback()
        raise
    finally:
        db.close()

# --- Database Initialization ---

def init_db() -> None:
    """
    Initializes the database by creating all tables defined in the models.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise

# --- Database Interaction Functions ---

def create_document_record(db: Session, doc_id: str, filename: str, user_id: int) -> Optional[Document]:
    """Creates a new document record in the database."""
    try:
        new_doc = Document(doc_id=doc_id, filename=filename, user_id=user_id, status='queued')
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        logger.info("Created document record", doc_id=doc_id, user_id=user_id)
        return new_doc
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Failed to create document record", doc_id=doc_id, error=str(e))
        return None

def update_document_status(db: Session, doc_id: str, status: str) -> bool:
    """Updates the status of a specific document."""
    try:
        result = db.query(Document).filter(Document.doc_id == doc_id).update({"status": status})
        if result > 0:
            logger.info("Updated document status", doc_id=doc_id, status=status)
            return True
        logger.warning("Document not found for status update", doc_id=doc_id)
        return False
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Failed to update document status", doc_id=doc_id, error=str(e))
        return False

def get_document_by_id(db: Session, doc_id: str) -> Optional[Document]:
    """Retrieves a document by its unique doc_id."""
    try:
        return db.query(Document).filter(Document.doc_id == doc_id).first()
    except SQLAlchemyError as e:
        logger.error("Failed to retrieve document", doc_id=doc_id, error=str(e))
        return None

def update_document_summary(db: Session, doc_id: str, summary_data: Dict[str, Any]) -> bool:
    """Updates the summary of a document."""
    try:
        summary_json = json.dumps(summary_data)
        result = db.query(Document).filter(Document.doc_id == doc_id).update({"summary": summary_json})
        return result > 0
    except (SQLAlchemyError, TypeError) as e:
        db.rollback()
        logger.error("Failed to update document summary", doc_id=doc_id, error=str(e))
        return False

def update_document_entities(db: Session, doc_id: str, entities_data: Dict[str, Any]) -> bool:
    """Updates the extracted entities of a document."""
    try:
        entities_json = json.dumps(entities_data)
        result = db.query(Document).filter(Document.doc_id == doc_id).update({"entities": entities_json})
        return result > 0
    except (SQLAlchemyError, TypeError) as e:
        db.rollback()
        logger.error("Failed to update document entities", doc_id=doc_id, error=str(e))
        return False

def delete_document_record(db: Session, doc_id: str) -> bool:
    """Deletes a document record from the database."""
    try:
        doc = db.query(Document).filter(Document.doc_id == doc_id).first()
        if doc:
            db.delete(doc)
            db.commit()
            logger.info("Deleted document record", doc_id=doc_id)
            return True
        return False
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Failed to delete document record", doc_id=doc_id, error=str(e))
        return False