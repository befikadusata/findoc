"""
SQLAlchemy Models for FinDocAI

This module defines SQLAlchemy ORM models for the document management system.
These models will be used with Alembic for database migrations.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


class Document(Base):
    """
    Document model representing a document in the system.
    """
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(255), unique=True, nullable=False)  # Unique document identifier
    filename = Column(String(500), nullable=False)  # Name of the document file
    status = Column(String(100), nullable=False, default='uploaded')  # Current processing status
    created_at = Column(DateTime, default=func.current_timestamp())  # Timestamp when record was created
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())  # Timestamp when record was last updated
    summary = Column(Text, nullable=True)  # JSON string containing document summary
    entities = Column(Text, nullable=True)  # JSON string containing extracted entities