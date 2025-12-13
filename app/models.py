"""
SQLAlchemy Models for FinDocAI

This module defines SQLAlchemy ORM models for the document management system.
These models will be used with Alembic for database migrations.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, func, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class User(Base):
    """
    User model representing a user in the system.
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())

    documents = relationship("Document", back_populates="owner")


class Document(Base):
    """
    Document model representing a document in the system.
    """
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(255), unique=True, nullable=False)  # Unique document identifier
    filename = Column(String(500), nullable=False)  # Name of the document file
    status = Column(String(100), nullable=False, default='uploaded')  # Current processing status
    doc_type = Column(String(100), nullable=True)  # Document type (e.g., '10-K', 'Invoice')
    created_at = Column(DateTime, default=func.current_timestamp())  # Timestamp when record was created
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())  # Timestamp when record was last updated
    summary = Column(Text, nullable=True)  # JSON string containing document summary
    entities = Column(Text, nullable=True)  # JSON string containing extracted entities

    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)  # Foreign key to users table
    owner = relationship("User", back_populates="documents")