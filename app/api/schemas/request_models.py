"""
API Schemas for Input Validation

This module contains Pydantic models for validating API inputs.
"""

from pydantic import BaseModel, Field, field_validator, EmailStr
from typing import Optional
import re


class UserCreate(BaseModel):
    """
    Validation schema for user creation.
    """
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)


class DocumentIdRequest(BaseModel):
    """
    Validation schema for document ID parameters.
    """
    doc_id: str = Field(
        ..., 
        min_length=1,
        max_length=255,
        description="Unique document identifier"
    )
    
    @field_validator('doc_id')
    @classmethod
    def validate_doc_id_format(cls, v):
        """
        Validate document ID format.
        Accepts UUID format or other alphanumeric identifiers with hyphens/underscores.
        """
        if not v:
            raise ValueError('Document ID cannot be empty')
        
        # Basic validation for allowed characters (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Document ID can only contain alphanumeric characters, hyphens, and underscores')
        
        return v


class QueryRequest(BaseModel):
    """
    Validation schema for document query parameters.
    """
    doc_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique document identifier"
    )
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,  # Reasonable limit for questions
        description="Question to ask about the document"
    )
    
    @field_validator('doc_id')
    @classmethod
    def validate_doc_id_format(cls, v):
        """
        Validate document ID format.
        """
        if not v:
            raise ValueError('Document ID cannot be empty')
        
        # Basic validation for allowed characters (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Document ID can only contain alphanumeric characters, hyphens, and underscores')
        
        return v
    
    @field_validator('question')
    @classmethod
    def validate_question_length(cls, v):
        """
        Validate question length and content.
        """
        if not v or len(v.strip()) == 0:
            raise ValueError('Question cannot be empty')
        
        if len(v) > 2000:
            raise ValueError('Question is too long (maximum 2000 characters)')
        
        return v.strip()


class DeleteDocumentRequest(BaseModel):
    """
    Validation schema for document deletion parameters.
    """
    doc_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique document identifier to delete"
    )
    
    @field_validator('doc_id')
    @classmethod
    def validate_doc_id_format(cls, v):
        """
        Validate document ID format.
        """
        if not v:
            raise ValueError('Document ID cannot be empty')
        
        # Basic validation for allowed characters (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Document ID can only contain alphanumeric characters, hyphens, and underscores')
        
        return v


class SummaryRequest(BaseModel):
    """
    Validation schema for document summary parameters.
    """
    doc_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique document identifier"
    )
    
    @field_validator('doc_id')
    @classmethod
    def validate_doc_id_format(cls, v):
        """
        Validate document ID format.
        """
        if not v:
            raise ValueError('Document ID cannot be empty')
        
        # Basic validation for allowed characters (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Document ID can only contain alphanumeric characters, hyphens, and underscores')
        
        return v