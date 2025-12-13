"""
Authentication and Authorization Module for FinDocAI

This module provides API key-based authentication and basic authorization functionality.
"""

import secrets
from typing import Optional
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security

from app.utils.logging_config import get_logger
from app.config import settings

logger = get_logger(__name__)

# For demo purposes, we'll use a simple API key approach
# In production, you'd want to store these in a secure database
API_KEYS = set()

# Add the main API key to the set if it's configured
if settings.gemini_api_key:
    # Generate a simple API key from the Gemini API key for demonstration
    # In a real system, you'd have separate API keys for your service
    main_key = secrets.token_urlsafe(32)  # Generate a random key
    API_KEYS.add(main_key)
    
    # Log the API key identifier (not the actual key) for debugging
    logger.info("API key generated", key_hash=hash(main_key))

# For demo purposes, we can also add some hardcoded keys
# In production, these would come from a database
demo_key = "demo-key-12345"
API_KEYS.add(demo_key)


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())):
    """
    Verify the API key provided in the Authorization header.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        True if the API key is valid
        
    Raises:
        HTTPException: If the API key is invalid
    """
    if credentials.credentials in API_KEYS:
        return True
    else:
        logger.warning("Invalid API key provided", key_hash=hash(credentials.credentials))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


def get_current_user(request: Request):
    """
    Get the current authenticated user based on the API key.
    In this implementation, we just verify the API key.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
        )
    
    api_key = auth_header[7:]  # Remove "Bearer " prefix
    if api_key in API_KEYS:
        logger.info("API key validated", endpoint=request.url.path)
        return {"api_key_valid": True}
    else:
        logger.warning("Invalid API key", endpoint=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )