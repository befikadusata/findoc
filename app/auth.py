"""
FastAPI Authentication Module

This module provides a robust authentication and authorization system for the FinDocAI API.
It replaces the original, insecure API key-based approach with a standard OAuth2-compliant
mechanism using JWT (JSON Web Tokens) for authenticating users.

Key Features:
- **Password Hashing**: Uses `passlib` with the bcrypt algorithm to securely hash and verify user passwords.
- **JWT Token Generation**: Creates short-lived access tokens upon successful login, which are used to authenticate subsequent API requests.
- **Token Verification**: Validates JWT tokens to ensure they are correctly signed, not expired, and belong to a valid user.
- **Dependency Injection**: Provides a `get_current_user` dependency that can be injected into API endpoints to protect them and retrieve the authenticated user's details.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import User
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- Configuration ---
SECRET_KEY = settings.jwt_secret_key.get_secret_value()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# --- Utility Functions ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against a hashed one."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a plain-text password."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Generates a new JWT access token.
    The token contains the provided data and has a defined expiration time.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.info("Access token created for user: %s", data.get("sub"))
    return encoded_jwt

# --- Database Interaction ---

def get_user(db: Session, username: str) -> Optional[User]:
    """
    Retrieves a user from the database by their username.
    Returns the User object or None if not found.
    """
    logger.debug("Attempting to retrieve user: %s", username)
    return db.query(User).filter(User.username == username).first()

# --- Authentication Dependency ---

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """
    Dependency to get the current authenticated user from a JWT token.
    This function is injected into protected API endpoints. It decodes the token,
    validates its signature and expiration, and retrieves the corresponding user
    from the database.

    Raises:
        HTTPException(401): If the token is invalid, expired, or the user is not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logger.warning("Token decoding failed: username (sub) is missing.")
            raise credentials_exception
    except JWTError as e:
        logger.error("JWT decoding error: %s", e)
        raise credentials_exception

    user = get_user(db, username=username)
    if user is None:
        logger.warning("User '%s' from token not found in the database.", username)
        raise credentials_exception
    
    logger.info("Authenticated user: %s", user.username)
    return user
