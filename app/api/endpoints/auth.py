"""
Authentication API Endpoints

This module defines the API routes for user authentication, including user registration
and login. These endpoints are crucial for the security of the FinDocAI application,
enabling users to create accounts and obtain JWT tokens for accessing protected routes.

- **/auth/register**: Allows new users to create an account by providing a username,
  email, and password. It ensures that usernames and emails are unique.
- **/auth/login**: Authenticates existing users based on their username and password.
  Upon successful authentication, it returns a JWT access token that the client can
  use for subsequent authenticated requests.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.auth import (
    create_access_token,
    get_password_hash,
    verify_password,
    get_user,
)
from app.api.schemas.request_models import UserCreate

router = APIRouter()

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user_create: UserCreate, db: Session = Depends(get_db)):
    """
    Handles new user registration.
    Hashes the password and creates a new user in the database.
    Prevents duplicate usernames or emails.
    """
    # Check if user already exists
    if get_user(db, username=user_create.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
    if db.query(User).filter(User.email == user_create.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create new user
    hashed_password = get_password_hash(user_create.password)
    new_user = User(
        username=user_create.username,
        email=user_create.email,
        hashed_password=hashed_password,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {"message": "User created successfully", "username": new_user.username}

@router.post("/login")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Authenticates a user and returns a JWT access token.
    It verifies the username and password against the database.
    """
    user = get_user(db, username=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}
