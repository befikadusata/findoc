#!/usr/bin/env python3
"""
Script to generate bcrypt hash for test password.
"""

from passlib.context import CryptContext

# Create the same context as in the auth module
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Generate the hash for 'testpassword'
hashed = pwd_context.hash("testpassword")
print(f"Hash for 'testpassword': {hashed}")