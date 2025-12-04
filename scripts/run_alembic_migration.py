#!/usr/bin/env python3
"""
Script to load environment variables from .env file and run Alembic commands
"""

import os
import sys
import subprocess

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def load_env_file(filepath):
    """Load environment variables from a .env file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

if __name__ == "__main__":
    # Load environment variables
    load_env_file('.env')

    # Now run the alembic command
    result = subprocess.run([
        'alembic', 'revision', '--autogenerate', '-m', 'Initial database schema'
    ], env=os.environ, cwd='.')

    exit(result.returncode)