#!/bin/bash

# FinDocAI Setup Script
# This script sets up the development environment for FinDocAI

set -e  # Exit immediately if a command exits with a non-zero status

echo "==========================================="
echo "FinDocAI Development Environment Setup"
echo "==========================================="

# Check if running on Ubuntu/Debian
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ $NAME != *"Ubuntu"* ]] && [[ $NAME != *"Debian"* ]]; then
        echo "Warning: This script is designed for Ubuntu/Debian systems."
        echo "Proceeding anyway, but some commands may need adjustment."
    fi
else
    echo "Warning: Could not determine OS. Proceeding anyway."
fi

# Check for required tools
echo "Checking for required tools..."

command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is required but not installed. Aborting."; exit 1; }
command -v pip >/dev/null 2>&1 || { echo "Error: pip is required but not installed. Aborting."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Error: docker is required but not installed. Aborting."; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "Error: docker-compose is required but not installed. Aborting."; exit 1; }

echo "✓ python3: $(python3 --version)"
echo "✓ pip: $(pip --version)"
echo "✓ docker: $(docker --version)"
echo "✓ docker-compose: $(docker-compose --version)"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not detected. Creating and activating one..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "✓ Created virtual environment in ./venv"
    else
        echo "✓ Virtual environment already exists"
    fi
    
    # Activate the virtual environment
    source venv/bin/activate
    echo "✓ Activated virtual environment"
else
    echo "✓ Using existing virtual environment: $VIRTUAL_ENV"
fi

# Install system dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y tesseract-ocr libtesseract-dev pkg-config

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Set up PostgreSQL database
echo "Setting up PostgreSQL database..."
if command -v sudo >/dev/null 2>&1; then
    # Check if postgres user exists
    if id "postgres" &>/dev/null; then
        # Try to create the user and database
        sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='findocai_user'" | grep -q 1 || \
        sudo -u postgres createuser --pwprompt findocai_user
        sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='findocai'" | grep -q 1 || \
        sudo -u postgres createdb findocai -O findocai_user
    else
        echo "PostgreSQL not found or postgres user doesn't exist. Make sure PostgreSQL is installed and running."
        echo "You can install it with: sudo apt install postgresql postgresql-contrib"
    fi
else
    echo "Could not run sudo commands. Please ensure PostgreSQL is installed and set up manually."
fi

# Initialize the application database
echo "Initializing application database..."
python scripts/init_db.py

echo ""
echo "==========================================="
echo "Setup completed successfully!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Set your GEMINI_API_KEY environment variable:"
echo "   export GEMINI_API_KEY='your_google_ai_api_key'"
echo ""
echo "2. To start all services, run:"
echo "   ./scripts/run.sh"
echo ""
echo "3. To stop all services, run:"
echo "   ./scripts/stop.sh"
echo ""
echo "The application will be available at http://localhost:8000"
echo "==========================================="