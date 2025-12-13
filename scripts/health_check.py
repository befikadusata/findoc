#!/usr/bin/env python3
"""
FinDocAI Health Check Script
This script performs a basic health check of the FinDocAI system by:
1. Checking if the API server is running
2. Testing basic functionality (including authenticated endpoints)
"""

import requests
import sys
import time
import os
import uuid

from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# --- Authentication for test requests ---
TEST_USERNAME = "healthcheck_user"
TEST_PASSWORD = "healthcheck_password"
AUTH_TOKEN = None
AUTH_HEADERS = {}

def get_auth_token():
    """Registers a test user and logs in to obtain an authentication token."""
    global AUTH_TOKEN, AUTH_HEADERS
    
    # Try to register
    register_payload = {
        "username": TEST_USERNAME,
        "email": f"{TEST_USERNAME}@example.com",
        "password": TEST_PASSWORD
    }
    try:
        register_response = requests.post("http://localhost:8000/auth/register", json=register_payload, timeout=10)
        if register_response.status_code == 201:
            logger.info("Test user registered successfully.")
        elif register_response.status_code == 400 and "already registered" in register_response.text:
            logger.info("Test user already registered, proceeding to login.")
        else:
            logger.error(f"Failed to register test user: {register_response.status_code} - {register_response.text}")
            return False
            
        # Login
        login_data = {
            "username": TEST_USERNAME,
            "password": TEST_PASSWORD
        }
        login_response = requests.post("http://localhost:8000/auth/login", data=login_data, timeout=10)
        if login_response.status_code == 200:
            AUTH_TOKEN = login_response.json()["access_token"]
            AUTH_HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}"}
            logger.info("Successfully obtained auth token for health check user.")
            return True
        else:
            logger.error(f"Failed to login test user: {login_response.status_code} - {login_response.text}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("Auth server not reachable for login.")
        return False
    except Exception as e:
        logger.error(f"Error during authentication setup: {e}")
        return False

def check_api_health():
    """Check if the API server is responding."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"API Health Check: {health_data}")
            return True
        else:
            logger.error(f"API Health Check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("API Health Check failed: Could not connect to server")
        return False
    except Exception as e:
        logger.error(f"API Health Check failed with error: {e}")
        return False

def check_metrics_endpoint():
    """Check if the metrics endpoint is available."""
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=10)
        if response.status_code == 200:
            logger.info("Metrics endpoint is accessible")
            return True
        else:
            logger.error(f"Metrics endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Metrics endpoint check failed: {e}")
        return False

def test_document_upload():
    """Test document upload functionality with a simple text file."""
    if not AUTH_HEADERS:
        logger.warning("Skipping document upload test: Authentication failed.")
        return False

    temp_file_name = f"test_upload_{uuid.uuid4().hex}.txt"
    try:
        # Create a temporary test file
        with open(temp_file_name, "w") as f:
            f.write("This is a test document for FinDocAI health check.\n" * 10)
        
        # Upload the test file
        with open(temp_file_name, "rb") as f:
            files = {"file": (temp_file_name, f, "text/plain")}
            response = requests.post("http://localhost:8000/upload", files=files, headers=AUTH_HEADERS, timeout=30)
        
        if response.status_code == 201: # Expect 201 Created
            result = response.json()
            logger.info(f"Document upload test: {result}")
            
            # Try to get the status of the uploaded document
            if "doc_id" in result:
                doc_id = result["doc_id"]
                status_response = requests.get(f"http://localhost:8000/status/{doc_id}", headers=AUTH_HEADERS, timeout=10)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    logger.info(f"Document status check: {status_data['status']}")
                else:
                    logger.warning(f"Document status check failed with status {status_response.status_code} - {status_response.text}")
            
            return True
        else:
            logger.error(f"Document upload test failed with status {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Document upload test failed: {e}")
        return False
    finally:
        # Clean up the temporary test file
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)

def main():
    """Main function to run health checks."""
    logger.info("FinDocAI Health Check")
    logger.info("="*50)
    
    all_checks_passed = True
    
    # First, try to get an auth token
    logger.info("0. Setting up authentication...")
    if not get_auth_token():
        all_checks_passed = False
        logger.error("Authentication setup failed. Skipping tests for protected endpoints.")
    
    logger.info("1. Checking API health...")
    if not check_api_health():
        all_checks_passed = False
    
    logger.info("\n2. Checking metrics endpoint...")
    if not check_metrics_endpoint():
        all_checks_passed = False
    
    logger.info("\n3. Testing document upload...")
    if not test_document_upload():
        all_checks_passed = False
    
    logger.info("\n" + "="*50)
    if all_checks_passed:
        logger.info("All health checks passed!")
        sys.exit(0)
    else:
        logger.error("Some health checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()