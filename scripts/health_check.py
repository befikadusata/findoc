#!/usr/bin/env python3
"""
FinDocAI Health Check Script
This script performs a basic health check of the FinDocAI system by:
1. Checking if the API server is running
2. Testing basic functionality
"""

import requests
import sys
import time
import os

def check_api_health():
    """Check if the API server is responding."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ API Health Check: {health_data}")
            return True
        else:
            print(f"✗ API Health Check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ API Health Check failed: Could not connect to server")
        return False
    except Exception as e:
        print(f"✗ API Health Check failed with error: {e}")
        return False

def check_metrics_endpoint():
    """Check if the metrics endpoint is available."""
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=10)
        if response.status_code == 200:
            print("✓ Metrics endpoint is accessible")
            return True
        else:
            print(f"✗ Metrics endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Metrics endpoint check failed: {e}")
        return False

def test_document_upload():
    """Test document upload functionality with a simple text file."""
    try:
        # Create a temporary test file
        with open("test_upload.txt", "w") as f:
            f.write("This is a test document for FinDocAI health check.\n" * 10)
        
        # Upload the test file
        with open("test_upload.txt", "rb") as f:
            files = {"file": f}
            response = requests.post("http://localhost:8000/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Document upload test: {result}")
            
            # Try to get the status of the uploaded document
            if "doc_id" in result:
                doc_id = result["doc_id"]
                status_response = requests.get(f"http://localhost:8000/status/{doc_id}", timeout=10)
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"✓ Document status check: {status_data['status']}")
            
            # Clean up the test file
            os.remove("test_upload.txt")
            return True
        else:
            print(f"✗ Document upload test failed with status {response.status_code}")
            print(f"Response: {response.text}")
            # Clean up the test file
            if os.path.exists("test_upload.txt"):
                os.remove("test_upload.txt")
            return False
            
    except Exception as e:
        print(f"✗ Document upload test failed: {e}")
        # Clean up the test file
        if os.path.exists("test_upload.txt"):
            os.remove("test_upload.txt")
        return False

def main():
    """Main function to run health checks."""
    print("FinDocAI Health Check")
    print("="*50)
    
    all_checks_passed = True
    
    print("1. Checking API health...")
    if not check_api_health():
        all_checks_passed = False
    
    print("\n2. Checking metrics endpoint...")
    if not check_metrics_endpoint():
        all_checks_passed = False
    
    print("\n3. Testing document upload...")
    if not test_document_upload():
        all_checks_passed = False
    
    print("\n" + "="*50)
    if all_checks_passed:
        print("✓ All health checks passed!")
        sys.exit(0)
    else:
        print("✗ Some health checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()