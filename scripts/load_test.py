"""
Load testing script for FinDocAI using Locust.

This script simulates multiple users uploading documents and querying them.
"""

import os
import uuid
from locust import HttpUser, TaskSet, task, between
import tempfile


class DocumentProcessingTasks(TaskSet):
    """Task set for document processing load testing."""

    def on_start(self):
        """Initialize the user session and log in."""
        self.doc_id = None
        self.uploaded_file = None

        # Register and login to get an authentication token
        # For a real load test, you might pre-create users or handle this more robustly
        self.username = f"loadtestuser_{uuid.uuid4().hex[:8]}"
        register_response = self.client.post("/auth/register", json={
            "username": self.username,
            "email": f"loadtest_{uuid.uuid4().hex[:8]}@example.com",
            "password": "testpassword"
        })
        if register_response.status_code == 201 or "Username already registered" in register_response.text:
            response = self.client.post("/auth/login", data={
                "username": self.username, # Use the dynamically generated username
                "password": "testpassword"
            })
            if response.status_code == 200:
                self.token = response.json()["access_token"]
                self.headers = {"Authorization": f"Bearer {self.token}"}
            else:
                print(f"Failed to login load test user: {response.status_code} - {response.text}")
                self.interrupt() # Stop this user if login fails
        else:
            print(f"Failed to register load test user: {register_response.status_code} - {register_response.text}")
            self.interrupt()
    
    @task(3)  # This task will be executed 3 times more often than others
    def health_check(self):
        """Check the health endpoint."""
        self.client.get("/health", headers=self.headers)
    
    @task(2)
    def upload_document(self):
        """Upload a document."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write(f"Test document content for load testing - {uuid.uuid4()}".encode('utf-8'))
        temp_file.flush()
        
        try:
            with open(temp_file.name, 'rb') as f:
                response = self.client.post(
                    "/upload",
                    files={"file": ("test_doc.txt", f, "text/plain")},
                    headers=self.headers
                )
                
                if response.status_code == 201: # Changed to 201 Created
                    data = response.json()
                    self.doc_id = data.get("doc_id")
                    self.uploaded_file = temp_file.name
                else:
                    print(f"Upload failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error uploading document: {e}")
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
    
    @task(1)
    def check_status(self):
        """Check the status of an uploaded document."""
        if self.doc_id:
            response = self.client.get(f"/status/{self.doc_id}", headers=self.headers)
            if response.status_code != 200:
                if response.json().get("detail") == "Document not found" or response.json().get("detail") == "Not authorized to access this document":
                    self.doc_id = None
    
    @task(1)
    def query_document(self):
        """Query a document if we have a doc_id."""
        if self.doc_id:
            params = {
                "doc_id": self.doc_id,
                "question": "What is this document about?"
            }
            response = self.client.post("/query", json=params, headers=self.headers) # Changed to POST
            if response.status_code != 200:
                print(f"Query failed: {response.status_code} - {response.text}")

    @task(1)
    def get_summary(self):
        """Get document summary if we have a doc_id."""
        if self.doc_id:
            response = self.client.get(f"/summary/{self.doc_id}", headers=self.headers)
            if response.status_code != 200 and response.status_code != 404:  # 404 is expected if summary not ready
                print(f"Summary request failed: {response.status_code} - {response.text}")

    @task(1)  # Lower frequency as delete is less common - using integer weight
    def delete_document(self):
        """Delete a document if we have a doc_id."""
        if self.doc_id:
            response = self.client.delete(f"/documents/{self.doc_id}", headers=self.headers)
            if response.status_code == 200:
                self.doc_id = None  # Clear doc_id since document is deleted
            elif response.status_code != 404:  # 404 is expected if document already deleted
                print(f"Document deletion failed: {response.status_code} - {response.text}")


class DocumentProcessingUser(HttpUser):
    """User class for simulating document processing load."""

    tasks = [DocumentProcessingTasks]
    wait_time = between(1, 3)  # Wait between 1 and 3 seconds between tasks
    host = "http://localhost:8000"  # Specify the host for the load test

    def on_start(self):
        """Actions to perform when the user starts."""
        print(f"Starting load test user: {self.environment.runner.user_count}")

    def on_stop(self):
        """Actions to perform when the user stops."""
        print("Stopping load test user")


# Additional task set for API endpoint load testing
class APITaskSet(TaskSet):
    """Task set for load testing API endpoints."""
    
    def on_start(self):
        # For API only tests, we might not need a logged in user for public endpoints
        # or we can assume a shared token for simplicity if not testing auth specifically
        self.headers = {} # No auth for public endpoints in this simplified example
        
    @task(5)
    def get_root(self):
        """Access the root endpoint."""
        self.client.get("/", headers=self.headers)

    @task(3)
    def get_health(self):
        """Check health endpoint."""
        self.client.get("/health", headers=self.headers)

    @task(1)
    def get_metrics(self):
        """Access the metrics endpoint."""
        self.client.get("/metrics", headers=self.headers)


class APIUser(HttpUser):
    """User class for API load testing."""

    tasks = [APITaskSet]
    wait_time = between(0.5, 2)
    host = "http://localhost:8000"  # Specify the host for the load test