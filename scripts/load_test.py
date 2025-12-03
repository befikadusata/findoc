"""
Load testing script for FinDocAI using Locust.

This script simulates multiple users uploading documents and querying them.
"""

import os
import uuid
from locust import HttpUser, TaskSet, task, between
from app.database import DB_PATH
import tempfile


class DocumentProcessingTasks(TaskSet):
    """Task set for document processing load testing."""
    
    def on_start(self):
        """Initialize the user session."""
        self.doc_id = None
        self.uploaded_file = None
    
    @task(3)  # This task will be executed 3 times more often than others
    def health_check(self):
        """Check the health endpoint."""
        self.client.get("/health")
    
    @task(2)
    def upload_document(self):
        """Upload a document."""
        # Create a temporary text file with some content
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write(f"Test document content for load testing - {uuid.uuid4()}")
        temp_file.flush()
        
        try:
            with open(temp_file.name, 'rb') as f:
                response = self.client.post(
                    "/upload",
                    files={"file": ("test_doc.txt", f, "text/plain")}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.doc_id = data.get("doc_id")
                    self.uploaded_file = temp_file.name
        except Exception as e:
            print(f"Error uploading document: {e}")
    
    @task(1)
    def check_status(self):
        """Check the status of an uploaded document."""
        if self.doc_id:
            response = self.client.get(f"/status/{self.doc_id}")
            if response.status_code != 200:
                # If the document doesn't exist, clear the ID
                if response.json().get("error") == "Document not found":
                    self.doc_id = None
    
    @task(1)
    def query_document(self):
        """Query a document if we have a doc_id."""
        if self.doc_id:
            params = {
                "doc_id": self.doc_id,
                "question": "What is this document about?"
            }
            response = self.client.get("/query", params=params)
            # Don't worry about the response since the doc might not be processed yet


class DocumentProcessingUser(HttpUser):
    """User class for simulating document processing load."""
    
    tasks = [DocumentProcessingTasks]
    wait_time = between(1, 3)  # Wait between 1 and 3 seconds between tasks
    
    def on_start(self):
        """Actions to perform when the user starts."""
        print(f"Starting load test user: {self.environment.runner.user_count}")
    
    def on_stop(self):
        """Actions to perform when the user stops."""
        print("Stopping load test user")


# Additional task set for API endpoint load testing
class APITaskSet(TaskSet):
    """Task set for load testing API endpoints."""
    
    @task(5)
    def get_root(self):
        """Access the root endpoint."""
        self.client.get("/")
    
    @task(3)
    def get_health(self):
        """Check health endpoint."""
        self.client.get("/health")
    
    @task(1)
    def get_metrics(self):
        """Access the metrics endpoint."""
        self.client.get("/metrics")


class APIUser(HttpUser):
    """User class for API load testing."""
    
    tasks = [APITaskSet]
    wait_time = between(0.5, 2)
    
    # Set a different host for this user type if needed
    # host = "http://localhost:8000"