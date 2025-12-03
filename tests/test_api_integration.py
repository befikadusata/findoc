"""Integration test for the document upload endpoint."""

import os
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest
import uuid
from app.main import app
from app.database import init_db, get_document_status


def test_upload_endpoint_integration():
    """Integration test for the document upload endpoint."""
    # Create a test client for the FastAPI app
    client = TestClient(app)

    # Create a temporary file to simulate document upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write("This is a test document for upload testing.")
        tmp_file_path = tmp_file.name

    try:
        # Mock the Celery task sending to avoid actual processing
        with patch('app.main.celery_app') as mock_celery:
            mock_task = MagicMock()
            mock_task.id = 'test-task-id'
            mock_celery.send_task.return_value = mock_task

            # Make the upload request
            with open(tmp_file_path, 'rb') as f:
                response = client.post(
                    "/upload",
                    files={"file": ("test_document.txt", f, "text/plain")}
                )

            # Verify the response
            assert response.status_code == 200
            response_data = response.json()
            assert "doc_id" in response_data
            assert response_data["filename"] == "test_document.txt"
            assert response_data["task_id"] == "test-task-id"
            assert "Document uploaded successfully and processing started" in response_data["message"]

            # Verify the document ID was generated as a valid UUID
            doc_id = response_data["doc_id"]
            try:
                uuid.UUID(doc_id)  # This will raise ValueError if invalid
                uuid_valid = True
            except ValueError:
                uuid_valid = False
            assert uuid_valid, "doc_id should be a valid UUID"

            # Verify that the file was saved to the expected location
            expected_file_path = f"./data/uploads/{doc_id}_test_document.txt"
            assert os.path.exists(expected_file_path)

            # Clean up the saved file
            if os.path.exists(expected_file_path):
                os.remove(expected_file_path)

            # Verify the Celery task was called with the correct arguments
            mock_celery.send_task.assert_called_once()
            args, kwargs = mock_celery.send_task.call_args
            assert args[0] == 'process_document'
            # The 'args' parameter is passed as keyword argument to send_task
            task_args = kwargs.get('args', [])
            assert len(task_args) == 2
            assert task_args[0] == doc_id  # First arg should be doc_id
            assert task_args[1] == expected_file_path  # Second arg should be file_path

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def test_upload_endpoint_database_integration():
    """Test that document upload correctly updates the database."""
    # Create a test client
    client = TestClient(app)

    # Create a temporary file to simulate document upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write("Test content for database integration.")
        tmp_file_path = tmp_file.name

    try:
        # Mock the Celery task to prevent actual processing
        with patch('app.main.celery_app') as mock_celery:
            mock_task = MagicMock()
            mock_task.id = 'test-task-id'
            mock_celery.send_task.return_value = mock_task

            # Upload the document
            with open(tmp_file_path, 'rb') as f:
                response = client.post(
                    "/upload",
                    files={"file": ("db_test_document.txt", f, "text/plain")}
                )

            assert response.status_code == 200
            response_data = response.json()
            doc_id = response_data["doc_id"]

            # Check the database directly to verify the record was created with 'queued' status
            # Using the database function instead of direct connection
            document_info = get_document_status(doc_id)

            assert document_info is not None, "Document record should exist in the database"
            assert document_info["status"] == "queued", "Document status should be 'queued' after upload"

            # Now test the status endpoint
            status_response = client.get(f"/status/{doc_id}")
            assert status_response.status_code == 200

            status_data = status_response.json()
            assert status_data["doc_id"] == doc_id
            assert status_data["filename"] == "db_test_document.txt"
            assert status_data["status"] == "queued"

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

        # Clean up the uploaded file if it was created
        if 'doc_id' in locals():
            expected_file_path = f"./data/uploads/{doc_id}_db_test_document.txt"
            if os.path.exists(expected_file_path):
                os.remove(expected_file_path)


def test_status_endpoint_for_nonexistent_document():
    """Test the status endpoint with a non-existent document ID."""
    client = TestClient(app)

    # Request status for a document that doesn't exist
    nonexistent_doc_id = "nonexistent-document-id"
    response = client.get(f"/status/{nonexistent_doc_id}")

    assert response.status_code == 200  # Endpoint returns 200 but with error in body
    response_data = response.json()
    assert "error" in response_data
    assert response_data["doc_id"] == nonexistent_doc_id
    assert "Document not found" in response_data["error"]