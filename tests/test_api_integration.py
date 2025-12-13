import os
import tempfile
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest
from app.main import create_app
from app.database import get_db
from app.models import User, Document

# --- Fixtures ---

@pytest.fixture
def client(db_session):
    """
    Provides a FastAPI TestClient with the database session dependency overridden.
    """
    app = create_app() # Call the factory function to get the app instance
    app.dependency_overrides[get_db] = lambda: db_session
    yield TestClient(app)
    app.dependency_overrides.clear()

@pytest.fixture
def auth_headers(db_session, monkeypatch):
    """
    Creates a test user directly in the database and returns authentication headers
    by manually generating a token. This bypasses the API endpoints for auth.
    """
    # Mock verify_password to avoid bcrypt issues, if needed for manual token validation within auth.py
    def mock_verify_password(plain_password, hashed_password):
        return plain_password == "testpassword"  # Simple check for tests

    import app.auth
    monkeypatch.setattr(app.auth, "verify_password", mock_verify_password)

    # Create user directly in the test database
    test_username = "testuser"
    test_password = "testpassword" # This will be hashed by create_access_token indirectly
    
    # Check if user already exists (from previous test runs in the same session)
    existing_user = db_session.query(User).filter_by(username=test_username).first()
    if not existing_user:
        hashed_password = app.auth.get_password_hash(test_password)
        user = User(username=test_username, email="test@example.com", hashed_password=hashed_password)
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

    # Create token directly using the auth module's function
    token = app.auth.create_access_token(data={"sub": test_username})
    return {"Authorization": f"Bearer {token}"}

# --- Tests ---

def test_upload_and_status_integration(client, auth_headers, db_session):
    """
    Integration test for uploading a document and checking its status.
    """
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
        tmp_file.write(b"test content")
        tmp_file.seek(0)
        
        with patch('app.main.celery_app.send_task') as mock_send_task, \
             patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_param') as mock_log_param, \
             patch('mlflow.log_artifact') as mock_log_artifact:
            
            # Configure mock_start_run to act as a context manager
            mock_start_run.return_value.__enter__.return_value = MagicMock()
            mock_start_run.return_value.__exit__.return_value = False

            # 1. Upload the document
            response = client.post(
                "/upload",
                files={"file": ("test.txt", tmp_file, "text/plain")},
                headers=auth_headers
            )
            assert response.status_code == 201
            data = response.json()
            doc_id = data["doc_id"]
            
            # 2. Verify the document was created in the database
            doc = db_session.query(Document).filter_by(doc_id=doc_id).one_or_none()
            assert doc is not None
            assert doc.filename == "test.txt"
            assert doc.status == "queued"
            
            # 3. Verify the Celery task was called
            mock_send_task.assert_called_once()
            mock_start_run.assert_called_once()
            mock_log_param.assert_called() # Called multiple times
            mock_log_artifact.assert_called_once()
            
            # 4. Check the status endpoint
            response = client.get(f"/status/{doc_id}", headers=auth_headers)
            assert response.status_code == 200
            assert response.json()["status"] == "queued"

def test_upload_path_traversal_prevention(client, auth_headers, db_session):
    """
    Tests that the server sanitizes filenames to prevent path traversal vulnerabilities.
    """
    malicious_filename = "../../../etc/passwd.txt"
    expected_sanitized_filename = "passwd.txt" # Based on werkzeug.utils.secure_filename behavior

    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
        tmp_file.write(b"malicious content")
        tmp_file.seek(0)
        
        with patch('app.main.celery_app.send_task') as mock_send_task, \
             patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_param') as mock_log_param, \
             patch('mlflow.log_artifact') as mock_log_artifact:

            # Configure mock_start_run to act as a context manager
            mock_start_run.return_value.__enter__.return_value = MagicMock()
            mock_start_run.return_value.__exit__.return_value = False

            response = client.post(
                "/upload",
                files={"file": (malicious_filename, tmp_file, "text/plain")},
                headers=auth_headers
            )
            assert response.status_code == 201
            data = response.json()
            doc_id = data["doc_id"]

            # Assert that the returned filename is sanitized
            assert data["filename"] == expected_sanitized_filename

            # Verify the filename stored in the database is sanitized
            doc = db_session.query(Document).filter_by(doc_id=doc_id).one_or_none()
            assert doc is not None
            assert doc.filename == expected_sanitized_filename
            
            # Verify the file on disk has the sanitized name
            uploaded_file_path = os.path.join("./data/uploads", f"{doc_id}_{expected_sanitized_filename}")
            assert os.path.exists(uploaded_file_path)
            
            # Clean up the created file
            os.remove(uploaded_file_path)

def test_upload_celery_not_configured(client, auth_headers, monkeypatch):
    """
    Tests that the /upload endpoint returns a 500 error if Celery is not configured.
    """
    # Temporarily set celery_app to None to simulate it not being configured
    monkeypatch.setattr('app.main.celery_app', None)

    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
        tmp_file.write(b"test content")
        tmp_file.seek(0)
        
        with patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_param') as mock_log_param, \
             patch('mlflow.log_artifact') as mock_log_artifact:

            # Configure mock_start_run to act as a context manager
            mock_start_run.return_value.__enter__.return_value = MagicMock()
            mock_start_run.return_value.__exit__.return_value = False
        
            response = client.post(
                "/upload",
                files={"file": ("test.txt", tmp_file, "text/plain")},
                headers=auth_headers
            )
            assert response.status_code == 500
            assert "Celery not configured" in response.json()["detail"]


def test_unauthorized_access(client):
    """
    Tests that endpoints are protected against unauthorized access.
    """
    response = client.post("/upload", files={"file": ("test.txt", b"content", "text/plain")})
    assert response.status_code == 401 # Unauthorized
    
    response = client.get("/status/some-doc-id")
    assert response.status_code == 401 # Unauthorized

def test_query_and_summary_authorization(client, auth_headers, db_session):
    """
    Tests that users can only query and get summaries for their own documents.
    """
    # 1. Create a document for another user
    other_user = User(username="otheruser", email="other@example.com", hashed_password="password")
    db_session.add(other_user)
    db_session.commit()
    
    other_doc = Document(doc_id="other-doc", filename="other.txt", user_id=other_user.id, summary="{}")
    db_session.add(other_doc)
    db_session.commit()
    
    # 2. Try to access the other user's document
    response = client.post("/query", json={"doc_id": "other-doc", "question": "test"}, headers=auth_headers)
    assert response.status_code == 403 # Forbidden
    
    response = client.get("/summary/other-doc", headers=auth_headers)
    assert response.status_code == 403 # Forbidden