import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import create_app
from app.database import get_db, delete_document_record
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

def test_delete_document_api_e2e(client, auth_headers, db_session):
    """
    End-to-end test for the document deletion API endpoint.
    """
    # 1. Create a document to be deleted
    user = db_session.query(User).filter_by(username="testuser").one()
    doc = Document(doc_id="delete-me", filename="delete_me.txt", user_id=user.id)
    db_session.add(doc)
    db_session.commit()

    # Create a dummy file that the endpoint expects to delete
    # This path is constructed the same way as in app/main.py
    upload_dir = "./data/uploads"
    file_path = os.path.join(upload_dir, f"{doc.doc_id}_{doc.filename}")
    os.makedirs(upload_dir, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(b"dummy content")
    
    # Mock the RAG pipeline deletion and file deletion
    with patch('app.rag.pipeline.delete_document_from_chromadb', return_value=True) as mock_chroma_delete, \
         patch('os.remove') as mock_os_remove:
        
        # 2. Call the delete endpoint
        response = client.delete(f"/documents/{doc.doc_id}", headers=auth_headers)
        
        # 3. Assert the response
        assert response.status_code == 200
        assert response.json()["message"] == "Document deleted successfully"
        
        # 4. Verify the mocks were called
        mock_chroma_delete.assert_called_once_with(doc.doc_id)
        mock_os_remove.assert_called_once_with(file_path) # Assert with the expected file_path
        
        # 5. Verify the document is gone from the database
        deleted_doc = db_session.query(Document).filter_by(doc_id=doc.doc_id).one_or_none()
        assert deleted_doc is None

    # Clean up the dummy file if os.remove was mocked and didn't actually remove it
    if os.path.exists(file_path):
        os.remove(file_path)

def test_delete_unauthorized(client, db_session, monkeypatch):
    """
    Test that a user cannot delete another user's document.
    """
    # Mock verify_password to avoid bcrypt issues when creating token for user1
    def mock_verify_password(plain_password, hashed_password):
        return plain_password == "testpassword"  # Simple check for tests

    import app.auth
    monkeypatch.setattr(app.auth, "verify_password", mock_verify_password)
    from app.auth import create_access_token

    # 1. Create a document owned by another user
    user1 = User(username="user1", email="user1@example.com", hashed_password="password")
    user2 = User(username="user2", email="user2@example.com", hashed_password="password")
    db_session.add_all([user1, user2])
    db_session.commit()
    
    doc = Document(doc_id="user2-doc", filename="user2.txt", user_id=user2.id)
    db_session.add(doc)
    db_session.commit()
    
    # 2. Create token for user1 manually
    token = create_access_token(data={"sub": user1.username})
    headers = {"Authorization": f"Bearer {token}"}
    
    # 3. Attempt to delete user2's document
    response = client.delete(f"/documents/{doc.doc_id}", headers=headers)
    
    # 4. Assert that it's forbidden
    assert response.status_code == 403

def test_delete_nonexistent_document(client, auth_headers):
    """
    Test deleting a document that does not exist.
    """
    response = client.delete("/documents/nonexistent-doc", headers=auth_headers)
    assert response.status_code == 404
