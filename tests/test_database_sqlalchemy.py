import pytest
from app import database as db_ops
from app.models import Document, User

def test_create_document_record(db_session):
    """Test creating a document record."""
    user = db_session.query(User).filter_by(username="testuser").one()
    doc = db_ops.create_document_record(db_session, "test-doc-1", "test.pdf", user.id)
    
    assert doc is not None
    assert doc.doc_id == "test-doc-1"
    assert doc.filename == "test.pdf"
    assert doc.user_id == user.id
    assert doc.status == "queued"

def test_update_document_status(db_session):
    """Test updating a document's status."""
    user = db_session.query(User).filter_by(username="testuser").one()
    doc = db_ops.create_document_record(db_session, "test-doc-2", "test2.pdf", user.id)
    
    updated = db_ops.update_document_status(db_session, "test-doc-2", "processing")
    assert updated is True
    
    retrieved_doc = db_ops.get_document_by_id(db_session, "test-doc-2")
    assert retrieved_doc.status == "processing"

def test_get_document_by_id(db_session):
    """Test retrieving a document by its ID."""
    user = db_session.query(User).filter_by(username="testuser").one()
    db_ops.create_document_record(db_session, "test-doc-3", "test3.pdf", user.id)
    
    retrieved_doc = db_ops.get_document_by_id(db_session, "test-doc-3")
    assert retrieved_doc is not None
    assert retrieved_doc.doc_id == "test-doc-3"

def test_update_document_summary(db_session):
    """Test updating a document's summary."""
    user = db_session.query(User).filter_by(username="testuser").one()
    doc = db_ops.create_document_record(db_session, "test-doc-4", "test4.pdf", user.id)
    
    summary_data = {"summary": "This is a test summary."}
    updated = db_ops.update_document_summary(db_session, doc.doc_id, summary_data)
    assert updated is True
    
    retrieved_doc = db_ops.get_document_by_id(db_session, doc.doc_id)
    import json
    assert json.loads(retrieved_doc.summary) == summary_data

def test_update_document_entities(db_session):
    """Test updating a document's entities."""
    user = db_session.query(User).filter_by(username="testuser").one()
    doc = db_ops.create_document_record(db_session, "test-doc-5", "test5.pdf", user.id)
    
    entities_data = {"entities": {"name": "test"}}
    updated = db_ops.update_document_entities(db_session, doc.doc_id, entities_data)
    assert updated is True
    
    retrieved_doc = db_ops.get_document_by_id(db_session, doc.doc_id)
    import json
    assert json.loads(retrieved_doc.entities) == entities_data

def test_delete_document_record(db_session):
    """Test deleting a document record."""
    user = db_session.query(User).filter_by(username="testuser").one()
    doc = db_ops.create_document_record(db_session, "test-doc-6", "test6.pdf", user.id)
    
    deleted = db_ops.delete_document_record(db_session, doc.doc_id)
    assert deleted is True
    
    retrieved_doc = db_ops.get_document_by_id(db_session, doc.doc_id)
    assert retrieved_doc is None
