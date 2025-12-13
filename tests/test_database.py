"""Unit tests for PostgreSQL database functions."""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from unittest.mock import patch, MagicMock
import pytest
from app.database_factory import database
from app.config import settings


def get_test_db_connection():
    """Get a connection to the PostgreSQL test database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except psycopg2.Error as e:
        print(f"Failed to connect to PostgreSQL database: {e}")
        raise


def setup_test_database():
    """Initialize the database and clean it before tests."""
    # Initialize the database
    database.init_db()

    # Clear any existing data from the documents table
    conn = get_test_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents;")
    conn.commit()
    conn.close()


def test_init_db():
    """Test database initialization."""
    # Initialize the database
    database.init_db()

    # Establish a connection to verify the database and table exist
    conn = get_test_db_connection()
    cursor = conn.cursor()

    # Check if the documents table exists
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name = 'documents'
    """)
    tables = cursor.fetchall()

    assert len(tables) == 1
    assert tables[0][0] == 'documents'

    # Check if the required columns exist
    cursor.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'documents'
    """)
    columns = [row[0] for row in cursor.fetchall()]

    expected_columns = ['id', 'doc_id', 'filename', 'status', 'created_at', 'updated_at', 'summary', 'entities']
    for col in expected_columns:
        assert col in columns

    conn.close()


def test_create_document_record():
    """Test creating a document record."""
    setup_test_database()

    # Test creating a document record
    doc_id = "test-doc-123"
    filename = "test_document.pdf"

    success = database.create_document_record(doc_id, filename)
    assert success is True

    # Verify the record exists in the database
    conn = get_test_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT doc_id, filename, status FROM documents WHERE doc_id = %s", (doc_id,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row['doc_id'] == doc_id
    assert row['filename'] == filename
    assert row['status'] == 'uploaded'  # Default status


def test_update_document_status():
    """Test updating document status."""
    setup_test_database()

    # Create a document record first
    doc_id = "test-doc-456"
    filename = "test_document2.pdf"
    database.create_document_record(doc_id, filename)

    # Test updating the document status
    new_status = "processing"
    success = database.update_document_status(doc_id, new_status)
    assert success is True

    # Verify the status was updated in the database
    conn = get_test_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT status FROM documents WHERE doc_id = %s", (doc_id,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row['status'] == new_status


def test_get_document_status():
    """Test retrieving document status."""
    setup_test_database()

    # Create a document record first
    doc_id = "test-doc-789"
    filename = "test_document3.pdf"
    database.create_document_record(doc_id, filename)
    database.update_document_status(doc_id, "processed")

    # Test getting the document status
    status_info = database.get_document_status(doc_id)
    assert status_info is not None
    assert status_info['doc_id'] == doc_id
    assert status_info['filename'] == filename
    assert status_info['status'] == "processed"


def test_get_document_status_not_found():
    """Test retrieving status for a non-existent document."""
    # Test with a document ID that doesn't exist
    status_info = get_document_status("non-existent-id")
    assert status_info is None


def test_get_all_documents():
    """Test retrieving all documents."""
    setup_test_database()

    # Create multiple document records
    docs_data = [
        ("doc-1", "file1.pdf"),
        ("doc-2", "file2.pdf"),
        ("doc-3", "file3.pdf")
    ]

    for doc_id, filename in docs_data:
        database.create_document_record(doc_id, filename)
        database.update_document_status(doc_id, "processed")

    # Test getting all documents
    all_docs = database.get_all_documents()
    assert len(all_docs) == 3

    # Verify the documents are in the returned list
    doc_ids = [doc['doc_id'] for doc in all_docs]
    for doc_id, _ in docs_data:
        assert doc_id in doc_ids


def test_update_document_summary():
    """Test updating document summary."""
    setup_test_database()

    # Create a document record first
    doc_id = "test-doc-summary"
    filename = "summary_test.pdf"
    create_document_record(doc_id, filename)

    # Test updating the document summary
    summary_data = {
        "summary": "This is a test summary",
        "key_points": ["Point 1", "Point 2"]
    }

    success = database.update_document_summary(doc_id, summary_data)
    assert success is True

    # Verify the summary was updated in the database
    conn = get_test_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT summary FROM documents WHERE doc_id = %s", (doc_id,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row['summary'] is not None
    import json
    stored_summary = json.loads(row['summary'])
    assert stored_summary == summary_data


def test_get_document_summary():
    """Test retrieving document summary."""
    setup_test_database()

    # Create a document record and update its summary
    doc_id = "test-doc-get-summary"
    filename = "get_summary_test.pdf"
    create_document_record(doc_id, filename)

    summary_data = {
        "summary": "This is a test summary for retrieval",
        "key_points": ["Retrieved Point 1", "Retrieved Point 2"]
    }
    database.update_document_summary(doc_id, summary_data)

    # Test retrieving the document summary
    retrieved_summary = database.get_document_summary(doc_id)
    assert retrieved_summary == summary_data


def test_update_document_entities():
    """Test updating document entities."""
    setup_test_database()

    # Create a document record first
    doc_id = "test-doc-entities"
    filename = "entities_test.pdf"
    database.create_document_record(doc_id, filename)

    # Test updating the document entities
    entities_data = {
        "invoice_number": "INV-123",
        "total_amount": 100.50,
        "customer_name": "John Doe"
    }

    success = database.update_document_entities(doc_id, entities_data)
    assert success is True

    # Verify the entities were updated in the database
    conn = get_test_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT entities FROM documents WHERE doc_id = %s", (doc_id,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row['entities'] is not None
    import json
    stored_entities = json.loads(row['entities'])
    assert stored_entities == entities_data


def test_get_document_entities():
    """Test retrieving document entities."""
    setup_test_database()

    # Create a document record and update its entities
    doc_id = "test-doc-get-entities"
    filename = "get_entities_test.pdf"
    create_document_record(doc_id, filename)

    entities_data = {
        "invoice_number": "INV-456",
        "total_amount": 200.75,
        "customer_name": "Jane Smith"
    }
    database.update_document_entities(doc_id, entities_data)

    # Test retrieving the document entities
    retrieved_entities = database.get_document_entities(doc_id)
    assert retrieved_entities == entities_data


def test_database_error_handling():
    """Test error handling in database operations."""
    # Mock psycopg2.connect to simulate connection errors
    with patch('psycopg2.connect', side_effect=psycopg2.Error("Connection failed")):
        # Test that functions return appropriate values when database errors occur
        result = database.create_document_record("doc-id", "filename.pdf")
        assert result is False

        result = database.update_document_status("doc-id", "status")
        assert result is False

        result = database.get_document_status("doc-id")
        assert result is None

        result = database.get_all_documents()
        assert result == []

        result = database.update_document_summary("doc-id", {"summary": "test"})
        assert result is False

        result = database.get_document_summary("doc-id")
        assert result is None

        result = database.update_document_entities("doc-id", {"entity": "test"})
        assert result is False

        result = database.get_document_entities("doc-id")
        assert result is None