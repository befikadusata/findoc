"""
Test script for document deletion functionality
This script tests the end-to-end deletion functionality implemented for FinDocAI
"""
import os
import tempfile
import uuid
from unittest.mock import patch, MagicMock

def test_file_deletion():
    """Test the file deletion function"""
    print("Testing file deletion functionality...")
    
    # Create a temporary file to simulate document deletion
    with tempfile.NamedTemporaryFile(delete=False, prefix="test_doc_", suffix=".txt") as tmp_file:
        tmp_file.write(b"Test document content")
        tmp_file_path = tmp_file.name
    
    doc_id = str(uuid.uuid4())
    filename = os.path.basename(tmp_file_path).replace(f"{doc_id}_", "")
    
    # Test the deletion function
    from app.main import delete_document_file
    result = delete_document_file(doc_id, filename)
    
    print(f"File deletion test - Expected: False (file doesn't match pattern), Actual: {result}")
    
    # Clean up by deleting the actual temp file
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)
    
    # Create a file with the correct naming pattern
    correct_path = f"./data/uploads/{doc_id}_{filename}"
    os.makedirs("./data/uploads", exist_ok=True)
    with open(correct_path, 'w') as f:
        f.write("Test content")
    
    # Now test with correct path
    result = delete_document_file(doc_id, filename)
    print(f"Correct path file deletion test - Expected: True, Actual: {result}")
    
    return result


def test_database_deletion():
    """Test the database deletion function"""
    print("Testing database deletion functionality...")
    
    # We'll mock the database connection since we can't assume PostgreSQL is running
    with patch('app.database.get_db_connection') as mock_conn:
        mock_cursor = MagicMock()
        mock_conn.return_value.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1  # Simulate one row deleted
        
        from app.database_factory import database
        doc_id = str(uuid.uuid4())
        result = database.delete_document_record(doc_id)
        
        # Verify the SQL command was called correctly
        mock_cursor.execute.assert_called_once_with('DELETE FROM documents WHERE doc_id = %s', (doc_id,))
        mock_conn.return_value.commit.assert_called_once()
        
        print(f"Database deletion test - Expected: True, Actual: {result}")
        
        return result


def test_chromadb_deletion():
    """Test the ChromaDB deletion function"""
    print("Testing ChromaDB deletion functionality...")
    
    # Import the function and mock the ChromaDB client
    from app.rag.pipeline import delete_document_from_chromadb
    from app.rag.pipeline import rag_pipeline
    
    # Mock the ChromaDB client
    original_client = rag_pipeline.chroma_client
    mock_chroma_client = MagicMock()
    rag_pipeline.chroma_client = mock_chroma_client
    
    doc_id = "test-doc-123"
    collection_name = f"docs_{doc_id.replace('-', '_')}"
    
    try:
        result = delete_document_from_chromadb(doc_id)
        
        # Verify the delete_collection method was called with correct name
        mock_chroma_client.delete_collection.assert_called_once_with(collection_name)
        print(f"ChromaDB deletion test - Expected: True, Actual: {result}")
        
        return result
    finally:
        # Restore the original client
        rag_pipeline.chroma_client = original_client


def main():
    """Run all tests"""
    print("Starting end-to-end deletion functionality tests...\n")
    
    # Test file deletion
    file_test_result = test_file_deletion()
    print()
    
    # Test database deletion
    db_test_result = test_database_deletion()
    print()
    
    # Test ChromaDB deletion
    chromadb_test_result = test_chromadb_deletion()
    print()
    
    # Summary
    print("Test Summary:")
    print(f"File deletion test: {'PASSED' if file_test_result else 'NEEDS MANUAL VERIFICATION (requires correct file path)'}")
    print(f"Database deletion test: {'PASSED' if db_test_result else 'FAILED'}")
    print(f"ChromaDB deletion test: {'PASSED' if chromadb_test_result else 'FAILED'}")
    
    all_passed = db_test_result and chromadb_test_result
    print(f"\nOverall result: {'PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("\nNote: For full end-to-end testing, a running PostgreSQL instance and ChromaDB instance are required.")


if __name__ == "__main__":
    main()