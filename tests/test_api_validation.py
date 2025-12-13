import pytest
from pydantic import ValidationError
from app.api.schemas.request_models import DocumentIdRequest, QueryRequest, DeleteDocumentRequest, SummaryRequest
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def test_validation_models():
    """Test the validation models directly."""
    logger.info("Testing validation models...")
    
    # Test valid document ID
    doc_req = DocumentIdRequest(doc_id="valid-doc-id-123")
    assert doc_req.doc_id == "valid-doc-id-123"
    logger.info(f"Valid document ID accepted: {doc_req.doc_id}")
    
    # Test valid query request
    query_req = QueryRequest(doc_id="valid-doc-456", question="What is this document about?")
    assert query_req.doc_id == "valid-doc-456"
    assert query_req.question == "What is this document about?"
    logger.info(f"Valid query request accepted: doc_id={query_req.doc_id}, question length={len(query_req.question)}")
    
    # Test valid summary request
    summary_req = SummaryRequest(doc_id="valid-summary-id")
    assert summary_req.doc_id == "valid-summary-id"
    logger.info(f"Valid summary request accepted: {summary_req.doc_id}")
    
    # Test valid delete request
    delete_req = DeleteDocumentRequest(doc_id="valid-delete-id")
    assert delete_req.doc_id == "valid-delete-id"
    logger.info(f"Valid delete request accepted: {delete_req.doc_id}")


def test_document_id_validation():
    """Test document ID validation logic."""
    logger.info("Testing document ID validation...")
    
    # Test valid IDs
    valid_ids = [
        "123e4567-e89b-12d3-a456-426614174000",  # UUID format
        "doc_123",  # With underscore
        "doc-123",  # With hyphen
        "doc123",   # Simple alphanumeric
        "a" * 255   # Maximum length
    ]
    
    for doc_id in valid_ids:
        req = DocumentIdRequest(doc_id=doc_id)
        assert req.doc_id == doc_id
        logger.info(f"Valid ID accepted: {doc_id[:20]}{'...' if len(doc_id) > 20 else ''}")
    
    # Test invalid IDs
    invalid_ids = [
        "",  # Empty
        "a" * 256,  # Too long
        "doc with spaces",  # Contains spaces
        "doc/with/slashes",  # Contains slashes
        "doc@invalid",  # Contains @
        None  # None value
    ]
    
    for doc_id in invalid_ids:
        with pytest.raises(ValidationError):
            DocumentIdRequest(doc_id=doc_id)
        logger.info(f"Invalid ID correctly rejected: {str(doc_id)[:20] if doc_id else 'None'}")


def test_query_validation():
    """Test query validation logic."""
    logger.info("Testing query validation...")
    
    # Test valid queries
    valid_queries = [
        ("valid-doc", "Short question?"),
        ("valid-doc", "A" * 2000),  # Maximum length question
    ]
    
    for doc_id, question in valid_queries:
        req = QueryRequest(doc_id=doc_id, question=question)
        assert req.doc_id == doc_id
        assert req.question == question
        logger.info(f"Valid query accepted: doc_id={doc_id}, question length={len(req.question)}")
    
    # Test invalid queries (too long)
    with pytest.raises(ValidationError):
        QueryRequest(doc_id="valid-doc", question="A" * 2001)  # Too long
    logger.info("Too-long question correctly rejected")
    
    # Test empty question
    with pytest.raises(ValidationError):
        QueryRequest(doc_id="valid-doc", question="")
    logger.info("Empty question correctly rejected")


def test_schema_imports():
    """Test that schemas can be imported properly."""
    logger.info("Testing schema imports...")
    
    from app.api.schemas import request_models
    assert request_models is not None
    logger.info("Schemas module imported successfully")
    
    # Check that all expected classes are present
    expected_classes = ['DocumentIdRequest', 'QueryRequest', 'DeleteDocumentRequest', 'SummaryRequest', 'UserCreate']
    for cls_name in expected_classes:
        assert hasattr(request_models, cls_name)
        logger.info(f"{cls_name} is available")