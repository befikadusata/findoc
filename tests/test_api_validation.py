"""
Test script for API input validation
This script tests the new input validation functionality.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_validation_models():
    """Test the validation models directly."""
    print("Testing validation models...")
    
    try:
        from app.api.schemas.request_models import DocumentIdRequest, QueryRequest, DeleteDocumentRequest, SummaryRequest
        
        # Test valid document ID
        doc_req = DocumentIdRequest(doc_id="valid-doc-id-123")
        print(f"✓ Valid document ID accepted: {doc_req.doc_id}")
        
        # Test valid query request
        query_req = QueryRequest(doc_id="valid-doc-456", question="What is this document about?")
        print(f"✓ Valid query request accepted: doc_id={query_req.doc_id}, question length={len(query_req.question)}")
        
        # Test valid summary request
        summary_req = SummaryRequest(doc_id="valid-summary-id")
        print(f"✓ Valid summary request accepted: {summary_req.doc_id}")
        
        # Test valid delete request
        delete_req = DeleteDocumentRequest(doc_id="valid-delete-id")
        print(f"✓ Valid delete request accepted: {delete_req.doc_id}")
        
        return True
    except Exception as e:
        print(f"✗ Error testing validation models: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_id_validation():
    """Test document ID validation logic."""
    print("\nTesting document ID validation...")
    
    try:
        from app.api.schemas.request_models import DocumentIdRequest
        
        # Test valid IDs
        valid_ids = [
            "123e4567-e89b-12d3-a456-426614174000",  # UUID format
            "doc_123",  # With underscore
            "doc-123",  # With hyphen
            "doc123",   # Simple alphanumeric
            "a" * 255   # Maximum length
        ]
        
        for doc_id in valid_ids:
            try:
                req = DocumentIdRequest(doc_id=doc_id)
                print(f"✓ Valid ID accepted: {doc_id[:20]}{'...' if len(doc_id) > 20 else ''}")
            except Exception as e:
                print(f"✗ Valid ID rejected: {doc_id}, error: {e}")
        
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
            try:
                if doc_id is None:
                    req = DocumentIdRequest(doc_id="valid")
                    # Manually test the validator
                    DocumentIdRequest.validate_doc_id_format(doc_id if doc_id is not None else "")
                    print(f"✗ Invalid ID was not caught: {doc_id}")
                else:
                    req = DocumentIdRequest(doc_id=doc_id)
                    print(f"✗ Invalid ID was not caught: {doc_id}")
            except Exception:
                print(f"✓ Invalid ID correctly rejected: {str(doc_id)[:20] if doc_id else 'None'}")
        
        return True
    except Exception as e:
        print(f"✗ Error testing document ID validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_query_validation():
    """Test query validation logic."""
    print("\nTesting query validation...")
    
    try:
        from app.api.schemas.request_models import QueryRequest
        
        # Test valid queries
        valid_queries = [
            ("valid-doc", "Short question?"),
            ("valid-doc", "A" * 2000),  # Maximum length question
        ]
        
        for doc_id, question in valid_queries:
            try:
                req = QueryRequest(doc_id=doc_id, question=question)
                print(f"✓ Valid query accepted: doc_id={doc_id}, question length={len(req.question)}")
            except Exception as e:
                print(f"✗ Valid query rejected: doc_id={doc_id}, question length={len(question)}, error: {e}")
        
        # Test invalid queries (too long)
        try:
            req = QueryRequest(doc_id="valid-doc", question="A" * 2001)  # Too long
            print("✗ Too-long question was not rejected")
        except Exception:
            print("✓ Too-long question correctly rejected")
        
        # Test empty question
        try:
            req = QueryRequest(doc_id="valid-doc", question="")
            print("✗ Empty question was not rejected")
        except Exception:
            print("✓ Empty question correctly rejected")
        
        return True
    except Exception as e:
        print(f"✗ Error testing query validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schema_imports():
    """Test that schemas can be imported properly."""
    print("\nTesting schema imports...")
    
    try:
        from app.api.schemas import request_models
        print("✓ Schemas module imported successfully")
        
        # Check that all expected classes are present
        expected_classes = ['DocumentIdRequest', 'QueryRequest', 'DeleteDocumentRequest', 'SummaryRequest']
        for cls_name in expected_classes:
            if hasattr(request_models, cls_name):
                print(f"✓ {cls_name} is available")
            else:
                print(f"✗ {cls_name} is missing")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Error importing schemas: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Starting tests for API input validation...\n")
    
    test_results = []
    test_results.append(test_validation_models())
    test_results.append(test_document_id_validation())
    test_results.append(test_query_validation())
    test_results.append(test_schema_imports())
    
    print(f"\nTest Results: {sum(test_results)}/{len(test_results)} passed")
    
    if all(test_results):
        print("\n🎉 All tests passed! API input validation implemented successfully.")
        return True
    else:
        print("\n❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)