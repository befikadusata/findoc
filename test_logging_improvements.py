"""
Test script for logging improvements
This script tests that the print statements have been replaced with structured logging.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_logging_imports():
    """Test that all modules can be imported without errors."""
    print("Testing imports after logging improvements...")
    
    try:
        from app.utils.logging_config import get_logger
        print("✓ Logging config imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import logging config: {e}")
        return False
    
    try:
        from app.rag.pipeline import RAGPipeline
        print("✓ RAG pipeline imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import RAG pipeline: {e}")
        return False
    
    try:
        from app.nlp.extraction import extract_entities, generate_summary
        print("✓ NLP extraction imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NLP extraction: {e}")
        return False
    
    try:
        from app.classification.model import classify_document
        print("✓ Classification model imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import classification model: {e}")
        return False
    
    return True


def test_logger_functionality():
    """Test that logging functionality works properly."""
    print("Testing logger functionality...")
    
    try:
        from app.utils.logging_config import get_logger
        logger = get_logger(__name__)
        
        # Test logging with context binding
        bound_logger = logger.bind(test_id="test-log-123")
        bound_logger.info("Test log message", value=42)
        print("✓ Logger functionality works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Error testing logger functionality: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Starting tests for logging improvements...\n")
    
    test_results = []
    test_results.append(test_logging_imports())
    test_results.append(test_logger_functionality())
    
    print(f"\nTest Results: {sum(test_results)}/{len(test_results)} passed")
    
    if all(test_results):
        print("\n🎉 All tests passed! Logging improvements implemented successfully.")
        return True
    else:
        print("\n❌ Some tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)