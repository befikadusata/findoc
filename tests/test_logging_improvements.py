import pytest
from unittest.mock import patch, MagicMock
from app.utils.logging_config import get_logger
from app.rag.pipeline import RAGPipeline
from app.nlp.extraction import extract_entities, generate_summary
from app.classification.model import classify_document

logger = get_logger(__name__)

def test_logging_imports():
    """Test that all modules can be imported without errors."""
    logger.info("Testing imports after logging improvements...")
    
    # These imports are now at the top of the file, so this test just asserts their presence
    assert get_logger is not None
    logger.info("Logging config imported successfully")
    
    assert RAGPipeline is not None
    logger.info("RAG pipeline imported successfully")
    
    assert extract_entities is not None and generate_summary is not None
    logger.info("NLP extraction imported successfully")
    
    assert classify_document is not None
    logger.info("Classification model imported successfully")


def test_logger_functionality():
    """Test that logging functionality works properly."""
    logger.info("Testing logger functionality...")
    
    # To properly test, we need to capture log output, which is complex with structlog.
    # For simplicity, we'll just ensure the logger can be instantiated and called without error.
    try:
        test_logger = get_logger("test.module")
        test_logger.info("Test log message from test_logger_functionality", value=123)
        test_logger.warning("Test warning message", extra_data="test_event_context")
        test_logger.error("Test error message", error_code=500)
        logger.info("Logger functionality works correctly (calls made without error).")
    except Exception as e:
        pytest.fail(f"Logger functionality test failed: {e}")