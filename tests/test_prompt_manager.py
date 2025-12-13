import pytest
from prompts.prompt_manager import PromptManager, render_rag_query_prompt, render_entity_extraction_prompt, render_summarization_prompt
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def test_prompt_manager_initialization_and_template_listing():
    """Test the basic functionality of the PromptManager initialization and template listing."""
    logger.info("Testing Prompt Manager initialization and template listing...")
    
    pm = PromptManager()
    assert pm is not None, "PromptManager instance should be created successfully"
    logger.info("PromptManager instance created successfully")
    
    templates = pm.list_available_templates()
    assert len(templates) > 0, "Should find available templates"
    logger.info(f"Found {len(templates)} available templates")
    for name, info in templates.items():
        logger.info(f"  - {name}: v{info['version']}")
    
    # Test version retrieval
    rag_version = pm.get_template_version("rag/rag_query.j2")
    extraction_version = pm.get_template_version("extraction/entity_extraction.j2")
    summary_version = pm.get_template_version("summarization/document_summarization.j2")
    
    assert rag_version is not None
    assert extraction_version is not None
    assert summary_version is not None
    logger.info(f"Template versions - RAG: v{rag_version}, Extraction: v{extraction_version}, Summary: v{summary_version}")


def test_template_rendering():
    """Test template rendering with various parameters."""
    logger.info("Testing template rendering with various parameters...")
    
    # Test RAG prompt with realistic values
    context = "The contract was signed on 2023-01-15. The total amount is $50,000. Payment is due within 30 days."
    query = "When is the payment due?"
    rag_result = render_rag_query_prompt(context=context, query=query)
    
    assert rag_result is not None
    assert context in rag_result
    assert query in rag_result
    logger.info("RAG prompt correctly includes context and query")
    
    # Test entity extraction prompt
    doc_type = "invoice"
    text = "Sample invoice text with invoice number INV-001 and amount $1000"
    extraction_result = render_entity_extraction_prompt(doc_type=doc_type, text=text)
    
    assert extraction_result is not None
    assert doc_type in extraction_result
    assert text in extraction_result
    logger.info("Entity extraction prompt correctly includes doc_type and text")
    
    # Test summarization prompt
    doc_type = "agreement"
    text_truncated = "Sample agreement text for summarization"
    max_length = 500
    summary_result = render_summarization_prompt(doc_type=doc_type, text_truncated=text_truncated, max_length=max_length)
    
    assert summary_result is not None
    assert doc_type in summary_result
    assert str(max_length) in summary_result
    logger.info("Summarization prompt correctly includes parameters")


def test_fallback_mechanism():
    """Test the fallback mechanism when templates don't exist."""
    logger.info("Testing fallback mechanism...")
    
    pm = PromptManager()
    
    # Try to render a non-existent template - should return None
    result = pm.render_template("non/existent/template.j2", some_param="test")
    assert result is None, "Should have returned None for non-existent template"
    logger.info("Correctly returned None for non-existent template")