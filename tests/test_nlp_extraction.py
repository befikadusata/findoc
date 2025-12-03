"""Unit tests for NLP extraction functions."""

import os
import json
from unittest.mock import patch, MagicMock
import pytest
from pydantic import ValidationError
from app.nlp.extraction import extract_entities, generate_summary, FinancialEntity, SummaryResult


def test_financial_entity_model():
    """Test the FinancialEntity Pydantic model."""
    # Test with valid data
    entity = FinancialEntity(
        invoice_number="INV-123",
        total_amount=100.50,
        customer_name="John Doe"
    )
    
    assert entity.invoice_number == "INV-123"
    assert entity.total_amount == 100.50
    assert entity.customer_name == "John Doe"
    
    # Test with invalid data type
    with pytest.raises(ValidationError):
        FinancialEntity(total_amount="not_a_number")


def test_summary_result_model():
    """Test the SummaryResult Pydantic model."""
    summary = SummaryResult(
        summary="This is a summary",
        key_points=["Point 1", "Point 2"],
        document_type="invoice"
    )
    
    assert summary.summary == "This is a summary"
    assert len(summary.key_points) == 2
    assert summary.document_type == "invoice"


def test_extract_entities():
    """Test entity extraction (mocked Gemini API)."""
    sample_text = """
    INVOICE
    Invoice Number: INV-2023-001
    Date: 2023-06-15
    Due Date: 2023-07-15
    Customer: John Smith
    Total: $4,860.00
    """
    
    mock_response_text = json.dumps({
        "invoice_number": "INV-2023-001",
        "invoice_date": "2023-06-15",
        "due_date": "2023-07-15",
        "customer_name": "John Smith",
        "total_amount": 4860.0
    })
    
    with patch('os.getenv', return_value='fake-api-key'), \
         patch('google.generativeai.configure'), \
         patch('google.generativeai.GenerativeModel') as mock_model_class:
        
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = mock_response_text
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model_instance
        
        result = extract_entities(sample_text, "invoice")
        
        assert result['invoice_number'] == "INV-2023-001"
        assert result['invoice_date'] == "2023-06-15"
        assert result['customer_name'] == "John Smith"
        assert result['total_amount'] == 4860.0


def test_extract_entities_api_key_missing():
    """Test entity extraction when API key is missing."""
    with patch('os.getenv', return_value=None):  # No API key
        result = extract_entities("sample text", "invoice")
        assert "error" in result
        assert "GEMINI_API_KEY" in result["error"]


def test_extract_entities_api_error():
    """Test entity extraction when API call fails."""
    with patch('os.getenv', return_value='fake-api-key'), \
         patch('google.generativeai.configure'), \
         patch('google.generativeai.GenerativeModel', side_effect=Exception("API Error")):
        
        result = extract_entities("sample text", "invoice")
        # Should return empty entity in case of error
        assert isinstance(result, dict)


def test_extract_entities_json_error():
    """Test entity extraction when response is not valid JSON."""
    with patch('os.getenv', return_value='fake-api-key'), \
         patch('google.generativeai.configure'), \
         patch('google.generativeai.GenerativeModel') as mock_model_class:
        
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Invalid JSON response"
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model_instance
        
        result = extract_entities("sample text", "invoice")
        # Should return empty entity in case of JSON error
        assert isinstance(result, dict)


def test_generate_summary():
    """Test summary generation (mocked Gemini API)."""
    sample_text = """
    INVOICE
    Invoice Number: INV-2023-001
    Date: 2023-06-15
    Due Date: 2023-07-15
    Customer: John Smith
    Total: $4,860.00
    """
    
    mock_response_text = json.dumps({
        "summary": "This is a summary of the invoice",
        "key_points": ["Invoice number INV-2023-001", "Total amount $4,860.00"],
        "document_type": "invoice",
        "document_date": "2023-06-15"
    })
    
    with patch('os.getenv', return_value='fake-api-key'), \
         patch('google.generativeai.configure'), \
         patch('google.generativeai.GenerativeModel') as mock_model_class:
        
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = mock_response_text
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model_instance
        
        result = generate_summary(sample_text, "invoice")
        
        assert result['summary'] == "This is a summary of the invoice"
        assert len(result['key_points']) == 2
        assert result['document_type'] == "invoice"


def test_generate_summary_api_key_missing():
    """Test summary generation when API key is missing."""
    with patch('os.getenv', return_value=None):  # No API key
        result = generate_summary("sample text", "invoice")
        assert "error" in result
        assert "GEMINI_API_KEY" in result["error"]


def test_generate_summary_api_error():
    """Test summary generation when API call fails."""
    with patch('os.getenv', return_value='fake-api-key'), \
         patch('google.generativeai.configure'), \
         patch('google.generativeai.GenerativeModel', side_effect=Exception("API Error")):
        
        result = generate_summary("sample text", "invoice")
        # Should return error dict in case of API error
        assert "error" in result


def test_generate_summary_json_error():
    """Test summary generation when response is not valid JSON."""
    with patch('os.getenv', return_value='fake-api-key'), \
         patch('google.generativeai.configure'), \
         patch('google.generativeai.GenerativeModel') as mock_model_class:
        
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Invalid JSON response"
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model_instance
        
        result = generate_summary("sample text", "invoice")
        # Should return default summary in case of JSON error
        assert isinstance(result, dict)


def test_generate_summary_with_long_text():
    """Test that long text is properly truncated for summary generation."""
    long_text = "This is a sample document. " * 2000  # Very long text
    mock_response_text = json.dumps({
        "summary": "This is a summary",
        "key_points": ["Point 1"],
        "document_type": "invoice"
    })
    
    with patch('os.getenv', return_value='fake-api-key'), \
         patch('google.generativeai.configure'), \
         patch('google.generativeai.GenerativeModel') as mock_model_class:
        
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = mock_response_text
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model_instance
        
        result = generate_summary(long_text, "invoice")
        
        # Verify the function works with long text
        assert 'summary' in result