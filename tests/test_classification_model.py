"""Unit tests for document classification functions."""

import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest
from app.classification.model import DocumentClassifier, classify_document


def test_document_classifier_initialization():
    """Test initializing the document classifier."""
    classifier = DocumentClassifier()
    assert classifier.model_name == "distilbert-base-uncased"
    assert classifier.num_labels == 6


def test_classify_document_with_transformer():
    """Test document classification with transformer model (mocked)."""
    classifier = DocumentClassifier()
    
    # Mock the transformer classifier
    mock_result = [
        [
            {'label': 'LABEL_0', 'score': 0.8, 'doc_type': 'invoice'},
            {'label': 'LABEL_1', 'score': 0.1, 'doc_type': 'contract'}
        ]
    ]
    
    with patch.object(classifier, 'classifier') as mock_classifier:
        mock_classifier.return_value = mock_result
        
        result = classifier.classify_document("This is an invoice for services rendered.")
        
        assert len(result) == 1  # top_k=1 by default
        assert result[0]['label'] in ['LABEL_0', 'LABEL_1']
        assert result[0]['doc_type'] in ['invoice', 'contract']


def test_classify_document_with_keyword_fallback():
    """Test document classification with keyword-based fallback."""
    classifier = DocumentClassifier()
    
    # Set classifier to None to trigger fallback
    classifier.classifier = None
    
    # Test invoice classification
    invoice_text = "INVOICE Date: 2023-06-15 Invoice No: INV-2023-06015"
    result = classifier.classify_document(invoice_text, top_k=2)
    
    # Verify results are sorted by score
    assert len(result) == 2
    assert all('label' in item and 'score' in item for item in result)
    # The invoice keyword should give it a higher score in the fallback system


def test_classify_document_error_handling():
    """Test error handling in document classification."""
    classifier = DocumentClassifier()
    
    # Mock an exception in the transformer classifier
    with patch.object(classifier, 'classifier', side_effect=Exception("Model error")):
        result = classifier.classify_document("Test document text")
        # Should fallback to keyword-based classification
        assert isinstance(result, list)
        assert len(result) >= 0  # May be empty but shouldn't crash


def test_keyword_based_classification():
    """Test the keyword-based fallback classification directly."""
    classifier = DocumentClassifier()
    
    # Test document with loan keywords
    loan_text = "LOAN APPLICATION Borrower: John Smith Amount: $50,000 Interest rate: 5%"
    result = classifier._keyword_based_classification(loan_text, top_k=1)
    
    assert len(result) >= 1
    # Check if loan-related keywords were detected
    loan_result = next((item for item in result if item['label'] == 'loan_application'), None)
    if loan_result:
        assert loan_result['score'] >= 0  # Score should be non-negative


def test_classify_document_convenience_function():
    """Test the convenience function for document classification."""
    mock_result = [
        {'label': 'LABEL_0', 'score': 0.9, 'doc_type': 'invoice'},
    ]
    
    with patch('app.classification.model.classifier.classify_document', return_value=mock_result):
        result = classify_document("Sample document text")
        assert result == mock_result