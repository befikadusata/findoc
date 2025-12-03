"""Unit tests for RAG pipeline functions."""

import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest
from app.rag.pipeline import RAGPipeline, index_document, query_document, generate_response_with_rag


def test_rag_pipeline_initialization():
    """Test initializing the RAG pipeline."""
    with patch('app.rag.pipeline.SentenceTransformer'), \
         patch('chromadb.PersistentClient'):
        pipeline = RAGPipeline()
        assert pipeline.model_name == "all-MiniLM-L6-v2"
        assert pipeline.persist_directory == "./data/chroma"


def test_chunk_text():
    """Test text chunking functionality."""
    pipeline = RAGPipeline()  # This won't try to initialize real models in this test
    
    long_text = "This is a sample text. " * 100  # Create a long text
    chunks = pipeline.chunk_text(long_text, chunk_size=50, chunk_overlap=10)
    
    assert len(chunks) > 0
    # Each chunk should be less than or equal to the chunk_size
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_create_embeddings():
    """Test creating embeddings (mocked)."""
    with patch('app.rag.pipeline.SentenceTransformer') as mock_transformer_class:
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_transformer_class.return_value = mock_encoder
        
        pipeline = RAGPipeline()
        texts = ["Sample text 1", "Sample text 2"]
        embeddings = pipeline.create_embeddings(texts)
        
        # Should return list of lists
        assert len(embeddings) == 2
        assert all(isinstance(embedding, list) for embedding in embeddings)


def test_create_embeddings_with_fallback():
    """Test embeddings with encoder failure."""
    pipeline = RAGPipeline()
    # Simulate encoder being None
    pipeline.encoder = None
    
    texts = ["Sample text 1", "Sample text 2"]
    embeddings = pipeline.create_embeddings(texts)
    
    # Should return empty embeddings as fallback
    assert len(embeddings) == 2
    assert all(embedding == [] for embedding in embeddings)


def test_index_document():
    """Test indexing a document (mocked)."""
    with patch('app.rag.pipeline.SentenceTransformer'), \
         patch('chromadb.PersistentClient') as mock_client_class:
        
        # Mock the ChromaDB client and collection
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        pipeline = RAGPipeline()
        
        # Mock the text chunking and embedding
        with patch.object(pipeline, 'chunk_text', return_value=["chunk1", "chunk2"]), \
             patch.object(pipeline, 'create_embeddings', return_value=[[0.1, 0.2], [0.3, 0.4]]):
            
            success = pipeline.index_document("doc123", "Sample document text", "invoice")
            
            # Should return True for success
            assert success is True
            
            # Verify the collection.add method was called
            mock_collection.add.assert_called()


def test_index_document_chromadb_failure():
    """Test indexing with ChromaDB initialization failure."""
    pipeline = RAGPipeline()
    # Simulate ChromaDB client being None
    pipeline.chroma_client = None
    
    success = pipeline.index_document("doc123", "Sample document text", "invoice")
    
    # Should return False
    assert success is False


def test_query_document():
    """Test querying a document (mocked)."""
    with patch('app.rag.pipeline.SentenceTransformer') as mock_transformer_class, \
         patch('chromadb.PersistentClient') as mock_client_class:
        
        # Mock the encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [[0.5, 0.6]]
        mock_transformer_class.return_value = mock_encoder
        
        # Mock the ChromaDB client
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_result = {
            'documents': [['retrieved chunk']],
            'metadatas': [[{'doc_type': 'invoice'}]],
            'distances': [[0.1]]
        }
        mock_collection.query.return_value = mock_result
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        pipeline = RAGPipeline()
        
        results = pipeline.query_document("doc123", "Sample query")
        
        # Should return formatted results
        assert len(results) == 1
        assert results[0]['document'] == 'retrieved chunk'
        assert results[0]['metadata']['doc_type'] == 'invoice'


def test_generate_response_with_rag():
    """Test generating response with RAG (mocked)."""
    with patch('app.rag.pipeline.SentenceTransformer'), \
         patch('chromadb.PersistentClient') as mock_client_class:
        
        # Set up mock client that returns query results
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_result = {
            'documents': [['relevant context chunk']],
            'metadatas': [[{'doc_type': 'invoice'}]],
            'distances': [[0.1]]
        }
        mock_collection.query.return_value = mock_result
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        pipeline = RAGPipeline()
        
        # Mock Gemini API
        with patch('os.getenv', return_value='fake-api-key'), \
             patch('google.generativeai.configure') as mock_configure, \
             patch('google.generativeai.GenerativeModel') as mock_model_class:
            
            mock_model_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Generated response based on context"
            mock_model_instance.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model_instance
            
            response = pipeline.generate_response_with_rag("doc123", "Sample question", use_llm=True)
            
            # Should return the generated response
            assert "Generated response" in response


def test_generate_response_with_rag_fallback():
    """Test RAG generation with Gemini API failure."""
    with patch('app.rag.pipeline.SentenceTransformer'), \
         patch('chromadb.PersistentClient') as mock_client_class:
        
        # Set up mock client that returns query results
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_result = {
            'documents': [['relevant context chunk']],
            'metadatas': [[{'doc_type': 'invoice'}]],
            'distances': [[0.1]]
        }
        mock_collection.query.return_value = mock_result
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        pipeline = RAGPipeline()
        
        # Simulate Gemini API error
        with patch('os.getenv', return_value='fake-api-key'), \
             patch('google.generativeai.configure', side_effect=Exception("API Error")):
            
            response = pipeline.generate_response_with_rag("doc123", "Sample question", use_llm=True)
            
            # Should return context as fallback
            assert "Based on the document:" in response


def test_convenience_functions():
    """Test the convenience functions for RAG operations."""
    # Test index_document convenience function
    with patch('app.rag.pipeline.rag_pipeline.index_document', return_value=True) as mock_method:
        result = index_document("doc123", "Sample text", "invoice")
        assert result is True
        mock_method.assert_called_once_with("doc123", "Sample text", "invoice", None)
    
    # Test query_document convenience function
    with patch('app.rag.pipeline.rag_pipeline.query_document', return_value=[{"document": "result"}]) as mock_method:
        result = query_document("doc123", "query text")
        assert result == [{"document": "result"}]
        mock_method.assert_called_once_with("doc123", "query text", 5)
    
    # Test generate_response_with_rag convenience function
    with patch('app.rag.pipeline.rag_pipeline.generate_response_with_rag', return_value="response") as mock_method:
        result = generate_response_with_rag("doc123", "question")
        assert result == "response"
        mock_method.assert_called_once_with("doc123", "question", 3, use_llm=True)