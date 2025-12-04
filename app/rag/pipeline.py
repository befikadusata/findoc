"""
RAG (Retrieval-Augmented Generation) Pipeline Module

This module provides functions for chunking documents, creating embeddings,
and indexing them in a vector database for retrieval-augmented generation.
"""

"""
RAG (Retrieval-Augmented Generation) Pipeline Module

This module provides functions for chunking documents, creating embeddings,
and indexing them in a vector database for retrieval-augmented generation.
"""

import os
import uuid
from typing import List, Dict, Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import centralized settings
from app.config import settings

# Import structured logging
from app.utils.logging_config import get_logger

class RAGPipeline:
    """
    RAG pipeline for document chunking, embedding, and indexing.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 persist_directory: str = "./data/chroma"):
        """
        Initialize the RAG pipeline.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
            persist_directory (str): Directory to persist the ChromaDB
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        
        # Ensure the persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize the sentence transformer model
        try:
            self.encoder = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Error initializing sentence transformer: {e}")
            # Fallback initialization
            self.encoder = None
        
        # Initialize ChromaDB client with persistence
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Error initializing ChromaDB client: {e}")
            self.chroma_client = None
    
    def chunk_text(self, text: str, 
                   chunk_size: int = 1000, 
                   chunk_overlap: int = 200) -> List[str]:
        """
        Split text into chunks using RecursiveCharacterTextSplitter.
        
        Args:
            text (str): The text to chunk
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        # Use LangChain's RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_text(text)
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts using the sentence transformer model.

        Args:
            texts (List[str]): List of texts to embed

        Returns:
            List[List[float]]: List of embeddings
        """
        if self.encoder is None:
            # Fallback: return empty embeddings
            return [[] for _ in texts]

        try:
            embeddings = self.encoder.encode(texts)
            # Convert to list of lists format
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger = get_logger(__name__)
            logger.error("Error creating embeddings", error=str(e))
            # Return empty embeddings in case of error
            return [[] for _ in texts]
    
    def index_document(self,
                      doc_id: str,
                      text: str,
                      doc_type: str = "unknown",
                      metadata: Optional[Dict] = None) -> bool:
        """
        Index a document in the vector database.

        Args:
            doc_id (str): Unique identifier for the document
            text (str): Text content of the document
            doc_type (str): Type of document (for metadata)
            metadata (Dict, optional): Additional metadata for the document

        Returns:
            bool: True if successful, False otherwise
        """
        logger = get_logger(__name__).bind(doc_id=doc_id, doc_type=doc_type)

        if self.chroma_client is None:
            logger.error("ChromaDB client not initialized")
            return False

        try:
            # Set up default metadata
            if metadata is None:
                metadata = {}

            # Add document type to metadata
            metadata['doc_type'] = doc_type
            metadata['doc_id'] = doc_id

            # Chunk the text
            chunks = self.chunk_text(text)
            logger.info("Document split into chunks", chunk_count=len(chunks))

            if not chunks:
                logger.warning("No chunks created for document")
                return False

            # Create embeddings for the chunks
            embeddings = self.create_embeddings(chunks)
            logger.info("Created embeddings", embedding_count=len(chunks))

            # Create a unique collection for this document
            collection_name = f"docs_{doc_id.replace('-', '_')}"

            # Get or create the collection
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

            # Prepare IDs, documents, embeddings, and metadata for storage
            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            documents = chunks
            metadatas = [metadata.copy() for _ in chunks]

            # Add the chunks to the collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info("Document indexed successfully", collection=collection_name, chunks=len(chunks))
            return True

        except Exception as e:
            logger.error("Error indexing document", error=str(e))
            return False
    
    def query_document(self,
                      doc_id: str,
                      query: str,
                      n_results: int = 5) -> List[Dict]:
        """
        Query a specific document collection for relevant chunks.

        Args:
            doc_id (str): ID of the document to query
            query (str): Query text
            n_results (int): Number of results to return

        Returns:
            List[Dict]: List of matching chunks with metadata
        """
        logger = get_logger(__name__).bind(doc_id=doc_id, query=query)

        if self.chroma_client is None:
            logger.error("ChromaDB client not initialized")
            return []

        try:
            # Create the collection name based on the document ID
            collection_name = f"docs_{doc_id.replace('-', '_')}"

            # Get the collection
            collection = self.chroma_client.get_collection(
                name=collection_name
            )

            # Create embedding for the query
            query_embedding = self.encoder.encode([query])

            # Ensure the embedding is in the right format
            query_embedding_list = query_embedding[0].tolist() if hasattr(query_embedding[0], 'tolist') else query_embedding[0]

            # Query the collection
            results = collection.query(
                query_embeddings=[query_embedding_list],
                n_results=n_results
            )

            # Format the results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })

            return formatted_results

        except Exception as e:
            logger.error("Error querying document", error=str(e))
            return []

    def generate_response_with_rag(self,
                                  doc_id: str,
                                  query: str,
                                  n_results: int = 3,
                                  use_llm: bool = True) -> str:
        """
        Generate response to a query using RAG (Retrieval-Augmented Generation).

        Args:
            doc_id (str): ID of the document to query
            query (str): Question/query to answer
            n_results (int): Number of relevant chunks to retrieve
            use_llm (bool): Whether to use LLM to generate response

        Returns:
            str: Generated response with retrieved context
        """
        # Query the document to get relevant chunks
        results = self.query_document(doc_id, query, n_results)

        if not results:
            return "No relevant information found in the document."

        # Combine the retrieved chunks to form context
        context_parts = []
        for result in results:
            context_parts.append(result['document'])

        context = "\n\n".join(context_parts)

        if use_llm:
            # Use the Gemini API to generate a response based on the context
            try:
                import google.generativeai as genai
                
                # Get the API key from centralized settings
                if settings.gemini_api_key is None:
                    return f"Based on the document:\n\n{context}"

                # Configure the API key
                genai.configure(api_key=settings.gemini_api_key.get_secret_value())

                # Select the model
                model = genai.GenerativeModel('gemini-pro')

                # Use the prompt manager to get the RAG query prompt
                from prompts.prompt_manager import render_rag_query_prompt
                rag_prompt = render_rag_query_prompt(context=context, query=query)

                if not rag_prompt:
                    # Fallback to default prompt if template rendering fails
                    rag_prompt = f"""
                    Answer the question based on the provided context from a document.
                    If the answer is not in the context, say "I don't have enough information to answer that question."

                    Context:
                    {context}

                    Question: {query}

                    Answer:
                    """

                # Generate the content
                response = model.generate_content(
                    rag_prompt,
                    request_options={"timeout": 60}
                )

                # Return the text part of the response
                if response.text:
                    return response.text.strip()
                else:
                    # If there's an issue with the response, return the context
                    return f"Based on the document:\n\n{context}"

            except Exception as e:
                logger = get_logger(__name__)
                logger.error("Error using Gemini API", error=str(e))
                # Fallback to returning the context if there's an error
                return f"Based on the document:\n\n{context}"
        else:
            # For backward compatibility, return the context as before
            return f"Based on the document:\n\n{context}"


# Global RAG pipeline instance for convenience
rag_pipeline = RAGPipeline()


def index_document(doc_id: str, text: str, doc_type: str = "unknown", 
                   metadata: Optional[Dict] = None) -> bool:
    """
    Convenience function to index a document.
    
    Args:
        doc_id (str): Unique identifier for the document
        text (str): Text content of the document
        doc_type (str): Type of document (for metadata)
        metadata (Dict, optional): Additional metadata for the document
        
    Returns:
        bool: True if successful, False otherwise
    """
    return rag_pipeline.index_document(doc_id, text, doc_type, metadata)


def query_document(doc_id: str, query: str, n_results: int = 5) -> List[Dict]:
    """
    Convenience function to query a document collection.

    Args:
        doc_id (str): ID of the document to query
        query (str): Query text
        n_results (int): Number of results to return

    Returns:
        List[Dict]: List of matching chunks with metadata
    """
    return rag_pipeline.query_document(doc_id, query, n_results)


def generate_response_with_rag(doc_id: str, query: str, n_results: int = 3) -> str:
    """
    Convenience function to generate a response using RAG.

    Args:
        doc_id (str): ID of the document to query
        query (str): Question/query to answer
        n_results (int): Number of relevant chunks to retrieve

    Returns:
        str: Generated response with retrieved context
    """
    return rag_pipeline.generate_response_with_rag(doc_id, query, n_results, use_llm=True)


def delete_document_from_chromadb(doc_id: str) -> bool:
    """
    Delete a document's collection from ChromaDB.

    Args:
        doc_id (str): ID of the document to delete

    Returns:
        bool: True if successful, False otherwise
    """
    logger = get_logger(__name__).bind(doc_id=doc_id)

    if rag_pipeline.chroma_client is None:
        logger.error("ChromaDB client not initialized")
        return False

    try:
        # Create the collection name based on the document ID
        collection_name = f"docs_{doc_id.replace('-', '_')}"

        # Delete the collection
        rag_pipeline.chroma_client.delete_collection(collection_name)
        logger.info("Successfully deleted collection", collection=collection_name)
        return True

    except Exception as e:
        logger.error("Error deleting collection", error=str(e))
        return False


if __name__ == "__main__":
    # Example usage
    logger = get_logger(__name__)
    sample_text = "This is a sample document for testing the RAG pipeline. " * 10
    success = index_document("test-doc-123", sample_text, "test")
    if success:
        logger.info("Document indexed successfully")

        # Query the document
        results = query_document("test-doc-123", "sample document", n_results=2)
        logger.info("Query results", results=results)
    else:
        logger.error("Failed to index document")
