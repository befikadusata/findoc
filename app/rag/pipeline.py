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
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
            print(f"Error initializing sentence transformer: {e}")
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
            print(f"Error initializing ChromaDB client: {e}")
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
            print(f"Error creating embeddings: {e}")
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
        if self.chroma_client is None:
            print("ChromaDB client not initialized")
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
            print(f"Document {doc_id} split into {len(chunks)} chunks")
            
            if not chunks:
                print(f"No chunks created for document {doc_id}")
                return False
            
            # Create embeddings for the chunks
            embeddings = self.create_embeddings(chunks)
            print(f"Created embeddings for {len(chunks)} chunks")
            
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
            
            print(f"Successfully indexed document {doc_id} with {len(chunks)} chunks in collection {collection_name}")
            return True
            
        except Exception as e:
            print(f"Error indexing document {doc_id}: {e}")
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
        if self.chroma_client is None:
            print("ChromaDB client not initialized")
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

            # Query the collection
            results = collection.query(
                query_embeddings=[query_embedding[0].tolist()],
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
            print(f"Error querying document {doc_id}: {e}")
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
                import os

                # Get the API key from environment variable
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    # If no API key is available, return the context as before
                    return f"Based on the document:\n\n{context}"

                # Configure the API key
                genai.configure(api_key=api_key)

                # Select the model
                model = genai.GenerativeModel('gemini-pro')

                # Create the RAG prompt template
                rag_prompt = f"""
                Answer the question based on the provided context from a document.
                If the answer is not in the context, say "I don't have enough information to answer that question."

                Context:
                {context}

                Question: {query}

                Answer:
                """

                # Generate the content
                response = model.generate_content(rag_prompt)

                # Return the text part of the response
                if response.text:
                    return response.text.strip()
                else:
                    # If there's an issue with the response, return the context
                    return f"Based on the document:\n\n{context}"

            except Exception as e:
                print(f"Error using Gemini API: {e}")
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


if __name__ == "__main__":
    # Example usage
    sample_text = "This is a sample document for testing the RAG pipeline. " * 10
    success = index_document("test-doc-123", sample_text, "test")
    if success:
        print("Document indexed successfully")

        # Query the document
        results = query_document("test-doc-123", "sample document", n_results=2)
        print(f"Query results: {results}")
    else:
        print("Failed to index document")