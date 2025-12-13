import os
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Depends
import time

# For Celery integration, we'll call the task by name to avoid import issues
from celery import Celery

# Import authentication
from app.auth import get_current_user
from app.config import settings

# Import the database module
from app.database_factory import database

# Import API schemas for input validation
from app.api.schemas.request_models import DocumentIdRequest, QueryRequest, DeleteDocumentRequest, SummaryRequest

# Import prometheus instrumentation
from prometheus_fastapi_instrumentator import Instrumentator

# Import structured logging
from app.utils.logging_config import get_logger
logger = get_logger(__name__)

# Import centralized settings
from app.config import settings

# Import MLflow
import mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Create a Celery instance for calling tasks
celery_app = Celery('findocai', broker=settings.redis_url)

app = FastAPI(title="FinDocAI", version="1.0.0", description="Intelligent Financial Document Processing API")

# Instrument the app with Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint that returns a welcome message and running status.

    Returns:
        Dict[str, str]: Welcome message and status
    """
    return {"message": "Welcome to FinDocAI - Intelligent Financial Document Processing API", "status": "running"}

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify the service is running.

    Returns:
        Dict[str, Any]: Health status and timestamp
    """
    return {"status": "healthy", "timestamp": int(time.time())}

# Authentication dependency - conditionally applied based on settings
if settings.require_auth:
    from app.auth import get_current_user
    auth_dependency = Depends(get_current_user)
else:
    # Create a dummy dependency that always passes
    async def no_auth():
        return {"api_key_valid": True, "user_id": "anonymous"}
    auth_dependency = Depends(no_auth)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), current_user: dict = auth_dependency) -> Dict[str, Any]:
    """
    Upload a document for processing.

    Args:
        file: The document file to upload
        current_user: Authenticated user (if authentication is enabled)

    Returns:
        Dict[str, Any]: Document ID, file info, and processing status
    """
    # Generate a unique document ID
    doc_id = str(uuid.uuid4())

    # Add document ID to logger context
    upload_logger = logger.bind(doc_id=doc_id, filename=file.filename)

    upload_logger.info("Starting document upload")

    # Create the uploads directory if it doesn't exist
    upload_dir = "./data/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Create the file path with the unique document ID
    file_path = os.path.join(upload_dir, f"{doc_id}_{file.filename}")

    try:
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Create a record in the database with 'uploaded' status
        if not database.create_document_record(doc_id, file.filename):
            upload_logger.error("Failed to create document record in database")
            return {
                "doc_id": doc_id,
                "filename": file.filename,
                "error": "Failed to register document in system"
            }

        # Update the document status to 'queued' before starting processing
        if not database.update_document_status(doc_id, 'queued'):
            upload_logger.error("Failed to update document status in database")
            return {
                "doc_id": doc_id,
                "filename": file.filename,
                "error": "Failed to update document status"
            }

        upload_logger.info("Document saved and status updated to queued")

        # Trigger the document processing task asynchronously
        # This will be executed by a Celery worker
        task = celery_app.send_task('process_document', args=[doc_id, file_path])

        upload_logger.info("Processing task queued", task_id=task.id)

    except Exception as e:
        upload_logger.error("Error during document upload processing", error=str(e))
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "error": f"Upload processing error: {str(e)}"
    try:
        upload_logger.info("Starting document upload")

        # Log the upload to MLflow
        with mlflow.start_run(run_name=f"document_upload_{doc_id}"):
            mlflow.log_param("document_id", doc_id)
            mlflow.log_param("filename", file.filename)
            mlflow.log_param("content_type", file.content_type)

            # Create the uploads directory if it doesn't exist
            upload_dir = "./data/uploads"
            os.makedirs(upload_dir, exist_ok=True)

            # Create the file path with the unique document ID
            file_path = os.path.join(upload_dir, f"{doc_id}_{file.filename}")

            # Read file content
            file_content = await file.read()
            mlflow.log_param("file_size", len(file_content))

            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                buffer.write(file_content)

            # Log the file as an artifact
            mlflow.log_artifact(file_path, "uploaded_files")

            # Create a record in the database with 'uploaded' status
            if not database.create_document_record(doc_id, file.filename):
                upload_logger.error("Failed to create document record in database")
                return {
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "error": "Failed to register document in system"
                }

            # Update the document status to 'queued' before starting processing
            if not database.update_document_status(doc_id, 'queued'):
                upload_logger.error("Failed to update document status in database")
                return {
                    "doc_id": doc_id,
                    "filename": file.filename,
                    "error": "Failed to update document status"
                }

            upload_logger.info("Document saved, status updated, and MLflow artifact logged")

            # Trigger the document processing task asynchronously
            # This will be executed by a Celery worker
            task = celery_app.send_task('process_document', args=[doc_id, file_path])

        upload_logger.info("Processing task queued", task_id=task.id)

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "file_path": file_path,
            "task_id": task.id,
            "message": "Document uploaded successfully and processing started"
        }

    except Exception as e:
        upload_logger.error("Error during document upload processing", error=str(e))
        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "error": f"Upload processing error: {str(e)}"
        }

@app.get("/status/{doc_id}")
async def get_document_status_endpoint(doc_id: str, current_user: dict = auth_dependency) -> Dict[str, Any]:
    """
    Get the status of a document by its ID.

    Args:
        doc_id: The unique identifier of the document
        current_user: Authenticated user (if authentication is enabled)

    Returns:
        Dict[str, Any]: Document status information or error
    """
    # Validate doc_id format
    try:
        DocumentIdRequest(doc_id=doc_id)
    except Exception as e:
        logger.error("Invalid document ID format", doc_id=doc_id, error=str(e))
        return {"error": "Invalid document ID format", "doc_id": doc_id}

    document_info = database.get_document_status(doc_id)

    if document_info is None:
        logger.warning("Document not found", doc_id=doc_id)
        return {"error": "Document not found", "doc_id": doc_id}

    return {
        "doc_id": document_info["doc_id"],
        "filename": document_info["filename"],
        "status": document_info["status"],
        "created_at": document_info["created_at"],
        "updated_at": document_info["updated_at"]
    }

@app.get("/query")
async def query_document_endpoint(doc_id: str, question: str, explain: bool = False, current_user: dict = auth_dependency) -> Dict[str, Any]:
    """
    Query a document and get an answer based on its content.

    Args:
        doc_id: The unique identifier of the document to query
        question: The question to ask about the document
        explain: Whether to include explanation data (confidence scores, source attribution)
        current_user: Authenticated user (if authentication is enabled)

    Returns:
        Dict[str, Any]: The answer to the question or error, with optional explanation data
    """
    # Validate input parameters
    try:
        QueryRequest(doc_id=doc_id, question=question)
    except Exception as e:
        return {"error": f"Invalid input parameters: {str(e)}", "doc_id": doc_id}

    # Verify that the document exists in our system
    document_info = database.get_document_status(doc_id)

    if document_info is None:
        logger.warning("Document not found for query", doc_id=doc_id, question=question)
        return {"error": "Document not found", "doc_id": doc_id}

    # Use the RAG pipeline to generate a response
    try:
        from app.rag.pipeline import generate_response_with_rag
        answer = generate_response_with_rag(doc_id, question)
    except Exception as e:
        logger.error("Error generating RAG response", doc_id=doc_id, question=question, error=str(e))
        return {"error": f"Error generating response: {str(e)}", "doc_id": doc_id}
    # Use the RAG pipeline to generate a response and log to MLflow
    with mlflow.start_run(run_name=f"document_query_{doc_id}"):
        mlflow.log_param("document_id", doc_id)
        mlflow.log_param("question", question)
        mlflow.log_param("query_timestamp", int(time.time()))
        mlflow.log_param("include_explanation", explain)

        from app.rag.pipeline import generate_response_with_rag
        result = generate_response_with_rag(doc_id, question, include_explanation=explain)

        # Log metrics based on result format
        if explain and isinstance(result, dict):
            answer = result.get("response", "")
            mlflow.log_metric("answer_length", len(answer) if answer else 0)
            mlflow.log_metric("confidence_score", result.get("explanation", {}).get("confidence", 0.0))
        else:
            answer_text = result if isinstance(result, str) else result.get("response", "")
            mlflow.log_metric("answer_length", len(answer_text) if answer_text else 0)

    if explain and isinstance(result, dict):
        # Return the full result with explanations
        return {
            "doc_id": doc_id,
            "question": question,
            "answer": result.get("response", ""),
            "explanation": result.get("explanation", {}),
            "retrieved_chunks": result.get("retrieved_chunks", [])
        }
    else:
        # Return just the answer for backward compatibility
        answer_text = result if isinstance(result, str) else result.get("response", "")
        return {
            "doc_id": doc_id,
            "question": question,
            "answer": answer_text
        }

@app.get("/summary/{doc_id}")
async def get_document_summary_endpoint(doc_id: str, current_user: dict = auth_dependency) -> Dict[str, Any]:
    """
    Get the summary of a document.

    Args:
        doc_id: The unique identifier of the document to get summary for
        current_user: Authenticated user (if authentication is enabled)

    Returns:
        Dict[str, Any]: Document summary or error
    """
    # Validate doc_id format
    try:
        SummaryRequest(doc_id=doc_id)
    except Exception as e:
        logger.error("Invalid document ID format for summary", doc_id=doc_id, error=str(e))
        return {"error": "Invalid document ID format", "doc_id": doc_id}

    # Verify that the document exists in our system
    document_info = database.get_document_status(doc_id)

    if document_info is None:
        logger.warning("Document not found for summary", doc_id=doc_id)
        return {"error": "Document not found", "doc_id": doc_id}

    # Get the summary from the database
    summary = database.get_document_summary(doc_id)

    if summary is None:
        logger.warning("Summary not available", doc_id=doc_id)
        return {"error": "Summary not available", "doc_id": doc_id}

    return {
        "doc_id": doc_id,
        "summary": summary
    }


@app.get("/explain/classification/{doc_id}")
async def get_classification_explanation(doc_id: str) -> Dict[str, Any]:
    """
    Get the classification explanation for a document if available.

    Args:
        doc_id: The unique identifier of the document to get explanation for

    Returns:
        Dict[str, Any]: Classification explanation or error
    """
    # Validate doc_id format
    try:
        DocumentIdRequest(doc_id=doc_id)
    except Exception:
        return {"error": "Invalid document ID format", "doc_id": doc_id}

    # Verify that the document exists in our system
    document_info = get_document_status(doc_id)

    if document_info is None:
        return {"error": "Document not found", "doc_id": doc_id}

    # Retrieve document entities which may contain classification explanation
    from app.database import get_document_entities
    entities = get_document_entities(doc_id)

    # We need to extract classification explanation from the document processing information
    # For now, we'll provide a way to classify fresh with explanation
    from app.classification.model import classifier

    # Get the document text from storage (this would require reading the file back)
    # Since we don't have a function to retrieve the original text, we'll return information
    # about what we know about the classification if it's available in the document status
    doc_type = document_info.get('status', '').replace('classified_as_', '') if 'classified_as_' in document_info.get('status', '') else 'unknown'

    return {
        "doc_id": doc_id,
        "classification_info": {
            "predicted_type": doc_type,
            "status": document_info.get('status'),
            "created_at": document_info.get('created_at'),
            "updated_at": document_info.get('updated_at')
        },
        "message": "Classification explanation endpoint - implementation would require access to original text for attention visualization"
    }


@app.delete("/documents/{doc_id}")
async def delete_document_endpoint(doc_id: str, current_user: dict = auth_dependency) -> Dict[str, Any]:
    """
    Delete a document and all its associated data.

    Args:
        doc_id: The unique identifier of the document to delete
        current_user: Authenticated user (if authentication is enabled)

    Returns:
        Dict[str, Any]: Deletion status information
    """
    # Validate doc_id format
    try:
        DeleteDocumentRequest(doc_id=doc_id)
    except Exception as e:
        logger.error("Invalid document ID format for deletion", doc_id=doc_id, error=str(e))
        return {"error": "Invalid document ID format", "doc_id": doc_id}

    # Verify that the document exists in our system
    document_info = database.get_document_status(doc_id)

    if document_info is None:
        logger.warning("Document not found for deletion", doc_id=doc_id)
        return {"error": "Document not found", "doc_id": doc_id}

    # Import required functions
    from app.rag.pipeline import delete_document_from_chromadb

    # Log the deletion attempt
    delete_logger = logger.bind(doc_id=doc_id)
    delete_logger.info("Starting document deletion process")

    try:
        # 1. Delete from ChromaDB
        chroma_deleted = delete_document_from_chromadb(doc_id)
        delete_logger.info("ChromaDB collection deletion attempted", success=chroma_deleted)

        # 2. Delete from PostgreSQL database
        db_deleted = database.delete_document_record(doc_id)
        delete_logger.info("Database record deletion attempted", success=db_deleted)

        # 3. Delete the actual file from storage
        file_deleted = delete_document_file(doc_id, document_info["filename"])
        delete_logger.info("File system deletion attempted", success=file_deleted)

        # Determine overall success
        overall_success = chroma_deleted and db_deleted and file_deleted

        if overall_success:
            delete_logger.info("Document deletion completed successfully")
            return {
                "message": "Document deleted successfully",
                "doc_id": doc_id,
                "deleted_from": {
                    "chromadb": chroma_deleted,
                    "database": db_deleted,
                    "file_system": file_deleted
                }
            }
        else:
            delete_logger.warning("Document deletion partially failed",
                                chromadb=chroma_deleted,
                                database=db_deleted,
                                file_system=file_deleted)
            return {
                "message": "Document deletion partially completed",
                "doc_id": doc_id,
                "deleted_from": {
                    "chromadb": chroma_deleted,
                    "database": db_deleted,
                    "file_system": file_deleted
                }
            }

    except Exception as e:
        delete_logger.error("Error occurred during document deletion", error=str(e))
        return {
            "error": f"Error occurred during document deletion: {str(e)}",
            "doc_id": doc_id
        }


def delete_document_file(doc_id: str, filename: str) -> bool:
    """
    Delete the actual document file from the file system.

    Args:
        doc_id: The document ID
        filename: The original filename

    Returns:
        bool: True if file was successfully deleted, False otherwise
    """
    import os

    file_path = os.path.join("./data/uploads", f"{doc_id}_{filename}")

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Document file deleted from file system", doc_id=doc_id, file_path=file_path)
            return True
        else:
            logger.warning("Document file not found in file system", doc_id=doc_id, file_path=file_path)
            return False
    except Exception as e:
        logger.error("Error deleting document file from file system",
                    doc_id=doc_id, file_path=file_path, error=str(e))
        return False


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)