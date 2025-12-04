import os
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File
import time

# For Celery integration, we'll call the task by name to avoid import issues
from celery import Celery

# Import the database module
from app.database import init_db, create_document_record, update_document_status, get_document_status

# Import prometheus instrumentation
from prometheus_fastapi_instrumentator import Instrumentator

# Import structured logging
from app.utils.logging_config import get_logger
logger = get_logger(__name__)

# Configure Celery app to match the worker configuration
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', '6380')  # Using port 6380 as per our docker-compose setup
redis_url = f'redis://{redis_host}:{redis_port}/0'

# Create a Celery instance for calling tasks
celery_app = Celery('findocai', broker=redis_url)

# Initialize the database
init_db()

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

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a document for processing.

    Args:
        file: The document file to upload

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

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Create a record in the database with 'uploaded' status
    create_document_record(doc_id, file.filename)

    # Update the document status to 'queued' before starting processing
    update_document_status(doc_id, 'queued')

    upload_logger.info("Document saved and status updated to queued")

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

@app.get("/status/{doc_id}")
async def get_document_status_endpoint(doc_id: str) -> Dict[str, Any]:
    """
    Get the status of a document by its ID.

    Args:
        doc_id: The unique identifier of the document

    Returns:
        Dict[str, Any]: Document status information or error
    """
    document_info = get_document_status(doc_id)

    if document_info is None:
        return {"error": "Document not found", "doc_id": doc_id}

    return {
        "doc_id": document_info["doc_id"],
        "filename": document_info["filename"],
        "status": document_info["status"],
        "created_at": document_info["created_at"],
        "updated_at": document_info["updated_at"]
    }

@app.get("/query")
async def query_document_endpoint(doc_id: str, question: str) -> Dict[str, Any]:
    """
    Query a document and get an answer based on its content.

    Args:
        doc_id: The unique identifier of the document to query
        question: The question to ask about the document

    Returns:
        Dict[str, Any]: The answer to the question or error
    """
    # Verify that the document exists in our system
    document_info = get_document_status(doc_id)

    if document_info is None:
        return {"error": "Document not found", "doc_id": doc_id}

    # Use the RAG pipeline to generate a response
    from app.rag.pipeline import generate_response_with_rag
    answer = generate_response_with_rag(doc_id, question)

    return {
        "doc_id": doc_id,
        "question": question,
        "answer": answer
    }

@app.get("/summary/{doc_id}")
async def get_document_summary_endpoint(doc_id: str) -> Dict[str, Any]:
    """
    Get the summary of a document.

    Args:
        doc_id: The unique identifier of the document to get summary for

    Returns:
        Dict[str, Any]: Document summary or error
    """
    # Verify that the document exists in our system
    document_info = get_document_status(doc_id)

    if document_info is None:
        return {"error": "Document not found", "doc_id": doc_id}

    # Get the summary from the database
    from app.database import get_document_summary
    summary = get_document_summary(doc_id)

    if summary is None:
        return {"error": "Summary not available", "doc_id": doc_id}

    return {
        "doc_id": doc_id,
        "summary": summary
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)