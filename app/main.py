import os
import uuid
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status

# For Celery integration
from celery import Celery

# Import authentication and database modules
from app.auth import get_current_user
from app.database import get_db, Session
from app.models import User, Document
from app.api.endpoints import auth as auth_endpoints

# Import API schemas for input validation
from app.api.schemas.request_models import QueryRequest

# Import prometheus instrumentation
from prometheus_fastapi_instrumentator import Instrumentator

# Import structured logging and settings
from app.utils.logging_config import get_logger
from app.config import settings

from app.utils.file_utils import sanitize_filename

logger = get_logger(__name__)

# Import MLflow
import mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Initialize Celery only if Redis URL is configured
celery_app = None
if settings.redis_url:
    celery_app = Celery('findocai', broker=settings.redis_url)
else:
    logger.warning("REDIS_URL not configured, Celery will not be initialized.")

def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application.
    This prevents the app from being initialized during pytest collection
    and allows for flexible testing configurations.
    """
    app = FastAPI(title="FinDocAI", version="1.0.0", description="Intelligent Financial Document Processing API")

    # Include the authentication router
    app.include_router(auth_endpoints.router, prefix="/auth", tags=["Authentication"])

    # Instrument the app with Prometheus metrics
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    @app.get("/")
    async def root() -> Dict[str, str]:
        """Root endpoint that returns a welcome message."""
        return {"message": "Welcome to FinDocAI - Intelligent Financial Document Processing API", "status": "running"}

    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint to verify the service is running."""
        import time
        return {"status": "healthy", "timestamp": int(time.time())}

    @app.post("/upload", status_code=status.HTTP_201_CREATED)
    async def upload_document(
        file: UploadFile = File(...), 
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """
        Upload a document for processing and associate it with the current user.
        """
        doc_id = str(uuid.uuid4())
        upload_logger = logger.bind(doc_id=doc_id, filename=file.filename, username=current_user.username)
        sanitized_filename = sanitize_filename(file.filename)

        with mlflow.start_run(run_name=f"document_upload_{doc_id}"):
            mlflow.log_param("document_id", doc_id)
            mlflow.log_param("filename", sanitized_filename)
            mlflow.log_param("user_id", current_user.id)

            upload_dir = "./data/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, f"{doc_id}_{sanitized_filename}")

            file_content = 0 # Temporarily set to 0. It will be the actual size from the streaming
            total_bytes_written = 0
            with open(file_path, "wb") as buffer:
                while chunk := await file.read(8192):  # Read in 8KB chunks
                    buffer.write(chunk)
                    total_bytes_written += len(chunk)
            mlflow.log_param("file_size", total_bytes_written)

            mlflow.log_artifact(file_path, "uploaded_files")

            # Create document record and associate with user
            new_document = Document(
                doc_id=doc_id,
                filename=sanitized_filename,
                status='queued',
                user_id=current_user.id
            )
            db.add(new_document)
            db.commit()
            db.refresh(new_document)
            
            upload_logger.info("Document saved and associated with user")

            if celery_app:
                task = celery_app.send_task('process_document', args=[doc_id, file_path])
                upload_logger.info("Processing task queued", task_id=task.id)
            else:
                upload_logger.warning("Celery app not initialized, document not queued for processing.")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Celery not configured")

        return {
            "doc_id": doc_id,
            "filename": sanitized_filename,
            "message": "Document uploaded successfully and processing has started."
        }


    @app.get("/status/{doc_id}")
    async def get_document_status_endpoint(
        doc_id: str, 
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """
        Get the status of a document, ensuring ownership.
        """
        doc = db.query(Document).filter(Document.doc_id == doc_id).first()

        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        
        if doc.user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this document")

        return {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "status": doc.status,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at
        }


    @app.post("/query")
    async def query_document_endpoint(
        request: QueryRequest,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """
        Query a document and get an answer, ensuring ownership.
        """
        doc = db.query(Document).filter(Document.doc_id == request.doc_id).first()

        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        
        if doc.user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to query this document")

        with mlflow.start_run(run_name=f"document_query_{request.doc_id}"):
            mlflow.log_param("document_id", request.doc_id)
            mlflow.log_param("question", request.question)

            from app.rag.pipeline import generate_response_with_rag
            result = generate_response_with_rag(request.doc_id, request.question)

            mlflow.log_metric("answer_length", len(result.get("response", "")))

        return {
            "doc_id": request.doc_id,
            "question": request.question,
            "answer": result.get("response", "")
        }

    @app.get("/summary/{doc_id}")
    async def get_document_summary_endpoint(
        doc_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """
        Get the summary of a document, ensuring ownership.
        """
        doc = db.query(Document).filter(Document.doc_id == doc_id).first()

        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        
        if doc.user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this summary")
            
        if not doc.summary:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Summary not available for this document")

        return {"doc_id": doc.doc_id, "summary": doc.summary}

    @app.delete("/documents/{doc_id}", status_code=status.HTTP_200_OK)
    async def delete_document_endpoint(
        doc_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """
        Delete a document and all its associated data, ensuring ownership.
        """
        doc = db.query(Document).filter(Document.doc_id == doc_id).first()

        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        
        if doc.user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete this document")

        delete_logger = logger.bind(doc_id=doc_id, username=current_user.username)
        delete_logger.info("Starting document deletion process")

        try:
            from app.rag.pipeline import delete_document_from_chromadb
            chroma_deleted = delete_document_from_chromadb(doc_id)
            
            db.delete(doc)
            db.commit()

            file_path = os.path.join("./data/uploads", f"{doc.doc_id}_{doc.filename}")
            if os.path.exists(file_path):
                os.remove(file_path)

            delete_logger.info("Document deleted successfully")
            return {"message": "Document deleted successfully", "doc_id": doc_id}

        except Exception as e:
            delete_logger.error("Error during document deletion", error=str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred: {e}")
    
    return app # Return the configured app instance

# Global app instance will be created only when app.main is run directly
# Otherwise, create_app() should be called explicitly (e.g., in tests)
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)