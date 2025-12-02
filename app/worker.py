from celery import Celery
import os

# Import the OCR module for text extraction
from app.ingestion.ocr import extract_text

# Configure the broker URL - using Redis
# When using docker-compose, the service name 'redis' is used as the hostname
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', '6380')  # Using port 6380 as per our docker-compose setup
redis_url = f'redis://{redis_host}:{redis_port}/0'

# Create Celery app instance
celery_app = Celery('findocai', broker=redis_url, backend=redis_url)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
)

@celery_app.task(bind=True, name='process_document')
def process_document(self, doc_id: str, filepath: str):
    """
    Process the uploaded document asynchronously.

    This function now includes OCR and text extraction as the first step,
    using a hybrid approach with PyPDF2 and Tesseract.
    """
    try:
        # Update document status to 'processing'
        from app.database import update_document_status
        update_document_status(doc_id, 'processing')

        # OCR and text extraction - the first step in the pipeline
        print(f"Starting OCR and text extraction for document {doc_id} at {filepath}")
        extracted_text = extract_text(filepath)

        # Update document status based on extraction results
        if extracted_text.strip():
            print(f"Successfully extracted {len(extracted_text)} characters from document {doc_id}")
            # Update document status to 'extracted' or 'processing' for next steps
            update_document_status(doc_id, 'extracted')
        else:
            print(f"No text extracted from document {doc_id}")
            # Update document status to 'failed_extraction' if needed
            update_document_status(doc_id, 'failed_extraction')
            return {"status": "failed", "doc_id": doc_id, "error": "No text could be extracted from the document"}

        # Here we would continue with the rest of the processing:
        # - Document classification
        # - Entity extraction
        # - Vector indexing
        # etc.

        # For now, just simulate the rest of the processing with a short delay
        import time
        time.sleep(2)

        # Update document status to 'completed'
        update_document_status(doc_id, 'completed')

        print(f"Completed processing for document {doc_id}")
        return {
            "status": "completed",
            "doc_id": doc_id,
            "processed_file": filepath,
            "extracted_text_length": len(extracted_text)
        }

    except Exception as e:
        print(f"Error processing document {doc_id}: {str(e)}")
        # Update document status to 'error'
        from app.database import update_document_status
        update_document_status(doc_id, 'error')

        # Update task state to reflect failure
        self.update_state(
            state='FAILURE',
            meta={'exc_type': type(e).__name__, 'exc_message': str(e)}
        )
        raise e

if __name__ == '__main__':
    celery_app.start()