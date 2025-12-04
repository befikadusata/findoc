from celery import Celery
import os
from datetime import datetime
from typing import Any, Dict

# Import prometheus client for custom metrics
from prometheus_client import Counter, Histogram, Gauge

# Import structured logging
from app.utils.logging_config import get_logger

# Import the OCR module for text extraction
from app.ingestion.ocr import extract_text

# Import the classification module
from app.classification.model import classify_document

# Import the RAG pipeline for indexing
from app.rag.pipeline import index_document

# Import the NLP module for entity extraction and summarization
from app.nlp.extraction import extract_entities, generate_summary

# Define custom Prometheus metrics
DOCUMENT_PROCESSING_TOTAL = Counter(
    'document_processing_total',
    'Total number of processed documents',
    ['status']
)

DOCUMENT_PROCESSING_DURATION = Histogram(
    'document_processing_duration_seconds',
    'Time spent processing documents'
)

DOCUMENT_PROCESSING_CURRENT = Gauge(
    'document_processing_current',
    'Number of currently processing documents'
)

# Configure the broker URL - using Redis
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
def process_document(self, doc_id: str, filepath: str) -> Dict[str, Any]:
    """
    Process the uploaded document asynchronously.

    This function includes OCR and text extraction as the first step,
    followed by document classification, vector indexing, entity extraction,
    and summarization.

    Args:
        doc_id: The unique identifier of the document
        filepath: Path to the document file

    Returns:
        Dict[str, Any]: Processing results with status and metrics
    """
    # Get a logger instance with document context
    task_logger = get_logger("worker.process_document").bind(doc_id=doc_id, filepath=filepath)

    # Start the timer and increment the gauge
    start_time = datetime.now()
    DOCUMENT_PROCESSING_CURRENT.inc()

    try:
        task_logger.info("Starting document processing")

        # Update document status to 'processing'
        from app.database import update_document_status
        update_document_status(doc_id, 'processing')

        # OCR and text extraction - the first step in the pipeline
        task_logger.info("Starting OCR and text extraction")
        extracted_text = extract_text(filepath)

        # Update document status based on extraction results
        if extracted_text.strip():
            task_logger.info("Text extraction successful", text_length=len(extracted_text))
            # Update document status to 'extracted' or 'processing' for next steps
            update_document_status(doc_id, 'extracted')
        else:
            task_logger.warning("No text extracted from document")
            # Update document status to 'failed_extraction' if needed
            update_document_status(doc_id, 'failed_extraction')
            return {"status": "failed", "doc_id": doc_id, "error": "No text could be extracted from the document"}

        # Document classification - the second step in the pipeline
        task_logger.info("Starting document classification")
        classification_results = classify_document(extracted_text, top_k=1)

        if classification_results:
            doc_type = classification_results[0]['doc_type']
            confidence = classification_results[0]['score']

            task_logger.info("Document classified", doc_type=doc_type, confidence=confidence)

            # Update document status with the document type
            update_document_status(doc_id, f'classified_as_{doc_type}')
        else:
            task_logger.warning("Document classification failed")
            # Update document status to 'classification_failed' if needed
            update_document_status(doc_id, 'classification_failed')
            doc_type = 'unknown'

        # RAG indexing - the third step in the pipeline
        task_logger.info("Starting RAG indexing")
        indexing_success = index_document(
            doc_id=doc_id,
            text=extracted_text,
            doc_type=doc_type,
            metadata={'source_file': os.path.basename(filepath)}
        )

        if indexing_success:
            task_logger.info("Document indexed successfully")
            # Update document status to indicate successful indexing
            update_document_status(doc_id, f'indexed_as_{doc_type}')
        else:
            task_logger.error("Document indexing failed")
            # Update document status to indicate indexing failure
            update_document_status(doc_id, 'indexing_failed')

        # Entity extraction and summarization - the fourth step in the pipeline
        task_logger.info("Starting entity extraction and summarization")

        # Extract entities
        entities = extract_entities(extracted_text, doc_type)
        entities_success = True
        if 'error' not in entities:
            from app.database import update_document_entities
            entities_success = update_document_entities(doc_id, entities)
            if entities_success:
                task_logger.info("Entities extracted and stored successfully")
            else:
                task_logger.error("Failed to store entities in database")
        else:
            task_logger.error("Entity extraction failed", error=entities['error'])
            entities_success = False

        # Generate summary
        summary = generate_summary(extracted_text, doc_type)
        summary_success = True
        if 'error' not in summary:
            from app.database import update_document_summary
            summary_success = update_document_summary(doc_id, summary)
            if summary_success:
                task_logger.info("Summary generated and stored successfully")
            else:
                task_logger.error("Failed to store summary in database")
        else:
            task_logger.error("Summary generation failed", error=summary['error'])
            summary_success = False

        # Update document status to 'completed'
        update_document_status(doc_id, 'completed')

        task_logger.info("Document processing completed")
        return {
            "status": "completed",
            "doc_id": doc_id,
            "processed_file": filepath,
            "extracted_text_length": len(extracted_text),
            "doc_type": doc_type,
            "classification_confidence": classification_results[0]['score'] if classification_results else 0.0,
            "indexing_success": indexing_success,
            "entities_success": entities_success,
            "summary_success": summary_success
        }

    except Exception as e:
        task_logger.error("Error processing document", error=str(e))
        # Update document status to 'error'
        from app.database import update_document_status
        update_document_status(doc_id, 'error')

        # Update task state to reflect failure
        self.update_state(
            state='FAILURE',
            meta={'exc_type': type(e).__name__, 'exc_message': str(e)}
        )
        raise e

    finally:
        # Calculate duration and update metrics
        duration = (datetime.now() - start_time).total_seconds()
        DOCUMENT_PROCESSING_DURATION.observe(duration)
        DOCUMENT_PROCESSING_CURRENT.dec()

        task_logger.info("Processing completed", duration=duration)

        # Increment the total counter based on the result
        # This would be incremented in the returning part, but we'll increment it here for now
        # In a real implementation, this would be handled based on final status
        DOCUMENT_PROCESSING_TOTAL.labels(status="completed").inc()


if __name__ == '__main__':
    celery_app.start()