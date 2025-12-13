from celery import Celery, chain, group
from celery.exceptions import MaxRetriesExceededError
import os
from datetime import datetime
from typing import Any, Dict, List

from kombu import Exchange, Queue

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

# Import centralized settings
from app.config import settings

# Import MLflow
import mlflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# --- Metrics Definition ---
DOCUMENT_PROCESSING_TOTAL = Counter('document_processing_total', 'Total number of processed documents', ['status'])
DOCUMENT_PROCESSING_DURATION = Histogram('document_processing_duration_seconds', 'Time spent processing documents')
DOCUMENT_PROCESSING_CURRENT = Gauge('document_processing_current', 'Number of currently processing documents')
DEAD_LETTER_TOTAL = Counter('dead_letter_tasks_total', 'Total number of tasks sent to dead letter queue')

# --- Celery and Queue Configuration ---
celery_app = Celery('findocai', broker=settings.redis_url, backend=settings.redis_url)

default_exchange = Exchange('default', type='direct')
dead_letter_exchange = Exchange('dead_letter', type='direct')

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    result_expires=3600,
    task_acks_late=True, # Acknowledge task after it completes
    task_reject_on_worker_lost=True, # Requeue task if worker is lost
    task_queues=(
        Queue('default', default_exchange, routing_key='default'),
        Queue('dead_letter', dead_letter_exchange, routing_key='dead_letter')
    ),
    task_routes={
        'app.worker.handle_task_failure': {'queue': 'dead_letter'},
    }
)

# --- Failure Handling Task ---
@celery_app.task(name='app.worker.handle_task_failure')
def handle_task_failure(request, exc, traceback, doc_id, filepath):
    """Logs and handles tasks that have failed permanently."""
    logger = get_logger("worker.handle_task_failure").bind(doc_id=doc_id, task_id=request.id)
    logger.error(
        "Task failed permanently and sent to dead letter queue.",
        exc=str(exc),
        traceback=traceback,
        filepath=filepath
    )
    from app.database import update_document_status
    update_document_status(doc_id, 'failed')
    DEAD_LETTER_TOTAL.inc()
    DOCUMENT_PROCESSING_TOTAL.labels(status='failed').inc()
    DOCUMENT_PROCESSING_CURRENT.dec()

# --- Chained Tasks ---
@celery_app.task(bind=True, autoretry_for=(Exception,), max_retries=3, retry_backoff=True, name='app.worker.task_extract_text')
def task_extract_text(self, doc_id: str, filepath: str) -> Dict[str, Any]:
    """Task 1: Extracts text from the document."""
    logger = get_logger("worker.task_extract_text").bind(doc_id=doc_id)
    logger.info("Starting OCR and text extraction")
    from app.database import update_document_status
    update_document_status(doc_id, 'processing_ocr')

    # Start MLflow run for this task
    with mlflow.start_run(run_name=f"ocr_extraction_{doc_id}"):
        extracted_text = extract_text(filepath)
        if not extracted_text.strip():
            raise ValueError("No text could be extracted from the document")

        # Log parameters and metrics to MLflow
        mlflow.log_param("document_id", doc_id)
        mlflow.log_param("file_path", filepath)
        mlflow.log_param("file_size", os.path.getsize(filepath))
        mlflow.log_metric("text_length", len(extracted_text))

        logger.info("Text extraction successful", text_length=len(extracted_text))
        update_document_status(doc_id, 'extracted')
        return {'text': extracted_text, 'doc_id': doc_id, 'filepath': filepath}

@celery_app.task(bind=True, autoretry_for=(Exception,), max_retries=3, retry_backoff=True, name='app.worker.task_classify_document')
def task_classify_document(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Task 2: Classifies the document based on extracted text."""
    doc_id, text = data['doc_id'], data['text']
    logger = get_logger("worker.task_classify_document").bind(doc_id=doc_id)
    logger.info("Starting document classification")
    from app.database import update_document_status
    update_document_status(doc_id, 'processing_classification')

    # Start MLflow run for this task
    with mlflow.start_run(run_name=f"document_classification_{doc_id}"):
        results = classify_document(text, top_k=1, include_explanation=True)
        doc_type = results[0]['doc_type'] if results else 'unknown'

        # Log parameters and metrics to MLflow
        mlflow.log_param("document_id", doc_id)
        mlflow.log_param("document_type", doc_type)
        mlflow.log_param("text_length", len(text))
        mlflow.log_metric("confidence_score", results[0]['confidence'] if results else 0.0)

        # Log explanation data if available
        explanation = results[0].get('explanation', {})
        if 'top_influential_tokens' in explanation:
            # Log top tokens for analysis
            top_tokens = [token_info['token'] for token_info in explanation['top_influential_tokens'][:5]]
            mlflow.log_param("top_influential_tokens", ", ".join(top_tokens))
            mlflow.log_metric("avg_attention", explanation.get('average_attention', 0.0))

        mlflow.log_artifact("app/classification/model.py", "model_source")  # Log model source

        logger.info("Document classified", doc_type=doc_type)
        update_document_status(doc_id, f'classified_as_{doc_type}')
        data['doc_type'] = doc_type

        # Add explanation data to the result
        if results and len(results) > 0:
            data['classification_explanation'] = results[0].get('explanation', None)

        return data

@celery_app.task(bind=True, name='app.worker.task_run_nlp_and_rag')
def task_run_nlp_and_rag(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Task 3: Orchestrates parallel NLP and RAG tasks."""
    doc_id, doc_type, text, filepath = data['doc_id'], data['doc_type'], data['text'], data['filepath']
    logger = get_logger("worker.task_run_nlp_and_rag").bind(doc_id=doc_id)
    logger.info("Starting parallel NLP and RAG processing.")

    # Log to MLflow about the parallel processing
    with mlflow.start_run(run_name=f"parallel_processing_{doc_id}"):
        mlflow.log_param("document_id", doc_id)
        mlflow.log_param("document_type", doc_type)
        mlflow.log_param("text_length", len(text))
        
        parallel_tasks = group(
            task_index_document.s(text, doc_type, doc_id, filepath),
            task_extract_entities.s(text, doc_type, doc_id),
            task_generate_summary.s(text, doc_type, doc_id)
        )
        parallel_tasks.apply_async()
        return data

@celery_app.task(autoretry_for=(Exception,), max_retries=3, retry_backoff=True, name='app.worker.task_index_document')
def task_index_document(text: str, doc_type: str, doc_id: str, filepath: str) -> bool:
    """Sub-task: Indexes document for RAG."""
    logger = get_logger("worker.task_index_document").bind(doc_id=doc_id)
    logger.info("Starting RAG indexing")
    
    # Track this indexing task with MLflow
    with mlflow.start_run(run_name=f"rag_indexing_{doc_id}"):
        success = index_document(doc_id=doc_id, text=text, doc_type=doc_type, metadata={'source_file': os.path.basename(filepath)})
        
        # Log parameters and metrics
        mlflow.log_param("document_id", doc_id)
        mlflow.log_param("document_type", doc_type)
        mlflow.log_param("index_success", success)
        mlflow.log_metric("text_length", len(text))
        
        if not success:
            logger.error("Document indexing failed")
        return success

@celery_app.task(autoretry_for=(Exception,), max_retries=3, retry_backoff=True, name='app.worker.task_extract_entities')
def task_extract_entities(text: str, doc_type: str, doc_id: str) -> bool:
    """Sub-task: Extracts entities from text."""
    logger = get_logger("worker.task_extract_entities").bind(doc_id=doc_id)
    logger.info("Starting entity extraction")
    from app.database import update_document_entities
    entities = extract_entities(text, doc_type)
    if 'error' in entities:
        logger.error("Entity extraction failed", error=entities['error'])
        return False
    
    # Track entity extraction with MLflow
    with mlflow.start_run(run_name=f"entity_extraction_{doc_id}"):
        mlflow.log_param("document_id", doc_id)
        mlflow.log_param("document_type", doc_type)
        mlflow.log_param("text_length", len(text))
        if 'entities' in entities and isinstance(entities['entities'], dict):
            mlflow.log_metric("entities_found", len(entities['entities']))
        
        success = update_document_entities(doc_id, entities)
        mlflow.log_param("update_success", success)
        return success

@celery_app.task(autoretry_for=(Exception,), max_retries=3, retry_backoff=True, name='app.worker.task_generate_summary')
def task_generate_summary(text: str, doc_type: str, doc_id: str) -> bool:
    """Sub-task: Generates a summary of the text."""
    logger = get_logger("worker.task_generate_summary").bind(doc_id=doc_id)
    logger.info("Starting summary generation")
    from app.database import update_document_summary
    summary = generate_summary(text, doc_type)
    if 'error' in summary:
        logger.error("Summary generation failed", error=summary['error'])
        return False
    
    # Track summary generation with MLflow
    with mlflow.start_run(run_name=f"summary_generation_{doc_id}"):
        mlflow.log_param("document_id", doc_id)
        mlflow.log_param("document_type", doc_type)
        mlflow.log_param("text_length", len(text))
        mlflow.log_metric("summary_length", len(summary) if isinstance(summary, str) else len(str(summary)))
        
        success = update_document_summary(doc_id, summary)
        mlflow.log_param("update_success", success)
        return success

@celery_app.task(bind=True, name='app.worker.task_finalize_processing')
def task_finalize_processing(self, data: Dict[str, Any], start_time_str: str):
    """Final Task: Marks processing as complete and records metrics."""
    doc_id = data['doc_id']
    logger = get_logger("worker.task_finalize_processing").bind(doc_id=doc_id)

    from app.database import update_document_status
    update_document_status(doc_id, 'completed')
    logger.info("Document processing completed successfully.")

    start_time = datetime.fromisoformat(start_time_str)
    duration = (datetime.now() - start_time).total_seconds()
    
    # Log final metrics to MLflow
    with mlflow.start_run(run_name=f"final_processing_{doc_id}"):
        mlflow.log_param("document_id", doc_id)
        mlflow.log_metric("total_processing_time", duration)
        mlflow.log_metric("processing_status", "completed")
    
    DOCUMENT_PROCESSING_DURATION.observe(duration)
    DOCUMENT_PROCESSING_CURRENT.dec()
    DOCUMENT_PROCESSING_TOTAL.labels(status='completed').inc()

# --- Main Orchestrator Task ---
@celery_app.task(bind=True, name='process_document')
def process_document(self, doc_id: str, filepath: str):
    """
    Orchestrates the document processing pipeline as a chain of tasks.
    """
    task_logger = get_logger("worker.process_document").bind(doc_id=doc_id)
    task_logger.info("Initializing document processing pipeline.")

    # Log the entire document processing job to MLflow
    with mlflow.start_run(run_name=f"document_processing_{doc_id}"):
        mlflow.log_param("document_id", doc_id)
        mlflow.log_param("file_path", filepath)
        mlflow.log_param("file_size", os.path.getsize(filepath))
        mlflow.log_param("processing_start_time", datetime.now().isoformat())
        
        from app.database import update_document_status
        update_document_status(doc_id, 'queued')

        DOCUMENT_PROCESSING_CURRENT.inc()
        start_time_iso = datetime.now().isoformat()

        # Define the workflow as a chain of tasks
        workflow = chain(
            task_extract_text.s(doc_id=doc_id, filepath=filepath),
            task_classify_document.s(),
            task_run_nlp_and_rag.s(),
            task_finalize_processing.s(start_time_str=start_time_iso)
        )

        # Define error handling for the entire chain
        workflow.link_error(handle_task_failure.s(doc_id=doc_id, filepath=filepath))

        # Execute the workflow
        workflow.apply_async()

if __name__ == '__main__':
    celery_app.start()