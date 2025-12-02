from celery import Celery
import os

# Import the OCR module for text extraction
from app.ingestion.ocr import extract_text

# Import the classification module
from app.classification.model import classify_document

# Import the RAG pipeline for indexing
from app.rag.pipeline import index_document

# Import the NLP module for entity extraction and summarization
from app.nlp.extraction import extract_entities, generate_summary

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
    followed by document classification, vector indexing, entity extraction,
    and summarization.
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

        # Document classification - the second step in the pipeline
        print(f"Starting document classification for document {doc_id}")
        classification_results = classify_document(extracted_text, top_k=1)

        if classification_results:
            doc_type = classification_results[0]['doc_type']
            confidence = classification_results[0]['score']

            print(f"Document {doc_id} classified as: {doc_type} with confidence: {confidence:.2f}")

            # Update document status with the document type
            update_document_status(doc_id, f'classified_as_{doc_type}')

            # Store the classification result in the database or pass it forward
            # For now, we'll just log it; in a real system you might extend the database schema
        else:
            print(f"Could not classify document {doc_id}")
            # Update document status to 'classification_failed' if needed
            update_document_status(doc_id, 'classification_failed')
            doc_type = 'unknown'

        # RAG indexing - the third step in the pipeline
        print(f"Starting RAG indexing for document {doc_id}")
        indexing_success = index_document(
            doc_id=doc_id,
            text=extracted_text,
            doc_type=doc_type,
            metadata={'source_file': os.path.basename(filepath)}
        )

        if indexing_success:
            print(f"Successfully indexed document {doc_id} in vector database")
            # Update document status to indicate successful indexing
            update_document_status(doc_id, f'indexed_as_{doc_type}')
        else:
            print(f"Failed to index document {doc_id} in vector database")
            # Update document status to indicate indexing failure
            update_document_status(doc_id, 'indexing_failed')

        # Entity extraction and summarization - the fourth step in the pipeline
        print(f"Starting entity extraction and summarization for document {doc_id}")

        # Extract entities
        entities = extract_entities(extracted_text, doc_type)
        entities_success = True
        if 'error' not in entities:
            from app.database import update_document_entities
            entities_success = update_document_entities(doc_id, entities)
            if entities_success:
                print(f"Successfully stored entities for document {doc_id}")
            else:
                print(f"Failed to store entities for document {doc_id}")
        else:
            print(f"Entity extraction failed for document {doc_id}: {entities['error']}")
            entities_success = False

        # Generate summary
        summary = generate_summary(extracted_text, doc_type)
        summary_success = True
        if 'error' not in summary:
            from app.database import update_document_summary
            summary_success = update_document_summary(doc_id, summary)
            if summary_success:
                print(f"Successfully stored summary for document {doc_id}")
            else:
                print(f"Failed to store summary for document {doc_id}")
        else:
            print(f"Summary generation failed for document {doc_id}: {summary['error']}")
            summary_success = False

        # Update document status to 'completed'
        update_document_status(doc_id, 'completed')

        print(f"Completed processing for document {doc_id}")
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