# Document Ingestion and Processing

This section covers the entry point for documents into the FinDocAI system and how they are queued for asynchronous processing.

## 2.1 Document Ingestion API

The ingestion API is the primary entry point for submitting new documents. It's designed to be non-blocking, immediately accepting the file, storing it, and queueing it for processing without making the client wait for the entire pipeline to complete.

**Endpoint:** `POST /upload`

```python
# app/main.py
from fastapi import FastAPI, UploadFile, BackgroundTasks
from app.worker import process_document
import uuid

app = FastAPI(title="FinDocAI")

@app.post("/upload")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    doc_id = str(uuid.uuid4())
    filepath = f"./data/uploads/{doc_id}_{file.filename}"
    
    # Save file
    with open(filepath, "wb") as f:
        f.write(await file.read())
    
    # Store metadata
    db.insert({
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "queued",
        "uploaded_at": datetime.utcnow()
    })
    
    # Queue async processing
    process_document.delay(doc_id, filepath)
    
    return {"doc_id": doc_id, "status": "processing"}
```

**Design Decisions:**
- **Asynchronous Handling:** Uses `async` and `await` for non-blocking I/O, and `BackgroundTasks` (delegated to Celery) to offload heavy processing.
- **Unique IDs:** Assigns a `UUID` to each document to prevent naming collisions and provide a reliable identifier for tracking.
- **Immediate Feedback:** Returns a `doc_id` and `processing` status to the client instantly.
- **Status Tracking:** The document's state is tracked in a database, transitioning from `queued` → `processing` → `completed`/`failed`.

## 2.2 Async Task Processing

Celery is used to manage the background processing of documents. This decouples the web server from the compute-intensive tasks, ensuring the API remains responsive.

```python
# app/worker.py
from celery import Celery
from prometheus_client import Histogram

celery_app = Celery('findocai', broker='redis://localhost:6379/0')

# Metrics
processing_time = Histogram(
    'document_processing_seconds',
    'Time to process document',
    ['doc_type']
)

@celery_app.task(bind=True, max_retries=3)
def process_document(self, doc_id: str, filepath: str):
    try:
        with processing_time.labels(doc_type='unknown').time():
            # 1. Extract text
            text = extract_text(filepath)
            
            # 2. Classify
            doc_type = classify_document(text)['doc_type']
            
            # 3. Extract entities
            entities = extract_entities(text, doc_type)
            
            # 4. Summarize
            summary = generate_summary(text, doc_type)
            
            # 5. Index for RAG
            index_document(doc_id, text)
            
            # 6. Update DB
            db.update(doc_id, {
                "status": "completed",
                "doc_type": doc_type,
                "entities": entities,
                "summary": summary,
                "processed_at": datetime.utcnow()
            })
            
    except Exception as e:
        db.update(doc_id, {"status": "failed", "error": str(e)})
        self.retry(exc=e, countdown=60)  # Retry after 1 min
```

**Best Practices Implemented:**
- **Idempotency:** Tasks are designed to be safely retried without causing unintended side effects.
- **Automatic Retries:** Failed tasks are automatically retried with an exponential backoff strategy to handle transient errors.
- **Error Handling:** Exceptions are caught, and the document status is updated to `failed` in the database.
- **Monitoring:** The `processing_time` histogram captures execution duration for observability.
- **Scalability:** Multiple Celery workers can be run concurrently to process documents in parallel.
- **Future Enhancements:** Could be extended with a dead-letter queue for tasks that repeatedly fail and priority queues for urgent documents.
