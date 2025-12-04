# API Reference

The FinDocAI API provides endpoints for uploading documents, checking their status, and querying their content.

### POST `/upload`

Upload a document for asynchronous processing.

**Request:**

The request should be a `multipart/form-data` payload with a `file` field containing the document.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/invoice.pdf"
```

**Response:**

A successful request immediately returns a JSON object with the document's unique ID, filename, file path, and task ID.

```json
{
  "doc_id": "a3f2c1b0-f5a8-4c3e-b4de-1ad6b1b7c8d0",
  "filename": "invoice.pdf",
  "file_path": "./data/uploads/a3f2c1b0-f5a8-4c3e-b4de-1ad6b1b7c8d0_invoice.pdf",
  "task_id": "task-abc123def456",
  "message": "Document uploaded successfully and processing started"
}
```

### GET `/status/{doc_id}`

Check the processing status and retrieve the results for a specific document.

**Request:**

```bash
curl http://localhost:8000/status/a3f2c1b0-f5a8-4c3e-b4de-1ad6b1b7c8d0
```

**Response:**

The response contains the document's metadata and status.

```json
{
  "doc_id": "a3f2c1b0-f5a8-4c3e-b4de-1ad6b1b7c8d0",
  "filename": "invoice.pdf",
  "status": "completed",
  "created_at": "2025-12-03 10:30:00",
  "updated_at": "2025-12-03 10:35:00"
}
```

**Error Response:**

If the document is not found, an error response is returned.

```json
{
  "error": "Document not found",
  "doc_id": "nonexistent-id"
}
}
```

### GET `/query`

Ask a natural language question about a specific document using the RAG pipeline.

**Request:**

The request requires the `doc_id` and the question as query parameters.

```bash
curl "http://localhost:8000/query?doc_id=a3f2c1b0-f5a8-4c3e-b4de-1ad6b1b7c8d0&question=What%20is%20the%20total%20amount?"
```

**Response:**

The API returns the document ID, question, and LLM-generated answer.

```json
{
  "doc_id": "a3f2c1b0-f5a8-4c3e-b4de-1ad6b1b7c8d0",
  "question": "What is the total amount?",
  "answer": "The total amount is $1,250.00 USD."
}
```

**Error Response:**

If the document is not found, an error response is returned.

```json
{
  "error": "Document not found",
  "doc_id": "nonexistent-id"
}
```

### GET `/summary/{doc_id}`

Retrieve a concise, LLM-generated summary and key points from the document.

**Request:**

```bash
curl http://localhost:8000/summary/a3f2c1b0-f5a8-4c3e-b4de-1ad6b1b7c8d0
```

**Response:**

```json
{
  "doc_id": "a3f2c1b0-f5a8-4c3e-b4de-1ad6b1b7c8d0",
  "summary": {
    "summary": "This is an invoice from XYZ Supplier to ABC Corp for the amount of $1,250, dated November 15, 2025...",
    "key_points": [
      "Total Amount: $1,250.00",
      "Due Date: December 15, 2025",
      "Vendor: XYZ Supplier"
    ],
    "document_type": "invoice",
    "document_date": "2025-11-15",
    "extracted_entities": {
      "invoice_number": "INV-12345",
      "total_amount": 1250.0
    }
  }
}
```

**Error Response:**

If the document is not found, an error response is returned.

```json
{
  "error": "Document not found",
  "doc_id": "nonexistent-id"
}
```

**Or if summary is not available:**

```json
{
  "error": "Summary not available",
  "doc_id": "unprocessed-document-id"
}
```
