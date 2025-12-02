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

A successful request immediately returns a JSON object with the document's unique ID and its initial status.

```json
{
  "doc_id": "a3f2c1b0-f5a8-4c3e-b4de-1ad6b1b7c8d0",
  "status": "processing"
}
```

### GET `/status/{doc_id}`

Check the processing status and retrieve the results for a specific document.

**Request:**

```bash
curl http://localhost:8000/status/a3f2c1b0-f5a8-4c3e-b4de-1ad6b1b7c8d0
```

**Response:**

Once processing is complete, the response contains the document's status, type, and the extracted entities.

```json
{
  "doc_id": "a3f2c1b0-f5a8-4c3e-b4de-1ad6b1b7c8d0",
  "status": "completed",
  "doc_type": "invoice",
  "processed_at": "2025-12-02T10:30:00Z",
  "entities": {
    "amount": 1250.00,
    "currency": "USD",
    "date": "2025-11-15",
    "parties": ["ABC Corp", "XYZ Supplier"],
    "account_number": null,
    "terms": "Net 30"
  }
}
```

### GET `/query`

Ask a natural language question about a specific document using the RAG pipeline.

**Request:**

The request requires the `doc_id` and the question `q` as query parameters.

```bash
curl "http://localhost:8000/query?doc_id=a3f2c1b0-f5a8-4c3e-b4de-1ad6b1b7c8d0&q=What%20is%20the%20total%20amount?"
```

**Response:**

The API returns the question, the LLM-generated answer, a confidence score, and the sources (document chunks) used to generate the answer.

```json
{
  "question": "What is the total amount?",
  "answer": "The total amount is $1,250.00 USD.",
  "confidence": 0.94,
  "sources": ["chunk_0", "chunk_3"]
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
  "summary": "This is an invoice from XYZ Supplier to ABC Corp for the amount of $1,250, dated November 15, 2025. The services provided include consulting and software licenses, with payment terms of Net 30.",
  "key_points": [
    "Total Amount: $1,250.00",
    "Due Date: December 15, 2025",
    "Vendor: XYZ Supplier"
  ]
}
```
