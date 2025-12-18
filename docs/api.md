# API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Most endpoints require authentication via JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <token>
```

## Endpoints

### Authentication

#### Register User
- **POST** `/auth/register`
- **Description**: Register a new user account
- **Request Body**:
```json
{
  "username": "string",
  "email": "user@example.com",
  "password": "securepassword"
}
```
- **Response**:
```json
{
  "id": "user_id",
  "username": "string",
  "email": "user@example.com"
}
```

#### Login User
- **POST** `/auth/login`
- **Description**: Authenticate and get JWT token
- **Request Body**:
```json
{
  "username": "string",
  "password": "securepassword"
}
```
- **Response**:
```json
{
  "access_token": "jwt_token",
  "token_type": "bearer",
  "user": {
    "id": "user_id",
    "username": "string",
    "email": "user@example.com"
  }
}
```

### Document Processing

#### Upload Document
- **POST** `/upload`
- **Description**: Upload a document for processing
- **Authentication**: Required
- **Request**: Multipart form data with file
- **Response**:
```json
{
  "doc_id": "uuid-string",
  "filename": "original_filename.pdf",
  "message": "Document uploaded successfully and processing has started."
}
```
- **Supported Formats**: PDF, images (JPEG, PNG), text files

#### Get Document Status
- **GET** `/status/{doc_id}`
- **Description**: Check the processing status of a document
- **Authentication**: Required
- **Path Parameters**:
  - `doc_id`: Document identifier
- **Response**:
```json
{
  "doc_id": "uuid-string",
  "filename": "filename.pdf",
  "status": "queued|processing_ocr|processing_classification|extracted|classified_as_invoice|completed|failed",
  "created_at": "2023-10-20T10:00:00Z",
  "updated_at": "2023-10-20T10:05:00Z"
}
```

### Document Query

#### Query Document
- **POST** `/query`
- **Description**: Ask questions about a document using RAG
- **Authentication**: Required
- **Request Body**:
```json
{
  "doc_id": "uuid-string",
  "question": "What is the total amount?"
}
```
- **Response**:
```json
{
  "doc_id": "uuid-string",
  "question": "What is the total amount?",
  "answer": "The total amount is $1,250.00."
}
```

#### Get Document Summary
- **GET** `/summary/{doc_id}`
- **Description**: Retrieve the summary of a processed document
- **Authentication**: Required
- **Path Parameters**:
  - `doc_id`: Document identifier
- **Response**:
```json
{
  "doc_id": "uuid-string",
  "summary": "This document is an invoice for consulting services..."
}
```

### Document Management

#### Delete Document
- **DELETE** `/documents/{doc_id}`
- **Description**: Delete a document and all associated data
- **Authentication**: Required
- **Path Parameters**:
  - `doc_id`: Document identifier
- **Response**:
```json
{
  "message": "Document deleted successfully",
  "doc_id": "uuid-string"
}
```

### System Endpoints

#### Health Check
- **GET** `/health`
- **Description**: Check system health
- **Authentication**: Not required
- **Response**:
```json
{
  "status": "healthy",
  "timestamp": 1697798400
}
```

#### Metrics
- **GET** `/metrics`
- **Description**: Prometheus metrics endpoint
- **Authentication**: Not required
- **Response**: Plain text metrics in Prometheus format

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message describing the issue"
}
```

### Common Status Codes

- `200`: Success for GET and PUT requests
- `201`: Resource created successfully (POST)
- `400`: Bad request - invalid input
- `401`: Unauthorized - authentication required
- `403`: Forbidden - insufficient permissions
- `404`: Not found - resource doesn't exist
- `422`: Unprocessable entity - validation error
- `500`: Internal server error

## Request Validation

All API requests are validated using Pydantic models:

- Document IDs: 1-255 characters, alphanumeric with hyphens/underscores
- Questions: 1-2000 characters
- File uploads: Maximum 50MB per file
- User credentials: Username 3-50 chars, password minimum 8 chars

## Example Usage

### Upload a document using curl

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/financial_document.pdf"
```

### Query a document

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "abc123-uuid-string",
    "question": "What is the invoice amount?"
  }'
```

## Rate Limiting

The API implements rate limiting to prevent abuse:
- 100 requests per minute per IP for unauthenticated endpoints
- 1000 requests per minute per user for authenticated endpoints

## File Upload Requirements

- Maximum file size: 50MB
- Supported formats: PDF, JPEG, PNG, TXT
- File names are sanitized to prevent path traversal attacks
- Each user can store up to 1GB of documents by default