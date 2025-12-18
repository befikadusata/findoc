# Demo Guide

## Quick Start: Running the Demo in 30 Seconds

This guide will help you get FinDocAI up and running quickly to demonstrate its capabilities.

### Prerequisites

- **Docker** and **Docker Compose** installed
- **Python 3.10+** (for API interaction)
- **Google Gemini API key** (optional for full functionality)

### Step 1: Clone and Setup (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/findocai.git
cd findocai

# Set up environment variables
cp .env.example .env
# Edit .env to add your GEMINI_API_KEY if you have one
```

### Step 2: Start Services (3 minutes)

```bash
# Start all services using Docker Compose
docker-compose up -d

# This starts:
# - PostgreSQL database
# - Redis for task queue
# - MLflow for experiment tracking
# - API service
# - Worker service
```

### Step 3: Verify Services (1 minute)

```bash
# Check that all services are running
docker-compose ps

# Verify the API is accessible
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": 1697798400
}
```

### Step 4: Register and Authenticate (2 minutes)

```bash
# Register a new user
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_user",
    "email": "demo@example.com",
    "password": "demopassword123"
  }'

# Login to get JWT token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "demo_user",
    "password": "demopassword123"
  }'
```

Save the returned token for subsequent API calls.

### Step 5: Process Your First Document (5 minutes)

```bash
# Upload a document (replace YOUR_TOKEN and path to your file)
curl -X POST "http://localhost:8000/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./sample_documents/invoice_sample.pdf"
```

Expected response:
```json
{
  "doc_id": "abc123-def456-ghi789",
  "filename": "invoice_sample.pdf",
  "message": "Document uploaded successfully and processing has started."
}
```

### Step 6: Monitor Processing (1 minute)

```bash
# Check the document status (replace with the doc_id from previous step)
curl -X GET "http://localhost:8000/status/abc123-def456-ghi789" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Keep checking until status becomes "completed" (this typically takes 1-3 minutes).

### Step 7: Query Your Document (1 minute)

```bash
# Ask questions about your document
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "doc_id": "abc123-def456-ghi789",
    "question": "What is the total amount in this document?"
  }'
```

### Step 8: Get Document Summary (30 seconds)

```bash
# Retrieve the auto-generated summary
curl -X GET "http://localhost:8000/summary/abc123-def456-ghi789" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Demo Scenarios

### Scenario 1: Financial Document Analysis
1. Upload an invoice or financial statement
2. Ask questions like "What is the total amount?" or "What is the due date?"
3. Review the generated summary

### Scenario 2: Contract Review
1. Upload a contract document
2. Ask questions like "What are the key terms?" or "What is the termination clause?"
3. See how the system identifies important contract elements

### Scenario 3: Bank Statement Processing
1. Upload a bank statement
2. Ask questions about transactions or balances
3. Observe the RAG system retrieving relevant information

## Alternative: Running Locally (Without Docker)

If you prefer to run the application locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your-api-key"
# Set other variables as needed

# Initialize database
python scripts/init_db.py

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In a separate terminal, start the worker
celery -A app.worker worker --loglevel=info
```

## Monitoring the Demo

Monitor the system's performance and metrics:

```bash
# View metrics at: http://localhost:8000/metrics
open http://localhost:8000/metrics

# View Docker logs
docker-compose logs -f
```

## Troubleshooting

### Common Issues:

1. **Port already in use**: The demo requires ports 8000, 5434, 6380. Ensure they're available.

2. **API key missing**: If you don't have a Gemini API key, the system will work but responses will be simpler.

3. **Redis connection errors**: Ensure Docker Compose services are running: `docker-compose ps`

4. **Document processing stuck**: Check worker logs: `docker-compose logs findocai-worker`

### Performance Metrics to Highlight:
- Document processing typically takes 30-120 seconds depending on document size
- Query response times of <1 second once documents are processed
- System can handle multiple concurrent document uploads

## Demo Tips for Presentations

1. **Prepare documents in advance**: Have sample financial documents ready in your `sample_documents/` folder
2. **Show the processing pipeline**: Explain how OCR, classification, and RAG work together
3. **Highlight security**: Mention JWT authentication and secure file handling
4. **Show scalability**: Point out the async processing with Celery and Redis
5. **Emphasize explainability**: If using XAI features, show how the system explains its decisions

## Stopping the Demo

When finished:

```bash
# Stop all services
docker-compose down

# Optionally, remove volumes (this WILL delete all data)
docker-compose down -v
```