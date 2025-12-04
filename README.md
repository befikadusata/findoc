# FinDocAI - Financial Document Processing System

An intelligent document processing system that extracts, analyzes, and answers questions about financial documents using modern LLM (Large Language Model) and RAG (Retrieval-Augmented Generation) architectures. Designed as a production-ready demonstration of AI engineering practices, focusing on financial document processing such as invoices, contracts, bank statements, and loan applications.

## Features

- **Document Ingestion**: Supports multiple document formats (PDF, images, text)
- **OCR & Text Extraction**: Hybrid approach using PyPDF2/pypdf and Tesseract OCR
- **Document Classification**: Transformer-based classification using DistilBERT
- **RAG Pipeline**: Context-aware question answering with ChromaDB vector storage
- **Entity Extraction**: Structured entity extraction using Gemini API
- **Document Summarization**: AI-powered document summarization
- **API Interface**: FastAPI web gateway with comprehensive endpoints
- **Async Processing**: Celery-based task queue for document processing
- **Observability**: Prometheus metrics and structured logging

## Architecture

### Tech Stack
- **Web Framework**: FastAPI
- **Task Queue**: Celery with Redis
- **Database**: PostgreSQL
- **Vector Store**: ChromaDB
- **ML Models**: Transformers, Sentence Transformers, PyTorch
- **OCR**: Pytesseract, pdf2image
- **Monitoring**: Prometheus, Grafana
- **Logging**: Structured logging with Structlog

### System Components
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   API Layer     │    │ Processing Layer │    │  Storage Layer  │
│                 │    │                  │    │                 │
│ • FastAPI       │    │ • OCR & Text     │    │ • PostgreSQL    │
│ • Health checks │    │   Extraction     │    │ • ChromaDB      │
│ • Doc Upload    │    │ • Classification │    │ • File Storage  │
│ • Query RAG     │    │ • RAG Pipeline   │    │                 │
└─────────────────┘    │ • Entity Extract │    └─────────────────┘
                       │ • Summarization  │
                       └──────────────────┘
```

## Installation & Setup

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- PostgreSQL (or run via Docker)
- Redis (or run via Docker)
- Google Gemini API key for LLM features

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/findocai.git
cd findocai
```

2. Set up the Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

4. Start the required services:
```bash
docker-compose up -d
```

5. Initialize the database:
```bash
python scripts/init_db.py
```

6. Start the FastAPI server:
```bash
uvicorn app.main:app --reload --port 8000
```

7. In a separate terminal, start the Celery worker:
```bash
celery -A app.worker worker --loglevel=info
```

## API Endpoints

### Document Upload
- `POST /upload` - Upload a financial document for processing
- Returns document ID and processing task ID

### Document Status
- `GET /status/{doc_id}` - Check processing status of a document

### Document Query
- `GET /query?doc_id=...&question=...` - Ask questions about a document using RAG

### Document Summary
- `GET /summary/{doc_id}` - Retrieve generated summary for a document

### Health & Metrics
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics endpoint

## Usage Examples

### Document Upload
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/financial_document.pdf"
```

### Query Document
```bash
curl -X GET "http://localhost:8000/query?doc_id=doc-123&question=What is the total amount?"
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Load Testing
```bash
# Run the load testing script
locust -f scripts/load_test.py
```

### Running with Monitoring
```bash
# Start all services including monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

## Project Structure

```
findocai/
├── app/                    # Main application code
│   ├── main.py            # FastAPI application entry point
│   ├── worker.py          # Celery worker implementation
│   ├── database.py        # PostgreSQL database module
│   ├── ingestion/         # OCR and text extraction modules
│   │   └── ocr.py
│   ├── classification/    # Document classification modules
│   │   └── model.py
│   ├── rag/              # RAG pipeline modules
│   │   └── pipeline.py
│   └── nlp/              # NLP modules
│       └── extraction.py
├── scripts/               # Utility and setup scripts
├── tests/                 # Test files
├── data/                  # Data directory
│   └── uploads/          # Document upload storage
├── docs/                 # Documentation files
├── docker-compose.yml    # Docker services definition
└── requirements.txt      # Python dependencies
```

## Configuration

The application can be configured using environment variables:

- `GEMINI_API_KEY` - Google Gemini API key for LLM features
- `DB_HOST` - PostgreSQL database host (default: localhost)
- `DB_PORT` - PostgreSQL database port (default: 5434)
- `DB_NAME` - PostgreSQL database name (default: findocai)
- `DB_USER` - PostgreSQL database user (default: findocai_user)
- `DB_PASSWORD` - PostgreSQL database password (default: findocai_password)
- `REDIS_HOST` - Redis host (default: localhost)
- `REDIS_PORT` - Redis port (default: 6380)

## Monitoring & Observability

The system includes comprehensive monitoring capabilities:
- **Metrics**: Prometheus metrics available at `/metrics`
- **Logging**: Structured JSON logs with contextual information
- **Health Checks**: `/health` endpoint for service health monitoring

## Production Considerations

For production deployment:
- Use environment-specific configurations
- Implement proper authentication and authorization
- Set up alerting based on metrics
- Use a production-grade PostgreSQL instance
- Implement proper backup strategies
- Set up SSL/TLS for API endpoints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## About

FinDocAI demonstrates modern AI engineering practices for processing financial documents using:
- Asynchronous processing pipelines
- Modern LLM integration
- Scalable architecture patterns
- Production-ready observability
- Comprehensive testing strategy