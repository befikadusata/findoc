# FinDocAI - Project Context and Development Guide

## Project Overview

FinDocAI is an intelligent document processing system that extracts, analyzes, and answers questions about financial documents using modern LLM (Large Language Model) and RAG (Retrieval-Augmented Generation) architectures. It is designed as a production-ready demonstration of AI engineering practices, focusing on financial document processing such as invoices, contracts, bank statements, and loan applications.

The system implements a modular, asynchronous pipeline architecture that includes:
- Document ingestion and processing
- OCR and text extraction
- Document classification
- Entity extraction and summarization
- Vector storage and retrieval (RAG)
- Question answering capabilities

### Key Features
- Production-ready MLOps pipeline with observability
- Real-time document processing with async task management using Celery
- Modern AI stack: RAG, embeddings, LLMs (Gemini API)
- Cloud-native design ready for AWS migration
- Comprehensive monitoring and error tracking
- Fintech-specific use cases (KYC, loan docs, statements)
- Structured logging with contextual information
- Advanced error handling with dead letter queues
- Parallel processing of NLP and RAG tasks
- **NEW: Explainable AI (XAI) with attention-based explanations**
- **NEW: Containerized deployment with Docker**
- **NEW: MLflow integration for experiment tracking**
- **NEW: CI/CD pipeline with GitHub Actions**

## Technical Architecture

The system follows a microservices-like architecture with these main components:

### API Layer
- FastAPI for the web gateway
- Async endpoints for document upload, querying, and status tracking
- Health checks and metrics exposure
- Input validation with Pydantic schemas
- **NEW: Explanations endpoints for XAI features**

### Processing Layer
- Message queue using Redis and Celery for task management
- OCR and text extraction pipeline (hybrid: PyPDF2/native PDF + Tesseract OCR)
- Document classification using transformer models (DistilBERT)
- Entity extraction and summarization using LLMs (Gemini API)
- Vector storage and retrieval for RAG functionality (ChromaDB)
- Chained and parallel task execution for efficient processing
- **NEW: Attention-based explanations for classification**
- **NEW: Confidence scoring and source attribution for RAG**

### Storage Layer
- PostgreSQL for metadata storage
- ChromaDB for vector embeddings storage
- Local filesystem for raw document storage
- JSON serialization for complex data structures
- MLflow backend for model experiment tracking

### Observability Stack
- Prometheus for metrics collection (custom metrics defined)
- Grafana for dashboards and visualization
- Structured logging throughout the application (using Structlog)
- Dead letter queue for failed tasks
- **NEW: MLflow for model and experiment tracking**

## Project Structure

```
findocai/
├── app/                    # Main application code
│   ├── main.py            # FastAPI application entry point
│   ├── worker.py          # Celery worker implementation with chained tasks
│   ├── database.py        # PostgreSQL database module
│   ├── config.py          # Application configuration using Pydantic Settings
│   ├── models.py          # Database models (if any)
│   ├── api/               # API-related modules
│   │   └── schemas/       # Request/response schemas
│   ├── classification/    # Document classification modules
│   │   └── model.py       # With attention-based XAI explanations
│   ├── ingestion/         # OCR and text extraction modules
│   │   └── ocr.py
│   ├── nlp/               # NLP modules
│   │   └── extraction.py
│   ├── rag/               # RAG pipeline modules
│   │   └── pipeline.py    # With confidence scoring and source attribution
│   ├── utils/             # Utility modules
│   │   └── logging_config.py # Structured logging configuration
│   │   └── attention_visualization.py # NEW: XAI visualization tools
│   └── __pycache__/       # Python cache
├── data/                  # Data directory
│   ├── uploads/           # Document upload storage
│   └── chroma/            # ChromaDB vector storage directory
├── docs/                 # Documentation files
│   ├── 00_introduction.md
│   ├── 01_system_architecture.md
│   ├── 02_implementation/
│   ├── 03_api_reference.md
│   ├── 04_operations/
│   ├── 05_project_context/
│   ├── 06_appendix/
│   ├── llm_providers/     # This documentation directory
│   ├── xai_implementation_plan.md  # NEW: XAI implementation details
│   └── project_archive/
├── monitoring/           # Monitoring stack configuration
│   ├── docker-compose.monitoring.yml
│   ├── grafana-dashboard.json
│   └── prometheus.yml
├── prompts/              # Prompt templates and management
│   ├── extraction/
│   ├── rag/
│   ├── summarization/
│   └── prompt_manager.py
├── scripts/              # Utility and setup scripts
│   ├── download_models.py
│   ├── health_check.py
│   ├── init_db.py
│   ├── load_test.py
│   ├── run_alembic_migration.py
│   ├── run.sh
│   ├── setup.sh
│   └── stop.sh
├── tests/                # Test files
│   ├── test_api_integration.py
│   ├── test_api_validation.py
│   ├── test_classification_model.py
│   ├── test_database.py
│   ├── test_deletion_functionality.py
│   ├── test_ingestion_ocr.py
│   ├── test_logging_improvements.py
│   ├── test_nlp_extraction.py
│   ├── test_prompt_manager.py
│   └── test_rag_pipeline.py
├── .github/workflows/    # NEW: CI/CD pipeline configuration
│   └── ci-cd.yml
├── alembic/              # Database migration files
├── alembic.ini
├── Dockerfile            # NEW: Containerization
├── docker-compose.yml    # NEW: Multi-service orchestration
├── docker-compose.monitoring.yml
├── pyproject.toml        # Python project configuration
├── requirements.txt      # Python dependencies (with MLflow, SHAP, Captum)
├── .env                  # Environment variables file
├── .git/                 # Git repository files
├── findocai.db           # Local SQLite database (alternative to PostgreSQL)
└── README.md             # Main project documentation
```

## Current Implementation State

The project is in an advanced state with most core features implemented:

### Implemented Features
- ✅ Document upload and processing API
- ✅ OCR and text extraction (hybrid approach)
- ✅ Document classification using transformer models
- ✅ RAG pipeline with ChromaDB vector storage
- ✅ Entity extraction and document summarization
- ✅ Asynchronous task processing with Celery
- ✅ PostgreSQL database integration
- ✅ Prometheus metrics and structured logging
- ✅ Input validation with Pydantic schemas
- ✅ Error handling and dead letter queue
- ✅ Parallel task execution for efficiency
- ✅ API endpoints for status, query, and deletion
- ✅ **NEW: Explainable AI (XAI) with attention-based explanations**
- ✅ **NEW: Containerization with Docker**
- ✅ **NEW: MLflow integration for experiment tracking**
- ✅ **NEW: CI/CD pipeline with GitHub Actions**

### Processing Pipeline
The document processing follows this chained workflow:
1. **Text Extraction**: Extract text from document using OCR/hybrid approach
2. **Classification**: Classify document type using transformer models with XAI explanations
3. **Parallel Processing**: Simultaneously:
   - Index document in ChromaDB (RAG) with confidence scoring
   - Extract entities using LLM
   - Generate document summary
4. **Completion**: Mark document as processed

### API Endpoints
- `POST /upload` - Upload a document for processing
- `GET /status/{doc_id}` - Check processing status of a document
- `GET /query?doc_id={doc_id}&question={question}&explain={true|false}` - Query a document using RAG with optional explanations
- `GET /summary/{doc_id}` - Retrieve document summary
- `GET /explain/classification/{doc_id}` - NEW: Retrieve XAI explanations for document classification
- `DELETE /documents/{doc_id}` - Delete a document and its data
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics endpoint

## Development Conventions

- Python code following PEP 8 standards
- Asynchronous processing for document handling
- Structured logging with contextual information using Structlog
- Modular architecture with clear separation of concerns
- Type hints for better code maintainability
- Comprehensive testing at all levels
- Git version control with clear commit messages
- Centralized configuration using Pydantic Settings
- Input validation using Pydantic schemas
- Error handling with dead letter queues for failed tasks
- **NEW: MLflow tracking for all model operations**
- **NEW: XAI data logging and visualization**

## Building and Running

The system is designed to be run locally with Docker for containerized services:

1. Set up Python virtual environment
2. Install dependencies (FastAPI, Uvicorn, Celery, Redis, etc.)
3. Configure environment variables (see `.env` file)
4. Start dependent services using Docker Compose
5. Initialize the database
6. Run the FastAPI server with Uvicorn
7. Start the Celery worker process separately

### Quick Start Commands:
```bash
# Install dependencies
pip install -r requirements.txt

# Start services (Redis, PostgreSQL, MLflow)
docker-compose up -d

# Initialize database
python scripts/init_db.py

# Start FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker (in separate terminal)
celery -A app.worker worker --loglevel=info
```

### Containerized Deployment:
```bash
# Build and start all services
docker-compose up --build -d

# View service logs
docker-compose logs -f
```

## Configuration

The application uses a centralized configuration system:

### Configurable Properties
- Redis connection settings (`redis_host`, `redis_port`)
- PostgreSQL database settings (`db_host`, `db_port`, `db_name`, `db_user`, `db_password`)
- Gemini API key (`gemini_api_key`)
- **NEW: MLflow tracking URI (`mlflow_tracking_uri`)**

### Environment Variables
- `GEMINI_API_KEY` - Google Gemini API key for LLM features
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` - Database credentials
- `REDIS_HOST`, `REDIS_PORT` - Redis connection settings
- `MLFLOW_TRACKING_URI` - URI for MLflow tracking server (default: http://localhost:5000)

## Testing

The project includes comprehensive tests covering:
- API integration and validation
- Classification model functionality
- Database operations
- Ingestion and OCR processes
- NLP extraction
- RAG pipeline
- Logging improvements
- Deletion functionality
- **NEW: XAI functionality tests**

Run tests using: `pytest tests/ -v`

## Monitoring and Operations

The system provides comprehensive monitoring capabilities:
- Custom Prometheus metrics (document processing, duration, current processing)
- Structured JSON logs with contextual information
- Dead letter queue for failed tasks
- Health check endpoint
- Performance metrics collection
- Operational scripts in the `scripts/` directory
- **NEW: MLflow for model and experiment tracking**

### MLflow Integration:
- Tracks document classification experiments with attention explanations
- Logs RAG query performance and confidence scores
- Records entity extraction and summarization results
- Captures model performance metrics over time

### CI/CD Pipeline:
- Automated testing with PostgreSQL and Redis services
- Docker image building and pushing
- Production deployment templates

## XAI (Explainable AI) Features

The system includes comprehensive XAI functionality:

### Classification Explanations
- Attention weight analysis for transformer models
- Identification of most influential tokens
- Confidence scoring with statistical measures
- Visualization of attention patterns

### RAG Explanations
- Confidence scoring based on similarity metrics
- Source attribution with chunk IDs and rankings
- Content preview for retrieved chunks
- Error attribution in case of failures

### Visualization Tools
- Attention weight visualizations
- HTML-based inline token highlighting
- Summary statistics for attention analysis

## Future Considerations

According to the architecture documentation, the current local/demo implementation uses open-source tools with clear production alternatives for cloud environments like AWS:
- API Gateway alternatives: AWS API Gateway + Lambda
- Task queue: AWS SQS + ECS Tasks
- OCR: AWS Textract
- Model hosting: SageMaker Endpoints
- Vector DB: Pinecone or OpenSearch
- Storage: S3 + Athena
- Database: Aurora PostgreSQL
- **NEW: MLOps: AWS SageMaker Experiments and Model Registry**