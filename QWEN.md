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
- Modern AI stack: RAG, embeddings, LLMs (Gemini/Mistral)
- Cloud-native design ready for AWS migration
- Comprehensive monitoring and error tracking
- Fintech-specific use cases (KYC, loan docs, statements)

## Technical Architecture

The system follows a microservices-like architecture with these main components:

### API Layer
- FastAPI for the web gateway
- Async endpoints for document upload, querying, and status tracking
- Health checks and metrics exposure

### Processing Layer
- Message queue using Redis and Celery for task management
- OCR and text extraction pipeline
- Document classification using transformer models
- Entity extraction and summarization using LLMs
- Vector storage and retrieval for RAG functionality

### Storage Layer
- SQLite for metadata storage
- ChromaDB for vector embeddings storage
- Local filesystem for raw document storage

### Observability Stack
- Prometheus for metrics collection
- Grafana for dashboards and visualization
- Structured logging throughout the application

## Project Structure

```
findocai/
├── app/                    # Main application code
│   ├── main.py            # FastAPI application entry point
│   ├── worker.py          # Celery worker implementation
│   ├── database.py        # Database module
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
├── pyproject.toml        # Python project configuration
├── requirements.txt      # Python dependencies
├── docker-compose.yml    # Docker services definition
└── TODO.md              # Implementation roadmap
```

## Implementation Plan

The project follows a phased implementation approach:

### Phase 1: Project Setup & Core Infrastructure
- Project structure initialization
- Python environment setup
- Dependency installation
- FastAPI application creation
- Docker setup for Redis

### Phase 2: Document Ingestion & Processing Pipeline
- Document upload API implementation
- Celery worker setup
- Metadata and status tracking
- OCR and text extraction functionality

### Phase 3: AI/ML Model Integration
- Document classification
- RAG - chunking and embedding
- RAG - retrieval and generation
- LLM-based entity extraction and summarization

### Phase 4: Observability & Testing
- Metrics implementation
- Monitoring stack setup
- Structured logging
- Test suite development

### Phase 5: Documentation & Finalization
- Project documentation completion
- Helper scripts finalization
- Code cleanup and review

## Development Conventions

- Python code following PEP 8 standards
- Asynchronous processing for document handling
- Structured logging with contextual information
- Modular architecture with clear separation of concerns
- Type hints for better code maintainability
- Comprehensive testing at all levels
- Git version control with clear commit messages

## Building and Running

The system is designed to be run locally with Docker for containerized services. The implementation plan indicates:

1. Set up Python virtual environment
2. Install dependencies (FastAPI, Uvicorn, Celery, Redis, etc.)
3. Run the FastAPI server with Uvicorn
4. Start the Celery worker process
5. Launch Redis using Docker Compose
6. Access the API endpoints for document processing

For the complete setup and run instructions, refer to the TODO.md file which contains the step-by-step implementation plan.

## Future Considerations

According to the architecture documentation, the local/demo implementation uses open-source tools with clear production alternatives for cloud environments like AWS:
- API Gateway alternatives: AWS API Gateway + Lambda
- Task queue: AWS SQS + ECS Tasks
- OCR: AWS Textract
- Model hosting: SageMaker Endpoints
- Vector DB: Pinecone or OpenSearch
- Storage: S3 + Athena
- Database: Aurora PostgreSQL