# System Architecture

## High-Level Design

The FinDocAI system is designed as a modular, asynchronous pipeline to handle the ingestion, processing, and querying of financial documents efficiently.

```
┌─────────────────┐
│  User/Client    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│           FastAPI Gateway (Async)                │
│  ┌──────────────┐  ┌──────────────────────┐    │
│  │ /upload      │  │ /query  /status      │    │
│  │ /summary     │  │ /metrics /health     │    │
│  └──────────────┘  └──────────────────────┘    │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│      Message Queue Layer (Redis + Celery)        │
│  ┌─────────────────────────────────────────┐   │
│  │  Task Queue: process_document()          │   │
│  │  • OCR & Text Extraction                 │   │
│  │  • Classification                        │   │
│  │  • Entity Extraction                     │   │
│  │  • Chunking & Embedding                  │   │
│  └─────────────────────────────────────────┘   │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────┐
│              Processing Pipeline                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Tesseract│→ │Classifier│→ │ LLM (Gemini/     │  │
│  │ PyPDF2   │  │DistilBERT│  │ Mistral-Ollama)  │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
└─────────┬────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────┐
│              Storage & Retrieval Layer                │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │  SQLite     │  │  Chroma DB  │  │ Local FS   │  │
│  │  (metadata) │  │  (vectors)  │  │ (raw docs) │  │
│  └─────────────┘  └─────────────┘  └────────────┘  │
└──────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────┐
│       Observability Stack (Prometheus + Grafana)      │
│  • Processing latency  • API throughput              │
│  • Error rates         • LLM token usage             │
│  • Queue depth         • Model performance           │
└──────────────────────────────────────────────────────┘
```

## Technology Stack

The technology stack is chosen to be modern, scalable, and adaptable. The local/demo implementation uses open-source tools, with clear production alternatives for a seamless transition to a cloud environment like AWS.

| Component | Technology | Production Alternative (Kifiya) |
|-----------|-----------|----------------------------------|
| **API Gateway** | FastAPI + Uvicorn | AWS API Gateway + Lambda |
| **Task Queue** | Celery + Redis | AWS SQS + ECS Tasks |
| **OCR** | Tesseract + PyPDF2 | AWS Textract |
| **ML Models** | HuggingFace Transformers | SageMaker Endpoints |
| **LLM** | Gemini API / Ollama (Mistral-7B) | Bedrock (Claude) / SageMaker |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | SageMaker Feature Store |
| **Vector DB** | Chroma (persistent) | Pinecone on AWS / OpenSearch |
| **Metadata Store** | PostgreSQL or other open-source tools | Aurora PostgreSQL |
| **File Storage** | Local filesystem other open-source tools similar with s3 | S3 + Athena |
| **Monitoring** | Prometheus + Grafana | CloudWatch + Custom Dashboards |
| **Infrastructure** | Docker Compose | EKS + Terraform |
