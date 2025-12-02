# FinDocAI: Implementation TODO

This document outlines the step-by-step implementation plan to build the FinDocAI project. The tasks are organized into logical phases, starting from project setup to a feature-complete, locally deployable application.

## Phase 1: Project Setup & Core Infrastructure

-   [x] **1.1: Initialize Project Structure**
    -   [x] Create the main project directory (`findocai`).
    -   [x] Initialize a Git repository (`git init`).
    -   [x] Create the application directory structure (`app`, `scripts`, `tests`, `data/uploads`).

-   [x] **1.2: Set Up Python Environment**
    -   [x] Create a Python virtual environment (`python3 -m venv venv`).
    -   [x] Activate the environment (`source venv/bin/activate`).
    -   [x] Create an initial `requirements.txt` file.

-   [x] **1.3: Install Core Dependencies**
    -   [x] Install FastAPI and Uvicorn: `pip install fastapi uvicorn[standard]`.
    -   [x] Install Celery and Redis client: `pip install celery redis`.
    -   [x] Add all installed packages to `requirements.txt`.

-   [x] **1.4: Initial FastAPI Application**
    -   [x] Create `app/main.py` with a basic FastAPI app instance.
    -   [x] Add a root endpoint (`/`) and a health check endpoint (`/health`).
    -   [x] Run the server with `uvicorn` to confirm it works.

-   [x] **1.5: Set Up Docker for Redis**
    -   [x] Create a `docker-compose.yml` file.
    -   [x] Define a `redis` service using the official Redis image.
    -   [x] Run `docker-compose up` to start the Redis container.

## Phase 2: Document Ingestion & Processing Pipeline

-   [x] **2.1: Implement Document Upload API**
    -   [x] Create the `POST /upload` endpoint in `app/main.py`.
    -   [x] Implement file handling to save the `UploadFile` to the `./data/uploads` directory.
    -   [x] Assign a unique `doc_id` (using `uuid`) to each file.

-   [x] **2.2: Set Up Celery Worker**
    -   [x] Create `app/worker.py` to define the Celery app instance and configure the broker URL.
    -   [x] Create a task `process_document(doc_id, filepath)` that will contain the main processing logic.
    -   [x] Call `process_document.delay()` from the `/upload` endpoint.

-   [x] **2.3: Implement Metadata and Status Tracking**
    -   [x] Set up a simple database module (e.g., `app/database.py`) using SQLite for the portfolio project.
    -   [x] Create a table to store document metadata (`doc_id`, `filename`, `status`, `created_at`).
    -   [x] Update the database with `queued` status upon upload.
    -   [x] Create the `GET /status/{doc_id}` endpoint to fetch and return the document's status.

-   [x] **2.4: Implement OCR & Text Extraction**
    -   [x] Create `app/ingestion/ocr.py`.
    -   [x] Install `pytesseract`, `pdf2image`, and `PyPDF2`.
    -   [x] Implement the `extract_text` function with the hybrid (PyPDF2 + Tesseract) approach.
    -   [x] Integrate the call to `extract_text` at the beginning of the `process_document` Celery task.

## Phase 3: AI/ML Model Integration

-   [x] **3.1: Document Classification**
    -   [x] Create `app/classification/model.py`.
    -   [x] Install `torch` and `transformers`.
    -   [x] Implement `classify_document` using a HuggingFace `pipeline` with a fine-tuned `distilbert-base-uncased` model.
    -   [x] Create a script `scripts/download_models.py` to download the classifier model from a hub.
    -   [x] Integrate into the Celery task and update the document status with the `doc_type`.

-   [x] **3.2: RAG - Chunking and Embedding**
    -   [x] Create `app/rag/pipeline.py`.
    -   [x] Install `sentence-transformers` and `chromadb`.
    -   [x] Implement the `index_document` function.
    -   [x] Inside the function, use `RecursiveCharacterTextSplitter` to chunk the extracted text.
    -   [x] Use `SentenceTransformer` to create embeddings for the chunks.
    -   [x] Initialize a persistent ChromaDB client and add the chunks, embeddings, and metadata to a collection.
    -   [x] Integrate `index_document` into the Celery task.

-   [x] **3.3: RAG - Retrieval and Generation**
    -   [x] Implement the `query_document` function in `app/rag/pipeline.py`.
    -   [x] This function should take a question, embed it, and query ChromaDB for the most relevant chunks.
    -   [x] Create the `GET /query` API endpoint that calls `query_document`.
    -   [x] Install `google-generativeai`.
    -   [x] In the `query_document` function, pass the retrieved context and question to the Gemini API using the RAG prompt template.
    -   [x] Return the LLM's answer from the API endpoint.

-   [x] **3.4: LLM-based Entity Extraction and Summarization**
    -   [x] Create `app/nlp/extraction.py`.
    -   [x] Implement `extract_entities` using the Gemini API with the structured JSON prompt.
    -   [x] Add Pydantic models for validation of the extracted data.
    -   [x] Implement `generate_summary` using a summarization prompt.
    -   [x] Integrate both functions into the Celery task and store their results in the database.
    -   [x] Create the `GET /summary/{doc_id}` endpoint.

## Phase 4: Observability & Testing

-   [ ] **4.1: Implement Metrics**
    -   [ ] Install `prometheus-fastapi-instrumentator`.
    -   [ ] Instrument the FastAPI app to expose a `/metrics` endpoint.
    -   [ ] Add custom Celery metrics (e.g., `processing_duration` histogram) using `prometheus_client`.

-   [ ] **4.2: Set Up Monitoring Stack**
    -   [ ] Create `docker-compose.monitoring.yml` to run Prometheus and Grafana.
    -   [ ] Configure Prometheus to scrape the FastAPI `/metrics` endpoint.
    -   [ ] Log in to Grafana, connect the Prometheus data source, and create a basic dashboard with key metrics (requests, latency, queue depth).

-   [ ] **4.3: Add Structured Logging**
    -   [ ] Install `structlog`.
    -   [ ] Configure `structlog` to output JSON-formatted logs for both FastAPI and Celery.
    -   [ ] Add contextual logging throughout the pipeline (e.g., `doc_id`, `doc_type`).

-   [ ] **4.4: Write Tests**
    -   [ ] Install `pytest`.
    -   [ ] Write unit tests for key functions (`extract_text`, `classify_document`).
    -   [ ] Write an integration test for the `POST /upload` endpoint that mocks the file and checks the final status via the `/status` endpoint.
    -   [ ] Install `locust` and create a `load_test.py` script to simulate user traffic.

## Phase 5: Documentation & Finalization

-   [ ] **5.1: Create Project README**
    -   [ ] Write a comprehensive `README.md` at the project root.
    -   [ ] Include a project overview, features, and detailed setup instructions.

-   [ ] **5.2: Finalize All Documentation**
    -   [ ] Review and refine all documents in the `docs/` directory to ensure they are accurate and professional.
    -   [ ] Add diagrams and code snippets where helpful.

-   [ ] **5.3: Create Helper Scripts**
    -   [ ] Finalize `scripts/init_db.py` to create the SQLite database and table.
    -   [ ] Finalize `scripts/download_models.py` to ensure all necessary ML models are downloaded.
    -   [ ] (Optional) Create a `run.sh` script to easily start all services (Uvicorn, Celery).

-   [ ] **5.4: Code Cleanup and Review**
    -   [ ] Format all code using `black` and `isort`.
    -   [ ] Add comments and type hints where necessary for clarity.
    -   [ ] Review all code for potential bugs or performance improvements.
