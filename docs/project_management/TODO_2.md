# FinDocAI: Implementation TODO - Phase 2 (Additions)

This document outlines additional steps to further enhance the FinDocAI project beyond the initial implementation, focusing on hardening it for production and improving its capabilities.

## Phase X: System Hardening & Advanced Features

-   [ ] **X.1: Security and Multi-Tenancy**
    -   [ ] Implement API Authentication (e.g., OAuth2 with JWTs).
    -   [ ] Implement API Authorization, associating documents with a `user_id` or `tenant_id`.
    -   [ ] Enforce document ownership checks on all data access endpoints (`/status`, `/query`, `/summary`).

-   [ ] **X.2: Robustness and Error Handling**
    -   [ ] Configure Celery to use a **Dead Letter Queue (DLQ)** for tasks that exhaust their retries.
    -   [ ] Refactor the `process_document` Celery task into a **chain of smaller, granular tasks** for better resilience and retriability.
    -   [ ] Implement network timeouts for all external API calls (e.g., Gemini API, sentence transformers download).

-   [x] **X.3: Configuration and Secrets Management**
    -   [x] Implement a **centralized configuration management** using Pydantic `BaseSettings` to load settings on application startup.
    -   [x] For production, integrate with a dedicated secrets management solution (e.g., AWS Secrets Manager).

-   [x] **X.4: Data Management and Governance**
    -   [x] Integrate a **database migration tool (e.g., Alembic)** to manage PostgreSQL schema evolution.
    -   [x] Implement a **comprehensive data deletion process** via a `DELETE /documents/{doc_id}` API endpoint, ensuring removal from all storage layers (filesystem/S3, PostgreSQL, ChromaDB).

-   [x] **X.5: MLOps and Model Lifecycle**
    -   [x] Implement a **prompt versioning and management system** for LLMs (e.g., externalizing prompts to templated files).

-   [x] **X.6: Logging Improvements**
    -   [x] Replace all `print()` statements with structured logging (using `structlog`) throughout the application.

-   [x] **X.7: API Input Validation**
    -   [x] Add Pydantic validation models to all API endpoints to enforce stricter input validation (e.g., `query` parameter length, `doc_id` format).
