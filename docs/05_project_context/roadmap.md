# Roadmap and Future Enhancements

This document outlines the planned trajectory for FinDocAI, evolving it from a powerful prototype into a comprehensive, enterprise-grade platform.

## Phase 1: Foundation (Current)

-   [x] End-to-end document processing pipeline (OCR, Classify, Extract, RAG).
-   [x] Asynchronous architecture with FastAPI and Celery.
-   [x] Core API for upload, status check, and querying.
-   [x] Foundational MLOps with Prometheus and Grafana.
-   [x] Production deployment strategy for AWS.
-   [x] Comprehensive testing suite (unit, integration, load).

## Phase 2: Enterprise Readiness (Weeks 6-8)

This phase focuses on enhancing the core feature set and improving user interaction.

-   **[ ] Multi-Language Support:**
    -   Integrate models and OCR settings for key regional languages like Amharic and Swahili.
-   **[ ] Fine-Tune LLM on Domain-Specific Data:**
    -   Collect and label a dataset of Kifiya's loan documents to fine-tune an open-source LLM (e.g., Llama 3) for improved accuracy on specialized terminology and formats.
-   **[ ] Document Comparison and Versioning:**
    -   Add functionality to compare two versions of a document (e.g., a draft contract and a final version), highlighting changes and discrepancies.
-   **[ ] Human-in-the-Loop (HITL) Review UI:**
    -   Develop a simple web interface where a human operator can review and correct low-confidence extractions. This feedback will be used to create high-quality training data for future model improvements.

## Phase 3: Advanced Capabilities (Months 3-4)

This phase focuses on scalability, advanced AI features, and deeper integration.

-   **[ ] Real-Time Streaming Ingestion:**
    -   Replace or augment the batch-upload API with a streaming pipeline using **Apache Kafka** or **AWS Kinesis**. This will enable real-time document processing as they arrive from various sources.
-   **[ ] Federated Learning for Privacy:**
    -   For scenarios with highly sensitive partner data, explore federated learning to train models across different data silos without centralizing the raw data, enhancing privacy and security.
-   **[ ] Core Platform Integration:**
    -   Develop deep, API-level integration with Kifiya's core Intelligent Data Decisioning (IDD) platform, making FinDocAI a seamless module within the larger ecosystem.
-   **[ ] Advanced Fraud Detection:**
    -   Train an anomaly detection model on document entities and patterns to automatically flag suspicious activities, such as unusual invoice amounts or forged bank statement entries.
