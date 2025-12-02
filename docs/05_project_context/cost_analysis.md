# Cost Analysis

This section provides a high-level cost estimate for running the FinDocAI system in a production environment on AWS, processing 10,000 documents per month.

## 8.1 Per-Document Cost (Production Estimates)

The cost per document is broken down by the major components in the processing pipeline.

| Component | Cost per Doc (USD) | Notes |
| :--- | :--- | :--- |
| **AWS Textract** | $0.015 | Assumes an average of 5 pages per document. |
| **Gemini API (Extraction)** | $0.001 | Using the `gemini-1.5-flash` model with ~2K tokens. |
| **Embeddings (SageMaker)** | $0.0005 | Cost for a batch inference job on a SageMaker endpoint. |
| **Pinecone (Vector Storage)** | $0.002 | Estimated cost per index write operation. |
| **Compute (ECS/EKS)** | $0.005 | Amortized cost of the container compute resources. |
| **Total** | **$0.0235** | **Approximately $23.50 per 1,000 documents.** |

*Note: These are estimates and can vary based on document complexity, usage patterns, and AWS pricing changes.*

## 8.2 Estimated Monthly Cost

Based on a workload of **10,000 documents per month**.

-   **Infrastructure:** ~$500 / month
    -   Includes EKS/ECS cluster management fees, Aurora/RDS database costs, and ElastiCache for Redis.
-   **AI Services:** ~$235 / month
    -   Calculated from the per-document cost ($0.0235 \* 10,000 documents).
-   **Total Estimated Monthly Cost:** **~$735 / month**

## 8.3 Cost Optimization Strategies

Several strategies can be employed to manage and reduce operational costs as the system scales.

-   **Use Spot Instances:** Configure the EKS/ECS cluster to use EC2 Spot Instances for Celery worker nodes. This can provide savings of up to 60-70% on compute costs for fault-tolerant workloads.
-   **Batch Processing:** For non-urgent documents, batching requests to AWS Textract and SageMaker endpoints can be more cost-effective than real-time processing.
-   **Cache Everything:**
    -   **OCR Results:** Cache the text output of Textract to avoid re-processing the same document.
    -   **Embeddings:** Store and reuse embeddings for identical text chunks.
    -   **LLM Responses:** Cache responses from Gemini/Bedrock for identical prompts.
-   **Right-Sizing:** Continuously monitor resource utilization (CPU, memory) and right-size the instances and pods to avoid paying for over-provisioned capacity.
-   **Intelligent Tiering:** Use tiered LLM models. Route simple extraction tasks to cheaper models (like Gemini Flash or a local Mistral instance) and reserve more powerful, expensive models (like Claude 3 Opus) for complex reasoning tasks.
