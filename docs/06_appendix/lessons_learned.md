# Appendix: Lessons Learned and Best Practices

This section documents key insights gained during the development of the FinDocAI prototype and highlights best practices and potential pitfalls to avoid in a production environment.

## 11.1 What Worked Well

1.  **Hybrid OCR Approach:** Starting with native PDF text extraction before falling back to Tesseract OCR proved highly effective. It is significantly faster and cheaper for modern, digitally-generated PDFs and avoids unnecessary OCR processing.
2.  **Prompt Versioning and Management:** Storing LLM prompts in a dedicated `/prompts` directory (or a similar structured format) allowed for rapid iteration and experimentation without changing application code.
3.  **Asynchronous Processing by Default:** Using Celery to decouple the API from the heavy-lifting (OCR, LLM calls) was a crucial architectural decision. It ensures the API remains responsive and the system is resilient to spikes in load.
4.  **Lightweight Vector Store for Prototyping:** Chroma DB was an excellent choice for local development. Its simple, file-based persistence and straightforward API made it easy to get the RAG pipeline up and running quickly.

## 11.2 Challenges and Solutions

| Challenge | Solution |
| :--- | :--- |
| **PDF Quality Variance** | An explicit multi-stage OCR pipeline was implemented. This involves quality checks and fallback mechanisms, but a production system would benefit from pre-processing steps like de-skewing and noise reduction. |
| **LLM Hallucination & Unstructured Output** | The combination of strong prompt engineering (requesting JSON output) and a strict validation layer using **Pydantic** was highly effective. The model is forced to adhere to a schema, and Pydantic ensures data integrity before it enters the database. |
| **Context Window and Token Limits** | An intelligent chunking strategy using `RecursiveCharacterTextSplitter` was essential. For very long documents, implementing a "sliding window" or summarization approach for context would be the next logical step. |
| **Cost and Latency Control** | The choice of **Gemini 1.5 Flash** provided a good balance of performance and cost. For a production system, a multi-tiered model approach (e.g., using a cheaper local model like Mistral for simple tasks and a powerful cloud model for complex ones) would be optimal. |

## 11.3 Production Gotchas to Avoid

-   **Storing Blobs in Databases:**
    -   **Mistake:** ❌ Storing the raw PDF files directly in a SQL database.
    -   **Best Practice:** ✅ Store the files in a dedicated object store like **S3** and save only the reference (e.g., S3 URI) in the database.
-   **Synchronous, Blocking API Calls:**
    -   **Mistake:** ❌ Making direct, synchronous calls to LLMs or other long-running processes from an API endpoint.
    -   **Best Practice:** ✅ Use a queue-based system (like Celery with Redis/SQS) to process tasks asynchronously with proper timeouts and retry logic.
-   **Single-Region Deployment:**
    -   **Mistake:** ❌ Deploying all infrastructure to a single availability zone (AZ).
    -   **Best Practice:** ✅ Design for High Availability (HA) by deploying services across multiple AZs to withstand single-zone failures.
-   **No Retry Logic:**
    -   **Mistake:** ❌ Assuming API calls and network requests will never fail.
    -   **Best Practice:** ✅ Implement **exponential backoff** for retrying failed tasks and configure a **Dead Letter Queue (DLQ)** to isolate and analyze tasks that repeatedly fail.
