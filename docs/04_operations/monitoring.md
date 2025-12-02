# Observability and Monitoring

A robust observability strategy is critical for maintaining a production-grade AI system. This involves collecting metrics, visualizing performance, and implementing a structured logging strategy.

## 3.1 Metrics Collection

The system is instrumented with Prometheus-compatible metrics to provide insight into its performance and health.

```python
# app/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Define key application metrics
doc_upload_total = Counter('documents_uploaded_total', 'Total docs uploaded')
llm_calls_total = Counter('llm_api_calls_total', 'LLM API calls', ['provider', 'model'])
extraction_success = Counter('entity_extraction_success_total', 'Successful extractions')
extraction_failure = Counter('entity_extraction_failure_total', 'Failed extractions')

processing_duration = Histogram(
    'document_processing_duration_seconds',
    'Processing time per document',
    ['doc_type'],
    buckets=[1, 5, 10, 30, 60, 120] # Buckets in seconds
)

queue_depth = Gauge('celery_queue_depth', 'Tasks in queue')

# Example usage within application code
@app.post("/upload")
async def upload_document(...):
    doc_upload_total.inc()
    # ... rest of code
```

These metrics are exposed via a `/metrics` endpoint that a Prometheus server can scrape.

## 3.2 Grafana Dashboard

A Grafana dashboard provides a visual overview of the system's health, using the Prometheus data source.

**Key Dashboard Panels:**
1.  **System Throughput:** A time-series graph showing documents processed per minute/hour.
2.  **Processing Latency:** P50, P95, and P99 processing times, bucketed by document type.
3.  **Error Rate:** A percentage gauge showing the ratio of failed tasks to total tasks over a rolling window.
4.  **LLM Costs & Usage:** Counters for token usage and API calls, which can be multiplied by cost-per-token to estimate expenses.
5.  **Queue Health:** A graph showing the Celery queue depth over time, indicating if workers are keeping up with the load.
6.  **Classifier Confidence:** A histogram displaying the confidence score distribution, which can help detect model performance degradation.

**Recommended Alerts:**
- **High Error Rate:** Alert if the task failure rate exceeds 5% over a 15-minute window.
- **High Latency:** Alert if the P95 processing latency surpasses 120 seconds.
- **Queue Overflow:** Alert if the Celery queue depth grows beyond 100 tasks, suggesting a worker bottleneck.

## 3.3 Logging Strategy

Structured logging is used to create machine-readable and easily searchable log entries.

```python
# app/utils/logging.py
import structlog

logger = structlog.get_logger()

# Example of a structured log entry
logger.info("document_processed",
    doc_id=doc_id,
    doc_type=doc_type,
    processing_time_sec=duration,
    entity_count=len(entities),
    chunk_count=len(chunks)
)
```

**Log Aggregation in Production:**
- **Collection:** In an AWS environment, logs from all services (EKS pods, Lambda, etc.) should be aggregated into **CloudWatch Logs**.
- **Analysis:** **CloudWatch Logs Insights** can be used for interactive searching, and **Amazon Athena** can be used for running complex SQL queries over logs stored in S3.
- **Retention Policy:** A common practice is to keep logs in a "hot" searchable tier for 30 days, then archive them to S3/Glacier for long-term retention (e.g., 1 year) for compliance and audit purposes.
