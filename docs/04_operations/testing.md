# Testing and Quality Assurance

A multi-layered testing strategy ensures the reliability, correctness, and performance of the FinDocAI system.

## 6.1 Unit Tests

Unit tests focus on isolating and verifying the smallest pieces of functionality, such as a single function or class. `pytest` is used as the testing framework.

**Example:**
This test validates that the `extract_entities` function correctly parses a simple string for a given document type.

```python
# tests/test_extraction.py
import pytest

def test_entity_extraction_invoice():
    """
    Given a simple text string for an invoice,
    When the extract_entities function is called,
    Then it should correctly identify the amount and date.
    """
    text = "Invoice #1234. Total: $500.00. Date: 2025-01-15"
    result = extract_entities(text, "invoice")
    
    assert result['amount'] == 500.00
    assert result['date'] == "2025-01-15"
```

## 6.2 Integration Tests

Integration tests verify that different components of the system work together as expected. This includes testing the full pipeline from API endpoint to database update.

**Example:**
This test simulates a file upload and polls the status endpoint to confirm the document is processed successfully from end to end.

```python
# tests/test_pipeline.py
import time
from fastapi.testclient import TestClient

# Assumes 'client' is a TestClient instance and 'mock_pdf' is a file-like object

def test_end_to_end_processing():
    """
    Given a mock PDF file,
    When it is uploaded to the /upload endpoint,
    Then the system should process it completely and update the status to 'completed'.
    """
    # 1. Upload mock PDF
    response = client.post("/upload", files={"file": mock_pdf})
    doc_id = response.json()['doc_id']
    
    # 2. Wait for async processing to complete
    time.sleep(30) # In a real test suite, use polling with a timeout
    
    # 3. Check final status
    status_response = client.get(f"/status/{doc_id}")
    status_data = status_response.json()
    
    assert status_data['status'] == 'completed'
    assert 'entities' in status_data
    assert status_data['doc_type'] is not None
```

## 6.3 Performance Tests

Performance tests are conducted using `Locust` to simulate user load and measure the system's throughput and latency under pressure.

**Example:**
This Locust test file defines a user behavior that continuously uploads a sample document to the `/upload` endpoint.

```python
# tests/load_test.py
from locust import HttpUser, task, between

class FinDocUser(HttpUser):
    wait_time = between(1, 3) # Wait 1-3 seconds between tasks
    
    @task
    def upload_document(self):
        # In a real test, you might vary the file
        with open("data/samples/sample_invoice.pdf", "rb") as f:
            self.client.post(
                "/upload",
                files={"file": f},
                name="/upload [invoice]" # Group requests in Locust UI
            )
```

**Target Performance Metrics:**
- **Throughput:** Sustain a load of at least 100 documents per hour.
- **Latency:** The P95 latency for the entire processing pipeline should remain under 120 seconds.
- **Success Rate:** The system should maintain a >99% success rate under load.
