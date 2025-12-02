# OCR, Text Extraction, and Classification

After ingestion, the first step in the processing pipeline is to extract raw text from the document and classify it to understand its type.

## 2.1 OCR & Text Extraction

A hybrid approach is used for text extraction to balance speed and accuracy. It first attempts to read native text from the PDF, which is fast and accurate. If that fails or yields insufficient text, it falls back to Optical Character Recognition (OCR) using Tesseract.

```python
# app/ingestion/ocr.py
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

def extract_text(filepath: str) -> str:
    """Hybrid OCR: native text + image OCR"""
    
    # Try native PDF text first (faster)
    try:
        reader = PdfReader(filepath)
        text = " ".join([page.extract_text() for page in reader.pages])
        if len(text.strip()) > 50:  # Sufficient text
            return text
    except:
        pass
    
    # Fallback to Tesseract for scanned PDFs
    images = convert_from_path(filepath, dpi=300)
    text_parts = []
    
    for img in images:
        text_parts.append(pytesseract.image_to_string(
            img, 
            lang='eng',
            config='--psm 6'  # Assume uniform text block
        ))
    
    return " ".join(text_parts)
```

**Production Considerations:**
- **Superior Accuracy:** For production, integrating **AWS Textract** (`boto3.client('textract').analyze_document()`) would provide higher accuracy, especially for complex financial document layouts.
- **Cost Management:** OCR results can be cached to avoid reprocessing identical documents. The Textract Queries API can also be used for targeted data extraction to reduce costs.
- **Language Support:** The current implementation is English-only. Production systems should incorporate multi-language models to handle international documents.

## 2.2 Document Classifier

Once text is extracted, a machine learning model classifies the document into a predefined category. This allows the system to tailor the subsequent extraction and analysis steps.

**Model:** A fine-tuned `distilbert-base-uncased` model.  
**Classes:** `invoice`, `contract`, `bank_statement`, `loan_application`

```python
# app/classification/model.py
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="./models/doc_classifier",
    tokenizer="distilbert-base-uncased"
)

def classify_document(text: str) -> dict:
    # Use first 512 tokens (BERT limit)
    truncated = text[:2000] # Truncate to ~500 tokens
    result = classifier(truncated)[0]
    
    return {
        "doc_type": result['label'],
        "confidence": result['score']
    }
```

**Model Training and Performance:**
- **Training Data:** The model was trained on a synthetic dataset generated from templates and augmented with GPT-4 to ensure variety.
- **Test Set Metrics:** Achieved **94.2% accuracy** and a **0.93 macro F1-score**.

**Production Enhancements:**
- **Continuous Improvement:** A human-in-the-loop system should be implemented to collect real, user-labeled data for continuous fine-tuning.
- **Active Learning:** Prioritize labeling uncertain or misclassified documents to improve model performance most efficiently.
- **Drift Detection:** Tools like Evidently AI can be used to monitor for data drift, ensuring the model remains accurate as document formats evolve.
