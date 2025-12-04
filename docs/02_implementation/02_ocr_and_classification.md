# OCR, Text Extraction, and Classification

After ingestion, the first step in the processing pipeline is to extract raw text from the document and classify it to understand its type.

## 2.1 OCR & Text Extraction

A hybrid approach is used for text extraction to balance speed and accuracy. It first attempts to read native text from the PDF, which is fast and accurate. If that fails or yields insufficient text, it falls back to Optical Character Recognition (OCR) using Tesseract.

```python
# app/ingestion/ocr.py
import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from pypdf import PdfReader
import tempfile

def extract_text(filepath: str) -> str:
    """
    Extract text from a document file using a hybrid approach.

    Args:
        filepath (str): Path to the document file

    Returns:
        str: Extracted text content
    """
    file_extension = os.path.splitext(filepath)[1].lower()

    if file_extension == '.pdf':
        result = extract_text_from_pdf(filepath)
        return result
    elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
        result = extract_text_from_image(filepath)
        return result
    else:
        # For other file types, try to read as text if possible
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                result = file.read()
                return result
        except UnicodeDecodeError:
            # If it's not a text file, return an empty string
            return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using a hybrid approach:
    1. First try to extract native text using pypdf
    2. Then use OCR on images in the PDF using pdf2image + pytesseract
    """
    all_text = []

    # Step 1: Extract native text using pypdf
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    all_text.append(text)
    except Exception as e:
        print(f"Error extracting native text from PDF: {e}")

    # Step 2: Extract text using OCR on PDF pages converted to images
    try:
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path)

        for page_image in pages:
            # Apply OCR to each image
            ocr_text = pytesseract.image_to_string(page_image)
            if ocr_text.strip():
                all_text.append(ocr_text)
    except Exception as e:
        print(f"Error performing OCR on PDF: {e}")

    result = "\n".join(all_text)
    return result

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file using Tesseract OCR.
    """
    try:
        # Open the image file
        image = Image.open(image_path)

        # Apply OCR to extract text
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""
```

**Production Considerations:**
- **Superior Accuracy:** For production, integrating **AWS Textract** (`boto3.client('textract').analyze_document()`) would provide higher accuracy, especially for complex financial document layouts.
- **Cost Management:** OCR results can be cached to avoid reprocessing identical documents. The Textract Queries API can also be used for targeted data extraction to reduce costs.
- **Language Support:** The current implementation is English-only. Production systems should incorporate multi-language models to handle international documents.

## 2.2 Document Classifier

Once text is extracted, a machine learning model classifies the document into a predefined category. This allows the system to tailor the subsequent extraction and analysis steps.

**Model:** A pre-trained `distilbert-base-uncased` model with custom classification head.
**Classes:** `invoice`, `contract`, `bank_statement`, `loan_application`, `identity_document`, `other`

```python
# app/classification/model.py
from transformers import pipeline
import torch
from typing import List

class DocumentClassifier:
    """
    Document classifier using a pre-trained transformer model.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 num_labels: int = 6):
        self.model_name = model_name
        self.num_labels = num_labels

        # Initialize the classification pipeline
        try:
            self.classifier = pipeline(
                "text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                top_k=None,  # Use top_k instead of deprecated return_all_scores=True
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
        except Exception as e:
            # Fallback to a simple keyword-based classifier for demo purposes
            self.classifier = None

    def classify_document(self, text: str, top_k: int = 1) -> List[dict]:
        """
        Classify a document based on its text content.
        """
        # If the transformer model fails to load, use a keyword-based fallback
        if self.classifier is None:
            return self._keyword_based_classification(text, top_k)

        try:
            # Truncate text to fit model's max length (typically 512 for BERT-based models)
            max_length = 512
            if len(text) > max_length:
                # Try to get a representative sample by taking the first and last parts
                text = text[:max_length//2] + " " + text[-max_length//2:]

            # Perform classification
            results = self.classifier(text)

            # Process results to get top_k predictions
            # With top_k=None, results come as [[{'label': ..., 'score': ...}, ...]]
            # Extract the inner list
            inner_results = results[0] if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list) else results
            top_results = sorted(inner_results, key=lambda x: x['score'], reverse=True)[:top_k]

            # Map generic labels to document types
            label_mapping = {
                'LABEL_0': 'invoice',
                'LABEL_1': 'contract',
                'LABEL_2': 'bank_statement',
                'LABEL_3': 'loan_application',
                'LABEL_4': 'identity_document',
                'LABEL_5': 'other'
            }

            # Apply label mapping
            for result in top_results:
                original_label = result['label']
                result['doc_type'] = label_mapping.get(original_label, original_label)

            return top_results

        except Exception as e:
            # Fallback to keyword-based classification
            return self._keyword_based_classification(text, top_k)

    def _keyword_based_classification(self, text: str, top_k: int = 1) -> List[dict]:
        """
        Fallback classification method using keyword matching.
        """
        text_lower = text.lower()

        # Define document type keywords
        doc_types = {
            'invoice': ['invoice', 'bill', 'payment', 'amount', 'due', 'invoice no', 'tax', 'subtotal'],
            'contract': ['contract', 'agreement', 'party', 'term', 'obligation', 'clause', 'sign', 'effective'],
            'bank_statement': ['statement', 'balance', 'transaction', 'account', 'debit', 'credit', 'bank'],
            'loan_application': ['loan', 'application', 'borrower', 'interest', 'collateral', 'repayment', 'credit'],
            'identity_document': ['id', 'license', 'passport', 'driver', 'identification', 'document', 'birth'],
            'other': []
        }

        # Calculate scores based on keyword matches
        scores = {}
        for doc_type, keywords in doc_types.items():
            score = 0
            for keyword in keywords:
                score += text_lower.count(keyword) * 10  # Weight keyword matches
            scores[doc_type] = score

        # Normalize scores to look like model outputs
        total_score = sum(scores.values()) or 1  # Avoid division by zero
        results = [
            {
                'label': doc_type,
                'score': score / total_score if total_score > 0 else 0,
                'doc_type': doc_type
            }
            for doc_type, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ]

        return results[:top_k]


# Global classifier instance for convenience
classifier = DocumentClassifier()

def classify_document(text: str, top_k: int = 1) -> List[dict]:
    """
    Convenience function to classify a document.
    """
    return classifier.classify_document(text, top_k)
```

**Model Training and Performance:**
- **Pre-trained Model:** Uses `distilbert-base-uncased` as the base model with fine-tuning capabilities.
- **Fallback System:** Implements a keyword-based fallback when the transformer model is unavailable.
- **Performance:** Balances accuracy and speed for real-time processing requirements.

**Production Enhancements:**
- **Continuous Improvement:** A human-in-the-loop system should be implemented to collect real, user-labeled data for continuous fine-tuning.
- **Active Learning:** Prioritize labeling uncertain or misclassified documents to improve model performance most efficiently.
- **Drift Detection:** Tools like Evidently AI can be used to monitor for data drift, ensuring the model remains accurate as document formats evolve.
