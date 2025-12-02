"""
Document Classification Module

This module provides functions for classifying documents using a pre-trained transformer model.
The implementation uses a HuggingFace pipeline with a DistilBERT model fine-tuned for document classification.
"""

import os
from typing import Optional, List
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


class DocumentClassifier:
    """
    Document classifier using a pre-trained transformer model.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 num_labels: int = 6,  # Adjust based on your document types
                 model_path: Optional[str] = None):
        """
        Initialize the document classifier.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            num_labels (int): Number of document categories
            model_path (str, optional): Path to a local fine-tuned model
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.model_path = model_path or model_name
        
        # Initialize the classification pipeline
        try:
            self.classifier = pipeline(
                "text-classification",
                model=self.model_path,
                tokenizer=self.model_path,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
        except Exception as e:
            print(f"Error initializing classifier: {e}")
            # Fallback to a simple keyword-based classifier for demo purposes
            self.classifier = None
    
    def classify_document(self, text: str, top_k: int = 1) -> List[dict]:
        """
        Classify a document based on its text content.
        
        Args:
            text (str): The text content of the document to classify
            top_k (int): Number of top predictions to return
            
        Returns:
            List[dict]: List of classification results with labels and confidence scores
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
            if isinstance(results[0], list):
                # Handle the case where return_all_scores=True returns a list of lists
                top_results = results[0][:top_k]
            else:
                # Handle other result formats
                top_results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
            
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
            print(f"Error during classification: {e}")
            # Fallback to keyword-based classification
            return self._keyword_based_classification(text, top_k)
    
    def _keyword_based_classification(self, text: str, top_k: int = 1) -> List[dict]:
        """
        Fallback classification method using keyword matching.
        
        Args:
            text (str): The text content of the document to classify
            top_k (int): Number of top predictions to return
            
        Returns:
            List[dict]: List of classification results with labels and confidence scores
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
    
    Args:
        text (str): The text content of the document to classify
        top_k (int): Number of top predictions to return
        
    Returns:
        List[dict]: List of classification results with labels and confidence scores
    """
    return classifier.classify_document(text, top_k)


if __name__ == "__main__":
    # Example usage
    sample_text = "INVOICE Date: 2023-06-15 Invoice No: INV-2023-06015 Customer: John Smith Item: Consulting Services Amount: $5,000.00 Due Date: 2023-07-15"
    result = classify_document(sample_text)
    print("Classification result:", result)