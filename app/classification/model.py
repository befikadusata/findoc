"""
Document Classification Module

This module provides functions for classifying documents using a pre-trained transformer model.
The implementation uses a HuggingFace pipeline with a DistilBERT model fine-tuned for document classification.
"""

import os
from typing import Optional, List, Dict, Tuple
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizer, DistilBertModel
import torch

# Import structured logging
from app.utils.logging_config import get_logger


class DocumentClassifier:
    """
    Document classifier using a pre-trained transformer model with attention-based explanations.
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
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.num_labels = num_labels
        self.model_path = model_path or model_name

        self.logger.info("Initializing document classifier", model_name=self.model_name, num_labels=self.num_labels)

        # Initialize tokenizer and model for explainability
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self.model = DistilBertModel.from_pretrained(self.model_path)

            # Initialize the classification pipeline for fallback
            self.classifier = pipeline(
                "text-classification",
                model=self.model_path,
                tokenizer=self.model_path,
                top_k=None,  # Use top_k instead of deprecated return_all_scores=True
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            self.logger.info("Classifier initialized successfully")
        except Exception as e:
            self.logger.error("Error initializing classifier", error=str(e))
            # Fallback to a simple keyword-based classifier for demo purposes
            self.tokenizer = None
            self.model = None
            self.classifier = None

    def classify_document(self, text: str, top_k: int = 1, include_explanation: bool = False) -> List[dict]:
        """
        Classify a document based on its text content with optional attention-based explanations.

        Args:
            text (str): The text content of the document to classify
            top_k (int): Number of top predictions to return
            include_explanation (bool): Whether to include attention-based explanations

        Returns:
            List[dict]: List of classification results with labels, confidence scores, and explanations
        """
        self.logger.info("Starting document classification", text_length=len(text), top_k=top_k)

        # If the transformer model fails to load, use a keyword-based fallback
        if self.classifier is None:
            self.logger.warning("Transformer classifier not available, using keyword-based fallback")
            return self._keyword_based_classification(text, top_k)

        try:
            # Truncate text to fit model's max length (typically 512 for BERT-based models)
            max_length = 512
            if len(text) > max_length:
                # Try to get a representative sample by taking the first and last parts
                text = text[:max_length//2] + " " + text[-max_length//2:]

            # Perform classification
            self.logger.debug("Performing transformer-based classification")
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

            # Apply label mapping and add explanations if requested
            for result in top_results:
                original_label = result['label']
                result['doc_type'] = label_mapping.get(original_label, original_label)

                # Add attention-based explanation if requested
                if include_explanation and self.tokenizer and self.model:
                    result['explanation'] = self._get_attention_explanation(text, result['label'])

            self.logger.info("Classification completed", results_count=len(top_results))
            return top_results

        except Exception as e:
            self.logger.error("Error during classification", error=str(e))
            # Fallback to keyword-based classification
            return self._keyword_based_classification(text, top_k)

    def _get_attention_explanation(self, text: str, predicted_label: str) -> Dict:
        """
        Generate attention-based explanations for the classification result.

        Args:
            text (str): Input text
            predicted_label (str): Predicted label for the text

        Returns:
            Dict: Explanation including most influential tokens
        """
        import torch
        import numpy as np

        try:
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Get model outputs with attention
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
                attentions = outputs.attentions  # Tuple of attention tensors for each layer

            # Calculate attention weights for each token
            # Get the attention from the last layer's first head (common approach)
            last_layer_attention = attentions[-1]  # Shape: [batch_size, num_heads, seq_len, seq_len]
            attention_weights = last_layer_attention[0, 0, 0, :].cpu().numpy()  # Average attention for first token (CLS)

            # Get tokens corresponding to the attention weights
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

            # Create a list of (token, attention_weight) pairs
            token_attention_pairs = [(token, float(attention_weights[i])) for i, token in enumerate(tokens) if token not in ['[CLS]', '[SEP]', '[PAD]']]

            # Sort by attention weight to find most influential tokens
            token_attention_pairs = sorted(token_attention_pairs, key=lambda x: x[1], reverse=True)

            # Return the top 10 most influential tokens
            top_tokens = token_attention_pairs[:10]

            # Calculate some statistics
            avg_attention = float(np.mean([weight for _, weight in top_tokens]))
            max_attention = float(max([weight for _, weight in top_tokens]))

            return {
                'predicted_label': predicted_label,
                'top_influential_tokens': [{'token': token, 'attention_weight': weight} for token, weight in top_tokens],
                'average_attention': avg_attention,
                'max_attention': max_attention,
                'explanation_method': 'attention_weights'
            }
        except Exception as e:
            self.logger.error("Error generating attention explanation", error=str(e))
            return {
                'predicted_label': predicted_label,
                'error': f'Could not generate explanation: {str(e)}',
                'explanation_method': 'attention_weights'
            }

    def generate_attention_visualization(self, text: str, predicted_label: str) -> Optional[Dict]:
        """
        Generate visualization for attention weights in the classification.

        Args:
            text (str): Input text
            predicted_label (str): Predicted label for the text

        Returns:
            Optional[Dict]: Visualization data or None if generation fails
        """
        try:
            # Get attention explanation
            explanation = self._get_attention_explanation(text, predicted_label)

            # Import visualization module
            from app.utils.attention_visualization import visualize_attention_weights, visualize_token_attention_inline, create_attention_summary

            # Create visualizations
            top_tokens = explanation.get('top_influential_tokens', [])

            if not top_tokens:
                return None

            # Generate visualization image
            image_url = visualize_attention_weights(top_tokens, f"Attention Weights for '{predicted_label}' Classification")

            # Create inline visualization
            inline_html = visualize_token_attention_inline(top_tokens)

            # Create summary
            summary = create_attention_summary(top_tokens)

            return {
                'image_url': image_url,
                'inline_html': inline_html,
                'summary': summary,
                'explanation_method': 'attention_weights_visualization'
            }
        except Exception as e:
            self.logger.error("Error generating attention visualization", error=str(e))
            return None

    def _keyword_based_classification(self, text: str, top_k: int = 1) -> List[dict]:
        """
        Fallback classification method using keyword matching.

        Args:
            text (str): The text content of the document to classify
            top_k (int): Number of top predictions to return

        Returns:
            List[dict]: List of classification results with labels and confidence scores
        """
        self.logger.debug("Using keyword-based classification fallback", text_length=len(text))
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

        self.logger.info("Keyword-based classification completed", results_count=len(results))
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
    logger = get_logger(__name__)
    logger.info("Classifying document via convenience function", text_length=len(text))
    return classifier.classify_document(text, top_k)


if __name__ == "__main__":
    # Example usage
    sample_text = "INVOICE Date: 2023-06-15 Invoice No: INV-2023-06015 Customer: John Smith Item: Consulting Services Amount: $5,000.00 Due Date: 2023-07-15"
    result = classify_document(sample_text)
    logger = get_logger(__name__)
    logger.info("Classification result", result=result)