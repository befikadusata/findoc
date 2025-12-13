#!/usr/bin/env python3
"""
Script to download required models for the FinDocAI application.

This script downloads the transformer models needed for document classification
and other NLP tasks. It ensures that models are available locally for faster
inference during document processing.
"""

import os
import sys
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def download_classification_model(
    model_name: str = "distilbert-base-uncased",
    cache_dir: str = "./models"
) -> str:
    """
    Download a pre-trained classification model from HuggingFace Hub.
    
    Args:
        model_name (str): Name of the model to download
        cache_dir (str): Directory to store the downloaded model
        
    Returns:
        str: Path to the downloaded model
    """
    logger.info(f"Downloading classification model: {model_name}")
    
    # Create models directory if it doesn't exist
    models_dir = Path(cache_dir)
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_path = models_dir / f"{model_name.replace('/', '_')}_tokenizer"
        tokenizer.save_pretrained(tokenizer_path)
        logger.info(f"Tokenizer saved to: {tokenizer_path}")
        
        # Download model
        logger.info("Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=6  # Adjust based on your classification needs
        )
        model_path = models_dir / f"{model_name.replace('/', '_')}_model"
        model.save_pretrained(model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Test the pipeline to ensure everything works
        logger.info("Testing the classification pipeline...")
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=tokenizer_path
        )
        
        # Test with a sample text
        test_text = "This is a sample document for testing."
        result = classifier(test_text)
        logger.info(f"Test classification result: {result}")
        
        logger.info(f"Successfully downloaded and verified model: {model_name}")
        return str(model_path)
        
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {str(e)}")
        raise e


def download_all_models():
    """Download all required models for the application."""
    logger.info("Starting model download process...")
    
    # Define models to download
    models_to_download = [
        "distilbert-base-uncased",  # Base model for classification
        # Additional models can be added here as needed
    ]
    
    downloaded_paths = []
    
    for model_name in models_to_download:
        try:
            model_path = download_classification_model(model_name)
            downloaded_paths.append(model_path)
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
            # Continue with other models even if one fails
    
    logger.info(f"Model download process completed!")
    logger.info(f"Downloaded {len(downloaded_paths)} models:")
    for path in downloaded_paths:
        logger.info(f"  - {path}")
    
    return downloaded_paths


if __name__ == "__main__":
    # Set environment variables to avoid some warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        downloaded_models = download_all_models()
        logger.info(f"Successfully downloaded {len(downloaded_models)} models!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during model download: {str(e)}")
        sys.exit(1)