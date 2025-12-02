"""
OCR and Text Extraction Module

This module provides functions for extracting text from various document types,
including PDFs, images, and other formats using a hybrid approach combining
PyPDF2 for native PDF text extraction and Tesseract for OCR.
"""

import os
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
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
        return extract_text_from_pdf(filepath)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
        return extract_text_from_image(filepath)
    else:
        # For other file types, try to read as text if possible
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # If it's not a text file, return an empty string
            return ""


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using a hybrid approach:
    1. First try to extract native text using PyPDF2
    2. Then use OCR on images in the PDF using pdf2image + pytesseract
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Combined extracted text content
    """
    all_text = []
    
    # Step 1: Extract native text using PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    all_text.append(text)
    except Exception as e:
        print(f"Error extracting native text from PDF {pdf_path}: {str(e)}")
    
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
        print(f"Error performing OCR on PDF {pdf_path}: {str(e)}")
    
    return "\n".join(all_text)


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file using Tesseract OCR.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text content
    """
    try:
        # Open the image file
        image = Image.open(image_path)
        
        # Apply OCR to extract text
        text = pytesseract.image_to_string(image)
        
        return text
    except Exception as e:
        print(f"Error extracting text from image {image_path}: {str(e)}")
        return ""


def save_extracted_text(text: str, output_path: str) -> bool:
    """
    Save extracted text to a file.
    
    Args:
        text (str): Text content to save
        output_path (str): Path where the text should be saved
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        return True
    except Exception as e:
        print(f"Error saving extracted text to {output_path}: {str(e)}")
        return False