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

# Import structured logging
from app.utils.logging_config import get_logger


def extract_text(filepath: str) -> str:
    """
    Extract text from a document file using a hybrid approach.

    Args:
        filepath (str): Path to the document file

    Returns:
        str: Extracted text content
    """
    logger = get_logger(__name__).bind(filepath=filepath)
    file_extension = os.path.splitext(filepath)[1].lower()

    logger.info("Starting text extraction", file_extension=file_extension)

    if file_extension == '.pdf':
        result = extract_text_from_pdf(filepath)
        logger.info("PDF text extraction completed", text_length=len(result))
        return result
    elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
        result = extract_text_from_image(filepath)
        logger.info("Image text extraction completed", text_length=len(result))
        return result
    else:
        # For other file types, try to read as text if possible
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                result = file.read()
                logger.info("Text file read completed", text_length=len(result))
                return result
        except UnicodeDecodeError:
            logger.warning("File could not be decoded as text")
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
    logger = get_logger(__name__).bind(pdf_path=pdf_path)
    all_text = []

    # Step 1: Extract native text using PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            logger.info("Processing PDF pages", page_count=len(pdf_reader.pages))
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    all_text.append(text)
    except Exception as e:
        logger.error("Error extracting native text from PDF", error=str(e))

    # Step 2: Extract text using OCR on PDF pages converted to images
    try:
        # Convert PDF pages to images
        logger.info("Converting PDF to images for OCR processing")
        pages = convert_from_path(pdf_path)

        for page_num, page_image in enumerate(pages):
            # Apply OCR to each image
            ocr_text = pytesseract.image_to_string(page_image)
            if ocr_text.strip():
                all_text.append(ocr_text)
                logger.debug("OCR processed page", page_number=page_num, text_length=len(ocr_text))
    except Exception as e:
        logger.error("Error performing OCR on PDF", error=str(e))

    result = "\n".join(all_text)
    logger.info("PDF text extraction completed", total_text_length=len(result))
    return result


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file using Tesseract OCR.

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Extracted text content
    """
    logger = get_logger(__name__).bind(image_path=image_path)
    logger.info("Starting OCR on image")

    try:
        # Open the image file
        image = Image.open(image_path)

        # Apply OCR to extract text
        text = pytesseract.image_to_string(image)

        logger.info("Image OCR completed", text_length=len(text))
        return text
    except Exception as e:
        logger.error("Error extracting text from image", error=str(e))
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
    logger = get_logger(__name__).bind(output_path=output_path)
    logger.info("Saving extracted text", text_length=len(text))

    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        logger.info("Text saved successfully")
        return True
    except Exception as e:
        logger.error("Error saving extracted text", error=str(e))
        return False