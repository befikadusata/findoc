"""
NLP Module for Entity Extraction and Summarization

This module provides functions for extracting structured entities from documents
and generating summaries using the Gemini API.
"""

import os
import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

import google.generativeai as genai

# Import centralized settings
from app.config import settings

# Import structured logging
from app.utils.logging_config import get_logger


class FinancialEntity(BaseModel):
    """
    Pydantic model for financial document entities with validation.
    """
    # Invoice related entities
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    invoice_date: Optional[str] = Field(None, description="Invoice date in YYYY-MM-DD format")
    due_date: Optional[str] = Field(None, description="Due date in YYYY-MM-DD format")
    
    # Amounts
    total_amount: Optional[float] = Field(None, description="Total amount")
    subtotal: Optional[float] = Field(None, description="Subtotal amount")
    tax_amount: Optional[float] = Field(None, description="Tax amount")
    discount_amount: Optional[float] = Field(None, description="Discount amount")
    
    # Parties
    customer_name: Optional[str] = Field(None, description="Customer or client name")
    customer_address: Optional[str] = Field(None, description="Customer address")
    vendor_name: Optional[str] = Field(None, description="Vendor or seller name")
    vendor_address: Optional[str] = Field(None, description="Vendor address")
    
    # Contract/loan specific
    contract_terms: Optional[str] = Field(None, description="Key contract terms")
    loan_amount: Optional[float] = Field(None, description="Loan amount")
    interest_rate: Optional[float] = Field(None, description="Interest rate")
    repayment_schedule: Optional[str] = Field(None, description="Repayment schedule")
    
    # Bank statement specific
    account_number: Optional[str] = Field(None, description="Account number")
    account_holder: Optional[str] = Field(None, description="Account holder name")
    opening_balance: Optional[float] = Field(None, description="Opening balance")
    closing_balance: Optional[float] = Field(None, description="Closing balance")
    
    # General entities
    document_type: Optional[str] = Field(None, description="Type of document")
    document_date: Optional[str] = Field(None, description="Date of document")


class SummaryResult(BaseModel):
    """
    Pydantic model for document summaries.
    """
    summary: str = Field(..., description="Brief summary of the document")
    key_points: List[str] = Field(default_factory=list, description="Key points from the document")
    document_type: str = Field(..., description="Type of document")
    document_date: Optional[str] = Field(None, description="Date of document")
    extracted_entities: Optional[Dict[str, Any]] = Field(None, description="Extracted entities if available")


def extract_entities(text: str, doc_type: str = "financial") -> Dict[str, Any]:
    """
    Extract structured entities from document text using the Gemini API.
    
    Args:
        text (str): Text content to extract entities from
        doc_type (str): Type of document (default: "financial")
        
    Returns:
        Dict[str, Any]: Extracted entities in structured format
    """
    try:
        # Get the API key from centralized settings
        if settings.gemini_api_key is None:
            return {"error": "GEMINI_API_KEY is not set in configuration"}
        
        genai.configure(api_key=settings.gemini_api_key.get_secret_value())
        
        # Select the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Use the prompt manager to get the entity extraction prompt
        from prompts.prompt_manager import render_entity_extraction_prompt
        prompt = render_entity_extraction_prompt(doc_type=doc_type, text=text)

        if not prompt:
            # Fallback to default prompt if template rendering fails
            prompt = f"""
            Extract structured information from the following {doc_type} document text.
            Return the result as a JSON object with the following fields (only include fields that are found in the document):
            - invoice_number
            - invoice_date (YYYY-MM-DD format)
            - due_date (YYYY-MM-DD format)
            - total_amount (as a number)
            - subtotal (as a number)
            - tax_amount (as a number)
            - discount_amount (as a number)
            - customer_name
            - customer_address
            - vendor_name
            - vendor_address
            - contract_terms
            - loan_amount (as a number)
            - interest_rate (as a number)
            - repayment_schedule
            - account_number
            - account_holder
            - opening_balance (as a number)
            - closing_balance (as a number)
            - document_type
            - document_date (YYYY-MM-DD format)

            The output should be a valid JSON object. Only include fields that are explicitly present in the text.
            For monetary values, extract as numbers without currency symbols.
            For dates, extract in YYYY-MM-DD format or as close to that as possible.

            Document text:
            {text}

            JSON Output:
            """
        
        # Generate the content
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            ),
            request_options={"timeout": 60}
        )
        
        # Parse the response
        if response.text:
            # Clean up the response if it contains JSON formatting markers
            result_text = response.text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:]  # Remove '```json'
            if result_text.endswith('```'):
                result_text = result_text[:-3]  # Remove '```'
            
            entities = json.loads(result_text)
            
            # Validate with Pydantic model
            validated_entities = FinancialEntity(**entities)
            return validated_entities.model_dump(exclude_unset=True)
        else:
            return FinancialEntity().dict(exclude_unset=True)
            
    except json.JSONDecodeError as e:
        logger = get_logger(__name__)
        logger.error("JSON decode error in entity extraction", error=str(e))
        return FinancialEntity().model_dump(exclude_unset=True)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error("Error extracting entities", error=str(e))
        return {"error": str(e)}


def generate_summary(text: str, doc_type: str = "financial", max_length: int = 300) -> Dict[str, Any]:
    """
    Generate a summary of the document using the Gemini API.
    
    Args:
        text (str): Text content to summarize
        doc_type (str): Type of document (default: "financial")
        max_length (int): Maximum length of the summary in characters
        
    Returns:
        Dict[str, Any]: Summary and key points in structured format
    """
    try:
        # Get the API key from centralized settings
        if settings.gemini_api_key is None:
            return {"error": "GEMINI_API_KEY is not set in configuration"}
        
        genai.configure(api_key=settings.gemini_api_key.get_secret_value())
        
        # Select the model
        model = genai.GenerativeModel('gemini-pro')
        
        # Use the prompt manager to get the summarization prompt
        from prompts.prompt_manager import render_summarization_prompt
        prompt = render_summarization_prompt(doc_type=doc_type, text_truncated=text[:4000], max_length=max_length)

        if not prompt:
            # Fallback to default prompt if template rendering fails
            prompt = f"""
            Create a concise summary of this {doc_type} document. The summary should be no more than {max_length} characters.
            Also extract 3-5 key points from the document.

            Document text:
            {text[:4000]}  # Limiting to 4000 chars to avoid token limits

            Provide the output as a JSON object with these fields:
            - summary: The summary text
            - key_points: Array of key points
            - document_type: The type of document
            - document_date: Document date if mentioned in the text
            - extracted_entities: Only include if you can identify significant entities

            JSON Output:
            """
        
        # Generate the content
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            ),
            request_options={"timeout": 60}
        )
        
        # Parse the response
        if response.text:
            # Clean up the response if it contains JSON formatting markers
            result_text = response.text.strip()
            if result_text.startswith('```json'):
                result_text = result_text[7:]  # Remove '```json'
            if result_text.endswith('```'):
                result_text = result_text[:-3]  # Remove '```'
            
            summary_data = json.loads(result_text)
            
            # Validate with Pydantic model
            validated_summary = SummaryResult(**summary_data)
            return validated_summary.model_dump(exclude_unset=True)
        else:
            return SummaryResult(summary="Could not generate summary", key_points=[], document_type=doc_type).model_dump(exclude_unset=True)

    except json.JSONDecodeError as e:
        logger = get_logger(__name__)
        logger.error("JSON decode error in summary generation", error=str(e))
        return SummaryResult(summary="Could not generate summary", key_points=[], document_type=doc_type).model_dump(exclude_unset=True)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error("Error generating summary", error=str(e))
        return {"error": str(e)}


if __name__ == "__main__":
    # Example usage
    logger = get_logger(__name__)
    sample_text = """
    INVOICE
    Invoice Number: INV-2023-001
    Date: 2023-06-15
    Due Date: 2023-07-15

    Customer: John Smith
    Address: 123 Main St, New York, NY

    Item: Consulting Services
    Subtotal: $4,500.00
    Tax: $360.00
    Total: $4,860.00
    """

    entities = extract_entities(sample_text, "invoice")
    logger.info("Extracted entities", entities=entities)

    summary = generate_summary(sample_text, "invoice")
    logger.info("Generated summary", summary=summary)


