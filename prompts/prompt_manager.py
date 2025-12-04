"""
Prompt Management Module for FinDocAI

This module provides functionality for loading, caching, and rendering prompt templates
with versioning support. It uses Jinja2 templates for prompt flexibility and management.
"""

import os
import logging
import re
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template

# Import structured logging
from app.utils.logging_config import get_logger
logger = get_logger(__name__)

class PromptManager:
    """
    Manages prompt templates with versioning and caching capabilities.
    """

    def __init__(self, prompts_dir: str = "./prompts"):
        """
        Initialize the prompt manager.

        Args:
            prompts_dir: Directory containing prompt templates
        """
        self.prompts_dir = prompts_dir
        self.environment = Environment(
            loader=FileSystemLoader(prompts_dir),
            autoescape=False  # We don't need HTML escaping for prompts
        )
        self._template_cache: Dict[str, Tuple[Template, str]] = {}  # Cache now stores (template, version)

        # Set up logging for the prompt manager
        logger.info("Prompt manager initialized", prompts_dir=prompts_dir)

    def extract_version_from_template(self, template_path: str) -> str:
        """
        Extract version information from the template file.

        Args:
            template_path: Path to the template file

        Returns:
            Version string, or '1.0' if not found
        """
        full_path = os.path.join(self.prompts_dir, template_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for version in the format {# Version: X.X #}
                match = re.search(r'\{#\s*Version:\s*([^\s#]+)\s*#\}', content)
                if match:
                    return match.group(1)
                else:
                    return "1.0"  # Default version
        except Exception as e:
            logger.error("Failed to extract version from template", template_path=template_path, error=str(e))
            return "1.0"

    def get_template(self, template_path: str) -> Optional[Tuple[Template, str]]:
        """
        Get a template by its path, with caching. Returns both template and version.

        Args:
            template_path: Path to the template file (relative to prompts_dir)

        Returns:
            Tuple of (Template object, version string) or None if not found
        """
        # Check cache first
        if template_path in self._template_cache:
            logger.debug("Template retrieved from cache", template_path=template_path)
            return self._template_cache[template_path]

        try:
            template = self.environment.get_template(template_path)
            version = self.extract_version_from_template(template_path)
            # Cache the template and version for future use
            self._template_cache[template_path] = (template, version)
            logger.info("Template loaded and cached", template_path=template_path, version=version)
            return (template, version)
        except Exception as e:
            logger.error("Failed to load template", template_path=template_path, error=str(e))
            return None

    def render_template(self, template_path: str, **kwargs) -> Optional[str]:
        """
        Render a template with the provided variables.

        Args:
            template_path: Path to the template file (relative to prompts_dir)
            **kwargs: Variables to pass to the template

        Returns:
            Rendered template string or None if rendering failed
        """
        template_result = self.get_template(template_path)
        if template_result is None:
            return None

        template, version = template_result
        try:
            rendered = template.render(**kwargs)
            logger.info("Template rendered successfully", template_path=template_path, version=version)
            return rendered
        except Exception as e:
            logger.error("Failed to render template", template_path=template_path, error=str(e))
            return None

    def get_template_version(self, template_path: str) -> str:
        """
        Get the version of a template without rendering it.

        Args:
            template_path: Path to the template file (relative to prompts_dir)

        Returns:
            Version string or '1.0' if not found
        """
        template_result = self.get_template(template_path)
        if template_result is None:
            return "1.0"
        return template_result[1]

    def invalidate_cache(self, template_path: Optional[str] = None) -> None:
        """
        Invalidate the template cache.

        Args:
            template_path: Path to specific template to invalidate, or None to clear all
        """
        if template_path:
            if template_path in self._template_cache:
                del self._template_cache[template_path]
                logger.info("Specific template cache invalidated", template_path=template_path)
        else:
            self._template_cache.clear()
            logger.info("Template cache cleared")

    def list_available_templates(self) -> Dict[str, Dict[str, str]]:
        """
        List all available template files with their paths and versions.

        Returns:
            Dictionary mapping template names to their info (path and version)
        """
        templates = {}
        for root, dirs, files in os.walk(self.prompts_dir):
            for file in files:
                if file.endswith('.j2'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.prompts_dir)
                    version = self.extract_version_from_template(rel_path)
                    templates[rel_path] = {
                        "path": full_path,
                        "version": version
                    }
        logger.info("Available templates listed", count=len(templates))
        return templates


# Global prompt manager instance for convenience
prompt_manager = PromptManager()


def render_rag_query_prompt(context: str, query: str) -> Optional[str]:
    """
    Render the RAG query prompt.

    Args:
        context: The retrieved context for the query
        query: The user's query

    Returns:
        Rendered prompt string or None if rendering failed
    """
    return prompt_manager.render_template(
        "rag/rag_query.j2",
        context=context,
        query=query
    )


def render_entity_extraction_prompt(doc_type: str, text: str) -> Optional[str]:
    """
    Render the entity extraction prompt.

    Args:
        doc_type: Type of document being processed
        text: Text content of the document

    Returns:
        Rendered prompt string or None if rendering failed
    """
    return prompt_manager.render_template(
        "extraction/entity_extraction.j2",
        doc_type=doc_type,
        text=text
    )


def render_summarization_prompt(doc_type: str, text_truncated: str, max_length: int) -> Optional[str]:
    """
    Render the summarization prompt.

    Args:
        doc_type: Type of document being processed
        text_truncated: Truncated text content of the document
        max_length: Maximum length for the summary

    Returns:
        Rendered prompt string or None if rendering failed
    """
    return prompt_manager.render_template(
        "summarization/document_summarization.j2",
        doc_type=doc_type,
        text_truncated=text_truncated,
        max_length=max_length
    )


def get_prompt_version(template_path: str) -> str:
    """
    Get the version of a prompt template.

    Args:
        template_path: Path to the template file (relative to prompts_dir)

    Returns:
        Version string of the prompt template
    """
    return prompt_manager.get_template_version(template_path)