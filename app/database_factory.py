"""
Database Factory Module for FinDocAI

This module provides a factory pattern to switch between different database implementations
(psycopg2 vs SQLAlchemy) with a unified interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from app.database import (
    init_db as init_db_psycopg2,
    create_document_record as create_document_record_psycopg2,
    update_document_status as update_document_status_psycopg2,
    get_document_status as get_document_status_psycopg2,
    get_all_documents as get_all_documents_psycopg2,
    update_document_summary as update_document_summary_psycopg2,
    get_document_summary as get_document_summary_psycopg2,
    update_document_entities as update_document_entities_psycopg2,
    get_document_entities as get_document_entities_psycopg2,
    delete_document_record as delete_document_record_psycopg2
)
from app.database_sqlalchemy import (
    init_db as init_db_sqlalchemy,
    create_document_record as create_document_record_sqlalchemy,
    update_document_status as update_document_status_sqlalchemy,
    get_document_status as get_document_status_sqlalchemy,
    get_all_documents as get_all_documents_sqlalchemy,
    update_document_summary as update_document_summary_sqlalchemy,
    get_document_summary as get_document_summary_sqlalchemy,
    update_document_entities as update_document_entities_sqlalchemy,
    get_document_entities as get_document_entities_sqlalchemy,
    delete_document_record as delete_document_record_sqlalchemy
)

class DatabaseInterface(ABC):
    """Abstract base class defining the database interface"""
    
    @abstractmethod
    def init_db(self) -> None:
        pass
    
    @abstractmethod
    def create_document_record(self, doc_id: str, filename: str) -> bool:
        pass
    
    @abstractmethod
    def update_document_status(self, doc_id: str, status: str) -> bool:
        pass
    
    @abstractmethod
    def get_document_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_all_documents(self) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def update_document_summary(self, doc_id: str, summary_data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def update_document_entities(self, doc_id: str, entities_data: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def get_document_entities(self, doc_id: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def delete_document_record(self, doc_id: str) -> bool:
        pass


class Psycopg2Database(DatabaseInterface):
    """Database implementation using psycopg2"""
    
    def init_db(self) -> None:
        init_db_psycopg2()
    
    def create_document_record(self, doc_id: str, filename: str) -> bool:
        return create_document_record_psycopg2(doc_id, filename)
    
    def update_document_status(self, doc_id: str, status: str) -> bool:
        return update_document_status_psycopg2(doc_id, status)
    
    def get_document_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return get_document_status_psycopg2(doc_id)
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        return get_all_documents_psycopg2()
    
    def update_document_summary(self, doc_id: str, summary_data: Dict[str, Any]) -> bool:
        return update_document_summary_psycopg2(doc_id, summary_data)
    
    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return get_document_summary_psycopg2(doc_id)
    
    def update_document_entities(self, doc_id: str, entities_data: Dict[str, Any]) -> bool:
        return update_document_entities_psycopg2(doc_id, entities_data)
    
    def get_document_entities(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return get_document_entities_psycopg2(doc_id)
    
    def delete_document_record(self, doc_id: str) -> bool:
        return delete_document_record_psycopg2(doc_id)


class SQLAlchemyDatabase(DatabaseInterface):
    """Database implementation using SQLAlchemy"""
    
    def init_db(self) -> None:
        init_db_sqlalchemy()
    
    def create_document_record(self, doc_id: str, filename: str) -> bool:
        return create_document_record_sqlalchemy(doc_id, filename)
    
    def update_document_status(self, doc_id: str, status: str) -> bool:
        return update_document_status_sqlalchemy(doc_id, status)
    
    def get_document_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return get_document_status_sqlalchemy(doc_id)
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        return get_all_documents_sqlalchemy()
    
    def update_document_summary(self, doc_id: str, summary_data: Dict[str, Any]) -> bool:
        return update_document_summary_sqlalchemy(doc_id, summary_data)
    
    def get_document_summary(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return get_document_summary_sqlalchemy(doc_id)
    
    def update_document_entities(self, doc_id: str, entities_data: Dict[str, Any]) -> bool:
        return update_document_entities_sqlalchemy(doc_id, entities_data)
    
    def get_document_entities(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return get_document_entities_sqlalchemy(doc_id)
    
    def delete_document_record(self, doc_id: str) -> bool:
        return delete_document_record_sqlalchemy(doc_id)


def get_database_implementation(use_sqlalchemy: bool = True) -> DatabaseInterface:
    """
    Factory function to get the appropriate database implementation.
    
    Args:
        use_sqlalchemy: If True, returns SQLAlchemy implementation, otherwise psycopg2
    
    Returns:
        DatabaseInterface implementation
    """
    if use_sqlalchemy:
        return SQLAlchemyDatabase()
    else:
        return Psycopg2Database()


# Create a global instance of the database using the preferred implementation
# For now, we'll default to SQLAlchemy which provides better maintainability
database = get_database_implementation(use_sqlalchemy=True)