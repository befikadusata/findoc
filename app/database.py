import sqlite3
from datetime import datetime
from typing import Optional
import os

# Database file path
DB_PATH = os.getenv('DATABASE_PATH', './data/findocai.db')

def init_db():
    """Initialize the database and create the documents table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create the documents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'uploaded',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create an index on doc_id for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_id ON documents (doc_id)')
    
    conn.commit()
    conn.close()

def create_document_record(doc_id: str, filename: str) -> bool:
    """Create a new document record in the database with 'uploaded' status."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO documents (doc_id, filename, status)
            VALUES (?, ?, ?)
        ''', (doc_id, filename, 'uploaded'))
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error:
        return False

def update_document_status(doc_id: str, status: str) -> bool:
    """Update the status of a document in the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE documents
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE doc_id = ?
        ''', (status, doc_id))
        
        conn.commit()
        conn.close()
        return cursor.rowcount > 0
    except sqlite3.Error:
        return False

def get_document_status(doc_id: str) -> Optional[dict]:
    """Get the status of a document from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This allows us to access columns by name
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT doc_id, filename, status, created_at, updated_at
            FROM documents
            WHERE doc_id = ?
        ''', (doc_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    except sqlite3.Error:
        return None

def get_all_documents() -> list:
    """Get all documents from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT doc_id, filename, status, created_at, updated_at FROM documents')
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []