#!/usr/bin/env python3
"""
Database system for PDF Redactor Tool
Stores file metadata and enables web access to processed files
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import hashlib

class RedactionDatabase:
    def __init__(self, db_path: str = "redaction_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processed_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_filename TEXT NOT NULL,
                    redacted_filename TEXT NOT NULL,
                    audit_filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    total_redactions INTEGER DEFAULT 0,
                    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_size_bytes INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'completed'
                )
            ''')
            
            # Create sessions table for web access
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS access_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            conn.commit()
    
    def add_processed_file(self, original_filename: str, redacted_filename: str, 
                          audit_filename: str, total_redactions: int, 
                          file_size_bytes: int) -> int:
        """Add a processed file to the database"""
        # Generate file hash for deduplication
        file_hash = self._generate_file_hash(redacted_filename)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if file already exists
            cursor.execute('''
                SELECT id FROM processed_files WHERE file_hash = ?
            ''', (file_hash,))
            
            existing = cursor.fetchone()
            if existing and existing[0] is not None:
                return int(existing[0])
            
            # Insert new file
            cursor.execute('''
                INSERT INTO processed_files 
                (original_filename, redacted_filename, audit_filename, file_hash, 
                 total_redactions, file_size_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (original_filename, redacted_filename, audit_filename, file_hash,
                  total_redactions, file_size_bytes))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_all_processed_files(self) -> List[Dict]:
        """Get all processed files"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM processed_files 
                ORDER BY processing_date DESC
            ''')
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_file_by_id(self, file_id: int) -> Optional[Dict]:
        """Get a specific file by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM processed_files WHERE id = ?
            ''', (file_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def create_access_session(self) -> str:
        """Create a new access session for web users"""
        import uuid
        session_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO access_sessions (session_id) VALUES (?)
            ''', (session_id,))
            
            conn.commit()
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate if a session is active"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT is_active FROM access_sessions 
                WHERE session_id = ? AND is_active = 1
            ''', (session_id,))
            
            return cursor.fetchone() is not None
    
    def update_session_access(self, session_id: str):
        """Update last access time for a session"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE access_sessions 
                SET last_accessed = CURRENT_TIMESTAMP 
                WHERE session_id = ?
            ''', (session_id,))
            
            conn.commit()
    
    def get_file_stats(self) -> Dict:
        """Get overall statistics about processed files"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_files,
                    SUM(total_redactions) as total_redactions,
                    SUM(file_size_bytes) as total_size_bytes,
                    MIN(processing_date) as first_processed,
                    MAX(processing_date) as last_processed
                FROM processed_files
            ''')
            
            row = cursor.fetchone()
            return {
                'total_files': row[0] or 0,
                'total_redactions': row[1] or 0,
                'total_size_mb': (row[2] or 0) / (1024 * 1024),
                'first_processed': row[3],
                'last_processed': row[4]
            }
    
    def _generate_file_hash(self, filepath: str) -> str:
        """Generate a hash for file deduplication"""
        if not os.path.exists(filepath):
            return hashlib.md5(filepath.encode()).hexdigest()
        
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

# Global database instance
db = RedactionDatabase() 