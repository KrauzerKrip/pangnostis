import sqlite3
import numpy as np
from typing import List, Optional
from pathlib import Path
from models import Transcription
import sqlite_vec

class TranscriptionRepository:
    def __init__(self, db_path: str = ".data/pangnostis.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _init_db(self):
        """Initialize the database and create the table if it doesn't exist."""
        # Ensure the directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    filename TEXT PRIMARY KEY,
                    transcription TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            conn.commit()

    def save(self, transcription: Transcription):
        """Save a transcription to the database."""
        # Ensure embedding is float32 and in native byte order (or little-endian as required by sqlite-vec)
        # sqlite-vec expects little-endian float32. 
        # numpy.ndarray.tobytes() uses system byte order. 
        # To be safe and explicit:
        embedding_blob = transcription.embedding.astype('<f4').tobytes()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO transcriptions (filename, transcription, summary, embedding)
                VALUES (?, ?, ?, ?)
            """, (
                transcription.filename,
                transcription.transcription,
                transcription.summary,
                embedding_blob
            ))
            conn.commit()

    def get_by_filename(self, filename: str) -> Optional[Transcription]:
        """Retrieve a transcription by filename."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, transcription, summary, embedding FROM transcriptions WHERE filename = ?", (filename,))
            row = cursor.fetchone()
            
            if row:
                return Transcription(
                    filename=row[0],
                    transcription=row[1],
                    summary=row[2],
                    embedding=np.frombuffer(row[3], dtype='<f4')
                )
            return None

    def get_all(self) -> List[Transcription]:
        """Retrieve all transcriptions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, transcription, summary, embedding FROM transcriptions")
            rows = cursor.fetchall()
            
            return [
                Transcription(
                    filename=row[0],
                    transcription=row[1],
                    summary=row[2],
                    embedding=np.frombuffer(row[3], dtype='<f4')
                )
                for row in rows
            ]
