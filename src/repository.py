import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np
import sqlite_vec

from models import Transcription


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
            # Check if table exists and has old schema
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='transcriptions'"
            )
            if cursor.fetchone():
                cursor.execute("PRAGMA table_info(transcriptions)")
                columns = [row[1] for row in cursor.fetchall()]
                # Drop table if 'who' exists (old schema)
                if "who" in columns:
                    cursor.execute("DROP TABLE transcriptions")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    transcription TEXT NOT NULL,
                    structured_data TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
            """)
            conn.commit()

    def save(self, transcription: Transcription):
        """Save a transcription to the database."""
        # Ensure embedding is float32 and in little-endian as required by sqlite-vec
        emb = transcription.embedding.astype("<f4").tobytes()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if transcription.id is not None:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO transcriptions (
                        id, filename, transcription, structured_data, embedding
                    )
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        transcription.id,
                        transcription.filename,
                        transcription.transcription,
                        transcription.structured_data,
                        emb,
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO transcriptions (
                        filename, transcription, structured_data, embedding
                    )
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        transcription.filename,
                        transcription.transcription,
                        transcription.structured_data,
                        emb,
                    ),
                )
                transcription.id = cursor.lastrowid
            conn.commit()

    def get_by_filename(self, filename: str) -> Optional[Transcription]:
        """Retrieve a transcription by filename."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, filename, transcription, structured_data, embedding
                FROM transcriptions WHERE filename = ?
            """,
                (filename,),
            )
            row = cursor.fetchone()

            if row:
                return Transcription(
                    id=row[0],
                    filename=row[1],
                    transcription=row[2],
                    structured_data=row[3],
                    embedding=np.frombuffer(row[4], dtype="<f4"),
                )
            return None

    def get_all(self) -> List[Transcription]:
        """Retrieve all transcriptions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, filename, transcription, structured_data, embedding
                FROM transcriptions
            """)
            rows = cursor.fetchall()

            return [
                Transcription(
                    id=row[0],
                    filename=row[1],
                    transcription=row[2],
                    structured_data=row[3],
                    embedding=np.frombuffer(row[4], dtype="<f4"),
                )
                for row in rows
            ]

    def get_by_id(self, id: int) -> Optional[Transcription]:
        """Retrieve a transcription by id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, filename, transcription, structured_data, embedding
                FROM transcriptions WHERE id = ?
            """,
                (id,),
            )
            row = cursor.fetchone()

            if row:
                return Transcription(
                    id=row[0],
                    filename=row[1],
                    transcription=row[2],
                    structured_data=row[3],
                    embedding=np.frombuffer(row[4], dtype="<f4"),
                )
            return None

    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding by exact text string."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT embedding FROM embedding_cache WHERE text = ?",
                (text,)
            )
            row = cursor.fetchone()
            if row:
                return np.frombuffer(row[0], dtype="<f4")
            return None

    def save_cached_embedding(self, text: str, embedding: np.ndarray):
        """Save an embedding to the cache."""
        # Ensure embedding is float32 and in little-endian as required
        emb_blob = embedding.astype("<f4").tobytes()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO embedding_cache (text, embedding)
                VALUES (?, ?)
                """,
                (text, emb_blob)
            )
            conn.commit()
