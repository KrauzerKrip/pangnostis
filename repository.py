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
                # Drop table if 'when' exists (old schema) or 'summary' exists (older schema)
                if "when" in columns or "summary" in columns:
                    cursor.execute("DROP TABLE transcriptions")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    transcription TEXT NOT NULL,
                    who TEXT NOT NULL,
                    what TEXT NOT NULL,
                    "where" TEXT NOT NULL,
                    context_vector_helper TEXT NOT NULL,
                    who_embedding BLOB NOT NULL,
                    what_embedding BLOB NOT NULL,
                    where_embedding BLOB NOT NULL,
                    context_embedding BLOB NOT NULL
                )
            """)
            conn.commit()

    def save(self, transcription: Transcription):
        """Save a transcription to the database."""
        # Ensure embeddings are float32 and in little-endian as required by sqlite-vec
        who_emb = transcription.who_embedding.astype("<f4").tobytes()
        what_emb = transcription.what_embedding.astype("<f4").tobytes()
        where_emb = transcription.where_embedding.astype("<f4").tobytes()
        context_emb = transcription.context_embedding.astype("<f4").tobytes()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if transcription.id is not None:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO transcriptions (
                        id, filename, transcription, who, what, "where",
                        context_vector_helper, who_embedding, what_embedding,
                        where_embedding, context_embedding
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        transcription.id,
                        transcription.filename,
                        transcription.transcription,
                        transcription.who,
                        transcription.what,
                        transcription.where,
                        transcription.context_vector_helper,
                        who_emb,
                        what_emb,
                        where_emb,
                        context_emb,
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO transcriptions (
                        filename, transcription, who, what, "where",
                        context_vector_helper, who_embedding, what_embedding,
                        where_embedding, context_embedding
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        transcription.filename,
                        transcription.transcription,
                        transcription.who,
                        transcription.what,
                        transcription.where,
                        transcription.context_vector_helper,
                        who_emb,
                        what_emb,
                        where_emb,
                        context_emb,
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
                SELECT id, filename, transcription, who, what, "where",
                       context_vector_helper, who_embedding, what_embedding,
                       where_embedding, context_embedding
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
                    who=row[3],
                    what=row[4],
                    where=row[5],
                    context_vector_helper=row[6],
                    who_embedding=np.frombuffer(row[7], dtype="<f4"),
                    what_embedding=np.frombuffer(row[8], dtype="<f4"),
                    where_embedding=np.frombuffer(row[9], dtype="<f4"),
                    context_embedding=np.frombuffer(row[10], dtype="<f4"),
                )
            return None

    def get_all(self) -> List[Transcription]:
        """Retrieve all transcriptions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, filename, transcription, who, what, "where",
                       context_vector_helper, who_embedding, what_embedding,
                       where_embedding, context_embedding
                FROM transcriptions
            """)
            rows = cursor.fetchall()

            return [
                Transcription(
                    id=row[0],
                    filename=row[1],
                    transcription=row[2],
                    who=row[3],
                    what=row[4],
                    where=row[5],
                    context_vector_helper=row[6],
                    who_embedding=np.frombuffer(row[7], dtype="<f4"),
                    what_embedding=np.frombuffer(row[8], dtype="<f4"),
                    where_embedding=np.frombuffer(row[9], dtype="<f4"),
                    context_embedding=np.frombuffer(row[10], dtype="<f4"),
                )
                for row in rows
            ]

    def get_by_id(self, id: int) -> Optional[Transcription]:
        """Retrieve a transcription by id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, filename, transcription, who, what, "where",
                       context_vector_helper, who_embedding, what_embedding,
                       where_embedding, context_embedding
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
                    who=row[3],
                    what=row[4],
                    where=row[5],
                    context_vector_helper=row[6],
                    who_embedding=np.frombuffer(row[7], dtype="<f4"),
                    what_embedding=np.frombuffer(row[8], dtype="<f4"),
                    where_embedding=np.frombuffer(row[9], dtype="<f4"),
                    context_embedding=np.frombuffer(row[10], dtype="<f4"),
                )
            return None
