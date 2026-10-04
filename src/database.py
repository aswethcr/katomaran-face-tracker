import json
import os
import sqlite3
from contextlib import contextmanager

import numpy as np


class FaceDatabase:
    """SQLite persistence for registered faces and entry/exit events."""

    def __init__(self, db_path="database/faces.db"):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self.connection = sqlite3.connect(
            self.db_path,
            timeout=30,
        )
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.create_tables()

    @contextmanager
    def transaction(self):
        try:
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def create_tables(self):
        with self.transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    face_id TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ('ENTRY','EXIT')),
                    timestamp TEXT NOT NULL,
                    image_path TEXT,
                    FOREIGN KEY(face_id) REFERENCES faces(face_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_face_id
                ON events(face_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON events(timestamp)
            """)

    @staticmethod
    def _to_embedding(value):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(arr)
        if arr.size == 0 or not np.isfinite(norm) or norm < 1e-6:
            return None
        return arr / norm

    def get_all_faces(self):
        rows = self.connection.execute(
            "SELECT face_id, embedding FROM faces ORDER BY face_id"
        ).fetchall()

        faces = []
        for face_id, embedding_json in rows:
            try:
                embedding = self._to_embedding(json.loads(embedding_json))
            except (TypeError, ValueError, json.JSONDecodeError):
                embedding = None
            if embedding is not None:
                faces.append({
                    "face_id": face_id,
                    "embedding": embedding,
                })
        return faces

    def _next_face_id(self):
        rows = self.connection.execute(
            "SELECT face_id FROM faces WHERE face_id LIKE 'FACE_%'"
        ).fetchall()

        highest = 0
        for (face_id,) in rows:
            try:
                highest = max(highest, int(face_id.split("_")[-1]))
            except (ValueError, IndexError):
                continue
        return f"FACE_{highest + 1:04d}"

    def add_face(self, face_id, embedding, timestamp):
        embedding = self._to_embedding(embedding)
        if embedding is None:
            raise ValueError("Cannot store an invalid embedding")

        embedding_json = json.dumps(embedding.astype(float).tolist())

        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO faces(face_id, first_seen, embedding)
                VALUES (?, ?, ?)
                """,
                (face_id, timestamp, embedding_json),
            )

    def register_face(self, embedding, timestamp):
        face_id = self._next_face_id()
        self.add_face(face_id, embedding, timestamp)
        return face_id

    def add_event(self, face_id, event_type, timestamp, image_path=None):
        if event_type not in ("ENTRY", "EXIT"):
            raise ValueError(f"Unsupported event type: {event_type}")

        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events(
                    face_id, event_type, timestamp, image_path
                )
                VALUES (?, ?, ?, ?)
                """,
                (face_id, event_type, timestamp, image_path),
            )
            return cursor.lastrowid

    def get_event_count(self, event_type=None):
        if event_type is None:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM events"
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM events WHERE event_type=?",
                (event_type,),
            ).fetchone()
        return int(row[0])

    def get_unique_face_count(self):
        row = self.connection.execute(
            "SELECT COUNT(*) FROM faces"
        ).fetchone()
        return int(row[0])

    def close(self):
        try:
            self.connection.commit()
        finally:
            self.connection.close()
