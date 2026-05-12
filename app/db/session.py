import sqlite3
from pathlib import Path

from app.core.config import Settings


FAMILY_MEMBERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS family_members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    relation TEXT NOT NULL,
    embedding BLOB NOT NULL,
    model_name TEXT NOT NULL
);
"""

VOICE_SESSIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_sessions (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    ended_at TEXT
);
"""

VOICE_SESSION_CHUNKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_session_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    final_decision TEXT NOT NULL,
    is_trusted_chunk INTEGER NOT NULL,
    is_spoofed INTEGER NOT NULL,
    spoof_score REAL NOT NULL,
    anti_spoofing_message TEXT NOT NULL,
    is_registered_family INTEGER NOT NULL,
    best_family_id INTEGER,
    best_family_name TEXT,
    best_family_relation TEXT,
    best_family_similarity REAL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES voice_sessions(id)
);
"""


def init_db(settings: Settings) -> None:
    """Create the SQLite database and required tables if they do not exist."""

    settings.database_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(settings.database_path) as connection:
        connection.execute(FAMILY_MEMBERS_SCHEMA)
        connection.execute(VOICE_SESSIONS_SCHEMA)
        connection.execute(VOICE_SESSION_CHUNKS_SCHEMA)
        connection.commit()


def get_connection(database_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name."""

    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    return connection
