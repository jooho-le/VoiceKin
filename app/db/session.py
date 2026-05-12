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
    is_analyzable INTEGER NOT NULL DEFAULT 1,
    quality_message TEXT NOT NULL DEFAULT 'not_recorded',
    duration_seconds REAL NOT NULL DEFAULT 0.0,
    rms_energy REAL NOT NULL DEFAULT 0.0,
    peak_amplitude REAL NOT NULL DEFAULT 0.0,
    speech_ratio REAL NOT NULL DEFAULT 0.0,
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
        _ensure_voice_session_chunk_columns(connection)
        connection.commit()


def get_connection(database_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name."""

    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_voice_session_chunk_columns(connection: sqlite3.Connection) -> None:
    """Add newly introduced columns when an older local SQLite DB already exists."""

    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(voice_session_chunks)").fetchall()
    }
    migrations = {
        "is_analyzable": "INTEGER NOT NULL DEFAULT 1",
        "quality_message": "TEXT NOT NULL DEFAULT 'not_recorded'",
        "duration_seconds": "REAL NOT NULL DEFAULT 0.0",
        "rms_energy": "REAL NOT NULL DEFAULT 0.0",
        "peak_amplitude": "REAL NOT NULL DEFAULT 0.0",
        "speech_ratio": "REAL NOT NULL DEFAULT 0.0",
    }

    for column_name, column_definition in migrations.items():
        if column_name not in columns:
            connection.execute(
                f"ALTER TABLE voice_session_chunks ADD COLUMN {column_name} {column_definition}"
            )
