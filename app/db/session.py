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


def init_db(settings: Settings) -> None:
    """Create the SQLite database and required tables if they do not exist."""

    settings.database_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(settings.database_path) as connection:
        connection.execute(FAMILY_MEMBERS_SCHEMA)
        connection.commit()


def get_connection(database_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name."""

    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    return connection
