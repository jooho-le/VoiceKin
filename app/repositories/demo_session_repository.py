from dataclasses import dataclass
from pathlib import Path

from app.db.session import get_connection
from app.repositories.voice_session_repository import utc_now_iso


@dataclass(frozen=True)
class DemoSessionRecord:
    id: str
    sample_path: Path
    actual_label: str
    status: str
    created_at: str
    answered_at: str | None
    user_guess: str | None


class DemoSessionRepository:
    """SQLite repository for the no-cost sample-based user demo."""

    def __init__(self, database_path: Path):
        self.database_path = database_path

    def create(
        self,
        session_id: str,
        sample_path: Path,
        actual_label: str,
    ) -> DemoSessionRecord:
        now = utc_now_iso()
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO demo_sessions (
                    id,
                    sample_path,
                    actual_label,
                    status,
                    created_at,
                    answered_at,
                    user_guess
                )
                VALUES (?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    session_id,
                    str(sample_path),
                    actual_label,
                    "active",
                    now,
                ),
            )
            connection.commit()

        record = self.get(session_id)
        if record is None:
            raise RuntimeError("failed to load created demo session")
        return record

    def get(self, session_id: str) -> DemoSessionRecord | None:
        with get_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT id, sample_path, actual_label, status, created_at, answered_at, user_guess
                FROM demo_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_record(row)

    def mark_answered(self, session_id: str, user_guess: str) -> DemoSessionRecord | None:
        now = utc_now_iso()
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                UPDATE demo_sessions
                SET status = ?, answered_at = ?, user_guess = ?
                WHERE id = ?
                """,
                ("answered", now, user_guess, session_id),
            )
            connection.commit()

        return self.get(session_id)

    @staticmethod
    def _row_to_record(row) -> DemoSessionRecord:
        return DemoSessionRecord(
            id=str(row["id"]),
            sample_path=Path(str(row["sample_path"])),
            actual_label=str(row["actual_label"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            answered_at=(
                str(row["answered_at"])
                if row["answered_at"] is not None
                else None
            ),
            user_guess=(
                str(row["user_guess"])
                if row["user_guess"] is not None
                else None
            ),
        )
