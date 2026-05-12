from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.db.session import get_connection


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string for SQLite records."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class VoiceSessionRecord:
    id: str
    status: str
    created_at: str
    updated_at: str
    ended_at: str | None


@dataclass(frozen=True)
class VoiceSessionChunkRecord:
    id: int
    session_id: str
    chunk_index: int
    final_decision: str
    is_trusted_chunk: bool
    is_spoofed: bool
    spoof_score: float
    anti_spoofing_message: str
    is_registered_family: bool
    best_family_id: int | None
    best_family_name: str | None
    best_family_relation: str | None
    best_family_similarity: float | None
    created_at: str


class VoiceSessionRepository:
    """SQLite repository for chunk-based continuous voice analysis sessions."""

    def __init__(self, database_path: Path):
        self.database_path = database_path

    def create(self, session_id: str) -> VoiceSessionRecord:
        now = utc_now_iso()
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO voice_sessions (id, status, created_at, updated_at, ended_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (session_id, "active", now, now),
            )
            connection.commit()

        record = self.get(session_id)
        if record is None:
            raise RuntimeError("failed to load created voice session")
        return record

    def get(self, session_id: str) -> VoiceSessionRecord | None:
        with get_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT id, status, created_at, updated_at, ended_at
                FROM voice_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_session(row)

    def end(self, session_id: str) -> VoiceSessionRecord | None:
        now = utc_now_iso()
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                UPDATE voice_sessions
                SET status = ?, updated_at = ?, ended_at = COALESCE(ended_at, ?)
                WHERE id = ?
                """,
                ("ended", now, now, session_id),
            )
            connection.commit()

        return self.get(session_id)

    def touch(self, session_id: str) -> None:
        now = utc_now_iso()
        with get_connection(self.database_path) as connection:
            connection.execute(
                """
                UPDATE voice_sessions
                SET updated_at = ?
                WHERE id = ?
                """,
                (now, session_id),
            )
            connection.commit()

    def count_chunks(self, session_id: str) -> int:
        with get_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS chunk_count
                FROM voice_session_chunks
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        return int(row["chunk_count"])

    def add_chunk(
        self,
        session_id: str,
        chunk_index: int,
        final_decision: str,
        is_trusted_chunk: bool,
        is_spoofed: bool,
        spoof_score: float,
        anti_spoofing_message: str,
        is_registered_family: bool,
        best_family_id: int | None,
        best_family_name: str | None,
        best_family_relation: str | None,
        best_family_similarity: float | None,
    ) -> VoiceSessionChunkRecord:
        now = utc_now_iso()
        with get_connection(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO voice_session_chunks (
                    session_id,
                    chunk_index,
                    final_decision,
                    is_trusted_chunk,
                    is_spoofed,
                    spoof_score,
                    anti_spoofing_message,
                    is_registered_family,
                    best_family_id,
                    best_family_name,
                    best_family_relation,
                    best_family_similarity,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    chunk_index,
                    final_decision,
                    int(is_trusted_chunk),
                    int(is_spoofed),
                    spoof_score,
                    anti_spoofing_message,
                    int(is_registered_family),
                    best_family_id,
                    best_family_name,
                    best_family_relation,
                    best_family_similarity,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE voice_sessions
                SET updated_at = ?
                WHERE id = ?
                """,
                (now, session_id),
            )
            connection.commit()
            chunk_id = int(cursor.lastrowid)

        chunk = self.get_chunk(chunk_id)
        if chunk is None:
            raise RuntimeError("failed to load created voice session chunk")
        return chunk

    def get_chunk(self, chunk_id: int) -> VoiceSessionChunkRecord | None:
        with get_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    session_id,
                    chunk_index,
                    final_decision,
                    is_trusted_chunk,
                    is_spoofed,
                    spoof_score,
                    anti_spoofing_message,
                    is_registered_family,
                    best_family_id,
                    best_family_name,
                    best_family_relation,
                    best_family_similarity,
                    created_at
                FROM voice_session_chunks
                WHERE id = ?
                """,
                (chunk_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_chunk(row)

    def list_chunks(self, session_id: str) -> list[VoiceSessionChunkRecord]:
        with get_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    session_id,
                    chunk_index,
                    final_decision,
                    is_trusted_chunk,
                    is_spoofed,
                    spoof_score,
                    anti_spoofing_message,
                    is_registered_family,
                    best_family_id,
                    best_family_name,
                    best_family_relation,
                    best_family_similarity,
                    created_at
                FROM voice_session_chunks
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return [self._row_to_chunk(row) for row in rows]

    @staticmethod
    def _row_to_session(row) -> VoiceSessionRecord:
        return VoiceSessionRecord(
            id=str(row["id"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            ended_at=str(row["ended_at"]) if row["ended_at"] is not None else None,
        )

    @staticmethod
    def _row_to_chunk(row) -> VoiceSessionChunkRecord:
        return VoiceSessionChunkRecord(
            id=int(row["id"]),
            session_id=str(row["session_id"]),
            chunk_index=int(row["chunk_index"]),
            final_decision=str(row["final_decision"]),
            is_trusted_chunk=bool(row["is_trusted_chunk"]),
            is_spoofed=bool(row["is_spoofed"]),
            spoof_score=float(row["spoof_score"]),
            anti_spoofing_message=str(row["anti_spoofing_message"]),
            is_registered_family=bool(row["is_registered_family"]),
            best_family_id=(
                int(row["best_family_id"])
                if row["best_family_id"] is not None
                else None
            ),
            best_family_name=(
                str(row["best_family_name"])
                if row["best_family_name"] is not None
                else None
            ),
            best_family_relation=(
                str(row["best_family_relation"])
                if row["best_family_relation"] is not None
                else None
            ),
            best_family_similarity=(
                float(row["best_family_similarity"])
                if row["best_family_similarity"] is not None
                else None
            ),
            created_at=str(row["created_at"]),
        )
