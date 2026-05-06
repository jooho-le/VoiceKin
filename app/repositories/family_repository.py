from dataclasses import dataclass
from pathlib import Path

from app.db.session import get_connection


@dataclass(frozen=True)
class FamilyMemberRecord:
    id: int
    name: str
    relation: str
    embedding: bytes
    model_name: str


class FamilyRepository:
    """SQLite repository for registered family voiceprints."""

    def __init__(self, database_path: Path):
        self.database_path = database_path

    def create(
        self,
        name: str,
        relation: str,
        embedding: bytes,
        model_name: str,
    ) -> FamilyMemberRecord:
        with get_connection(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO family_members (name, relation, embedding, model_name)
                VALUES (?, ?, ?, ?)
                """,
                (name, relation, embedding, model_name),
            )
            connection.commit()
            family_id = int(cursor.lastrowid)

        record = self.get(family_id)
        if record is None:
            raise RuntimeError("failed to load created family member")
        return record

    def list_all(self) -> list[FamilyMemberRecord]:
        with get_connection(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT id, name, relation, embedding, model_name
                FROM family_members
                ORDER BY id ASC
                """
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def get(self, family_id: int) -> FamilyMemberRecord | None:
        with get_connection(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT id, name, relation, embedding, model_name
                FROM family_members
                WHERE id = ?
                """,
                (family_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_record(row)

    def delete(self, family_id: int) -> bool:
        with get_connection(self.database_path) as connection:
            cursor = connection.execute(
                "DELETE FROM family_members WHERE id = ?",
                (family_id,),
            )
            connection.commit()
            return cursor.rowcount > 0

    @staticmethod
    def _row_to_record(row) -> FamilyMemberRecord:
        return FamilyMemberRecord(
            id=int(row["id"]),
            name=str(row["name"]),
            relation=str(row["relation"]),
            embedding=bytes(row["embedding"]),
            model_name=str(row["model_name"]),
        )
