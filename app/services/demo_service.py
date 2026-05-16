import random
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.core.config import Settings
from app.repositories.demo_session_repository import (
    DemoSessionRecord,
    DemoSessionRepository,
)


class DemoSampleNotFoundError(Exception):
    """Raised when no real/fake demo samples are available."""


class DemoSessionNotFoundError(Exception):
    """Raised when a demo session id does not exist."""


class DemoSampleFileMissingError(Exception):
    """Raised when a sample file recorded in a session no longer exists."""


@dataclass(frozen=True)
class DemoSample:
    path: Path
    label: str


@dataclass(frozen=True)
class DemoStartResult:
    session_id: str
    audio_url: str
    playback_seconds: int
    message: str


class DemoService:
    """No-cost demo service using pre-generated local real/fake audio samples."""

    def __init__(
        self,
        settings: Settings,
        demo_session_repository: DemoSessionRepository,
    ):
        self.settings = settings
        self.demo_session_repository = demo_session_repository

    def start_demo_session(self) -> DemoStartResult:
        sample = random.choice(self._list_samples())
        session_id = uuid4().hex
        record = self.demo_session_repository.create(
            session_id=session_id,
            sample_path=sample.path,
            actual_label=sample.label,
        )

        return DemoStartResult(
            session_id=record.id,
            audio_url=f"{self.settings.api_v1_prefix}/demo-sessions/{record.id}/audio",
            playback_seconds=10,
            message="demo_session_started",
        )

    def get_session(self, session_id: str) -> DemoSessionRecord:
        record = self.demo_session_repository.get(session_id)
        if record is None:
            raise DemoSessionNotFoundError("demo session not found")
        if not record.sample_path.exists():
            raise DemoSampleFileMissingError("demo sample file is missing")
        return record

    def submit_answer(self, session_id: str, user_guess: str) -> DemoSessionRecord:
        normalized_guess = self._normalize_guess(user_guess)
        record = self.get_session(session_id)
        updated = self.demo_session_repository.mark_answered(
            session_id=record.id,
            user_guess=normalized_guess,
        )
        if updated is None:
            raise DemoSessionNotFoundError("demo session not found")
        return updated

    def _list_samples(self) -> list[DemoSample]:
        samples: list[DemoSample] = []
        allowed_extensions = {
            extension.lower().lstrip(".")
            for extension in self.settings.allowed_audio_extensions
        }

        for label in ("real", "fake"):
            sample_dir = self.settings.demo_sample_dir / label
            if not sample_dir.exists():
                continue

            for path in sorted(sample_dir.rglob("*")):
                if path.is_file() and path.suffix.lower().lstrip(".") in allowed_extensions:
                    samples.append(DemoSample(path=path.resolve(), label=label))

        if not samples:
            raise DemoSampleNotFoundError(
                "no demo samples found. Put audio files under demo_samples/real and demo_samples/fake"
            )
        return samples

    @staticmethod
    def _normalize_guess(user_guess: str) -> str:
        value = user_guess.strip().lower()
        if value in {"real", "human", "actual", "recording", "실제", "사람"}:
            return "real"
        if value in {"fake", "ai", "spoof", "synthetic", "deepvoice", "기계", "ai음성"}:
            return "fake"
        raise ValueError("user_guess must be real or fake")
