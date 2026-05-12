from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.repositories.family_repository import FamilyRepository
from app.repositories.voice_session_repository import (
    VoiceSessionChunkRecord,
    VoiceSessionRecord,
    VoiceSessionRepository,
)
from app.services.anti_spoofing_service import AntiSpoofingResult, AntiSpoofingService
from app.services.speaker_service import SpeakerVerificationService
from app.services.voiceprint_service import (
    FamilyVerificationResult,
    FamilyVoiceMatch,
    VoiceprintService,
)
from app.utils.audio_quality import AudioQualityResult


class VoiceSessionNotFoundError(Exception):
    """Raised when a voice analysis session id does not exist."""


class VoiceSessionClosedError(Exception):
    """Raised when a chunk is uploaded to an ended session."""


@dataclass(frozen=True)
class VoiceSessionStatus:
    session_id: str
    status: str
    created_at: str
    updated_at: str
    ended_at: str | None
    chunks_analyzed: int
    total_chunks: int
    analyzable_chunks: int
    skipped_chunks: int
    is_spoofed: bool
    is_registered_family: bool
    risk_level: str
    message: str
    max_spoof_score: float
    max_spoof_chunk_index: int | None
    suspicious_chunks: int
    required_spoof_chunks: int
    strong_spoof_score: float
    best_family_match: FamilyVoiceMatch | None
    family_match_chunks: int
    required_family_match_chunks: int
    speaker_threshold: float
    anti_spoofing_threshold: float


@dataclass(frozen=True)
class VoiceSessionChunkAnalysis:
    session_id: str
    chunk_index: int
    is_analyzable: bool
    quality: AudioQualityResult
    is_trusted_chunk: bool
    final_decision: str
    family_verification: FamilyVerificationResult | None
    anti_spoofing: AntiSpoofingResult | None
    rolling_result: VoiceSessionStatus


class VoiceSessionService:
    """Service for near-real-time style voice analysis with uploaded chunks."""

    def __init__(
        self,
        voice_session_repository: VoiceSessionRepository,
        family_repository: FamilyRepository,
        speaker_threshold: float,
        anti_spoofing_threshold: float,
        repeated_spoof_chunks: int,
        strong_spoof_score: float,
        family_confirm_chunks: int,
        speaker_service: SpeakerVerificationService | None = None,
        anti_spoofing_service: AntiSpoofingService | None = None,
    ):
        self.voice_session_repository = voice_session_repository
        self.family_repository = family_repository
        self.speaker_threshold = speaker_threshold
        self.anti_spoofing_threshold = anti_spoofing_threshold
        self.repeated_spoof_chunks = repeated_spoof_chunks
        self.strong_spoof_score = strong_spoof_score
        self.family_confirm_chunks = family_confirm_chunks
        self.speaker_service = speaker_service
        self.anti_spoofing_service = anti_spoofing_service

    def start_session(self) -> VoiceSessionStatus:
        """Create a new active session that will receive audio chunks."""

        session = self.voice_session_repository.create(uuid4().hex)
        return self._build_status(session)

    def get_status(self, session_id: str) -> VoiceSessionStatus:
        """Return rolling status computed from all stored chunks."""

        session = self.voice_session_repository.get(session_id)
        if session is None:
            raise VoiceSessionNotFoundError("voice session not found")
        return self._build_status(session)

    def end_session(self, session_id: str) -> VoiceSessionStatus:
        """Mark a session as ended and return its final rolling status."""

        session = self.voice_session_repository.end(session_id)
        if session is None:
            raise VoiceSessionNotFoundError("voice session not found")
        return self._build_status(session)

    def analyze_chunk(
        self,
        session_id: str,
        wav_path: Path,
        audio_quality: AudioQualityResult,
        chunk_index: int | None = None,
    ) -> VoiceSessionChunkAnalysis:
        """Analyze one normalized wav chunk and update the rolling session result."""

        session = self.voice_session_repository.get(session_id)
        if session is None:
            raise VoiceSessionNotFoundError("voice session not found")
        if session.status != "active":
            raise VoiceSessionClosedError("voice session is already ended")

        if chunk_index is None:
            chunk_index = self.voice_session_repository.count_chunks(session_id)
        if chunk_index < 0:
            raise ValueError("chunk_index must be greater than or equal to 0")

        if not audio_quality.is_analyzable:
            self.voice_session_repository.add_chunk(
                session_id=session_id,
                chunk_index=chunk_index,
                is_analyzable=False,
                quality_message=audio_quality.message,
                duration_seconds=audio_quality.duration_seconds,
                rms_energy=audio_quality.rms_energy,
                peak_amplitude=audio_quality.peak_amplitude,
                speech_ratio=audio_quality.speech_ratio,
                final_decision="low_quality_chunk_skipped",
                is_trusted_chunk=False,
                is_spoofed=False,
                spoof_score=0.0,
                anti_spoofing_message="not_analyzed",
                is_registered_family=False,
                best_family_id=None,
                best_family_name=None,
                best_family_relation=None,
                best_family_similarity=None,
            )
            return VoiceSessionChunkAnalysis(
                session_id=session_id,
                chunk_index=chunk_index,
                is_analyzable=False,
                quality=audio_quality,
                is_trusted_chunk=False,
                final_decision="low_quality_chunk_skipped",
                family_verification=None,
                anti_spoofing=None,
                rolling_result=self.get_status(session_id),
            )

        if self.speaker_service is None:
            raise RuntimeError("speaker service is required to analyze a voice chunk")
        if self.anti_spoofing_service is None:
            raise RuntimeError("anti-spoofing service is required to analyze a voice chunk")

        voiceprint_service = VoiceprintService(
            family_repository=self.family_repository,
            speaker_service=self.speaker_service,
        )

        family_result = voiceprint_service.verify_family_voice(wav_path)
        anti_spoofing_result = self.anti_spoofing_service.detect_file(wav_path)

        is_trusted_chunk, final_decision = self._make_secure_decision(
            is_registered_family=family_result.is_registered_family,
            is_spoofed=anti_spoofing_result.is_spoofed,
        )
        best_match = family_result.best_match

        self.voice_session_repository.add_chunk(
            session_id=session_id,
            chunk_index=chunk_index,
            is_analyzable=True,
            quality_message=audio_quality.message,
            duration_seconds=audio_quality.duration_seconds,
            rms_energy=audio_quality.rms_energy,
            peak_amplitude=audio_quality.peak_amplitude,
            speech_ratio=audio_quality.speech_ratio,
            final_decision=final_decision,
            is_trusted_chunk=is_trusted_chunk,
            is_spoofed=anti_spoofing_result.is_spoofed,
            spoof_score=anti_spoofing_result.spoof_score,
            anti_spoofing_message=anti_spoofing_result.message,
            is_registered_family=family_result.is_registered_family,
            best_family_id=best_match.family_id if best_match else None,
            best_family_name=best_match.name if best_match else None,
            best_family_relation=best_match.relation if best_match else None,
            best_family_similarity=best_match.similarity if best_match else None,
        )

        rolling_result = self.get_status(session_id)
        return VoiceSessionChunkAnalysis(
            session_id=session_id,
            chunk_index=chunk_index,
            is_analyzable=True,
            quality=audio_quality,
            is_trusted_chunk=is_trusted_chunk,
            final_decision=final_decision,
            family_verification=family_result,
            anti_spoofing=anti_spoofing_result,
            rolling_result=rolling_result,
        )

    def _build_status(self, session: VoiceSessionRecord) -> VoiceSessionStatus:
        chunks = self.voice_session_repository.list_chunks(session.id)
        total_chunks = len(chunks)
        analyzable_chunks = [chunk for chunk in chunks if chunk.is_analyzable]
        chunks_analyzed = len(analyzable_chunks)
        skipped_chunks = total_chunks - chunks_analyzed

        if not analyzable_chunks:
            message = "no_chunks_analyzed" if not chunks else "waiting_for_clear_voice"
            return VoiceSessionStatus(
                session_id=session.id,
                status=session.status,
                created_at=session.created_at,
                updated_at=session.updated_at,
                ended_at=session.ended_at,
                chunks_analyzed=0,
                total_chunks=total_chunks,
                analyzable_chunks=0,
                skipped_chunks=skipped_chunks,
                is_spoofed=False,
                is_registered_family=False,
                risk_level="unknown",
                message=message,
                max_spoof_score=0.0,
                max_spoof_chunk_index=None,
                suspicious_chunks=0,
                required_spoof_chunks=self.repeated_spoof_chunks,
                strong_spoof_score=self.strong_spoof_score,
                best_family_match=None,
                family_match_chunks=0,
                required_family_match_chunks=self.family_confirm_chunks,
                speaker_threshold=self.speaker_threshold,
                anti_spoofing_threshold=self.anti_spoofing_threshold,
            )

        max_spoof_chunk = max(analyzable_chunks, key=lambda chunk: chunk.spoof_score)
        suspicious_chunks = [
            chunk
            for chunk in analyzable_chunks
            if chunk.spoof_score >= self.anti_spoofing_threshold
        ]
        suspicious_chunk_count = len(suspicious_chunks)

        best_family_match = self._find_best_family_match(analyzable_chunks)
        family_match_chunks = self._count_best_family_matches(
            chunks=analyzable_chunks,
            best_family_match=best_family_match,
        )

        is_spoofed = (
            max_spoof_chunk.spoof_score >= self.strong_spoof_score
            or suspicious_chunk_count >= self.repeated_spoof_chunks
        )
        is_registered_family = (
            best_family_match is not None
            and best_family_match.similarity >= self.speaker_threshold
            and family_match_chunks >= self.family_confirm_chunks
        )
        risk_level, message = self._make_rolling_decision(
            is_registered_family=is_registered_family,
            is_spoofed=is_spoofed,
            has_spoof_warning=suspicious_chunk_count > 0,
            has_family_warning=(
                best_family_match is not None
                and best_family_match.similarity >= self.speaker_threshold
            ),
        )

        return VoiceSessionStatus(
            session_id=session.id,
            status=session.status,
            created_at=session.created_at,
            updated_at=session.updated_at,
            ended_at=session.ended_at,
            chunks_analyzed=chunks_analyzed,
            total_chunks=total_chunks,
            analyzable_chunks=chunks_analyzed,
            skipped_chunks=skipped_chunks,
            is_spoofed=is_spoofed,
            is_registered_family=is_registered_family,
            risk_level=risk_level,
            message=message,
            max_spoof_score=round(max_spoof_chunk.spoof_score, 4),
            max_spoof_chunk_index=max_spoof_chunk.chunk_index,
            suspicious_chunks=suspicious_chunk_count,
            required_spoof_chunks=self.repeated_spoof_chunks,
            strong_spoof_score=self.strong_spoof_score,
            best_family_match=best_family_match,
            family_match_chunks=family_match_chunks,
            required_family_match_chunks=self.family_confirm_chunks,
            speaker_threshold=self.speaker_threshold,
            anti_spoofing_threshold=self.anti_spoofing_threshold,
        )

    @staticmethod
    def _find_best_family_match(
        chunks: list[VoiceSessionChunkRecord],
    ) -> FamilyVoiceMatch | None:
        family_chunks = [
            chunk
            for chunk in chunks
            if chunk.best_family_id is not None
            and chunk.best_family_name is not None
            and chunk.best_family_relation is not None
            and chunk.best_family_similarity is not None
        ]
        if not family_chunks:
            return None

        best_chunk = max(
            family_chunks,
            key=lambda chunk: chunk.best_family_similarity or -1.0,
        )
        return FamilyVoiceMatch(
            family_id=best_chunk.best_family_id or 0,
            name=best_chunk.best_family_name or "",
            relation=best_chunk.best_family_relation or "",
            similarity=round(best_chunk.best_family_similarity or 0.0, 4),
        )

    def _count_best_family_matches(
        self,
        chunks: list[VoiceSessionChunkRecord],
        best_family_match: FamilyVoiceMatch | None,
    ) -> int:
        if best_family_match is None:
            return 0

        return sum(
            1
            for chunk in chunks
            if chunk.best_family_id == best_family_match.family_id
            and chunk.best_family_similarity is not None
            and chunk.best_family_similarity >= self.speaker_threshold
        )

    @staticmethod
    def _make_secure_decision(
        is_registered_family: bool,
        is_spoofed: bool,
    ) -> tuple[bool, str]:
        if is_registered_family and not is_spoofed:
            return True, "trusted_family_voice"
        if is_registered_family and is_spoofed:
            return False, "spoofed_family_like_voice"
        if not is_registered_family and not is_spoofed:
            return False, "unknown_real_voice"
        return False, "spoofed_unknown_voice"

    @staticmethod
    def _make_rolling_decision(
        is_registered_family: bool,
        is_spoofed: bool,
        has_spoof_warning: bool,
        has_family_warning: bool,
    ) -> tuple[str, str]:
        if is_registered_family and not is_spoofed:
            return "low", "registered_family_likely"
        if is_registered_family and is_spoofed:
            return "high", "spoofed_family_like_voice"
        if is_spoofed:
            return "high", "spoofed_unknown_voice"
        if has_spoof_warning:
            return "medium", "spoof_warning_needs_more_chunks"
        if has_family_warning:
            return "medium", "family_match_needs_more_chunks"
        if not is_registered_family:
            return "medium", "unknown_real_voice"
        return "high", "spoofed_unknown_voice"
