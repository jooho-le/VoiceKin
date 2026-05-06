from functools import lru_cache

from app.core.config import get_settings
from app.services.anti_spoofing_service import AntiSpoofingService
from app.services.speaker_service import SpeakerVerificationService


@lru_cache(maxsize=1)
def get_speaker_service() -> SpeakerVerificationService:
    """Create one speaker model instance and reuse it across all API routes."""

    return SpeakerVerificationService(get_settings())


@lru_cache(maxsize=1)
def get_anti_spoofing_service() -> AntiSpoofingService:
    """Create one anti-spoofing model instance and reuse it across API routes."""

    return AntiSpoofingService(get_settings())
