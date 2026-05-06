from functools import lru_cache

from app.core.config import get_settings
from app.services.speaker_service import SpeakerVerificationService


@lru_cache(maxsize=1)
def get_speaker_service() -> SpeakerVerificationService:
    """Create one speaker model instance and reuse it across all API routes."""

    return SpeakerVerificationService(get_settings())
