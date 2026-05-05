import logging
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.schemas.voice import VoiceCompareResponse
from app.services.speaker_service import (
    SpeakerVerificationError,
    SpeakerVerificationService,
)
from app.utils.audio import (
    AudioDecodingError,
    AudioTooShortError,
    AudioValidationError,
    MissingAudioFileError,
    UnsupportedAudioFormatError,
    UploadFileTooLargeError,
    cleanup_temp_files,
    convert_audio_to_standard_wav,
    save_upload_file_to_temp,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])


@lru_cache(maxsize=1)
def get_speaker_service() -> SpeakerVerificationService:
    """Create one model service instance and reuse it for all requests."""

    return SpeakerVerificationService(get_settings())


@router.post("/compare", response_model=VoiceCompareResponse)
async def compare_voice(
    audio_file_1: UploadFile | None = File(
        default=None,
        description="First audio file. Supported: wav, mp3, m4a.",
    ),
    audio_file_2: UploadFile | None = File(
        default=None,
        description="Second audio file. Supported: wav, mp3, m4a.",
    ),
    settings: Settings = Depends(get_settings),
) -> VoiceCompareResponse:
    """Compare two uploaded voices with pretrained speaker embeddings."""

    temp_paths: list[Path | None] = []

    try:
        if audio_file_1 is None or audio_file_2 is None:
            raise MissingAudioFileError("audio_file_1 and audio_file_2 are required")

        original_1 = await save_upload_file_to_temp(
            upload_file=audio_file_1,
            allowed_extensions=settings.allowed_audio_extensions,
            max_size_bytes=settings.max_upload_size_bytes,
        )
        temp_paths.append(original_1)

        original_2 = await save_upload_file_to_temp(
            upload_file=audio_file_2,
            allowed_extensions=settings.allowed_audio_extensions,
            max_size_bytes=settings.max_upload_size_bytes,
        )
        temp_paths.append(original_2)

        wav_1 = convert_audio_to_standard_wav(
            input_path=original_1,
            target_sample_rate=settings.target_sample_rate,
            min_audio_seconds=settings.min_audio_seconds,
        )
        temp_paths.append(wav_1)

        wav_2 = convert_audio_to_standard_wav(
            input_path=original_2,
            target_sample_rate=settings.target_sample_rate,
            min_audio_seconds=settings.min_audio_seconds,
        )
        temp_paths.append(wav_2)

        # Load the pretrained model only after the request files are valid.
        # The first valid request may take longer because the model is downloaded.
        speaker_service = get_speaker_service()
        result = speaker_service.compare_files(wav_1, wav_2)

        return VoiceCompareResponse(
            similarity=result.similarity,
            threshold=result.threshold,
            is_same_speaker=result.is_same_speaker,
            message=result.message,
            model_name=result.model_name,
        )

    except MissingAudioFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except UnsupportedAudioFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except UploadFileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except AudioTooShortError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AudioDecodingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except SpeakerVerificationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during voice comparison")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error during voice comparison",
        ) from exc
    finally:
        cleanup_temp_files(temp_paths)
