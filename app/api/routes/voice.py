import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.repositories.family_repository import FamilyRepository
from app.schemas.voice import (
    FamilyCandidateResponse,
    VerifyFamilyResponse,
    VoiceCompareResponse,
)
from app.services.model_provider import get_speaker_service
from app.services.speaker_service import SpeakerVerificationError
from app.services.voiceprint_service import FamilyVoiceMatch, VoiceprintService
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


def get_family_repository(settings: Settings = Depends(get_settings)) -> FamilyRepository:
    return FamilyRepository(settings.database_path)


def _to_candidate_response(match: FamilyVoiceMatch) -> FamilyCandidateResponse:
    return FamilyCandidateResponse(
        family_id=match.family_id,
        name=match.name,
        relation=match.relation,
        similarity=match.similarity,
    )


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


@router.post("/verify-family", response_model=VerifyFamilyResponse)
async def verify_family_voice(
    audio_file: UploadFile | None = File(
        default=None,
        description="New call voice file to compare against all registered family voiceprints.",
    ),
    settings: Settings = Depends(get_settings),
    family_repository: FamilyRepository = Depends(get_family_repository),
) -> VerifyFamilyResponse:
    """Compare one uploaded voice against every registered family voiceprint."""

    temp_paths: list[Path | None] = []

    try:
        if audio_file is None:
            raise MissingAudioFileError("audio_file is required")

        original_file = await save_upload_file_to_temp(
            upload_file=audio_file,
            allowed_extensions=settings.allowed_audio_extensions,
            max_size_bytes=settings.max_upload_size_bytes,
        )
        temp_paths.append(original_file)

        wav_file = convert_audio_to_standard_wav(
            input_path=original_file,
            target_sample_rate=settings.target_sample_rate,
            min_audio_seconds=settings.min_audio_seconds,
        )
        temp_paths.append(wav_file)

        # Load the pretrained model only after upload validation and wav conversion.
        voiceprint_service = VoiceprintService(
            family_repository=family_repository,
            speaker_service=get_speaker_service(),
        )
        result = voiceprint_service.verify_family_voice(wav_file)

        best_match = (
            _to_candidate_response(result.best_match)
            if result.best_match is not None
            else None
        )

        return VerifyFamilyResponse(
            is_registered_family=result.is_registered_family,
            best_match=best_match,
            threshold=result.threshold,
            candidates=[
                _to_candidate_response(candidate)
                for candidate in result.candidates
            ],
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
        logger.exception("Unexpected error during family voice verification")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error during family voice verification",
        ) from exc
    finally:
        cleanup_temp_files(temp_paths)
