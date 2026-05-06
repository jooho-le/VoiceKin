import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.repositories.family_repository import FamilyRepository
from app.schemas.family import (
    FamilyDeleteResponse,
    FamilyListResponse,
    FamilyMemberResponse,
    FamilyRegisterResponse,
)
from app.services.model_provider import get_speaker_service
from app.services.speaker_service import SpeakerVerificationError
from app.services.voiceprint_service import RegisteredVoiceprint, VoiceprintService
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

router = APIRouter(prefix="/family", tags=["family"])


def get_family_repository(settings: Settings = Depends(get_settings)) -> FamilyRepository:
    return FamilyRepository(settings.database_path)


def _to_member_response(voiceprint: RegisteredVoiceprint) -> FamilyMemberResponse:
    return FamilyMemberResponse(
        family_id=voiceprint.family_id,
        name=voiceprint.name,
        relation=voiceprint.relation,
        model_name=voiceprint.model_name,
    )


@router.post("/register", response_model=FamilyRegisterResponse)
async def register_family_voice(
    name: str = Form(..., description="Family member display name. Example: 엄마"),
    relation: str = Form(..., description="Relation label. Example: mother"),
    audio_file: UploadFile | None = File(
        default=None,
        description="Family member voice file. Supported: wav, mp3, m4a.",
    ),
    settings: Settings = Depends(get_settings),
    family_repository: FamilyRepository = Depends(get_family_repository),
) -> FamilyRegisterResponse:
    """Register one family voiceprint from an uploaded voice sample."""

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
        registered = voiceprint_service.register_family_voice(
            name=name,
            relation=relation,
            wav_path=wav_file,
        )

        return FamilyRegisterResponse(
            family_id=registered.family_id,
            name=registered.name,
            relation=registered.relation,
            model_name=registered.model_name,
            message="voiceprint_registered",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
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
        logger.exception("Unexpected error during family voice registration")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error during family voice registration",
        ) from exc
    finally:
        cleanup_temp_files(temp_paths)


@router.get("", response_model=FamilyListResponse)
async def list_family_members(
    family_repository: FamilyRepository = Depends(get_family_repository),
) -> FamilyListResponse:
    """List registered family members without exposing voiceprint embeddings."""

    service = VoiceprintService(
        family_repository=family_repository,
    )
    members = [_to_member_response(member) for member in service.list_family_members()]
    return FamilyListResponse(members=members)


@router.get("/{family_id}", response_model=FamilyMemberResponse)
async def get_family_member(
    family_id: int,
    family_repository: FamilyRepository = Depends(get_family_repository),
) -> FamilyMemberResponse:
    """Get one registered family member by id."""

    service = VoiceprintService(
        family_repository=family_repository,
    )
    member = service.get_family_member(family_id)
    if member is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="family member not found",
        )

    return _to_member_response(member)


@router.delete("/{family_id}", response_model=FamilyDeleteResponse)
async def delete_family_member(
    family_id: int,
    family_repository: FamilyRepository = Depends(get_family_repository),
) -> FamilyDeleteResponse:
    """Delete one registered family member and its stored voiceprint."""

    deleted = family_repository.delete(family_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="family member not found",
        )

    return FamilyDeleteResponse(family_id=family_id, message="family_member_deleted")
