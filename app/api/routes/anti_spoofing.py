import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.schemas.anti_spoofing import AntiSpoofingLabelScore, AntiSpoofingResponse
from app.services.anti_spoofing_service import AntiSpoofingError, AntiSpoofingResult
from app.services.model_provider import get_anti_spoofing_service
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

router = APIRouter(prefix="/anti-spoofing", tags=["anti-spoofing"])


def anti_spoofing_result_to_response(result: AntiSpoofingResult) -> AntiSpoofingResponse:
    return AntiSpoofingResponse(
        is_spoofed=result.is_spoofed,
        spoof_score=result.spoof_score,
        threshold=result.threshold,
        predicted_label=result.predicted_label,
        predicted_score=result.predicted_score,
        message=result.message,
        model_name=result.model_name,
        analyzed_segments=result.analyzed_segments,
        max_spoof_segment_index=result.max_spoof_segment_index,
        segment_seconds=result.segment_seconds,
        label_scores=[
            AntiSpoofingLabelScore(label=label_score.label, score=label_score.score)
            for label_score in result.label_scores
        ],
    )


@router.post("/detect", response_model=AntiSpoofingResponse)
async def detect_spoofed_voice(
    audio_file: UploadFile | None = File(
        default=None,
        description="Voice file to classify as bonafide or spoof/deepfake.",
    ),
    settings: Settings = Depends(get_settings),
) -> AntiSpoofingResponse:
    """Detect whether one uploaded voice looks spoofed/deepfake."""

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

        # Load the model only after the upload is valid and converted.
        result = get_anti_spoofing_service().detect_file(wav_file)
        return anti_spoofing_result_to_response(result)

    except MissingAudioFileError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except AudioDecodingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except AudioValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except AntiSpoofingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during anti-spoofing detection")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="internal server error during anti-spoofing detection",
        ) from exc
    finally:
        cleanup_temp_files(temp_paths)
