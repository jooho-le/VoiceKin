import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Settings:
    """Application settings.

    Values can be overridden with environment variables or a local .env file.
    Example:
        VOICEKIN_SPEAKER_THRESHOLD=0.72
        VOICEKIN_MAX_UPLOAD_SIZE_MB=20
    """

    app_name: str = "VoiceKin Speaker Verification API"
    api_v1_prefix: str = "/api/v1"

    # SpeechBrain ECAPA-TDNN model published on Hugging Face.
    speaker_model_name: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_model_dir: Path = Path("pretrained_models/spkrec-ecapa-voxceleb")

    # Tune this value with real VoiceKin validation data later.
    speaker_threshold: float = 0.75

    # CPU is the safest default. Set VOICEKIN_DEVICE=cuda on a CUDA machine.
    device: str = "cpu"

    allowed_audio_extensions: Tuple[str, ...] = ("wav", "mp3", "m4a")
    max_upload_size_mb: int = 25
    min_audio_seconds: float = 1.0
    target_sample_rate: int = 16000

    def __post_init__(self) -> None:
        if not -1.0 <= self.speaker_threshold <= 1.0:
            raise ValueError("VOICEKIN_SPEAKER_THRESHOLD must be between -1.0 and 1.0")
        if self.max_upload_size_mb < 1:
            raise ValueError("VOICEKIN_MAX_UPLOAD_SIZE_MB must be greater than or equal to 1")
        if self.min_audio_seconds < 0.1:
            raise ValueError("VOICEKIN_MIN_AUDIO_SECONDS must be greater than or equal to 0.1")
        if self.target_sample_rate < 8000:
            raise ValueError("VOICEKIN_TARGET_SAMPLE_RATE must be greater than or equal to 8000")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


def _load_dotenv(dotenv_path: Path = Path(".env")) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from .env without adding extra dependencies."""

    values: dict[str, str] = {}
    if not dotenv_path.exists():
        return values

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def _get_env(name: str, default: str, dotenv_values: dict[str, str]) -> str:
    """Read VOICEKIN_* setting from environment first, then .env, then default."""

    return os.getenv(name) or dotenv_values.get(name, default)


def _get_int_env(name: str, default: int, dotenv_values: dict[str, str]) -> int:
    return int(_get_env(name, str(default), dotenv_values))


def _get_float_env(name: str, default: float, dotenv_values: dict[str, str]) -> float:
    return float(_get_env(name, str(default), dotenv_values))


def _get_tuple_env(
    name: str,
    default: Tuple[str, ...],
    dotenv_values: dict[str, str],
) -> Tuple[str, ...]:
    raw_value = _get_env(name, ",".join(default), dotenv_values)
    return tuple(item.strip().lower().lstrip(".") for item in raw_value.split(",") if item.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings so every module uses the same config instance."""

    dotenv_values = _load_dotenv()

    return Settings(
        app_name=_get_env("VOICEKIN_APP_NAME", "VoiceKin Speaker Verification API", dotenv_values),
        api_v1_prefix=_get_env("VOICEKIN_API_V1_PREFIX", "/api/v1", dotenv_values),
        speaker_model_name=_get_env(
            "VOICEKIN_SPEAKER_MODEL_NAME",
            "speechbrain/spkrec-ecapa-voxceleb",
            dotenv_values,
        ),
        speaker_model_dir=Path(
            _get_env(
                "VOICEKIN_SPEAKER_MODEL_DIR",
                "pretrained_models/spkrec-ecapa-voxceleb",
                dotenv_values,
            )
        ),
        speaker_threshold=_get_float_env("VOICEKIN_SPEAKER_THRESHOLD", 0.75, dotenv_values),
        device=_get_env("VOICEKIN_DEVICE", "cpu", dotenv_values),
        allowed_audio_extensions=_get_tuple_env(
            "VOICEKIN_ALLOWED_AUDIO_EXTENSIONS",
            ("wav", "mp3", "m4a"),
            dotenv_values,
        ),
        max_upload_size_mb=_get_int_env("VOICEKIN_MAX_UPLOAD_SIZE_MB", 25, dotenv_values),
        min_audio_seconds=_get_float_env("VOICEKIN_MIN_AUDIO_SECONDS", 1.0, dotenv_values),
        target_sample_rate=_get_int_env("VOICEKIN_TARGET_SAMPLE_RATE", 16000, dotenv_values),
    )
