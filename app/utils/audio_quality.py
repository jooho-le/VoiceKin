import array
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path


class AudioQualityError(Exception):
    """Raised when a normalized wav file cannot be inspected."""


@dataclass(frozen=True)
class AudioQualityResult:
    is_analyzable: bool
    message: str
    duration_seconds: float
    rms_energy: float
    peak_amplitude: float
    speech_ratio: float


def analyze_standard_wav_quality(
    wav_path: Path,
    target_sample_rate: int,
    min_analyzable_seconds: float,
    min_rms_energy: float,
    min_speech_ratio: float,
) -> AudioQualityResult:
    """Estimate whether a 16kHz mono PCM wav has enough speech-like signal."""

    samples = _read_standard_wav_samples(wav_path, target_sample_rate)
    duration_seconds = len(samples) / float(target_sample_rate)

    if not samples:
        return _result(False, "empty_audio", duration_seconds, 0.0, 0.0, 0.0)

    rms_energy = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    peak_amplitude = max(abs(sample) for sample in samples)
    speech_ratio = _compute_speech_ratio(
        samples=samples,
        sample_rate=target_sample_rate,
        min_rms_energy=min_rms_energy,
    )

    if duration_seconds < min_analyzable_seconds:
        return _result(
            False,
            "too_short_for_chunk_analysis",
            duration_seconds,
            rms_energy,
            peak_amplitude,
            speech_ratio,
        )
    if rms_energy < min_rms_energy:
        return _result(
            False,
            "low_energy_or_silence",
            duration_seconds,
            rms_energy,
            peak_amplitude,
            speech_ratio,
        )
    if speech_ratio < min_speech_ratio:
        return _result(
            False,
            "too_little_speech",
            duration_seconds,
            rms_energy,
            peak_amplitude,
            speech_ratio,
        )

    return _result(
        True,
        "analyzable",
        duration_seconds,
        rms_energy,
        peak_amplitude,
        speech_ratio,
    )


def _compute_speech_ratio(
    samples: list[float],
    sample_rate: int,
    min_rms_energy: float,
) -> float:
    frame_size = max(1, int(sample_rate * 0.02))
    frame_count = 0
    speech_like_frames = 0
    frame_threshold = min_rms_energy

    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if len(frame) < frame_size:
            continue

        frame_count += 1
        frame_rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame))
        if frame_rms >= frame_threshold:
            speech_like_frames += 1

    if frame_count == 0:
        return 0.0
    return speech_like_frames / frame_count


def _read_standard_wav_samples(wav_path: Path, target_sample_rate: int) -> list[float]:
    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channel_count = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            raw_audio = wav_file.readframes(frame_count)
    except Exception as exc:
        raise AudioQualityError("failed to read wav for audio quality analysis") from exc

    if sample_rate != target_sample_rate:
        raise AudioQualityError(
            f"audio sample rate is {sample_rate}, expected {target_sample_rate}"
        )
    if sample_width != 2:
        raise AudioQualityError(
            f"audio sample width is {sample_width} bytes, expected 2 bytes PCM16"
        )
    if frame_count <= 0 or not raw_audio:
        return []

    pcm_samples = array.array("h")
    pcm_samples.frombytes(raw_audio)
    if sys.byteorder == "big":
        pcm_samples.byteswap()

    if channel_count <= 1:
        return [sample / 32768.0 for sample in pcm_samples]

    mono_samples: list[float] = []
    for index in range(0, len(pcm_samples), channel_count):
        frame = pcm_samples[index : index + channel_count]
        mono_samples.append((sum(frame) / len(frame)) / 32768.0)
    return mono_samples


def _result(
    is_analyzable: bool,
    message: str,
    duration_seconds: float,
    rms_energy: float,
    peak_amplitude: float,
    speech_ratio: float,
) -> AudioQualityResult:
    return AudioQualityResult(
        is_analyzable=is_analyzable,
        message=message,
        duration_seconds=round(duration_seconds, 3),
        rms_energy=round(rms_energy, 6),
        peak_amplitude=round(peak_amplitude, 6),
        speech_ratio=round(speech_ratio, 4),
    )
