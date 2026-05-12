from pydantic import BaseModel, Field

from app.schemas.anti_spoofing import AntiSpoofingResponse
from app.schemas.voice import FamilyCandidateResponse, VerifyFamilyResponse


class VoiceSessionStatusResponse(BaseModel):
    """Rolling status for one chunk-based voice analysis session."""

    session_id: str = Field(..., examples=["9d4a5f2f4d2b4ab6bb5a7b957b04e51d"])
    status: str = Field(..., examples=["active"])
    created_at: str
    updated_at: str
    ended_at: str | None = None
    chunks_analyzed: int = Field(..., examples=[4])
    is_spoofed: bool = Field(
        ...,
        description="True when any analyzed chunk reaches the anti-spoofing threshold.",
        examples=[False],
    )
    is_registered_family: bool = Field(
        ...,
        description="True when the best family similarity across chunks reaches the speaker threshold.",
        examples=[True],
    )
    risk_level: str = Field(..., examples=["low"])
    message: str = Field(..., examples=["registered_family_likely"])
    max_spoof_score: float = Field(..., examples=[0.12])
    max_spoof_chunk_index: int | None = Field(default=None, examples=[2])
    best_family_match: FamilyCandidateResponse | None = None
    speaker_threshold: float = Field(..., examples=[0.75])
    anti_spoofing_threshold: float = Field(..., examples=[0.07])


class VoiceSessionStartResponse(VoiceSessionStatusResponse):
    """Response returned after starting a new voice analysis session."""


class VoiceSessionChunkResponse(BaseModel):
    """Result returned after analyzing one uploaded audio chunk."""

    session_id: str = Field(..., examples=["9d4a5f2f4d2b4ab6bb5a7b957b04e51d"])
    chunk_index: int = Field(..., examples=[0])
    is_trusted_chunk: bool = Field(
        ...,
        description="True only when this chunk matches a registered family member and is not spoofed.",
        examples=[True],
    )
    final_decision: str = Field(..., examples=["trusted_family_voice"])
    family_verification: VerifyFamilyResponse
    anti_spoofing: AntiSpoofingResponse
    rolling_result: VoiceSessionStatusResponse
