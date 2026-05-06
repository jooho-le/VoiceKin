from pydantic import BaseModel, ConfigDict, Field

from app.schemas.anti_spoofing import AntiSpoofingResponse


class VoiceCompareResponse(BaseModel):
    """Response body for speaker verification."""

    model_config = ConfigDict(protected_namespaces=())

    similarity: float = Field(
        ...,
        description="Cosine similarity between two speaker embeddings.",
        examples=[0.83],
    )
    threshold: float = Field(
        ...,
        description="Decision threshold used for same/different speaker.",
        examples=[0.75],
    )
    is_same_speaker: bool = Field(
        ...,
        description="True when similarity is greater than or equal to threshold.",
        examples=[True],
    )
    message: str = Field(
        ...,
        description="same_speaker or different_speaker.",
        examples=["same_speaker"],
    )
    model_name: str = Field(
        ...,
        description="Pretrained speaker verification model name.",
        examples=["speechbrain/spkrec-ecapa-voxceleb"],
    )


class FamilyCandidateResponse(BaseModel):
    """Similarity result for one registered family voiceprint."""

    family_id: int = Field(..., examples=[1])
    name: str = Field(..., examples=["엄마"])
    relation: str = Field(..., examples=["mother"])
    similarity: float = Field(..., examples=[0.86])


class VerifyFamilyResponse(BaseModel):
    """Response for comparing one call voice against all registered family voices."""

    model_config = ConfigDict(protected_namespaces=())

    is_registered_family: bool = Field(
        ...,
        description="True when the best match similarity is greater than or equal to threshold.",
        examples=[True],
    )
    best_match: FamilyCandidateResponse | None = Field(
        default=None,
        description="Most similar registered family member. Null when no family member exists.",
    )
    threshold: float = Field(..., examples=[0.75])
    candidates: list[FamilyCandidateResponse]
    message: str = Field(
        ...,
        examples=["registered_family_matched"],
    )
    model_name: str = Field(
        ...,
        examples=["speechbrain/spkrec-ecapa-voxceleb"],
    )


class SecureVoiceVerificationResponse(BaseModel):
    """Combined family verification and anti-spoofing result."""

    is_trusted: bool = Field(
        ...,
        description="True only when the voice matches a registered family member and is not spoofed.",
        examples=[True],
    )
    final_decision: str = Field(..., examples=["trusted_family_voice"])
    family_verification: VerifyFamilyResponse
    anti_spoofing: AntiSpoofingResponse
