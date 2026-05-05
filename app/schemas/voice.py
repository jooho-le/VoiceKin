from pydantic import BaseModel, ConfigDict, Field


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
