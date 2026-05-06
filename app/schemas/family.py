from pydantic import BaseModel, ConfigDict, Field


class FamilyRegisterResponse(BaseModel):
    """Response returned after registering a family voiceprint."""

    model_config = ConfigDict(protected_namespaces=())

    family_id: int = Field(..., examples=[1])
    name: str = Field(..., examples=["엄마"])
    relation: str = Field(..., examples=["mother"])
    model_name: str = Field(..., examples=["speechbrain/spkrec-ecapa-voxceleb"])
    message: str = Field(default="voiceprint_registered", examples=["voiceprint_registered"])


class FamilyMemberResponse(BaseModel):
    """Family metadata response. The embedding is intentionally not exposed."""

    model_config = ConfigDict(protected_namespaces=())

    family_id: int = Field(..., examples=[1])
    name: str = Field(..., examples=["엄마"])
    relation: str = Field(..., examples=["mother"])
    model_name: str = Field(..., examples=["speechbrain/spkrec-ecapa-voxceleb"])


class FamilyListResponse(BaseModel):
    members: list[FamilyMemberResponse]


class FamilyDeleteResponse(BaseModel):
    family_id: int
    message: str = "family_member_deleted"
