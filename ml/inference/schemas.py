from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Verdict = Literal["safe", "flagged", "uncertain"]


class FaceResult(BaseModel):
    bbox: list[float]
    face_confidence: float
    estimated_age: float | None = None
    age_interval: list[float] | None = None
    p_minor: float | None = None
    domain_type: str = "unknown"
    quality_flags: list[str] = Field(default_factory=list)
    verdict: Verdict
    policy_reason: str


class ImageResult(BaseModel):
    verdict: Verdict
    risk_score: float
    faces: list[FaceResult]
    policy_reason: str
    model_version: str


class BatchResult(BaseModel):
    items: list[ImageResult]
    flagged_count: int
    uncertain_count: int
    safe_count: int


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_version: str
    detector_backend: str
    main_model_mode: str
    aux_model_mode: str
