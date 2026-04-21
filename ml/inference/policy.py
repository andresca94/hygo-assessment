from __future__ import annotations

from dataclasses import dataclass


VERDICT_PRIORITY = {"safe": 0, "uncertain": 1, "flagged": 2}


@dataclass
class FacePolicyInput:
    estimated_age: float
    age_interval: tuple[float, float]
    p_minor: float
    face_confidence: float
    quality_flags: list[str]
    domain_type: str
    conflict_score: float
    face_size_ratio: float


@dataclass
class PolicyConfig:
    safe_threshold: float
    flagged_threshold: float
    adult_safe_age_lower_bound: float
    minimum_face_confidence: float
    low_face_area_threshold: float
    max_conflict_score: float


class PolicyEngine:
    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def decide_face(self, payload: FacePolicyInput) -> tuple[str, str]:
        if payload.face_confidence < self.config.minimum_face_confidence:
            return "uncertain", "low_face_confidence"
        if payload.face_size_ratio < self.config.low_face_area_threshold:
            return "uncertain", "low_face_area"
        if payload.conflict_score > self.config.max_conflict_score:
            return "uncertain", "model_conflict"
        if payload.domain_type in {"anime", "cartoon", "three_d"}:
            return "uncertain", "non_photographic_domain"
        if payload.p_minor >= self.config.flagged_threshold or payload.age_interval[0] < 18.0:
            return "flagged", "minor_risk"
        if (
            payload.p_minor < self.config.safe_threshold
            and payload.age_interval[0] >= self.config.adult_safe_age_lower_bound
            and not payload.quality_flags
        ):
            return "safe", "confident_adult_face"
        return "uncertain", "boundary_or_quality_case"

    def decide_image(self, faces: list[dict]) -> tuple[str, str]:
        if not faces:
            return "uncertain", "uncertain_no_face"
        worst = max(faces, key=lambda item: VERDICT_PRIORITY[item["verdict"]])
        return worst["verdict"], worst["policy_reason"]
