from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CSV_COLUMNS = [
    "image_id",
    "face_id",
    "image_path",
    "source_dataset",
    "license_type",
    "domain_type",
    "num_faces",
    "has_detectable_face",
    "quality_tags",
    "split",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "face_size_ratio",
    "pose_tag",
    "occlusion_tag",
    "blur_tag",
    "age_value",
    "age_bucket",
    "minor_label",
    "label_confidence",
    "label_status",
    "gender",
    "race",
]


@dataclass
class FaceRecord:
    image_id: str
    face_id: str
    image_path: str
    source_dataset: str
    license_type: str
    domain_type: str
    num_faces: int
    has_detectable_face: bool
    quality_tags: str
    split: str
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    face_size_ratio: float
    pose_tag: str
    occlusion_tag: str
    blur_tag: str
    age_value: float | None
    age_bucket: str
    minor_label: int | None
    label_confidence: float
    label_status: str
    gender: str
    race: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: payload.get(key) for key in CSV_COLUMNS}
