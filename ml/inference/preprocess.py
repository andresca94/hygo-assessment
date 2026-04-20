from __future__ import annotations

import io
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch


@dataclass
class FaceCrop:
    bbox: list[float]
    face_confidence: float
    crop: Image.Image
    face_size_ratio: float
    quality_flags: list[str]


class FacePreprocessor:
    def __init__(self) -> None:
        self.backend_name = "fallback"
        self._detector = None
        self._detector_error = None
        self._load_detector()

    def _load_detector(self) -> None:
        if os.getenv("ENABLE_INSIGHTFACE", "1") != "1":
            return
        try:
            from insightface.app import FaceAnalysis

            model_pack = os.getenv("INSIGHTFACE_MODEL_PACK", "buffalo_l")
            root = os.getenv("INSIGHTFACE_ROOT")
            allowed_modules_raw = os.getenv("INSIGHTFACE_ALLOWED_MODULES", "detection")
            allowed_modules = [item.strip() for item in allowed_modules_raw.split(",") if item.strip()]
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            kwargs = {
                "name": model_pack,
                "providers": providers,
            }
            if root:
                kwargs["root"] = root
            if allowed_modules:
                kwargs["allowed_modules"] = allowed_modules
            app = FaceAnalysis(**kwargs)
            app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            self._detector = app
            self.backend_name = "insightface"
        except Exception as exc:  # pragma: no cover
            self._detector_error = str(exc)
            self._detector = None
            self.backend_name = "fallback"

    @staticmethod
    def load_image(image_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def extract_faces(self, image: Image.Image) -> list[FaceCrop]:
        if self._detector is not None:
            return self._extract_with_insightface(image)
        return self._fallback_extract(image)

    def _extract_with_insightface(self, image: Image.Image) -> list[FaceCrop]:
        np_image = np.asarray(image)[:, :, ::-1]
        detections = self._detector.get(np_image)
        width, height = image.size
        results: list[FaceCrop] = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox.astype(float).tolist()
            crop = image.crop((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
            face_area = max(1.0, (x2 - x1) * (y2 - y1))
            results.append(
                FaceCrop(
                    bbox=[x1, y1, x2, y2],
                    face_confidence=float(detection.det_score),
                    crop=crop,
                    face_size_ratio=float(face_area / max(width * height, 1)),
                    quality_flags=[],
                )
            )
        return results

    def _fallback_extract(self, image: Image.Image) -> list[FaceCrop]:
        width, height = image.size
        if width < 64 or height < 64:
            return []
        margin_x = width * 0.1
        margin_y = height * 0.1
        x1, y1, x2, y2 = margin_x, margin_y, width - margin_x, height - margin_y
        crop = image.crop((x1, y1, x2, y2))
        face_area = max(1.0, (x2 - x1) * (y2 - y1))
        return [
            FaceCrop(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                face_confidence=0.55,
                crop=crop,
                face_size_ratio=float(face_area / max(width * height, 1)),
                quality_flags=["detector_fallback"],
            )
        ]
