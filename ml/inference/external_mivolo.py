from __future__ import annotations

import math
import os
from dataclasses import dataclass

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from PIL import Image


@dataclass
class ExternalMiVOLOResult:
    estimated_age: float
    p_minor: float
    spread_years: float
    gender: str | None
    gender_probability: float | None


class ExternalMiVOLOHFPredictor:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        dtype_name: str = "float16",
        age_to_minor_margin: float = 2.5,
        default_spread_years: float = 4.0,
    ) -> None:
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

        self.device = device
        self.inference_dtype = getattr(torch, dtype_name, torch.float16)
        self.age_to_minor_margin = age_to_minor_margin
        self.default_spread_years = default_spread_years
        self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=self.inference_dtype,
        ).to(device)
        self.model.eval()
        self.image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)

    def predict(self, image: Image.Image, bbox: list[float]) -> ExternalMiVOLOResult:
        bgr_image = np.asarray(image.convert("RGB"))[:, :, ::-1]
        face_crop = self._crop(bgr_image, bbox)
        body_crop = self._crop(bgr_image, self._expand_body_bbox(bgr_image, bbox))

        face_input = self.image_processor(images=[face_crop])["pixel_values"]
        body_input = self.image_processor(images=[body_crop])["pixel_values"]
        face_input = face_input.to(dtype=self.inference_dtype, device=self.device)
        body_input = body_input.to(dtype=self.inference_dtype, device=self.device)

        with torch.no_grad():
            output = self.model(faces_input=face_input, body_input=body_input)

        age = float(output.age_output[0].item())
        p_minor = 1.0 / (1.0 + math.exp((age - 18.0) / self.age_to_minor_margin))

        gender = None
        gender_probability = None
        if hasattr(output, "gender_class_idx") and hasattr(output, "gender_probs"):
            id2label = getattr(self.config, "gender_id2label", {})
            index = int(output.gender_class_idx[0].item())
            gender = str(id2label.get(index, "unknown"))
            gender_probability = float(output.gender_probs[0].item())

        return ExternalMiVOLOResult(
            estimated_age=age,
            p_minor=float(p_minor),
            spread_years=self.default_spread_years,
            gender=gender,
            gender_probability=gender_probability,
        )

    @staticmethod
    def _crop(image: np.ndarray, bbox: list[float]) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return image
        return crop

    @staticmethod
    def _expand_body_bbox(image: np.ndarray, bbox: list[float]) -> list[float]:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        face_width = max(1.0, x2 - x1)
        face_height = max(1.0, y2 - y1)
        expanded = [
            max(0.0, x1 - 1.25 * face_width),
            max(0.0, y1 - 0.75 * face_height),
            min(float(width), x2 + 1.25 * face_width),
            min(float(height), y2 + 4.0 * face_height),
        ]
        if expanded[2] <= expanded[0] or expanded[3] <= expanded[1]:
            return [0.0, 0.0, float(width), float(height)]
        return expanded
