from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.inference.external_mivolo import ExternalMiVOLOHFPredictor
from ml.inference.policy import FacePolicyInput, PolicyConfig, PolicyEngine
from ml.inference.preprocess import FaceCrop, FacePreprocessor
from ml.training.models.dinov2_head import DINOv2AuxiliaryModel
from ml.training.models.mivolo_wrapper import MiVOLOAgeEstimator
from ml.training.utils import AGE_BUCKETS


class AgeSafetyPredictor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_version = os.getenv("MODEL_VERSION", "age-safety-v1.0.0")
        self.main_checkpoint_path = Path(os.getenv("MAIN_MODEL_CHECKPOINT", "ml/training/outputs/exported/main_best.pt"))
        self.aux_checkpoint_path = Path(os.getenv("AUX_MODEL_CHECKPOINT", "ml/training/outputs/exported/aux_best.pt"))
        self.calibration_path = Path(os.getenv("CALIBRATION_FILE", "ml/training/outputs/exported/calibration.json"))
        self.policy_path = Path(os.getenv("POLICY_FILE", "ml/training/outputs/exported/policy.json"))
        self.preprocessor = FacePreprocessor()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.main_model = None
        self.aux_model = None
        self.external_mivolo = None
        self.temperature = 1.0
        self.main_model_mode = "unavailable"
        self.aux_model_mode = "unavailable"
        self.policy = PolicyEngine(self._load_policy_config())
        self._ready = False
        self._load_models()

    def _load_policy_config(self) -> PolicyConfig:
        defaults = {
            "safe_threshold": float(os.getenv("POLICY_SAFE_THRESHOLD", "0.05")),
            "flagged_threshold": float(os.getenv("POLICY_FLAGGED_THRESHOLD", "0.40")),
            "adult_safe_age_lower_bound": float(os.getenv("POLICY_SAFE_AGE_LOWER_BOUND", "20.0")),
            "minimum_face_confidence": 0.80,
            "low_face_area_threshold": 0.05,
            "max_conflict_score": 0.25,
        }
        if self.policy_path.exists():
            defaults.update(json.loads(self.policy_path.read_text()))
        return PolicyConfig(**defaults)

    def _load_models(self) -> None:
        if os.getenv("ENABLE_EXTERNAL_MIVOLO_HF", "0") == "1":
            try:
                self.external_mivolo = ExternalMiVOLOHFPredictor(
                    model_id=os.getenv("MIVOLO_HF_MODEL_ID", "iitolstykh/mivolo_v2"),
                    device=self.device,
                    dtype_name=os.getenv("MIVOLO_HF_DTYPE", "float16"),
                )
                self.main_model_mode = "mivolo_hf"
            except Exception:
                self.external_mivolo = None

        if self.calibration_path.exists() and self.external_mivolo is None:
            self.temperature = float(json.loads(self.calibration_path.read_text()).get("temperature", 1.0))

        if self.external_mivolo is None and self.main_checkpoint_path.exists():
            payload = torch.load(self.main_checkpoint_path, map_location=self.device)
            config = payload.get("config", {})
            main_config = config.get("model", {}).get("main", {})
            self.main_model = MiVOLOAgeEstimator(
                backbone_name=main_config.get("backbone_name", "tf_efficientnet_b0"),
                pretrained=False,
                num_buckets=len(AGE_BUCKETS),
                dropout=float(main_config.get("dropout", 0.2)),
            ).to(self.device)
            self.main_model.load_state_dict(payload["model_state_dict"])
            self.main_model.eval()
            self.main_model_mode = "custom_checkpoint"

        if self.aux_checkpoint_path.exists():
            payload = torch.load(self.aux_checkpoint_path, map_location=self.device)
            config = payload.get("config", {})
            aux_config = config.get("model", {}).get("aux", {})
            domain_classes = config.get("data", {}).get(
                "domain_classes",
                ["real", "ai_generated", "cartoon", "anime", "three_d", "edited", "unknown"],
            )
            self.domain_classes = domain_classes
            self.aux_model = DINOv2AuxiliaryModel(
                backbone_name=aux_config.get("backbone_name", "dinov2_vits14_reg"),
                backbone_source=aux_config.get("backbone_source", "auto"),
                repo_dir=aux_config.get("repo_dir", os.getenv("DINOV2_REPO_DIR")),
                pretrained=False,
                freeze_encoder=bool(aux_config.get("freeze_encoder", True)),
                num_domains=len(domain_classes),
                dropout=float(aux_config.get("dropout", 0.2)),
            ).to(self.device)
            self.aux_model.load_state_dict(payload["model_state_dict"])
            self.aux_model.eval()
            self.aux_model_mode = "checkpoint"
        else:
            self.domain_classes = ["real", "ai_generated", "cartoon", "anime", "three_d", "edited", "unknown"]
            self.aux_model_mode = "unavailable"

        self._ready = self.external_mivolo is not None or self.main_model is not None

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def detector_backend(self) -> str:
        return self.preprocessor.backend_name

    def health(self) -> dict:
        return {
            "status": "ok" if self.ready else "degraded",
            "model_ready": self.ready,
            "model_version": self.model_version,
            "detector_backend": self.detector_backend,
            "main_model_mode": self.main_model_mode,
            "aux_model_mode": self.aux_model_mode,
        }

    def predict_image(self, image_bytes: bytes) -> dict:
        image = self.preprocessor.load_image(image_bytes)
        faces = self.preprocessor.extract_faces(image)
        if not faces:
            return {
                "verdict": "uncertain",
                "risk_score": 1.0,
                "faces": [],
                "policy_reason": "uncertain_no_face",
                "model_version": self.model_version,
            }

        face_results = [self.predict_face(face, image=image) for face in faces]
        verdict, reason = self.policy.decide_image(face_results)
        risk_score = max(item["p_minor"] if item["p_minor"] is not None else 1.0 for item in face_results)
        return {
            "verdict": verdict,
            "risk_score": risk_score,
            "faces": face_results,
            "policy_reason": reason,
            "model_version": self.model_version,
        }

    def predict_face(self, face: FaceCrop, image=None) -> dict:
        if not self.ready:
            return {
                "bbox": face.bbox,
                "face_confidence": face.face_confidence,
                "estimated_age": None,
                "age_interval": None,
                "p_minor": None,
                "domain_type": "unknown",
                "quality_flags": sorted(set(face.quality_flags + ["model_not_ready"])),
                "verdict": "uncertain",
                "policy_reason": "model_not_ready",
            }

        tensor = self.transform(face.crop).unsqueeze(0).to(self.device)
        if self.external_mivolo is not None and image is not None:
            result = self.external_mivolo.predict(image=image, bbox=face.bbox)
            age = result.estimated_age
            p_minor = result.p_minor
            external_spread = result.spread_years
        else:
            with torch.no_grad():
                main_outputs = self.main_model(tensor)
                age = float(main_outputs["age"].squeeze(0).item())
                minor_logit = float((main_outputs["minor_logit"].squeeze(0) / self.temperature).item())
                p_minor = float(torch.sigmoid(torch.tensor(minor_logit)).item())
            external_spread = 2.5

        aux_p_minor = p_minor
        uncertainty_score = 0.5
        domain_type = "unknown"
        with torch.no_grad():
            if self.aux_model is not None:
                aux_outputs = self.aux_model(tensor)
                aux_p_minor = float(torch.sigmoid(aux_outputs["minor_logit"].squeeze(0)).item())
                uncertainty_score = float(torch.sigmoid(aux_outputs["uncertainty_logit"].squeeze(0)).item())
                domain_index = int(aux_outputs["domain_logits"].argmax(dim=1).item())
                domain_type = self.domain_classes[domain_index]

        conflict_score = abs(p_minor - aux_p_minor)
        spread = external_spread + 5.0 * uncertainty_score + 2.0 * conflict_score
        age_interval = [max(0.0, age - spread), age + spread]
        policy_input = FacePolicyInput(
            estimated_age=age,
            age_interval=(age_interval[0], age_interval[1]),
            p_minor=p_minor,
            face_confidence=face.face_confidence,
            quality_flags=face.quality_flags,
            domain_type=domain_type,
            conflict_score=conflict_score,
            face_size_ratio=face.face_size_ratio,
        )
        verdict, reason = self.policy.decide_face(policy_input)
        return {
            "bbox": face.bbox,
            "face_confidence": face.face_confidence,
            "estimated_age": age,
            "age_interval": age_interval,
            "p_minor": p_minor,
            "domain_type": domain_type,
            "quality_flags": face.quality_flags,
            "verdict": verdict,
            "policy_reason": reason,
        }
