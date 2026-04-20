from __future__ import annotations

import os

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


def build_feature_extractor(backbone_name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    if timm is not None:
        try:
            model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
            feature_dim = getattr(model, "num_features", 512)
            return model, feature_dim
        except Exception:
            pass

    fallback = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    feature_dim = fallback.fc.in_features
    fallback.fc = nn.Identity()
    return fallback, feature_dim


class MiVOLOAgeEstimator(nn.Module):
    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b0",
        pretrained: bool = True,
        num_buckets: int = 6,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone, feature_dim = build_feature_extractor(backbone_name, pretrained=pretrained)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.age_head = nn.Linear(feature_dim, 1)
        self.bucket_head = nn.Linear(feature_dim, num_buckets)
        self.minor_head = nn.Linear(feature_dim, 1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return self.dropout(self.norm(features))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode(images)
        return {
            "features": features,
            "age": self.age_head(features).squeeze(-1),
            "bucket_logits": self.bucket_head(features),
            "minor_logit": self.minor_head(features).squeeze(-1),
        }
