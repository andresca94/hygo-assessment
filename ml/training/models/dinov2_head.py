from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


def build_aux_encoder(
    backbone_name: str,
    pretrained: bool = True,
    backbone_source: str = "auto",
    repo_dir: str | None = None,
) -> tuple[nn.Module, int]:
    if backbone_source in {"auto", "torch_hub"} and backbone_name.startswith("dinov2_"):
        try:
            local_repo = Path(repo_dir).expanduser() if repo_dir else None
            if local_repo and local_repo.exists():
                model = torch.hub.load(str(local_repo), backbone_name, source="local")
            else:
                model = torch.hub.load("facebookresearch/dinov2", backbone_name)
            feature_dim = getattr(model, "embed_dim", getattr(model, "num_features", 384))
            return model, int(feature_dim)
        except Exception:
            if backbone_source == "torch_hub":
                pass

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


class DINOv2AuxiliaryModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "dinov2_vits14_reg",
        backbone_source: str = "auto",
        repo_dir: str | None = None,
        pretrained: bool = True,
        freeze_encoder: bool = True,
        num_domains: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder, feature_dim = build_aux_encoder(
            backbone_name,
            pretrained=pretrained,
            backbone_source=backbone_source,
            repo_dir=repo_dir,
        )
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        hidden_dim = feature_dim // 2
        self.minor_head = nn.Linear(hidden_dim, 1)
        self.domain_head = nn.Linear(hidden_dim, num_domains)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        hidden = self.head(features)
        return {
            "features": hidden,
            "minor_logit": self.minor_head(hidden).squeeze(-1),
            "domain_logits": self.domain_head(hidden),
            "uncertainty_logit": self.uncertainty_head(hidden).squeeze(-1),
        }
