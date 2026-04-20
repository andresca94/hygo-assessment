from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from ml.training.utils import bucket_to_index, listify_quality_flags


class ManifestFaceDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: int = 224,
        train: bool = False,
        include_weak_labels: bool = True,
    ) -> None:
        dataframe = pd.read_csv(manifest_path)
        dataframe = dataframe[dataframe["split"] == split].copy()

        if not include_weak_labels:
            dataframe = dataframe[dataframe["label_status"] == "trusted"].copy()
        else:
            dataframe = dataframe[dataframe["label_status"].fillna("excluded") != "excluded"].copy()

        dataframe = dataframe[dataframe["image_path"].notna()].reset_index(drop=True)
        self.dataframe = dataframe
        self.repo_root = Path(__file__).resolve().parents[3]
        self.transform = self._build_transform(image_size=image_size, train=train)

    def _resolve_image_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if path.exists():
            return path

        known_prefixes = [
            Path("/workspace/hygo-assessment"),
            Path("/workspace"),
        ]
        for prefix in known_prefixes:
            try:
                relative = path.relative_to(prefix)
            except ValueError:
                continue
            candidate = self.repo_root / relative
            if candidate.exists():
                return candidate

        return path

    @staticmethod
    def _build_transform(image_size: int, train: bool) -> transforms.Compose:
        steps: list[transforms.Transform] = [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        ]
        if train:
            steps.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                ]
            )
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | list[str]]:
        row = self.dataframe.iloc[index]
        image_path = self._resolve_image_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)

        age_value = row["age_value"] if pd.notna(row["age_value"]) else -1.0
        minor_label = row["minor_label"] if pd.notna(row["minor_label"]) else -1
        age_bucket = row["age_bucket"] if pd.notna(row["age_bucket"]) else "26+"
        quality_flags = listify_quality_flags(row["quality_tags"])

        boundary_weight = 2.0 if 15 <= float(age_value) <= 21 else 1.0
        quality_penalty = 0.2 * len(quality_flags)
        sample_weight = max(1.0, boundary_weight + quality_penalty)

        return {
            "image": tensor,
            "age_value": torch.tensor(float(age_value), dtype=torch.float32),
            "has_age_value": torch.tensor(1 if age_value >= 0 else 0, dtype=torch.float32),
            "bucket_index": torch.tensor(bucket_to_index(age_bucket), dtype=torch.long),
            "minor_label": torch.tensor(float(minor_label if minor_label >= 0 else 0), dtype=torch.float32),
            "has_minor_label": torch.tensor(1 if minor_label >= 0 else 0, dtype=torch.float32),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "quality_score": torch.tensor(max(0.0, 1.0 - 0.15 * len(quality_flags)), dtype=torch.float32),
            "face_size_ratio": torch.tensor(float(row["face_size_ratio"]), dtype=torch.float32),
            "image_path": str(image_path),
            "domain_type": str(row["domain_type"]),
            "age_bucket": age_bucket,
            "gender": str(row.get("gender", "unknown")),
            "race": str(row.get("race", "unknown")),
            "quality_flags": ",".join(quality_flags),
            "source_dataset": str(row["source_dataset"]),
        }
