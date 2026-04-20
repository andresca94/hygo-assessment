from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import yaml

AGE_BUCKETS = ["0-12", "13-15", "16-17", "18-20", "21-25", "26+"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    base_config = data.pop("base_config", None)
    if base_config:
        base_path = (path.parent / base_config).resolve()
        base_data = load_yaml(base_path)
        data = deep_merge(base_data, data)
    return data


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def age_to_bucket(age_value: float | int | None) -> str:
    if age_value is None:
        return "unknown"
    age = float(age_value)
    if age <= 12:
        return "0-12"
    if age <= 15:
        return "13-15"
    if age <= 17:
        return "16-17"
    if age <= 20:
        return "18-20"
    if age <= 25:
        return "21-25"
    return "26+"


def bucket_to_index(bucket: str) -> int:
    try:
        return AGE_BUCKETS.index(bucket)
    except ValueError:
        return len(AGE_BUCKETS) - 1


def fairface_bucket_to_range(label: str) -> tuple[float | None, str]:
    mapping = {
        "0-2": (1.0, "0-12"),
        "3-9": (6.0, "0-12"),
        "10-19": (16.0, "16-17"),
        "20-29": (24.0, "21-25"),
        "30-39": (34.0, "26+"),
        "40-49": (44.0, "26+"),
        "50-59": (54.0, "26+"),
        "60-69": (64.0, "26+"),
        "more than 70": (72.0, "26+"),
    }
    return mapping.get(label, (None, "unknown"))


def normalize_domain(domain_type: str | None) -> str:
    if not domain_type:
        return "unknown"
    value = domain_type.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "3d": "three_d",
        "3d_render": "three_d",
        "synthetic": "ai_generated",
        "fake": "ai_generated",
    }
    return aliases.get(value, value)


def listify_quality_flags(raw_value: str | list[str] | None) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [flag for flag in raw_value if flag]
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def encode_quality_flags(flags: list[str]) -> str:
    return ",".join(sorted(set(flags)))
