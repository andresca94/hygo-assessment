from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.utils import ensure_dir, load_yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CSV_SUFFIXES = {".csv"}


@dataclass
class SourceValidationResult:
    source_name: str
    role: str
    required_for_trial: bool
    blocking: bool
    expected_raw_dir: str
    exists: bool
    image_count: int
    csv_count: int
    usable: bool
    reason: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate raw dataset folders before preprocessing starts.")
    parser.add_argument("--config", required=True, help="Path to the dataset source config YAML.")
    parser.add_argument("--raw-root", default="data/raw", help="Root directory that contains raw datasets.")
    parser.add_argument(
        "--strict-robustness",
        action="store_true",
        help="Fail when recommended robustness datasets are missing, not only when supervision is missing.",
    )
    return parser


def count_files(root: Path, suffixes: set[str]) -> int:
    if not root.exists():
        return 0
    count = 0
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            count += 1
    return count


def evaluate_source(source_name: str, source_data: dict, raw_root: Path, strict_robustness: bool) -> SourceValidationResult:
    target_dir = raw_root / source_data["expected_raw_dir"]
    role = str(source_data.get("role", "unspecified"))
    required_for_trial = bool(source_data.get("required_for_trial", False))
    acquisition = str(source_data.get("acquisition", "manual"))

    image_count = count_files(target_dir, IMAGE_SUFFIXES)
    csv_count = count_files(target_dir, CSV_SUFFIXES)

    blocking = False
    if strict_robustness and role == "robustness_eval" and acquisition != "manual_review":
        blocking = required_for_trial

    exists = target_dir.exists()
    usable = False
    reason = ""

    if source_name == "utkface":
        usable = image_count > 0
        reason = "expected face images named like UTKFace files" if not usable else "ok"
    elif source_name == "fairface":
        usable = csv_count > 0 and image_count > 0
        reason = "expected at least one label CSV and FairFace images" if not usable else "ok"
    elif source_name == "appa_real":
        usable = csv_count > 0 and image_count > 0
        reason = "expected APPA-REAL annotation CSVs plus image files" if not usable else "ok"
    else:
        usable = image_count > 0
        reason = "expected at least one image file for robustness evaluation" if not usable else "ok"

    if acquisition == "manual_review" and not usable:
        blocking = False

    if not exists:
        usable = False
        reason = "directory does not exist"

    return SourceValidationResult(
        source_name=source_name,
        role=role,
        required_for_trial=required_for_trial,
        blocking=blocking,
        expected_raw_dir=str(target_dir),
        exists=exists,
        image_count=image_count,
        csv_count=csv_count,
        usable=usable,
        reason=reason,
    )


def format_summary(results: list[SourceValidationResult]) -> str:
    lines = ["Raw dataset validation summary:"]
    for result in results:
        status = "OK" if result.usable else ("BLOCKING" if result.blocking else "WARN")
        lines.append(
            f"- {status}: {result.source_name} -> {result.expected_raw_dir} "
            f"(images={result.image_count}, csvs={result.csv_count})"
        )
        if result.reason != "ok":
            lines.append(f"  reason: {result.reason}")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    raw_root = ensure_dir(args.raw_root)
    manifests_dir = ensure_dir(ROOT_DIR / "ml/training/outputs/manifests")

    results = [
        evaluate_source(source_name, source_data, raw_root, args.strict_robustness)
        for source_name, source_data in config.get("sources", {}).items()
    ]

    payload = {
        "raw_root": str(raw_root),
        "results": [asdict(result) for result in results],
    }
    output_path = manifests_dir / "source_validation.json"
    output_path.write_text(json.dumps(payload, indent=2))

    print(format_summary(results))
    print(f"Wrote source validation report to {output_path}")

    primary_usable = any(result.usable and result.role == "primary_supervision" for result in results)

    if not primary_usable:
        raise SystemExit(
            "No usable primary supervision dataset was found under data/raw. "
            "Populate at least one of utkface, fairface, or appa_real and rerun."
        )
    blocking_failures = [result for result in results if result.blocking and not result.usable]
    if blocking_failures:
        names = ", ".join(result.source_name for result in blocking_failures)
        raise SystemExit(
            f"Required raw dataset folders are missing or empty: {names}. "
            "Populate the listed directories under data/raw and rerun."
        )


if __name__ == "__main__":
    main()
