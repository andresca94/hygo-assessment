from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.datasets.manifest import write_manifest
from ml.training.datasets.schemas import FaceRecord
from ml.training.utils import ensure_dir, normalize_domain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize non-real robustness datasets.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def infer_domain_from_path(path: Path) -> str:
    lowered = str(path).lower()
    parts = {part.lower() for part in path.parts}
    if "trueface" in parts:
        if "real" in parts:
            return "real"
        if "gan" in parts or "synthetic" in parts or "fake" in parts:
            return "ai_generated"
        if "social" in parts or "edited" in parts:
            return "edited"
    if "sfhq" in lowered or "generated_photos" in lowered or "sfhq_t2i" in lowered:
        return "ai_generated"
    if "digiface" in lowered:
        return "three_d"
    if "anime" in lowered:
        return "anime"
    if "cartoon" in lowered or "icartoon" in lowered:
        return "cartoon"
    if "deepfake" in lowered or "synthetic" in lowered or "fake" in lowered:
        return "ai_generated"
    if "3d" in lowered or "render" in lowered:
        return "three_d"
    return "edited"


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = ensure_dir(args.output_dir)
    records: list[dict] = []

    for image_path in sorted(raw_dir.rglob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        domain_type = normalize_domain(infer_domain_from_path(image_path))
        record = FaceRecord(
            image_id=image_path.stem,
            face_id=f"{image_path.stem}_face0",
            image_path=str(image_path.resolve()),
            source_dataset=image_path.parents[0].name.lower(),
            license_type="see_source_manifest",
            domain_type=domain_type,
            num_faces=1,
            has_detectable_face=True,
            quality_tags="robustness_only",
            split="robustness",
            bbox_x1=0.0,
            bbox_y1=0.0,
            bbox_x2=1.0,
            bbox_y2=1.0,
            face_size_ratio=1.0,
            pose_tag="unknown",
            occlusion_tag="unknown",
            blur_tag="unknown",
            age_value=None,
            age_bucket="unknown",
            minor_label=None,
            label_confidence=0.2,
            label_status="ambiguous",
            gender="unknown",
            race="unknown",
        )
        records.append(record.to_dict())

    write_manifest(records, output_dir / "manifest.csv")
    print(f"Wrote {len(records)} robustness records to {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
