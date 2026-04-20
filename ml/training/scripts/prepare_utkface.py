from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.datasets.manifest import write_manifest
from ml.training.datasets.schemas import FaceRecord
from ml.training.utils import age_to_bucket, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize UTKFace into the common manifest schema.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def parse_filename(file_path: Path) -> tuple[float | None, str, str]:
    parts = file_path.stem.split("_")
    if len(parts) < 4:
        return None, "unknown", "unknown"
    age_value = float(parts[0])
    gender = "male" if parts[1] == "0" else "female"
    race = {
        "0": "white",
        "1": "black",
        "2": "asian",
        "3": "indian",
        "4": "other",
    }.get(parts[2], "unknown")
    return age_value, gender, race


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = ensure_dir(args.output_dir)
    records: list[dict] = []

    for image_path in sorted(raw_dir.rglob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        age_value, gender, race = parse_filename(image_path)
        if age_value is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size

        record = FaceRecord(
            image_id=image_path.stem,
            face_id=f"{image_path.stem}_face0",
            image_path=str(image_path.resolve()),
            source_dataset="utkface",
            license_type="see_source_manifest",
            domain_type="real",
            num_faces=1,
            has_detectable_face=True,
            quality_tags="",
            split="unassigned",
            bbox_x1=0.0,
            bbox_y1=0.0,
            bbox_x2=float(width),
            bbox_y2=float(height),
            face_size_ratio=1.0,
            pose_tag="frontal",
            occlusion_tag="none",
            blur_tag="unknown",
            age_value=age_value,
            age_bucket=age_to_bucket(age_value),
            minor_label=1 if age_value < 18 else 0,
            label_confidence=1.0,
            label_status="trusted",
            gender=gender,
            race=race,
        )
        records.append(record.to_dict())

    write_manifest(records, output_dir / "manifest.csv")
    print(f"Wrote {len(records)} UTKFace records to {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
