from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.datasets.manifest import write_manifest
from ml.training.datasets.schemas import FaceRecord
from ml.training.utils import ensure_dir, fairface_bucket_to_range


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize FairFace into the common manifest schema.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def locate_csvs(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.rglob("*.csv") if "label" in path.name.lower())


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = ensure_dir(args.output_dir)
    records: list[dict] = []

    for csv_path in locate_csvs(raw_dir):
        frame = pd.read_csv(csv_path)
        for row in frame.to_dict(orient="records"):
            relative_path = row.get("file") or row.get("img_path") or row.get("image")
            if not relative_path:
                continue
            image_path = (csv_path.parent / str(relative_path)).resolve()
            if not image_path.exists():
                image_path = (raw_dir / str(relative_path)).resolve()
            if not image_path.exists():
                continue

            age_value, age_bucket = fairface_bucket_to_range(str(row.get("age", "")))
            minor_label = 1 if age_value is not None and age_value < 18 else 0
            label_status = "trusted" if age_value is not None else "weak"
            label_confidence = 0.8
            if str(row.get("age", "")).strip() == "10-19":
                age_value = None
                age_bucket = "unknown"
                minor_label = None
                label_status = "ambiguous"
                label_confidence = 0.3

            with Image.open(image_path) as image:
                width, height = image.size

            record = FaceRecord(
                image_id=image_path.stem,
                face_id=f"{image_path.stem}_face0",
                image_path=str(image_path),
                source_dataset="fairface",
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
                pose_tag="unknown",
                occlusion_tag="unknown",
                blur_tag="unknown",
                age_value=age_value,
                age_bucket=age_bucket,
                minor_label=minor_label,
                label_confidence=label_confidence,
                label_status=label_status,
                gender=str(row.get("gender", "unknown")).lower(),
                race=str(row.get("race", "unknown")).lower(),
            )
            records.append(record.to_dict())

    write_manifest(records, output_dir / "manifest.csv")
    print(f"Wrote {len(records)} FairFace records to {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
