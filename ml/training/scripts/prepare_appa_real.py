from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.datasets.manifest import write_manifest
from ml.training.datasets.schemas import FaceRecord
from ml.training.utils import age_to_bucket, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize APPA-REAL into the common manifest schema.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def locate_annotations(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.rglob("*.csv"))


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = ensure_dir(args.output_dir)
    records: list[dict] = []

    for csv_path in locate_annotations(raw_dir):
        frame = pd.read_csv(csv_path)
        columns = {column.lower(): column for column in frame.columns}
        age_key = columns.get("real_age") or columns.get("age") or columns.get("apparent_age_avg")
        image_key = columns.get("file_name") or columns.get("image_name") or columns.get("image")
        if not age_key or not image_key:
            continue

        for row in frame.to_dict(orient="records"):
            image_path = (raw_dir / str(row[image_key])).resolve()
            if not image_path.exists():
                continue
            age_value = float(row[age_key])
            record = FaceRecord(
                image_id=image_path.stem,
                face_id=f"{image_path.stem}_face0",
                image_path=str(image_path),
                source_dataset="appa_real",
                license_type="see_source_manifest",
                domain_type="real",
                num_faces=1,
                has_detectable_face=True,
                quality_tags="",
                split="unassigned",
                bbox_x1=0.0,
                bbox_y1=0.0,
                bbox_x2=1.0,
                bbox_y2=1.0,
                face_size_ratio=1.0,
                pose_tag="unknown",
                occlusion_tag="unknown",
                blur_tag="unknown",
                age_value=age_value,
                age_bucket=age_to_bucket(age_value),
                minor_label=1 if age_value < 18 else 0,
                label_confidence=0.85,
                label_status="trusted",
                gender=str(row.get(columns.get("gender", ""), "unknown")).lower() if columns.get("gender") else "unknown",
                race="unknown",
            )
            records.append(record.to_dict())

    write_manifest(records, output_dir / "manifest.csv")
    print(f"Wrote {len(records)} APPA-REAL records to {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
