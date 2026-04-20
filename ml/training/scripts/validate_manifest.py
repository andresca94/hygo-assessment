from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.utils import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the combined manifest before model training starts.")
    parser.add_argument("--manifest", required=True, help="Path to the merged master manifest CSV.")
    parser.add_argument("--require-val-test", action="store_true", help="Fail if validation or test supervised splits are empty.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    frame = pd.read_csv(manifest_path)
    supervised = frame[frame["age_bucket"].fillna("unknown") != "unknown"].copy()
    split_counts = frame["split"].fillna("missing").value_counts().to_dict()
    supervised_counts = supervised["split"].fillna("missing").value_counts().to_dict()

    payload = {
        "manifest_path": str(manifest_path),
        "row_count": int(len(frame)),
        "supervised_row_count": int(len(supervised)),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "supervised_split_counts": {str(key): int(value) for key, value in supervised_counts.items()},
    }

    manifests_dir = ensure_dir(ROOT_DIR / "ml/training/outputs/manifests")
    output_path = manifests_dir / "manifest_validation.json"
    output_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"Wrote manifest validation report to {output_path}")

    if frame.empty:
        raise SystemExit(
            "The master manifest is empty. This usually means the data/raw folders are empty or the prep scripts "
            "could not find the expected files."
        )
    if supervised.empty:
        raise SystemExit(
            "The master manifest has no supervised rows. Populate UTKFace, FairFace, or APPA-REAL with valid files and rerun."
        )
    if int(supervised_counts.get("train", 0)) == 0:
        raise SystemExit("The master manifest has no supervised train rows, so training cannot start.")
    if args.require_val_test:
        if int(supervised_counts.get("val", 0)) == 0 or int(supervised_counts.get("test", 0)) == 0:
            raise SystemExit(
                "The master manifest does not contain supervised val/test rows. Add more labeled supervision data and rerun."
            )


if __name__ == "__main__":
    main()
