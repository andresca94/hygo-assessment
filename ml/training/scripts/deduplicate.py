from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.datasets.manifest import load_manifest
from ml.training.utils import ensure_dir, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove exact and simple perceptual duplicates.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    return parser


def md5_file(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def average_hash(path: str) -> str:
    with Image.open(path) as image:
        image = image.convert("L").resize((8, 8))
        pixels = np.asarray(image)
    threshold = pixels.mean()
    bits = pixels > threshold
    return "".join("1" if bit else "0" for bit in bits.flatten())


def main() -> None:
    args = build_parser().parse_args()
    dataframe = load_manifest(args.input_manifest)
    dataframe = dataframe[dataframe["image_path"].notna()].copy()

    dataframe["exact_hash"] = dataframe["image_path"].map(md5_file)
    dataframe["ahash"] = dataframe["image_path"].map(average_hash)
    before_count = len(dataframe)
    dataframe = dataframe.drop_duplicates(subset=["exact_hash", "ahash"], keep="first").copy()
    after_count = len(dataframe)
    dataframe = dataframe.drop(columns=["exact_hash", "ahash"])
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output_manifest, index=False)

    report_dir = ensure_dir("reports/metrics")
    write_json(
        {
            "input_manifest": args.input_manifest,
            "output_manifest": args.output_manifest,
            "before_count": before_count,
            "after_count": after_count,
            "removed_duplicates": before_count - after_count,
        },
        report_dir / "deduplication_report.json",
    )
    print(f"Removed {before_count - after_count} duplicate rows")


if __name__ == "__main__":
    main()
