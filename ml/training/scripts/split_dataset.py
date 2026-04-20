from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.datasets.manifest import load_manifests
from ml.training.utils import ensure_dir, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the master manifest and stratified splits.")
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    return parser


def assign_splits(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["split"] = "robustness"

    supervised = dataframe[dataframe["age_bucket"].fillna("unknown") != "unknown"].copy()
    robustness = dataframe[dataframe["age_bucket"].fillna("unknown") == "unknown"].copy()

    if len(supervised) < 10:
        supervised.loc[:, "split"] = "train"
        return pd.concat([supervised, robustness], ignore_index=True)

    stratify_key = supervised["domain_type"].fillna("unknown") + "_" + supervised["age_bucket"].fillna("unknown")
    try:
        train_df, temp_df = train_test_split(supervised, test_size=0.2, random_state=42, stratify=stratify_key)
        temp_stratify = temp_df["domain_type"].fillna("unknown") + "_" + temp_df["age_bucket"].fillna("unknown")
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_stratify)
    except ValueError:
        train_df, temp_df = train_test_split(supervised, test_size=0.2, random_state=42, shuffle=True)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

    train_df.loc[:, "split"] = "train"
    val_df.loc[:, "split"] = "val"
    test_df.loc[:, "split"] = "test"
    return pd.concat([train_df, val_df, test_df, robustness], ignore_index=True)


def main() -> None:
    args = build_parser().parse_args()
    processed_root = Path(args.processed_root)
    manifest_paths = sorted(processed_root.rglob("manifest.csv"))
    dataframe = load_manifests(manifest_paths)
    dataframe = assign_splits(dataframe)
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_manifest, index=False)

    counts = dataframe.groupby(["split", "domain_type"], dropna=False).size().reset_index(name="count")
    report_dir = ensure_dir("reports/metrics")
    counts.to_csv(report_dir / "split_summary.csv", index=False)
    write_json(
        {
            "output_manifest": str(output_manifest),
            "row_count": int(len(dataframe)),
            "manifest_count": len(manifest_paths),
        },
        report_dir / "split_summary.json",
    )
    print(f"Wrote master manifest to {output_manifest}")


if __name__ == "__main__":
    main()
