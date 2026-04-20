from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build subgroup evaluation reports from saved predictions and manifest metadata.")
    parser.add_argument("--manifest", required=True, help="Path to the merged manifest CSV.")
    parser.add_argument("--predictions", required=True, help="Path to a predictions CSV emitted by evaluate.py.")
    parser.add_argument("--output", required=True, help="Path to write the subgroup metrics CSV.")
    return parser


def normalize_image_path(path: str) -> str:
    value = str(path).replace("\\", "/")
    marker = "/data/raw/"
    if marker in value:
        return value.split(marker, 1)[1]
    workspace_marker = "/workspace/hygo-assessment/"
    if workspace_marker in value:
        return value.split(workspace_marker, 1)[1]
    return value.lstrip("/")


def compute_group_metrics(frame: pd.DataFrame, group_columns: list[str], threshold: float = 0.5) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        y_true = group["minor_label"].astype(int)
        y_score = group["p_minor"]
        y_pred = (y_score >= threshold).astype(int)

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        row = {
            "group_type": " x ".join(group_columns),
            "group_value": " | ".join(str(value) for value in keys),
            "count": int(len(group)),
            "minor_count": int((y_true == 1).sum()),
            "adult_count": int((y_true == 0).sum()),
            "mean_p_minor": float(y_score.mean()),
            "minor_precision": float(tp / max(tp + fp, 1)),
            "minor_recall": float(tp / max(tp + fn, 1)),
            "minor_f1": float((2 * tp) / max((2 * tp) + fp + fn, 1)),
            "minor_false_negative_rate": float(fn / max(int((y_true == 1).sum()), 1)),
        }
        for column, value in zip(group_columns, keys):
            row[column] = value
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    predictions_path = Path(args.predictions)
    output_path = Path(args.output)

    manifest = pd.read_csv(manifest_path, low_memory=False)
    predictions = pd.read_csv(predictions_path, low_memory=False)

    metadata = manifest[
        ["image_path", "source_dataset", "split", "gender", "race", "age_bucket", "minor_label"]
    ].copy()
    metadata["image_key"] = metadata["image_path"].map(normalize_image_path)
    metadata = metadata.drop_duplicates(subset=["image_key"])

    predictions["image_key"] = predictions["image_path"].map(normalize_image_path)
    enriched = predictions.merge(
        metadata[["image_key", "split", "gender", "race"]],
        on="image_key",
        how="left",
        validate="many_to_one",
    )

    missing = int(enriched["gender"].isna().sum())
    if missing:
        raise RuntimeError(
            f"Failed to map manifest metadata for {missing} prediction rows. "
            "Check that the manifest and predictions come from the same run."
        )

    labeled = enriched[enriched["minor_label"].notna()].copy()
    subgroup_frames = [
        compute_group_metrics(labeled, ["age_bucket"]),
        compute_group_metrics(labeled, ["gender"]),
        compute_group_metrics(labeled, ["race"]),
        compute_group_metrics(labeled, ["gender", "age_bucket"]),
        compute_group_metrics(labeled, ["race", "age_bucket"]),
    ]
    subgroup_metrics = pd.concat(subgroup_frames, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subgroup_metrics.to_csv(output_path, index=False)
    print(f"Wrote subgroup metrics to {output_path}")


if __name__ == "__main__":
    main()
