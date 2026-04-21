#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from tempfile import gettempdir

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "hygo-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
CHARTS_DIR = REPORTS_DIR / "charts"
CONFUSION_DIR = REPORTS_DIR / "confusion_matrices"
RELIABILITY_DIR = REPORTS_DIR / "reliability"
TABLES_DIR = REPORTS_DIR / "tables"
GALLERIES_DIR = REPORTS_DIR / "galleries"
MANIFESTS_DIR = ROOT_DIR / "ml" / "training" / "outputs" / "manifests"
HISTORY_DIR = ROOT_DIR / "ml" / "training" / "outputs" / "history"
REVIEWER_DIR = ROOT_DIR / "reviewer_samples"

PALETTE = {
    "navy": "#1f3b5b",
    "slate": "#59708a",
    "teal": "#2a7f9e",
    "green": "#4a9d69",
    "amber": "#d59c43",
    "red": "#c45a4b",
    "gray": "#d9dee7",
    "ink": "#223042",
    "paper": "#f7f4ee",
}


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#d0d7e2",
        "axes.labelcolor": PALETTE["ink"],
        "axes.titleweight": "semibold",
        "axes.titlecolor": PALETTE["ink"],
        "grid.color": "#d8dde6",
        "grid.linestyle": "-",
        "grid.alpha": 0.4,
        "font.size": 11,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
    }
)


def ensure_dirs() -> None:
    for directory in (CHARTS_DIR, CONFUSION_DIR, RELIABILITY_DIR, TABLES_DIR, GALLERIES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def thousands(value: object) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, float)):
        if int(value) == value:
            return f"{int(value):,}"
        return f"{float(value):.4f}"
    return str(value)


def shorten(value: object, width: int) -> str:
    if pd.isna(value):
        return "-"
    return textwrap.shorten(str(value), width=width, placeholder="...")


def wrap_title(text: str, width: int = 26) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def save_close(fig: plt.Figure, path: Path, dpi: int = 180) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_table(
    frame: pd.DataFrame,
    path: Path,
    title: str,
    subtitle: str | None = None,
    col_widths: list[float] | None = None,
    font_size: int = 10,
) -> None:
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(lambda value: str(value))

    fig_height = max(2.8, 0.52 * (len(display) + 3))
    fig_width = max(8.5, 1.45 * len(display.columns) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_facecolor(PALETTE["paper"])

    ax.text(0.0, 1.10, title, transform=ax.transAxes, fontsize=16, fontweight="semibold", color=PALETTE["ink"])
    if subtitle:
        ax.text(0.0, 1.03, subtitle, transform=ax.transAxes, fontsize=10.5, color=PALETTE["slate"])

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="left",
        loc="upper left",
        bbox=[0.0, 0.0, 1.0, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    ncols = len(display.columns)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dde3ec")
        if row == 0:
            cell.set_facecolor("#e9eef5")
            cell.set_text_props(weight="semibold", color=PALETTE["ink"])
            cell.set_height(0.09)
        else:
            cell.set_facecolor("white" if row % 2 else "#f9fbfd")
            cell.set_height(0.078)
            cell.set_text_props(color=PALETTE["ink"])
        if col_widths and col < len(col_widths):
            cell.set_width(col_widths[col])
        else:
            cell.set_width(1.0 / max(ncols, 1))

    save_close(fig, path, dpi=200)


def plot_roc_and_pr(split: str) -> None:
    frame = pd.read_csv(METRICS_DIR / f"{split}_predictions.csv")
    frame = frame[frame["minor_label"].notna()].copy()
    if frame.empty:
        return

    y_true = frame["minor_label"].astype(int)
    y_score = frame["p_minor"].astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot(fpr, tpr, color=PALETTE["teal"], linewidth=2.5, label=f"{split.upper()} ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["slate"], linewidth=1.2)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{split.upper()} ROC curve")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True)
    save_close(fig, CHARTS_DIR / f"{split}_roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot(recall, precision, color=PALETTE["green"], linewidth=2.5, label=f"{split.upper()} PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{split.upper()} precision-recall curve")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(True)
    save_close(fig, CHARTS_DIR / f"{split}_pr_curve.png")


def plot_confusion_matrix(split: str) -> None:
    matrix_path = CONFUSION_DIR / f"{split}_confusion_matrix.json"
    if not matrix_path.exists():
        return

    payload = load_json(matrix_path)
    matrix = payload["matrix"]
    labels = ["adult", "minor"]

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(2), labels)
    ax.set_yticks(range(2), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{split.upper()} confusion matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color=PALETTE["ink"], fontweight="semibold")

    save_close(fig, CONFUSION_DIR / f"{split}_confusion_matrix.png")


def plot_split_domain_counts() -> None:
    frame = pd.read_csv(METRICS_DIR / "split_summary.csv")
    pivot = frame.pivot(index="split", columns="domain_type", values="count").fillna(0)
    pivot = pivot.loc[[item for item in ("train", "val", "test", "robustness") if item in pivot.index]]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[PALETTE["navy"], PALETTE["amber"], PALETTE["teal"], "#7db6cc", "#d1a759", "#73839a"][: len(pivot.columns)],
    )
    ax.set_title("Split composition by domain")
    ax.set_ylabel("Rows")
    ax.set_xlabel("Split")
    ax.legend(title="Domain", frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    save_close(fig, CHARTS_DIR / "split_domain_counts.png")


def plot_race_fnr(split: str) -> None:
    frame = pd.read_csv(METRICS_DIR / f"{split}_subgroup_metrics.csv")
    frame = frame[frame["group_type"] == "race"].copy()
    if frame.empty:
        return
    frame = frame.sort_values("minor_false_negative_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.bar(frame["group_value"], frame["minor_false_negative_rate"], color=PALETTE["teal"])
    ax.set_ylabel("Minor false negative rate")
    ax.set_title(f"{split.upper()} minor false negative rate by race")
    ax.tick_params(axis="x", rotation=30)
    save_close(fig, CHARTS_DIR / f"{split}_race_fnr.png")


def plot_robustness_verdicts() -> None:
    frame = pd.read_csv(METRICS_DIR / "robustness_predictions.csv")
    counts = frame["verdict"].value_counts()

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    colors = {
        "flagged": PALETTE["red"],
        "uncertain": PALETTE["amber"],
        "safe": PALETTE["green"],
    }
    ax.bar(counts.index, counts.values, color=[colors.get(label, PALETTE["slate"]) for label in counts.index])
    ax.set_ylabel("Count")
    ax.set_title("Robustness split verdict counts")
    save_close(fig, CHARTS_DIR / "robustness_verdict_counts.png")


def plot_robustness_policy_reasons() -> None:
    frame = pd.read_csv(METRICS_DIR / "robustness_predictions.csv")
    counts = frame["policy_reason"].value_counts().head(5)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.bar(counts.index, counts.values, color=PALETTE["navy"])
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("Count")
    ax.set_title("Top robustness policy reasons")
    save_close(fig, CHARTS_DIR / "robustness_policy_reasons.png")


def plot_source_validation_table() -> None:
    payload = load_json(MANIFESTS_DIR / "source_validation.json")
    frame = pd.DataFrame(payload["results"])[["source_name", "role", "image_count", "usable", "reason"]].copy()
    frame["source_name"] = frame["source_name"].str.replace("_", " ")
    frame["role"] = frame["role"].str.replace("_", " ")
    frame["image_count"] = frame["image_count"].map(thousands)
    frame["usable"] = frame["usable"].map(lambda item: "yes" if item else "no")
    frame["reason"] = frame["reason"].map(lambda text: shorten(text, 38))
    frame.columns = ["Source", "Role", "Images seen", "Usable", "Validation note"]
    render_table(
        frame,
        TABLES_DIR / "source_validation_table.png",
        title="Dataset source validation snapshot",
        subtitle="Parsed from ml/training/outputs/manifests/source_validation.json",
        col_widths=[0.18, 0.22, 0.16, 0.10, 0.34],
        font_size=9.5,
    )


def plot_split_summary_table() -> None:
    manifest = pd.read_csv(MANIFESTS_DIR / "master_manifest.csv", low_memory=False)
    rows = []
    for split in ("train", "val", "test", "robustness"):
        subset = manifest[manifest["split"] == split]
        if subset.empty:
            continue
        domains = ", ".join(sorted(subset["domain_type"].dropna().unique()))
        supervision = "trusted supervision" if split != "robustness" else "robustness only"
        rows.append(
            {
                "Split": split,
                "Rows": thousands(len(subset)),
                "Domains": domains,
                "Label status": supervision,
            }
        )
    frame = pd.DataFrame(rows)
    render_table(
        frame,
        TABLES_DIR / "split_summary_table.png",
        title="Merged manifest split summary",
        subtitle="Derived from ml/training/outputs/manifests/master_manifest.csv",
        col_widths=[0.16, 0.14, 0.28, 0.42],
        font_size=10,
    )


def plot_manifest_preview_table() -> None:
    manifest = pd.read_csv(MANIFESTS_DIR / "master_manifest.csv", low_memory=False)
    sample_rows: list[pd.Series] = []
    selectors = [
        ("train", "real"),
        ("val", "real"),
        ("test", "real"),
        ("robustness", "ai_generated"),
        ("robustness", "cartoon"),
    ]
    for split, domain in selectors:
        subset = manifest[(manifest["split"] == split) & (manifest["domain_type"] == domain)]
        if not subset.empty:
            sample_rows.append(subset.iloc[0])
    if not sample_rows:
        return

    frame = pd.DataFrame(sample_rows)[
        ["source_dataset", "domain_type", "split", "age_bucket", "minor_label", "label_status", "gender", "race", "quality_tags"]
    ].copy()
    frame["minor_label"] = frame["minor_label"].map(lambda value: "-" if pd.isna(value) else str(int(value)))
    frame["quality_tags"] = frame["quality_tags"].map(lambda value: shorten(value, 20))
    frame.columns = ["Source", "Domain", "Split", "Age bucket", "Minor", "Label status", "Gender", "Race", "Quality tags"]
    render_table(
        frame,
        TABLES_DIR / "manifest_preview_table.png",
        title="Manifest row preview",
        subtitle="A small stratified slice of the shipped master_manifest.csv schema",
        col_widths=[0.12, 0.12, 0.10, 0.10, 0.08, 0.12, 0.10, 0.14, 0.12],
        font_size=9.2,
    )


def plot_manifest_source_counts() -> None:
    manifest = pd.read_csv(MANIFESTS_DIR / "master_manifest.csv", low_memory=False)
    counts = manifest["source_dataset"].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(counts.index, counts.values, color=[PALETTE["navy"], PALETTE["teal"], PALETTE["amber"]][: len(counts)])
    ax.set_xlabel("Rows")
    ax.set_title("Rows by source dataset in the shipped manifest")
    save_close(fig, CHARTS_DIR / "manifest_source_dataset_counts.png")


def plot_trusted_age_bucket_counts() -> None:
    manifest = pd.read_csv(MANIFESTS_DIR / "master_manifest.csv", low_memory=False)
    subset = manifest[(manifest["split"].isin(["train", "val", "test"])) & (manifest["label_status"] == "trusted")].copy()
    if subset.empty:
        return
    ordered = ["0-12", "13-15", "16-17", "18-20", "21-25", "26+"]
    pivot = (
        subset.groupby(["split", "age_bucket"])
        .size()
        .reset_index(name="count")
        .pivot(index="split", columns="age_bucket", values="count")
        .reindex(columns=[bucket for bucket in ordered if bucket in subset["age_bucket"].unique()])
        .fillna(0)
    )
    pivot = pivot.loc[[item for item in ("train", "val", "test") if item in pivot.index]]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#7cc4a4", "#93b7d4", "#d4b483", "#c7cfd8", "#6fa8dc", "#355c7d"][: len(pivot.columns)])
    ax.set_ylabel("Trusted rows")
    ax.set_xlabel("Split")
    ax.set_title("Trusted age-bucket coverage across supervised splits")
    ax.legend(title="Age bucket", frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    save_close(fig, CHARTS_DIR / "trusted_age_bucket_counts.png")


def plot_label_status_breakdown() -> None:
    manifest = pd.read_csv(MANIFESTS_DIR / "master_manifest.csv", low_memory=False)
    counts = manifest["label_status"].value_counts()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.bar(counts.index, counts.values, color=[PALETTE["green"], PALETTE["amber"], PALETTE["slate"]][: len(counts)])
    ax.set_ylabel("Rows")
    ax.set_title("Trusted vs ambiguous supervision in the merged manifest")
    save_close(fig, CHARTS_DIR / "label_status_breakdown.png")


def plot_failure_reasons(split: str = "test") -> None:
    path = REPORTS_DIR / "failure_analysis" / f"{split}_failures.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    counts = frame["policy_reason"].value_counts().head(5)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.bar(counts.index, counts.values, color=PALETTE["red"])
    ax.tick_params(axis="x", rotation=18)
    ax.set_ylabel("Failure rows")
    ax.set_title(f"Top {split} failure reasons")
    save_close(fig, CHARTS_DIR / f"{split}_failure_reasons.png")


def plot_training_dynamics(model_name: str) -> None:
    history_path = HISTORY_DIR / f"{model_name}_history.json"
    if not history_path.exists():
        return

    frame = pd.DataFrame(load_json(history_path))
    if frame.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    epochs = frame["epoch"]

    axes[0].plot(epochs, frame["train_loss"], marker="o", linewidth=2.2, color=PALETTE["navy"], label="train loss")
    axes[0].plot(epochs, frame["val_loss"], marker="o", linewidth=2.2, color=PALETTE["amber"], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name.upper()} optimization")
    axes[0].legend(frameon=False)
    axes[0].grid(True)

    axes[1].plot(epochs, frame["val_minor_recall"], marker="o", linewidth=2.2, color=PALETTE["green"], label="val recall")
    if "val_minor_precision" in frame.columns:
        axes[1].plot(epochs, frame["val_minor_precision"], marker="o", linewidth=2.2, color=PALETTE["teal"], label="val precision")
    best_index = frame["val_minor_recall"].astype(float).idxmax()
    best_row = frame.loc[best_index]
    axes[1].scatter([best_row["epoch"]], [best_row["val_minor_recall"]], s=80, color=PALETTE["red"], zorder=5)
    axes[1].annotate(
        f"best recall: epoch {int(best_row['epoch'])}",
        xy=(best_row["epoch"], best_row["val_minor_recall"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9.5,
        color=PALETTE["ink"],
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.82 if model_name == "aux" else 0.85, 1.01)
    axes[1].set_title(f"{model_name.upper()} validation metrics")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(True)

    save_close(fig, CHARTS_DIR / f"{model_name}_training_dynamics.png")


def candidate_raw_roots() -> list[Path]:
    roots = [ROOT_DIR / "data" / "raw"]
    env_root = os.environ.get("HYGO_RAW_ROOT")
    if env_root:
        roots.insert(0, Path(env_root))
    return [root for root in roots if root.exists()]


def list_images(directory: Path, limit: int = 2) -> list[Path]:
    matches = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        matches.extend(sorted(directory.rglob(pattern)))
    return matches[:limit]


def plot_dataset_or_reviewer_gallery() -> None:
    gallery_path = GALLERIES_DIR / "domain_example_gallery.png"
    source_specs = [
        ("fairface", "Real-photo supervision"),
        ("utkface", "UTKFace ablation"),
        ("deepfakeface", "AI-generated robustness"),
        ("icartoonface", "Cartoon robustness"),
    ]

    raw_samples: list[tuple[Path, str]] = []
    for raw_root in candidate_raw_roots():
        for source_name, label in source_specs:
            source_dir = raw_root / source_name
            if not source_dir.exists():
                source_dir = raw_root / "nonreal" / source_name
            if source_dir.exists():
                for image_path in list_images(source_dir, limit=2):
                    raw_samples.append((image_path, label))

    if raw_samples:
        items = raw_samples[:8]
        title = "Representative raw-dataset examples"
        subtitle = "Automatically sampled from data/raw when the source images are present locally."
    else:
        sample_manifest = load_json(REVIEWER_DIR / "sample_manifest.json")
        preferred = [
            "adult_face_1.jpg",
            "minor_face_1.jpg",
            "adult_face_AI.jpg",
            "minor_face_AI.jpg",
            "cartoon_face_1.jpg",
            "anime_face_1.jpg",
            "multi_face_1.jpg",
            "no_face_1.jpg",
        ]
        manifest_by_name = {item["filename"]: item for item in sample_manifest}
        items = []
        for filename in preferred:
            path = REVIEWER_DIR / filename
            if path.exists():
                meta = manifest_by_name.get(filename, {})
                label = meta.get("category", filename.replace(".jpg", "")).replace("_", " ")
                items.append((path, label))
        title = "Representative deployed-input examples"
        subtitle = "Fallback gallery from reviewer_samples because the raw training datasets are not vendored in this repo snapshot."

    cols = 4
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13.5, 3.6 * rows))
    if not isinstance(axes, (list, tuple)):
        axes = axes.ravel()
    else:
        axes = axes.ravel()

    fig.suptitle(title, fontsize=18, fontweight="semibold", color=PALETTE["ink"], y=1.02)
    fig.text(0.01, 0.98, subtitle, fontsize=10.5, color=PALETTE["slate"], ha="left", va="top")

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#f7f9fc")

    for ax, (image_path, label) in zip(axes, items):
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax.set_title(wrap_title(label.title(), width=18), fontsize=11.5, color=PALETTE["ink"], pad=8)
        ax.axis("off")

    save_close(fig, gallery_path, dpi=180)


def main() -> None:
    ensure_dirs()
    for split in ("val", "test"):
        plot_roc_and_pr(split)
        plot_confusion_matrix(split)
        plot_race_fnr(split)
    plot_split_domain_counts()
    plot_robustness_verdicts()
    plot_robustness_policy_reasons()
    plot_source_validation_table()
    plot_split_summary_table()
    plot_manifest_preview_table()
    plot_manifest_source_counts()
    plot_trusted_age_bucket_counts()
    plot_label_status_breakdown()
    plot_failure_reasons("test")
    plot_training_dynamics("main")
    plot_training_dynamics("aux")
    plot_dataset_or_reviewer_gallery()


if __name__ == "__main__":
    main()
