#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from pathlib import Path
from tempfile import gettempdir

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "hygo-matplotlib-cache"))

import matplotlib
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
CHARTS_DIR = REPORTS_DIR / "charts"
CONFUSION_DIR = REPORTS_DIR / "confusion_matrices"
RELIABILITY_DIR = REPORTS_DIR / "reliability"


def ensure_dirs() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc_and_pr(split: str) -> None:
    frame = pd.read_csv(METRICS_DIR / f"{split}_predictions.csv")
    frame = frame[frame["minor_label"].notna()].copy()
    if frame.empty:
        return

    y_true = frame["minor_label"].astype(int)
    y_score = frame["p_minor"].astype(float)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"{split.upper()} ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split.upper()} ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"{split}_roc_curve.png", dpi=160)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"{split.upper()} PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{split.upper()} Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"{split}_pr_curve.png", dpi=160)
    plt.close()


def plot_confusion_matrix(split: str) -> None:
    matrix_path = CONFUSION_DIR / f"{split}_confusion_matrix.json"
    if not matrix_path.exists():
        return

    payload = json.loads(matrix_path.read_text())
    matrix = payload["matrix"]
    labels = ["adult", "minor"]

    plt.figure(figsize=(4.5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.xticks(range(2), labels)
    plt.yticks(range(2), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{split.upper()} Confusion Matrix")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(CONFUSION_DIR / f"{split}_confusion_matrix.png", dpi=160)
    plt.close()


def plot_split_domain_counts() -> None:
    frame = pd.read_csv(METRICS_DIR / "split_summary.csv")
    pivot = frame.pivot(index="split", columns="domain_type", values="count").fillna(0)
    pivot = pivot.loc[[index for index in ["train", "val", "test", "robustness"] if index in pivot.index]]

    ax = pivot.plot(kind="bar", stacked=True, figsize=(7, 4), colormap="tab20c")
    ax.set_title("Split Composition by Domain")
    ax.set_ylabel("Row count")
    ax.set_xlabel("Split")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "split_domain_counts.png", dpi=160)
    plt.close()


def plot_race_fnr(split: str) -> None:
    frame = pd.read_csv(METRICS_DIR / f"{split}_subgroup_metrics.csv")
    frame = frame[frame["group_type"] == "race"].copy()
    if frame.empty:
        return
    frame = frame.sort_values("minor_false_negative_rate", ascending=False)

    plt.figure(figsize=(7, 4))
    plt.bar(frame["group_value"], frame["minor_false_negative_rate"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Minor false negative rate")
    plt.title(f"{split.upper()} Minor False Negative Rate by Race")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"{split}_race_fnr.png", dpi=160)
    plt.close()


def plot_robustness_verdicts() -> None:
    frame = pd.read_csv(METRICS_DIR / "robustness_predictions.csv")
    counts = frame["verdict"].value_counts()

    plt.figure(figsize=(5, 4))
    plt.bar(counts.index, counts.values, color=["#c44e52", "#8172b2", "#55a868"][: len(counts)])
    plt.ylabel("Count")
    plt.title("Robustness Split Verdict Counts")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "robustness_verdict_counts.png", dpi=160)
    plt.close()


def plot_robustness_policy_reasons() -> None:
    frame = pd.read_csv(METRICS_DIR / "robustness_predictions.csv")
    counts = frame["policy_reason"].value_counts().head(5)

    plt.figure(figsize=(7, 4))
    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Count")
    plt.title("Top Robustness Policy Reasons")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "robustness_policy_reasons.png", dpi=160)
    plt.close()


def main() -> None:
    ensure_dirs()
    for split in ("val", "test"):
        plot_roc_and_pr(split)
        plot_confusion_matrix(split)
        plot_race_fnr(split)
    plot_split_domain_counts()
    plot_robustness_verdicts()
    plot_robustness_policy_reasons()


if __name__ == "__main__":
    main()
