#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "hygo-mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT_DIR = Path(__file__).resolve().parents[1]
DIAGRAMS_DIR = ROOT_DIR / "reports" / "diagrams"


@dataclass(frozen=True)
class Theme:
    bg: str = "#f6f1e8"
    ink: str = "#1b2436"
    muted: str = "#667187"
    line: str = "#2e3b56"
    shadow: str = "#cfc5b8"
    blue: str = "#e6eefc"
    blue_edge: str = "#6188d8"
    gold: str = "#f9edcf"
    gold_edge: str = "#c48a1c"
    green: str = "#e3efe1"
    green_edge: str = "#4f9268"
    plum: str = "#ece7fb"
    plum_edge: str = "#7868cf"
    rose: str = "#f5e5ec"
    rose_edge: str = "#c56f97"
    slate: str = "#f1f3f8"
    slate_edge: str = "#7a869d"
    red: str = "#f6e0e0"
    red_edge: str = "#c9606c"
    lane_blue: str = "#f2f6fd"
    lane_rose: str = "#fcf4f7"
    lane_plum: str = "#f6f3fd"


THEME = Theme()


def base_figure(width: float, height: float):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(THEME.bg)
    ax.set_facecolor(THEME.bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def add_card(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle="",
    *,
    fill,
    edge,
    accent=None,
    title_size=18,
    subtitle_size=11,
    title_align="left",
    subtitle_align=None,
):
    shadow = FancyBboxPatch(
        (x + 0.004, y - 0.004),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.024",
        linewidth=0,
        facecolor=THEME.shadow,
        alpha=0.13,
        zorder=1,
    )
    ax.add_patch(shadow)

    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.024",
        linewidth=2.1,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(card)

    accent_color = accent or edge
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.018, y + h - 0.023),
            max(0.05, w * 0.62),
            0.011,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0,
            facecolor=accent_color,
            zorder=3,
        )
    )

    if title_align == "center":
        tx = x + w / 2
        tha = "center"
    else:
        tx = x + 0.024
        tha = "left"

    ax.text(
        tx,
        y + (0.58 * h if subtitle else 0.50 * h),
        title,
        ha=tha,
        va="center",
        color=THEME.ink,
        fontsize=title_size,
        fontweight="bold",
        linespacing=1.05,
        zorder=4,
        wrap=True,
    )

    if subtitle:
        subtitle_align = subtitle_align or title_align
        if subtitle_align == "center":
            sx = x + w / 2
            sha = "center"
        else:
            sx = x + 0.024
            sha = "left"
        ax.text(
            sx,
            y + 0.28 * h,
            subtitle,
            ha=sha,
            va="center",
            color=THEME.muted,
            fontsize=subtitle_size,
            linespacing=1.15,
            zorder=4,
            wrap=True,
        )


def add_chip(ax, x, y, text, *, fill, edge, text_color=None, width=None, height=0.046, size=11):
    width = width or max(0.10, 0.0105 * len(text) + 0.03)
    chip = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=fill,
        zorder=5,
    )
    ax.add_patch(chip)
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.01, y + height - 0.012),
            width - 0.02,
            0.006,
            boxstyle="round,pad=0.002,rounding_size=0.01",
            linewidth=0,
            facecolor=edge,
            zorder=6,
            alpha=0.85,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2 - 0.002,
        text,
        ha="center",
        va="center",
        color=text_color or edge,
        fontsize=size,
        fontweight="bold",
        zorder=6,
    )


def add_lane(ax, x, y, w, h, *, fill, edge, accent, label, label_fill, label_edge):
    frame = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.028",
        linewidth=2.0,
        edgecolor=edge,
        facecolor=fill,
        zorder=1,
    )
    ax.add_patch(frame)
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.014, y + h - 0.020),
            w - 0.028,
            0.010,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            linewidth=0,
            facecolor=accent,
            zorder=2,
        )
    )
    add_chip(ax, x + 0.03, y + h + 0.012, label, fill=label_fill, edge=label_edge, width=max(0.18, 0.0108 * len(label) + 0.08), size=12)


def add_panel(ax, x, y, w, h, *, title, fill, edge):
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014,rounding_size=0.028",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=fill,
        zorder=0,
    )
    ax.add_patch(panel)
    ax.text(
        x + 0.03,
        y + h - 0.06,
        title.upper(),
        ha="left",
        va="center",
        color=edge,
        fontsize=14,
        fontweight="bold",
        zorder=2,
    )
    ax.plot(
        [x + 0.03, x + w - 0.03],
        [y + h - 0.09, y + h - 0.09],
        color=edge,
        linewidth=2.0,
        alpha=0.45,
        zorder=1,
    )


def add_arrow(ax, start, end, *, color=None, lw=2.5, curve=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=lw,
        color=color or THEME.line,
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=2,
        shrinkB=2,
        zorder=4,
    )
    ax.add_patch(arrow)


def add_orthogonal_arrow(ax, points, *, color=None, lw=2.7):
    color = color or THEME.line
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=lw,
            solid_capstyle="round",
            zorder=4,
        )
    add_arrow(ax, points[-2], points[-1], color=color, lw=lw)


def add_label(ax, x, y, text, *, size=10.5, color=None, weight="normal"):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=color or THEME.muted,
        fontsize=size,
        fontweight=weight,
        zorder=6,
    )


def save(fig, name: str):
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.25)
    fig.savefig(DIAGRAMS_DIR / name, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def system_diagram():
    fig, ax = base_figure(18.6, 7.1)

    add_panel(ax, 0.04, 0.11, 0.25, 0.74, title="Perception", fill=THEME.lane_blue, edge=THEME.blue_edge)
    add_panel(ax, 0.375, 0.11, 0.27, 0.74, title="Dual-model scoring", fill="#f7f5fb", edge=THEME.plum_edge)
    add_panel(ax, 0.73, 0.11, 0.23, 0.74, title="Policy decision", fill=THEME.lane_plum, edge=THEME.plum_edge)

    add_card(ax, 0.08, 0.58, 0.17, 0.14, "Input image", "reviewer or product upload", fill=THEME.blue, edge=THEME.blue_edge, title_size=19, subtitle_size=11, title_align="center", subtitle_align="center")
    add_card(ax, 0.08, 0.37, 0.17, 0.17, "InsightFace", "detect\nalign\nface confidence", fill=THEME.gold, edge=THEME.gold_edge, title_size=20, subtitle_size=12, title_align="center", subtitle_align="center")
    add_card(ax, 0.08, 0.16, 0.17, 0.13, "Face crop", "normalized 224 x 224", fill=THEME.slate, edge=THEME.slate_edge, title_size=18, subtitle_size=11, title_align="center", subtitle_align="center")

    add_arrow(ax, (0.165, 0.58), (0.165, 0.54), lw=2.5)
    add_arrow(ax, (0.165, 0.37), (0.165, 0.29), lw=2.5)

    add_card(ax, 0.44, 0.54, 0.17, 0.16, "Main model", "age + minor risk\nfine-tuned EfficientNet-B0", fill=THEME.green, edge=THEME.green_edge, title_size=19, subtitle_size=11, title_align="center", subtitle_align="center")
    add_card(ax, 0.44, 0.25, 0.17, 0.16, "Aux model", "domain + uncertainty\nfrozen DINOv2 encoder", fill=THEME.rose, edge=THEME.rose_edge, title_size=19, subtitle_size=11, title_align="center", subtitle_align="center")

    branch_x, branch_y = 0.30, 0.225
    join_x, join_y = 0.66, 0.475

    add_arrow(ax, (0.25, 0.225), (branch_x, branch_y), lw=2.8)
    add_orthogonal_arrow(ax, [(branch_x, branch_y), (0.37, branch_y), (0.37, 0.62), (0.44, 0.62)], lw=2.8)
    add_orthogonal_arrow(ax, [(branch_x, branch_y), (0.37, branch_y), (0.37, 0.33), (0.44, 0.33)], lw=2.8)
    add_label(ax, 0.36, 0.47, "age / minor-risk path", size=10.3)
    add_label(ax, 0.35, 0.185, "domain / uncertainty path", size=10.3)

    add_card(ax, 0.775, 0.53, 0.15, 0.14, "Fusion + calibration", "temperature scale\nage interval\nconflict features", fill=THEME.plum, edge=THEME.plum_edge, title_size=17, subtitle_size=10.5, title_align="center", subtitle_align="center")
    add_card(ax, 0.775, 0.33, 0.15, 0.12, "Policy gate", "safe vs abstain vs flag", fill=THEME.gold, edge=THEME.gold_edge, title_size=17, subtitle_size=10.5, title_align="center", subtitle_align="center")
    add_card(ax, 0.775, 0.14, 0.15, 0.12, "API output", "verdict\nrisk_score\npolicy_reason", fill=THEME.slate, edge=THEME.slate_edge, title_size=16, subtitle_size=10, title_align="center", subtitle_align="center")

    add_orthogonal_arrow(ax, [(0.61, 0.62), (0.64, 0.62), (join_x, 0.62), (join_x, join_y)], lw=2.8)
    add_orthogonal_arrow(ax, [(0.61, 0.33), (0.64, 0.33), (join_x, 0.33), (join_x, join_y)], lw=2.8)
    add_orthogonal_arrow(ax, [(join_x, join_y), (0.72, join_y), (0.72, 0.60), (0.775, 0.60)], lw=2.8)
    add_arrow(ax, (0.85, 0.53), (0.85, 0.45), lw=2.6)
    add_arrow(ax, (0.85, 0.33), (0.85, 0.26), lw=2.6)

    add_chip(ax, 0.77, 0.032, "safe", fill=THEME.green, edge=THEME.green_edge, width=0.052, height=0.036, size=9.5)
    add_chip(ax, 0.83, 0.032, "uncertain", fill=THEME.gold, edge=THEME.gold_edge, width=0.074, height=0.036, size=9.5)
    add_chip(ax, 0.912, 0.032, "flagged", fill=THEME.red, edge=THEME.red_edge, width=0.062, height=0.036, size=9.5)

    add_label(ax, 0.50, 0.045, "Shipped inference path: perception -> dual-model scoring -> calibrated policy verdict", size=12)
    save(fig, "system_inference_architecture.png")


def main_model_diagram():
    fig, ax = base_figure(12.6, 8.8)

    add_chip(ax, 0.04, 0.94, "MAIN MODEL ARCHITECTURE", fill=THEME.green, edge=THEME.green_edge, width=0.26, size=12)
    add_chip(ax, 0.71, 0.94, "full fine-tuning", fill="#fdfdfc", edge=THEME.green_edge, width=0.15, size=11)
    add_chip(ax, 0.87, 0.94, "three heads", fill="#fdfdfc", edge=THEME.plum_edge, width=0.10, size=11)

    add_card(ax, 0.16, 0.83, 0.68, 0.10, "224 x 224 RGB face crop", fill=THEME.blue, edge=THEME.blue_edge, title_size=19, title_align="center")
    add_card(ax, 0.12, 0.63, 0.76, 0.14, "EfficientNet-B0 backbone", "pretrained timm backbone reused for feature extraction", fill=THEME.green, edge=THEME.green_edge, title_size=24, subtitle_size=12, title_align="center", subtitle_align="center")
    add_card(ax, 0.23, 0.46, 0.54, 0.11, "Global pooled embedding", "compact feature vector passed to task heads", fill=THEME.slate, edge=THEME.slate_edge, title_size=19, subtitle_size=11, title_align="center", subtitle_align="center")
    add_card(ax, 0.28, 0.31, 0.44, 0.09, "LayerNorm + Dropout(0.2)", fill=THEME.plum, edge=THEME.plum_edge, title_size=18, title_align="center")

    add_card(ax, 0.05, 0.08, 0.24, 0.15, "Age head", "scalar age regression", fill=THEME.green, edge=THEME.green_edge, title_size=20, subtitle_size=12, title_align="center", subtitle_align="center")
    add_card(ax, 0.38, 0.08, 0.24, 0.15, "Bucket head", "6 age-bucket logits", fill=THEME.gold, edge=THEME.gold_edge, title_size=20, subtitle_size=12, title_align="center", subtitle_align="center")
    add_card(ax, 0.71, 0.08, 0.24, 0.15, "Minor head", "minor-risk logit", fill=THEME.red, edge=THEME.red_edge, title_size=20, subtitle_size=12, title_align="center", subtitle_align="center")

    add_arrow(ax, (0.50, 0.83), (0.50, 0.77))
    add_arrow(ax, (0.50, 0.63), (0.50, 0.57))
    add_arrow(ax, (0.50, 0.46), (0.50, 0.40))
    add_arrow(ax, (0.50, 0.31), (0.17, 0.23))
    add_arrow(ax, (0.50, 0.31), (0.50, 0.23))
    add_arrow(ax, (0.50, 0.31), (0.83, 0.23))

    add_label(ax, 0.50, 0.02, "Objective: age regression + bucket classification + minor-risk BCE", size=12)
    save(fig, "main_model_architecture.png")


def aux_model_diagram():
    fig, ax = base_figure(12.8, 9.0)

    add_chip(ax, 0.04, 0.94, "AUXILIARY MODEL ARCHITECTURE", fill=THEME.rose, edge=THEME.rose_edge, width=0.31, size=12)
    add_chip(ax, 0.71, 0.94, "encoder frozen", fill="#fdfdfc", edge=THEME.rose_edge, width=0.14, size=11)
    add_chip(ax, 0.86, 0.94, "head trainable", fill="#fdfdfc", edge=THEME.green_edge, width=0.13, size=11)

    add_card(ax, 0.17, 0.84, 0.66, 0.10, "224 x 224 RGB face crop", fill=THEME.blue, edge=THEME.blue_edge, title_size=19, title_align="center")
    add_card(ax, 0.10, 0.66, 0.80, 0.13, "DINOv2 encoder", "strong pretrained representation reused without backbone updates", fill=THEME.rose, edge=THEME.rose_edge, title_size=24, subtitle_size=12, title_align="center", subtitle_align="center")
    add_card(ax, 0.22, 0.49, 0.56, 0.11, "Encoder embedding", "frozen features forwarded to the lightweight head", fill=THEME.slate, edge=THEME.slate_edge, title_size=19, subtitle_size=11, title_align="center", subtitle_align="center")
    add_card(ax, 0.20, 0.32, 0.60, 0.12, "MLP head", "LayerNorm -> Dropout -> Linear -> GELU -> Dropout", fill=THEME.plum, edge=THEME.plum_edge, title_size=21, subtitle_size=11, title_align="center", subtitle_align="center")

    add_card(ax, 0.04, 0.08, 0.25, 0.16, "Minor branch", "aux minor-risk logit", fill=THEME.red, edge=THEME.red_edge, title_size=20, subtitle_size=12, title_align="center", subtitle_align="center")
    add_card(ax, 0.375, 0.08, 0.25, 0.16, "Domain branch", "7 domain logits", fill=THEME.gold, edge=THEME.gold_edge, title_size=20, subtitle_size=12, title_align="center", subtitle_align="center")
    add_card(ax, 0.71, 0.08, 0.25, 0.16, "Uncertainty branch", "uncertainty logit", fill=THEME.green, edge=THEME.green_edge, title_size=20, subtitle_size=12, title_align="center", subtitle_align="center")

    add_arrow(ax, (0.50, 0.84), (0.50, 0.79))
    add_arrow(ax, (0.50, 0.66), (0.50, 0.60))
    add_arrow(ax, (0.50, 0.49), (0.50, 0.44))
    add_arrow(ax, (0.50, 0.32), (0.165, 0.24))
    add_arrow(ax, (0.50, 0.32), (0.50, 0.24))
    add_arrow(ax, (0.50, 0.32), (0.835, 0.24))

    add_label(ax, 0.50, 0.02, "Objective: minor-risk + domain classification + uncertainty estimation", size=12)
    save(fig, "aux_model_architecture.png")


def fine_tuning_diagram():
    fig, ax = base_figure(16.8, 7.6)

    add_chip(ax, 0.04, 0.94, "FINE-TUNING PROCESS", fill=THEME.blue, edge=THEME.blue_edge, width=0.20, size=12)

    add_lane(
        ax,
        0.03,
        0.63,
        0.94,
        0.21,
        fill=THEME.lane_blue,
        edge=THEME.line,
        accent=THEME.blue_edge,
        label="1. Main model training",
        label_fill=THEME.blue,
        label_edge=THEME.blue_edge,
    )
    add_lane(
        ax,
        0.17,
        0.38,
        0.70,
        0.15,
        fill=THEME.lane_plum,
        edge=THEME.line,
        accent=THEME.plum_edge,
        label="2. Calibration and export",
        label_fill=THEME.plum,
        label_edge=THEME.plum_edge,
    )
    add_lane(
        ax,
        0.03,
        0.10,
        0.94,
        0.21,
        fill=THEME.lane_rose,
        edge=THEME.line,
        accent=THEME.rose_edge,
        label="3. Auxiliary model training",
        label_fill=THEME.rose,
        label_edge=THEME.rose_edge,
    )

    main_y = 0.66
    aux_y = 0.13
    calib_y = 0.40

    add_card(ax, 0.06, main_y, 0.16, 0.10, "Initialize", "pretrained EfficientNet-B0", fill=THEME.blue, edge=THEME.blue_edge, title_size=18, subtitle_size=11, title_align="center", subtitle_align="center")
    add_card(ax, 0.28, main_y, 0.16, 0.10, "Fine-tune", "all backbone blocks + heads", fill=THEME.green, edge=THEME.green_edge, title_size=18, subtitle_size=11, title_align="center", subtitle_align="center")
    add_card(ax, 0.50, main_y, 0.18, 0.10, "Optimize", "AdamW + AMP + grad accumulation", fill=THEME.gold, edge=THEME.gold_edge, title_size=18, subtitle_size=11, title_align="center", subtitle_align="center")
    add_card(ax, 0.75, main_y, 0.16, 0.10, "Checkpoint", "best validation minor recall", fill=THEME.green, edge=THEME.green_edge, title_size=18, subtitle_size=11, title_align="center", subtitle_align="center")

    add_card(ax, 0.22, calib_y, 0.17, 0.09, "Validation inference", "export val predictions", fill=THEME.plum, edge=THEME.plum_edge, title_size=17, subtitle_size=10.5, title_align="center", subtitle_align="center")
    add_card(ax, 0.44, calib_y, 0.17, 0.09, "Temperature scaling", "LBFGS on validation logits", fill=THEME.plum, edge=THEME.plum_edge, title_size=17, subtitle_size=10.5, title_align="center", subtitle_align="center")
    add_card(ax, 0.66, calib_y, 0.17, 0.09, "Export assets", "checkpoint + calibration + policy", fill=THEME.green, edge=THEME.green_edge, title_size=17, subtitle_size=10, title_align="center", subtitle_align="center")

    add_card(ax, 0.06, aux_y, 0.16, 0.10, "Initialize", "pretrained DINOv2", fill=THEME.rose, edge=THEME.rose_edge, title_size=18, subtitle_size=11, title_align="center", subtitle_align="center")
    add_card(ax, 0.28, aux_y, 0.16, 0.10, "Freeze encoder", "train only the MLP head", fill=THEME.rose, edge=THEME.rose_edge, title_size=18, subtitle_size=11, title_align="center", subtitle_align="center")
    add_card(ax, 0.50, aux_y, 0.18, 0.10, "Optimize", "minor + domain + uncertainty", fill=THEME.gold, edge=THEME.gold_edge, title_size=18, subtitle_size=11, title_align="center", subtitle_align="center")
    add_card(ax, 0.75, aux_y, 0.16, 0.10, "Checkpoint", "best validation minor recall", fill=THEME.green, edge=THEME.green_edge, title_size=18, subtitle_size=11, title_align="center", subtitle_align="center")

    for y in (main_y + 0.05, aux_y + 0.05):
        add_arrow(ax, (0.22, y), (0.28, y))
        add_arrow(ax, (0.44, y), (0.50, y))
        add_arrow(ax, (0.68, y), (0.75, y))

    add_arrow(ax, (0.83, main_y), (0.31, calib_y + 0.09), curve=0.05)
    add_arrow(ax, (0.39, calib_y + 0.045), (0.44, calib_y + 0.045))
    add_arrow(ax, (0.61, calib_y + 0.045), (0.66, calib_y + 0.045))

    add_label(ax, 0.50, 0.04, "Shipped setup: main model fully fine-tuned, auxiliary encoder frozen, policy calibrated after validation inference", size=12)
    save(fig, "fine_tuning_map.png")


def main():
    system_diagram()
    main_model_diagram()
    aux_model_diagram()
    fine_tuning_diagram()


if __name__ == "__main__":
    main()
