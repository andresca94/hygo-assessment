from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.datasets.face_dataset import ManifestFaceDataset
from ml.training.models.mivolo_wrapper import MiVOLOAgeEstimator
from ml.training.utils import AGE_BUCKETS, ensure_dir, load_yaml, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the primary age-safety model.")
    parser.add_argument("--config", required=True)
    return parser


def compute_loss(outputs: dict, batch: dict, config: dict) -> tuple[torch.Tensor, dict[str, float]]:
    regression_mask = batch["has_age_value"] > 0
    binary_mask = batch["has_minor_label"] > 0
    sample_weight = batch["sample_weight"]

    if regression_mask.any():
        regression_loss = F.smooth_l1_loss(outputs["age"][regression_mask], batch["age_value"][regression_mask], reduction="none")
        regression_loss = (regression_loss * sample_weight[regression_mask]).mean()
    else:
        regression_loss = outputs["age"].mean() * 0

    bucket_loss = F.cross_entropy(outputs["bucket_logits"], batch["bucket_index"], reduction="none")
    bucket_loss = (bucket_loss * sample_weight).mean()

    if binary_mask.any():
        binary_loss = F.binary_cross_entropy_with_logits(
            outputs["minor_logit"][binary_mask],
            batch["minor_label"][binary_mask],
            reduction="none",
        )
        binary_loss = (binary_loss * sample_weight[binary_mask]).mean()
    else:
        binary_loss = outputs["minor_logit"].mean() * 0

    total_loss = (
        config["loss"]["regression_weight"] * regression_loss
        + config["loss"]["bucket_weight"] * bucket_loss
        + config["loss"]["binary_weight"] * binary_loss
    )
    metrics = {
        "regression_loss": float(regression_loss.detach().cpu()),
        "bucket_loss": float(bucket_loss.detach().cpu()),
        "binary_loss": float(binary_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }
    return total_loss, metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, config: dict) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    true_positive = 0
    false_negative = 0
    predicted_positive = 0

    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["image"])
        loss, metrics = compute_loss(outputs, batch, config)
        losses.append(metrics["total_loss"])
        minor_probs = torch.sigmoid(outputs["minor_logit"])
        predictions = (minor_probs >= 0.5).float()
        labels = batch["minor_label"]
        mask = batch["has_minor_label"] > 0
        if mask.any():
            preds = predictions[mask]
            targets = labels[mask]
            true_positive += int(((preds == 1) & (targets == 1)).sum().item())
            false_negative += int(((preds == 0) & (targets == 1)).sum().item())
            predicted_positive += int((preds == 1).sum().item())

    recall = true_positive / max(true_positive + false_negative, 1)
    precision = true_positive / max(predicted_positive, 1)
    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "minor_recall": recall,
        "minor_precision": precision,
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["seed"]))

    manifest_path = config["paths"]["manifest"]
    checkpoints_dir = ensure_dir(config["paths"]["checkpoints_dir"])
    history_dir = ensure_dir(config["paths"]["history_dir"])
    exported_dir = ensure_dir(config["paths"]["exported_dir"])

    train_dataset = ManifestFaceDataset(
        manifest_path=manifest_path,
        split="train",
        image_size=int(config["data"]["image_size"]),
        train=True,
        include_weak_labels=bool(config["data"]["include_weak_labels"]),
    )
    val_dataset = ManifestFaceDataset(
        manifest_path=manifest_path,
        split="val",
        image_size=int(config["data"]["image_size"]),
        train=False,
        include_weak_labels=bool(config["data"]["include_weak_labels"]),
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"No train samples were found in {manifest_path}. Populate data/raw, rerun the dataset prep scripts, "
            "and rebuild the master manifest."
        )
    if len(val_dataset) == 0:
        raise RuntimeError(
            f"No validation samples were found in {manifest_path}. Add more supervised data so the split step can "
            "create a val set before training."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiVOLOAgeEstimator(
        backbone_name=config["model"]["main"]["backbone_name"],
        pretrained=bool(config["model"]["main"]["pretrained"]),
        num_buckets=len(AGE_BUCKETS),
        dropout=float(config["model"]["main"]["dropout"]),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["training"]["amp"]) and device.type == "cuda")
    accumulation_steps = int(config["training"]["gradient_accumulation_steps"])

    history: list[dict[str, float | int]] = []
    best_recall = -1.0
    best_loss = float("inf")

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            with torch.cuda.amp.autocast(enabled=bool(config["training"]["amp"]) and device.type == "cuda"):
                outputs = model(batch["image"])
                loss, _ = compute_loss(outputs, batch, config)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            if step % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu()) * accumulation_steps

        if len(train_loader) % accumulation_steps != 0 and len(train_loader) > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        val_metrics = evaluate(model, val_loader, device, config)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(len(train_loader), 1),
            "val_loss": val_metrics["loss"],
            "val_minor_recall": val_metrics["minor_recall"],
            "val_minor_precision": val_metrics["minor_precision"],
        }
        history.append(epoch_record)

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch + 1,
            "metrics": epoch_record,
        }
        torch.save(checkpoint_payload, checkpoints_dir / "main_last.pt")

        improved = val_metrics["minor_recall"] > best_recall or (
            val_metrics["minor_recall"] == best_recall and val_metrics["loss"] < best_loss
        )
        if improved:
            best_recall = val_metrics["minor_recall"]
            best_loss = val_metrics["loss"]
            torch.save(checkpoint_payload, checkpoints_dir / "main_best.pt")
            torch.save(checkpoint_payload, exported_dir / "main_best.pt")

        print(json.dumps(epoch_record))

    (history_dir / "main_history.json").write_text(json.dumps(history, indent=2))
    print(f"Main model training complete. Best recall={best_recall:.4f}")


if __name__ == "__main__":
    main()
