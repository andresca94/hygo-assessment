from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.datasets.face_dataset import ManifestFaceDataset
from ml.training.models.dinov2_head import DINOv2AuxiliaryModel
from ml.training.utils import ensure_dir, load_yaml, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the auxiliary domain-robust model.")
    parser.add_argument("--config", required=True)
    return parser


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def domain_index(domain_type: str, domain_classes: list[str]) -> int:
    try:
        return domain_classes.index(domain_type)
    except ValueError:
        return domain_classes.index("unknown")


def compute_uncertainty_target(batch: dict) -> torch.Tensor:
    scores = 1.0 - batch["quality_score"]
    return torch.clamp(scores, 0.0, 1.0)


@torch.no_grad()
def evaluate(model, loader, device, domain_classes) -> dict[str, float]:
    model.eval()
    losses = []
    true_positive = 0
    false_negative = 0

    for batch in loader:
        batch = move_batch(batch, device)
        domain_targets = torch.tensor([domain_index(item, domain_classes) for item in batch["domain_type"]], device=device)
        uncertainty_targets = compute_uncertainty_target(batch)
        outputs = model(batch["image"])

        minor_loss = F.binary_cross_entropy_with_logits(outputs["minor_logit"], batch["minor_label"])
        domain_loss = F.cross_entropy(outputs["domain_logits"], domain_targets)
        uncertainty_loss = F.binary_cross_entropy_with_logits(outputs["uncertainty_logit"], uncertainty_targets)
        losses.append(float((minor_loss + domain_loss + uncertainty_loss).detach().cpu()))

        minor_probs = torch.sigmoid(outputs["minor_logit"])
        predictions = (minor_probs >= 0.5).float()
        true_positive += int(((predictions == 1) & (batch["minor_label"] == 1)).sum().item())
        false_negative += int(((predictions == 0) & (batch["minor_label"] == 1)).sum().item())

    recall = true_positive / max(true_positive + false_negative, 1)
    return {"loss": sum(losses) / max(len(losses), 1), "minor_recall": recall}


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["seed"]))

    manifest_path = config["paths"]["manifest"]
    checkpoints_dir = ensure_dir(config["paths"]["checkpoints_dir"])
    history_dir = ensure_dir(config["paths"]["history_dir"])
    exported_dir = ensure_dir(config["paths"]["exported_dir"])
    domain_classes = list(config["data"]["domain_classes"])

    train_dataset = ManifestFaceDataset(
        manifest_path=manifest_path,
        split="train",
        image_size=int(config["data"]["image_size"]),
        train=True,
        include_weak_labels=True,
    )
    val_dataset = ManifestFaceDataset(
        manifest_path=manifest_path,
        split="val",
        image_size=int(config["data"]["image_size"]),
        train=False,
        include_weak_labels=True,
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
    model = DINOv2AuxiliaryModel(
        backbone_name=config["model"]["aux"]["backbone_name"],
        backbone_source=config["model"]["aux"].get("backbone_source", "auto"),
        repo_dir=config["model"]["aux"].get("repo_dir"),
        pretrained=bool(config["model"]["aux"]["pretrained"]),
        freeze_encoder=bool(config["model"]["aux"]["freeze_encoder"]),
        num_domains=len(domain_classes),
        dropout=float(config["model"]["aux"]["dropout"]),
    ).to(device)

    optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=float(config["training"]["lr"]))
    history = []
    best_recall = -1.0
    best_loss = float("inf")

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            domain_targets = torch.tensor([domain_index(item, domain_classes) for item in batch["domain_type"]], device=device)
            uncertainty_targets = compute_uncertainty_target(batch)

            outputs = model(batch["image"])
            minor_loss = F.binary_cross_entropy_with_logits(outputs["minor_logit"], batch["minor_label"])
            domain_loss = F.cross_entropy(outputs["domain_logits"], domain_targets)
            uncertainty_loss = F.binary_cross_entropy_with_logits(outputs["uncertainty_logit"], uncertainty_targets)
            loss = 0.45 * minor_loss + 0.35 * domain_loss + 0.20 * uncertainty_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        val_metrics = evaluate(model, val_loader, device, domain_classes)
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(len(train_loader), 1),
            "val_loss": val_metrics["loss"],
            "val_minor_recall": val_metrics["minor_recall"],
        }
        history.append(epoch_record)

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch + 1,
            "metrics": epoch_record,
        }
        torch.save(checkpoint_payload, checkpoints_dir / "aux_last.pt")

        improved = val_metrics["minor_recall"] > best_recall or (
            val_metrics["minor_recall"] == best_recall and val_metrics["loss"] < best_loss
        )
        if improved:
            best_recall = val_metrics["minor_recall"]
            best_loss = val_metrics["loss"]
            torch.save(checkpoint_payload, checkpoints_dir / "aux_best.pt")
            torch.save(checkpoint_payload, exported_dir / "aux_best.pt")

        print(json.dumps(epoch_record))

    (history_dir / "aux_history.json").write_text(json.dumps(history, indent=2))
    print(f"Aux model training complete. Best recall={best_recall:.4f}")


if __name__ == "__main__":
    main()
