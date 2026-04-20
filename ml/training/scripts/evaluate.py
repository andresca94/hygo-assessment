from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.datasets.face_dataset import ManifestFaceDataset
from ml.training.models.dinov2_head import DINOv2AuxiliaryModel
from ml.training.models.mivolo_wrapper import MiVOLOAgeEstimator
from ml.training.models.policy import FacePolicyInput, PolicyConfig, PolicyEngine
from ml.training.utils import AGE_BUCKETS, ensure_dir, load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the scaffold on a given split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "robustness"])
    return parser


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def maybe_load_calibration(exported_dir: Path) -> float:
    calibration_path = exported_dir / "calibration.json"
    if not calibration_path.exists():
        return 1.0
    payload = json.loads(calibration_path.read_text())
    return float(payload.get("temperature", 1.0))


def export_policy(config: dict, exported_dir: Path) -> None:
    policy_payload = {
        "safe_threshold": config["policy"]["safe_threshold"],
        "flagged_threshold": config["policy"]["flagged_threshold"],
        "minimum_face_confidence": config["policy"]["minimum_face_confidence"],
        "low_face_area_threshold": config["policy"]["low_face_area_threshold"],
        "max_conflict_score": config["policy"]["max_conflict_score"],
    }
    (exported_dir / "policy.json").write_text(json.dumps(policy_payload, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    exported_dir = ensure_dir(config["paths"]["exported_dir"])
    reports_dir = ensure_dir(config["paths"]["reports_dir"])
    metrics_dir = ensure_dir(reports_dir / "metrics")
    charts_dir = ensure_dir(reports_dir / "charts")
    confusion_dir = ensure_dir(reports_dir / "confusion_matrices")
    reliability_dir = ensure_dir(reports_dir / "reliability")
    failure_dir = ensure_dir(reports_dir / "failure_analysis")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ManifestFaceDataset(
        manifest_path=config["paths"]["manifest"],
        split=args.split,
        image_size=int(config["data"]["image_size"]),
        train=False,
        include_weak_labels=True,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples were found for split '{args.split}' in {config['paths']['manifest']}. "
            "Rebuild the manifest or add more data before evaluation."
        )
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
    )

    main_checkpoint = load_checkpoint(checkpoints_dir / "main_best.pt")
    if main_checkpoint is None:
        raise RuntimeError("Missing main model checkpoint at ml/training/outputs/checkpoints/main_best.pt")

    main_model = MiVOLOAgeEstimator(
        backbone_name=config["model"]["main"]["backbone_name"],
        pretrained=False,
        num_buckets=len(AGE_BUCKETS),
        dropout=float(config["model"]["main"]["dropout"]),
    ).to(device)
    main_model.load_state_dict(main_checkpoint["model_state_dict"])
    main_model.eval()

    aux_checkpoint = load_checkpoint(checkpoints_dir / "aux_best.pt")
    aux_model = None
    if aux_checkpoint is not None:
        aux_model = DINOv2AuxiliaryModel(
            backbone_name=config["model"]["aux"]["backbone_name"],
            backbone_source=config["model"]["aux"].get("backbone_source", "auto"),
            repo_dir=config["model"]["aux"].get("repo_dir"),
            pretrained=False,
            freeze_encoder=bool(config["model"]["aux"]["freeze_encoder"]),
            num_domains=len(config["data"]["domain_classes"]),
            dropout=float(config["model"]["aux"]["dropout"]),
        ).to(device)
        aux_model.load_state_dict(aux_checkpoint["model_state_dict"])
        aux_model.eval()

    temperature = maybe_load_calibration(exported_dir)
    policy = PolicyEngine(PolicyConfig(**config["policy"]))
    export_policy(config, exported_dir)

    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            main_outputs = main_model(batch["image"])
            aux_outputs = aux_model(batch["image"]) if aux_model is not None else None

            ages = main_outputs["age"].detach().cpu()
            minor_logits = (main_outputs["minor_logit"] / temperature).detach().cpu()
            quality_scores = batch["quality_score"].detach().cpu()
            face_size_ratios = batch["face_size_ratio"].detach().cpu()

            aux_minor_logits = torch.zeros_like(minor_logits)
            aux_uncertainty = torch.full_like(minor_logits, 0.5)
            source_domains = list(batch["domain_type"])
            predicted_domains = list(source_domains)
            if aux_outputs is not None:
                aux_minor_logits = aux_outputs["minor_logit"].detach().cpu()
                aux_uncertainty = torch.sigmoid(aux_outputs["uncertainty_logit"].detach().cpu())
                domain_indices = aux_outputs["domain_logits"].detach().cpu().argmax(dim=1).tolist()
                predicted_domains = [config["data"]["domain_classes"][index] for index in domain_indices]

            for index in range(len(ages)):
                estimated_age = float(ages[index].item())
                p_minor = float(torch.sigmoid(minor_logits[index]).item())
                aux_p_minor = float(torch.sigmoid(aux_minor_logits[index]).item())
                uncertainty_score = float(aux_uncertainty[index].item())
                conflict_score = abs(p_minor - aux_p_minor)
                spread = 2.5 + 5.0 * uncertainty_score + 2.0 * conflict_score
                age_interval = [max(0.0, estimated_age - spread), estimated_age + spread]
                raw_flags = str(batch["quality_flags"][index]).strip()
                quality_flags = [flag for flag in raw_flags.split(",") if flag]
                if quality_scores[index].item() < 0.5:
                    quality_flags = sorted(set(quality_flags + ["quality_warning"]))
                face_confidence = float(max(0.5, min(0.99, quality_scores[index].item())))

                verdict, reason = policy.decide_face(
                    FacePolicyInput(
                        estimated_age=estimated_age,
                        age_interval=(age_interval[0], age_interval[1]),
                        p_minor=p_minor,
                        face_confidence=face_confidence,
                        quality_flags=quality_flags,
                        domain_type=predicted_domains[index],
                        conflict_score=conflict_score,
                        face_size_ratio=float(face_size_ratios[index].item()),
                    )
                )

                rows.append(
                    {
                        "image_path": batch["image_path"][index],
                        "source_dataset": batch["source_dataset"][index],
                        "source_domain_type": source_domains[index],
                        "predicted_domain_type": predicted_domains[index],
                        "domain_type": predicted_domains[index],
                        "age_bucket": batch["age_bucket"][index],
                        "quality_flags": ",".join(quality_flags),
                        "estimated_age": estimated_age,
                        "age_interval_low": age_interval[0],
                        "age_interval_high": age_interval[1],
                        "minor_logit_raw": float(main_outputs["minor_logit"].detach().cpu()[index].item()),
                        "p_minor": p_minor,
                        "aux_p_minor": aux_p_minor,
                        "uncertainty_score": uncertainty_score,
                        "conflict_score": conflict_score,
                        "minor_label": float(batch["minor_label"].detach().cpu()[index].item())
                        if float(batch["has_minor_label"].detach().cpu()[index].item()) > 0
                        else math.nan,
                        "verdict": verdict,
                        "policy_reason": reason,
                    }
                )

    predictions = pd.DataFrame(rows)
    predictions_path = metrics_dir / f"{args.split}_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    labeled = predictions[predictions["minor_label"].notna()].copy()
    metrics_payload: dict[str, object] = {"split": args.split, "row_count": int(len(predictions))}
    metrics_payload["verdict_counts"] = {str(key): int(value) for key, value in predictions["verdict"].value_counts().to_dict().items()}
    metrics_payload["source_dataset_counts"] = {
        str(key): int(value) for key, value in predictions["source_dataset"].value_counts().to_dict().items()
    }
    metrics_payload["source_domain_counts"] = {
        str(key): int(value) for key, value in predictions["source_domain_type"].value_counts().to_dict().items()
    }
    metrics_payload["predicted_domain_counts"] = {
        str(key): int(value) for key, value in predictions["predicted_domain_type"].value_counts().to_dict().items()
    }
    metrics_payload["policy_reason_counts"] = {
        str(key): int(value) for key, value in predictions["policy_reason"].value_counts().to_dict().items()
    }
    metrics_payload["mean_p_minor"] = float(predictions["p_minor"].mean())
    metrics_payload["mean_uncertainty_score"] = float(predictions["uncertainty_score"].mean())

    if not labeled.empty and labeled["minor_label"].nunique() > 1:
        y_true = labeled["minor_label"].astype(int)
        y_score = labeled["p_minor"]
        y_pred = (y_score >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        metrics_payload.update(
            {
                "minor_precision": float(precision),
                "minor_recall": float(recall),
                "minor_f1": float(f1),
                "roc_auc": float(roc_auc_score(y_true, y_score)),
                "pr_auc": float(average_precision_score(y_true, y_score)),
                "minor_false_negative_rate": float(((y_true == 1) & (y_pred == 0)).sum() / max((y_true == 1).sum(), 1)),
            }
        )

        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        (confusion_dir / f"{args.split}_confusion_matrix.json").write_text(json.dumps({"labels": [0, 1], "matrix": matrix}, indent=2))

        plt.figure(figsize=(5, 4))
        plt.hist([labeled[y_true == 0]["p_minor"], labeled[y_true == 1]["p_minor"]], bins=20, label=["adult", "minor"], alpha=0.7)
        plt.legend()
        plt.xlabel("Predicted p_minor")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(charts_dir / f"{args.split}_score_histogram.png")
        plt.close()

        reliability = labeled.copy()
        reliability["bucket"] = pd.cut(reliability["p_minor"], bins=10, include_lowest=True)
        reliability_frame = reliability.groupby("bucket").agg(confidence=("p_minor", "mean"), observed=("minor_label", "mean")).reset_index()
        reliability_frame.to_csv(reliability_dir / f"{args.split}_reliability.csv", index=False)

        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.plot(reliability_frame["confidence"], reliability_frame["observed"], marker="o")
        plt.xlabel("Predicted confidence")
        plt.ylabel("Observed frequency")
        plt.tight_layout()
        plt.savefig(reliability_dir / f"{args.split}_reliability.png")
        plt.close()

        failures = labeled[((labeled["minor_label"] == 1) & (y_pred == 0)) | ((labeled["minor_label"] == 0) & (y_pred == 1))]
        failures.to_csv(failure_dir / f"{args.split}_failures.csv", index=False)

    slice_source = labeled if not labeled.empty else predictions
    slice_metrics = slice_source.groupby(["source_domain_type", "predicted_domain_type", "age_bucket"]).agg(
        count=("p_minor", "size"),
        mean_p_minor=("p_minor", "mean"),
    )
    slice_metrics.reset_index().to_csv(metrics_dir / f"{args.split}_slice_metrics.csv", index=False)

    (metrics_dir / f"{args.split}_metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
