from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.utils import ensure_dir, load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a temperature scaler on validation logits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", default=None)
    return parser


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, temperature_init: float = 1.0) -> float:
    temperature = torch.nn.Parameter(torch.tensor(float(temperature_init)))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        clipped = torch.clamp(temperature, 0.05, 10.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / clipped, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(temperature.detach(), 0.05, 10.0).item())


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    exported_dir = ensure_dir(config["paths"]["exported_dir"])
    predictions_path = Path(args.predictions or "reports/metrics/val_predictions.csv")

    frame = pd.read_csv(predictions_path)
    frame = frame[frame["minor_label"].notna()].copy()
    if frame.empty:
        raise RuntimeError("No labeled validation predictions were found for calibration.")

    logits = torch.tensor(frame["minor_logit_raw"].to_numpy(), dtype=torch.float32)
    labels = torch.tensor(frame["minor_label"].to_numpy(), dtype=torch.float32)
    temperature = fit_temperature(logits, labels, float(config["calibration"]["temperature_init"]))

    before_probs = torch.sigmoid(logits)
    after_probs = torch.sigmoid(logits / temperature)
    before_brier = float(torch.mean((before_probs - labels) ** 2).item())
    after_brier = float(torch.mean((after_probs - labels) ** 2).item())

    payload = {
        "method": "temperature",
        "temperature": temperature,
        "before_brier": before_brier,
        "after_brier": after_brier,
        "predictions_path": str(predictions_path),
    }

    (exported_dir / "calibration.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
