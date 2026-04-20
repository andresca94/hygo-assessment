from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.utils import ensure_dir, load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit dataset acquisition instructions.")
    parser.add_argument("--config", required=True, help="Path to the dataset source config YAML.")
    parser.add_argument("--raw-root", default="data/raw", help="Local raw dataset root.")
    parser.add_argument("--emit-instructions", action="store_true", help="Write a Markdown instruction file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    raw_root = ensure_dir(args.raw_root)
    manifests_dir = ensure_dir("ml/training/outputs/manifests")

    instructions: list[str] = ["# Dataset Acquisition Instructions", ""]
    serialized: dict[str, dict] = {}

    for source_name, source_data in config.get("sources", {}).items():
        target_dir = raw_root / source_data["expected_raw_dir"]
        ensure_dir(target_dir)
        serialized[source_name] = {
            "url": source_data["url"],
            "acquisition": source_data["acquisition"],
            "expected_raw_dir": str(target_dir),
            "role": source_data.get("role", "unspecified"),
            "required_for_trial": bool(source_data.get("required_for_trial", False)),
            "license_notes": source_data["license_notes"],
        }
        instructions.extend(
            [
                f"## {source_name}",
                f"- URL: {source_data['url']}",
                f"- Acquisition: {source_data['acquisition']}",
                f"- Role: {source_data.get('role', 'unspecified')}",
                f"- Required for trial: {bool(source_data.get('required_for_trial', False))}",
                f"- Raw target: `{target_dir}`",
                f"- Notes: {source_data['license_notes']}",
                "",
            ]
        )

    (manifests_dir / "source_acquisition.json").write_text(json.dumps(serialized, indent=2))
    if args.emit_instructions:
        (manifests_dir / "source_acquisition.md").write_text("\n".join(instructions))

    print(f"Wrote source manifest to {manifests_dir / 'source_acquisition.json'}")


if __name__ == "__main__":
    main()
