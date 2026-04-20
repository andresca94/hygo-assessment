from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from ml.training.utils import ensure_dir, load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare external repos and public model assets for RunPod.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--emit-instructions", action="store_true")
    parser.add_argument("--prepare-dirs", action="store_true")
    parser.add_argument("--clone-repos", action="store_true")
    parser.add_argument("--download-public", action="store_true")
    parser.add_argument("--prewarm-dinov2", action="store_true")
    return parser


def run_command(command: list[str], cwd: str | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd)


def clone_repo(clone_url: str, local_dir: Path) -> None:
    if local_dir.exists() and any(local_dir.iterdir()):
        return
    ensure_dir(local_dir.parent)
    run_command(["git", "clone", clone_url, str(local_dir)])


def install_editable(local_dir: Path) -> None:
    run_command([sys.executable, "-m", "pip", "install", "-e", str(local_dir)])


def download_hf_file(repo_id: str, filename: str, local_path: Path) -> None:
    from huggingface_hub import hf_hub_download

    ensure_dir(local_path.parent)
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_path.parent), local_dir_use_symlinks=False)
    source_path = Path(downloaded)
    if source_path.resolve() != local_path.resolve():
        shutil.copy2(source_path, local_path)


def download_hf_snapshot(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    ensure_dir(local_dir)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)


def download_direct_zip(url: str, local_dir: Path) -> None:
    import requests

    ensure_dir(local_dir)
    archive_path = local_dir / Path(url).name
    if not archive_path.exists():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        archive_path.write_bytes(response.content)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(local_dir)


def prewarm_dinov2(local_repo_dir: Path | None, model_name: str) -> None:
    import torch

    if local_repo_dir and local_repo_dir.exists():
        torch.hub.load(str(local_repo_dir), model_name, source="local")
    else:
        torch.hub.load("facebookresearch/dinov2", model_name)


def resolve_workspace_path(path_value: str, root_dir: Path) -> Path:
    if path_value.startswith("/workspace"):
        relative = path_value.removeprefix("/workspace").lstrip("/")
        return (root_dir / relative).resolve()
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root_dir / path).resolve()


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(args.config)
    root_dir = Path(args.root_dir or ROOT_DIR).resolve()
    manifest_dir = ensure_dir("ml/training/outputs/manifests")

    instruction_lines = ["# Model Asset Bootstrap", ""]
    manifest: dict[str, dict] = {"repos": {}, "assets": {}}

    for name, repo_config in config.get("repos", {}).items():
        local_dir = resolve_workspace_path(repo_config["local_dir"], root_dir)
        manifest["repos"][name] = {**repo_config, "exists": local_dir.exists()}
        instruction_lines.extend(
            [
                f"## repo:{name}",
                f"- clone_url: {repo_config['clone_url']}",
                f"- local_dir: `{local_dir}`",
                f"- install_editable: {repo_config['install_editable']}",
                "",
            ]
        )
        if args.prepare_dirs:
            ensure_dir(local_dir.parent)
        if args.clone_repos:
            clone_repo(repo_config["clone_url"], local_dir)
            if repo_config.get("install_editable"):
                install_editable(local_dir)

    for name, asset_config in config.get("assets", {}).items():
        manifest["assets"][name] = dict(asset_config)
        instruction_lines.extend(
            [
                f"## asset:{name}",
                f"- type: {asset_config['asset_type']}",
                f"- notes: {asset_config['notes']}",
            ]
        )
        if "repo_id" in asset_config:
            instruction_lines.append(f"- repo_id: {asset_config['repo_id']}")
        if "filename" in asset_config:
            instruction_lines.append(f"- filename: {asset_config['filename']}")
        if "url" in asset_config:
            instruction_lines.append(f"- url: {asset_config['url']}")
        if "local_dir" in asset_config:
            instruction_lines.append(f"- local_dir: `{asset_config['local_dir']}`")
        if "local_path" in asset_config:
            instruction_lines.append(f"- local_path: `{asset_config['local_path']}`")
        instruction_lines.append("")

        if args.prepare_dirs:
            if "local_dir" in asset_config:
                ensure_dir(resolve_workspace_path(asset_config["local_dir"], root_dir))
            if "local_path" in asset_config:
                ensure_dir(resolve_workspace_path(asset_config["local_path"], root_dir).parent)

        if args.download_public:
            asset_type = asset_config["asset_type"]
            if asset_type == "huggingface_snapshot":
                download_hf_snapshot(asset_config["repo_id"], resolve_workspace_path(asset_config["local_dir"], root_dir))
            elif asset_type == "huggingface_file":
                download_hf_file(
                    asset_config["repo_id"],
                    asset_config["filename"],
                    resolve_workspace_path(asset_config["local_path"], root_dir),
                )
            elif asset_type == "direct_zip":
                download_direct_zip(asset_config["url"], resolve_workspace_path(asset_config["local_dir"], root_dir))

    if args.prewarm_dinov2:
        repo_dir = resolve_workspace_path(config["repos"]["dinov2"]["local_dir"], root_dir)
        prewarm_dinov2(repo_dir if repo_dir.exists() else None, config["prewarm"]["dinov2_model_name"])

    (manifest_dir / "model_assets.json").write_text(json.dumps(manifest, indent=2))
    if args.emit_instructions:
        (manifest_dir / "model_assets.md").write_text("\n".join(instruction_lines))
    print(f"Wrote asset manifest to {manifest_dir / 'model_assets.json'}")


if __name__ == "__main__":
    main()
