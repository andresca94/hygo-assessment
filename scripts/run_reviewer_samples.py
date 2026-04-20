#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_DIR = ROOT_DIR / "reviewer_samples"
DEFAULT_MANIFEST = DEFAULT_SAMPLE_DIR / "sample_manifest.json"
DEFAULT_OUTPUT_DIR = DEFAULT_SAMPLE_DIR / "results"
DEFAULT_API_BASE = "http://127.0.0.1:3000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reviewer sample set against the public API.")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE)
    parser.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--skip-batch", action="store_true")
    return parser.parse_args()


def request_json(url: str, *, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_multipart(files: list[tuple[str, Path]]) -> tuple[bytes, str]:
    boundary = f"----CodexBoundary{uuid.uuid4().hex}"
    body = bytearray()
    for field_name, path in files:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{path.name}"\r\n'.encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(path.read_bytes())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), boundary


def post_files(url: str, files: list[tuple[str, Path]], *, timeout: float) -> dict:
    body, boundary = build_multipart(files)
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def print_result(filename: str, category: str, payload: dict) -> None:
    verdict = payload.get("verdict", "unknown")
    reason = payload.get("policy_reason", "unknown")
    risk = payload.get("risk_score", "n/a")
    print(f"{filename} [{category}] -> verdict={verdict} risk_score={risk} policy_reason={reason}")


def main() -> int:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        health = request_json(f"{args.api_base_url}/v1/age-safety/health", timeout=args.timeout)
    except urllib.error.URLError as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)
        return 1

    print("Health:", json.dumps(health, indent=2))
    if not health.get("model_ready"):
        print("Model is not ready. Start the stack first.", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    available_files: list[Path] = []

    for entry in manifest:
        filename = entry["filename"]
        category = entry["category"]
        image_path = sample_dir / filename
        if not image_path.exists():
            print(f"{filename} [{category}] -> missing")
            continue

        try:
            payload = post_files(
                f"{args.api_base_url}/v1/age-safety/check",
                [("file", image_path)],
                timeout=args.timeout,
            )
        except urllib.error.URLError as exc:
            print(f"{filename} [{category}] -> request failed: {exc}", file=sys.stderr)
            continue

        (output_dir / f"{image_path.stem}.json").write_text(json.dumps(payload, indent=2))
        print_result(filename, category, payload)
        available_files.append(image_path)

    if args.skip_batch or not available_files:
        return 0

    try:
        batch_payload = post_files(
            f"{args.api_base_url}/v1/age-safety/check-batch",
            [("files", path) for path in available_files],
            timeout=args.timeout,
        )
    except urllib.error.URLError as exc:
        print(f"Batch request failed: {exc}", file=sys.stderr)
        return 1

    (output_dir / "batch_results.json").write_text(json.dumps(batch_payload, indent=2))
    print(
        "Batch summary:",
        json.dumps(
            {
                "safe_count": batch_payload.get("safe_count"),
                "uncertain_count": batch_payload.get("uncertain_count"),
                "flagged_count": batch_payload.get("flagged_count"),
            },
            indent=2,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
