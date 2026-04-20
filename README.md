# Hygo Age Safety Assessment

This repository contains a conservative, auditable age-safety proof of concept built around four components:

1. A Python dataset and training pipeline.
2. A Python inference microservice.
3. A NestJS public API.
4. RunPod-oriented debug and export scripts for RTX 4090 24GB pods.

The system is intentionally policy-driven and safety-first. It only returns `safe` when adult evidence is strong. Ambiguous, low-quality, stylized, synthetic, or conflicting cases are pushed to `uncertain`, and high-risk cases are returned as `flagged`.

## Architecture

The intended production path is:

`face detection -> face alignment/crops -> main age estimator -> auxiliary domain-robust encoder -> calibration -> policy engine -> safe/flagged/uncertain`

Current repository decisions:

- `InsightFace` is the preferred face detection and alignment stack.
- `MiVOLO` is the preferred primary age estimator.
- `DINOv2` is the preferred auxiliary encoder for robustness under domain shift.
- The shipped code includes resilient fallbacks so the scaffold can boot before final model weights are available.
- The inference service supports an optional external MiVOLO V2 loading path through Hugging Face weights.
- The auxiliary encoder supports official DINOv2 loading through PyTorch Hub, using a local repo clone when present.

## Repository Layout

```text
.
├── api/
│   └── nestjs-service/
├── ml/
│   ├── inference/
│   └── training/
├── reports/
├── scripts/
├── DATA_CARD.md
├── DECISIONS.md
├── docker-compose.yml
└── README.md
```

## Quick Start

### 1. Local development with Docker Compose

```bash
cp .env.example .env
python ml/training/scripts/bootstrap_model_assets.py --config ml/training/configs/model_assets.yaml --emit-instructions --prepare-dirs
docker compose up --build
```

Services:

- NestJS API: `http://localhost:3000`
- Python inference service: `http://localhost:8000`

### 2. RunPod RTX 4090 workflow

Recommended pod shape:

- GPU: `RTX 4090 24GB`
- CPU: `8-16 vCPU`
- RAM: `64GB`
- Storage: `500GB+ SSD`

Recommended template:

- `ghcr.io/ai-dock/jupyter-pytorch:2.2.1-py3.10-cuda-11.8.0-runtime-22.04`
- Set `JUPYTER_PASSWORD` before startup.
- Ensure the pod `WORKSPACE` env var matches the actual mounted workspace path.
- Leave `INSTALL_API_DEPS=0` on this image unless you install Node and npm yourself.

Inside the pod:

```bash
cp .env.example .env
bash scripts/runpod_install_deps.sh
bash scripts/runpod_fetch_minimum_datasets.sh
bash scripts/runpod_prepare_assets.sh
bash scripts/runpod_train_full.sh
```

Before `runpod_train_full.sh`, the training pod must already contain real files under `data/raw/...`. The script now fails fast if the raw dataset folders are empty or if the merged manifest cannot produce supervised `train`, `val`, and `test` rows.

That script is intentionally conservative:

- It prepares external repos, cache directories, and model asset instructions.
- It snapshots the environment.
- It prepares manifests and splits.
- It trains the main and auxiliary models.
- It calibrates and evaluates.
- It packs a debug bundle you can download from the pod.

### 3. Download artifacts from the pod

From your local machine:

```bash
bash scripts/sync_artifacts_from_pod.sh user@your-pod-host
```

Or from inside the pod, create a compact handoff bundle:

```bash
bash scripts/runpod_collect_debug_bundle.sh
```

The generated archive contains:

- checkpoints
- calibration files
- metrics and charts
- failure-analysis CSVs
- service configs
- Git status
- `nvidia-smi` output
- Python and Node environment snapshots

## RunPod Readiness

You are ready to clone this repo on RunPod and run it there if these conditions are true:

- the raw datasets are available under `data/raw` with the expected folder names,
- Python dependencies can be installed from the pod,
- public external assets can be downloaded or manually placed,
- you accept that `TrueFace` remains a manual-review source unless you have a clean acquisition path.

Recommended RunPod sequence:

```bash
cp .env.example .env
bash scripts/runpod_install_deps.sh
bash scripts/runpod_prepare_assets.sh
AUTO_DOWNLOAD_PUBLIC_ASSETS=0 bash scripts/runpod_train_full.sh
```

Quick sanity check before training:

```bash
find data/raw -maxdepth 3 -type f | head -100
python ml/training/scripts/validate_dataset_sources.py --config ml/training/configs/datasets.yaml --raw-root data/raw
```

If you want the fastest first successful run, use the built-in FairFace fetch helper:

```bash
bash scripts/runpod_fetch_minimum_datasets.sh
```

That script downloads the official FairFace train/validation images and label CSVs from the official project links, stages them under `data/raw/fairface`, and reruns dataset validation.

If you want the external MiVOLO V2 inference path after training, set:

```bash
ENABLE_EXTERNAL_MIVOLO_HF=1
```

## Dataset Strategy

The project follows a data-first and failure-slice-driven process.

Primary supervision:

- UTKFace
- FairFace
- APPA-REAL

Robustness evaluation only:

- SFHQ
- SFHQ-T2I
- Generated Photos Synthetic Face Images Academic Dataset
- Anime Face Dataset
- iCartoonFace
- DeepFakeFace
- DigiFace-1M
- TrueFace if access and license review are resolved for the run

The non-real sets are used to stress test domain shift and abstention behavior, not as the main source of exact age labels.

## Main Commands

### Training pipeline

```bash
python ml/training/scripts/download_sources.py --config ml/training/configs/datasets.yaml --raw-root data/raw --emit-instructions
python ml/training/scripts/bootstrap_model_assets.py --config ml/training/configs/model_assets.yaml --emit-instructions --prepare-dirs
python ml/training/scripts/prepare_utkface.py --raw-dir data/raw/utkface --output-dir data/processed/utkface
python ml/training/scripts/prepare_fairface.py --raw-dir data/raw/fairface --output-dir data/processed/fairface
python ml/training/scripts/prepare_appa_real.py --raw-dir data/raw/appa_real --output-dir data/processed/appa_real
python ml/training/scripts/prepare_nonreal_eval.py --raw-dir data/raw/nonreal --output-dir data/processed/nonreal
python ml/training/scripts/split_dataset.py --processed-root data/processed --output-manifest ml/training/outputs/manifests/master_manifest.csv
python ml/training/scripts/deduplicate.py --input-manifest ml/training/outputs/manifests/master_manifest.csv --output-manifest ml/training/outputs/manifests/master_manifest.csv
python ml/training/scripts/train_main.py --config ml/training/configs/runpod_4090.yaml
python ml/training/scripts/train_aux.py --config ml/training/configs/runpod_4090.yaml
python ml/training/scripts/evaluate.py --config ml/training/configs/runpod_4090.yaml --split val
python ml/training/scripts/calibrate.py --config ml/training/configs/runpod_4090.yaml
python ml/training/scripts/evaluate.py --config ml/training/configs/runpod_4090.yaml --split test
```

### Inference service

```bash
uvicorn ml.inference.app:app --host 0.0.0.0 --port 8000
```

### NestJS API

```bash
cd api/nestjs-service
npm install
npm run start:dev
```

## API Contract

Public API endpoints:

- `POST /v1/age-safety/check`
- `POST /v1/age-safety/check-batch`
- `GET /v1/age-safety/health`

Representative response:

```json
{
  "verdict": "safe",
  "risk_score": 0.03,
  "faces": [
    {
      "bbox": [101, 88, 245, 260],
      "face_confidence": 0.99,
      "estimated_age": 28.4,
      "age_interval": [24.7, 33.1],
      "p_minor": 0.01,
      "domain_type": "real",
      "quality_flags": []
    }
  ],
  "policy_reason": "confident_adult_face",
  "model_version": "age-safety-v1.0.0"
}
```

## Current Scope

This repository is a production-oriented scaffold, not a claim that the final model is already trained. The code is structured so you can:

- prepare and document datasets,
- train incrementally,
- calibrate thresholds,
- audit slices,
- export artifacts from a pod,
- and wire the resulting model into a public NestJS API.

## Model Assets

External asset bootstrap is documented and scripted for:

- MiVOLO V2 Hugging Face weights
- official MiVOLO repository checkout
- official DINOv2 repository checkout
- InsightFace `buffalo_l` detection pack

See:

- [`ml/training/configs/model_assets.yaml`](ml/training/configs/model_assets.yaml)
- [`RUNPOD_CHECKLIST.md`](RUNPOD_CHECKLIST.md)

## Reviewer Handoff

After training on RunPod you have two clean submission options.

### Option A: submit with weights included

```bash
bash scripts/runpod_export_artifacts.sh
INCLUDE_WEIGHTS=1 bash scripts/prepare_submission_bundle.sh
```

This creates a submission archive that already contains `ml/training/outputs/exported`.

### Option B: submit without weights, but with a one-step install path

```bash
bash scripts/runpod_export_artifacts.sh
INCLUDE_WEIGHTS=0 bash scripts/prepare_submission_bundle.sh
```

Then the reviewer can run:

```bash
bash scripts/install_inference_bundle.sh <bundle.tar.gz-or-url>
```

The exported inference bundle produced by training is:

```text
exports/<model_version>_inference_bundle.tar.gz
```

## Documentation

- Dataset and provenance rationale: [`DATA_CARD.md`](DATA_CARD.md)
- System and policy rationale: [`DECISIONS.md`](DECISIONS.md)
- Pod bootstrap checklist: [`RUNPOD_CHECKLIST.md`](RUNPOD_CHECKLIST.md)
