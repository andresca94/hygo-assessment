# RunPod Checklist

Recommended template:

- `ghcr.io/ai-dock/jupyter-pytorch:2.2.1-py3.10-cuda-11.8.0-runtime-22.04`
- Keep `INSTALL_API_DEPS=0` unless you explicitly install Node and npm on the pod.

## Pod shape

- GPU: `RTX 4090 24GB`
- CPU: `8-16 vCPU`
- RAM: `64GB`
- Disk: `500GB+`

## Before starting training

1. Clone the repo anywhere inside your mounted RunPod workspace.
2. Copy `.env.example` to `.env`.
3. Set a `JUPYTER_PASSWORD` value in the pod template before startup.
4. Ensure the pod `WORKSPACE` environment variable matches the actual mountpoint you are using for storage.
5. Run `bash scripts/runpod_install_deps.sh`.
6. Run `bash scripts/runpod_prepare_assets.sh`.
7. Review `ml/training/outputs/manifests/source_acquisition.md`.
8. Place or download the required datasets into `data/raw` under the repo root.
9. Confirm the raw folders are not empty with `find data/raw -maxdepth 3 -type f | head -100`.
10. Run `python ml/training/scripts/validate_dataset_sources.py --config ml/training/configs/datasets.yaml --raw-root data/raw`.
11. Confirm free disk space with `df -h`.
12. Confirm the GPU is visible with `nvidia-smi`.
13. If you want the external MiVOLO V2 inference path, set `ENABLE_EXTERNAL_MIVOLO_HF=1` in `.env`.

## Required dataset folders

- `data/raw/utkface`
- `data/raw/fairface`
- `data/raw/appa_real`
- `data/raw/nonreal/sfhq`
- `data/raw/nonreal/sfhq_t2i`
- `data/raw/nonreal/generated_photos`
- `data/raw/nonreal/deepfakeface`
- `data/raw/nonreal/digiface1m`
- `data/raw/nonreal/anime_face_dataset`
- `data/raw/nonreal/icartoonface`
- `data/raw/nonreal/trueface` if access is approved for the run

## Asset folders

- `assets/insightface`
- `assets/mivolo`
- `third_party/dinov2`
- `third_party/MiVOLO`

## Recommended first run

```bash
cp .env.example .env
bash scripts/runpod_install_deps.sh
bash scripts/runpod_prepare_assets.sh
bash scripts/runpod_train_full.sh
```

If the validator reports zero primary supervision data, do not continue. Populate at least one of:

- `data/raw/utkface`
- `data/raw/fairface`
- `data/raw/appa_real`

## After training

1. Run `bash scripts/runpod_smoke_test.sh` if both services are up.
2. Run `bash scripts/runpod_collect_debug_bundle.sh --mode final`.
3. Download:
   - `debug_bundles/*.tar.gz`
   - `exports/*.tar.gz`

## What to send back

- the latest debug bundle archive
- the latest exported artifacts archive
- any failure logs if the run stopped before evaluation
