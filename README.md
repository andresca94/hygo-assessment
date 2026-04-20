# Hygo Age Safety Assessment

This repository contains a conservative, auditable age-safety proof of concept built around four components:

1. A Python dataset and training pipeline.
2. A Python inference microservice.
3. A NestJS public API.
4. RunPod-oriented debug and export scripts for RTX 4090 24GB pods.

The system is intentionally policy-driven and safety-first. It only returns `safe` when adult evidence is strong. Ambiguous, low-quality, stylized, synthetic, or conflicting cases are pushed to `uncertain`, and high-risk cases are returned as `flagged`.

As of April 20, 2026, the recommended shipped checkpoint in this repo is the `FairFace`-only baseline, not the later `UTKFace` ablation. The `UTKFace` experiment improved precision but reduced minor recall in the `13-17` slice, so it is kept as an ablation rather than the final candidate.

## Architecture

The intended production path is:

`face detection -> face alignment/crops -> main age estimator -> auxiliary domain-robust encoder -> calibration -> policy engine -> safe/flagged/uncertain`

Repository-supported model families:

- [`InsightFace`](https://github.com/deepinsight/insightface) provides face detection and alignment.
- [`MiVOLO`](https://github.com/WildChlamydia/MiVOLO) informed the main-model design, and an optional external MiVOLO V2 Hugging Face path is available at inference time.
- [`DINOv2`](https://github.com/facebookresearch/dinov2) provides the auxiliary encoder used for domain and uncertainty estimation.
- The shipped code includes resilient fallbacks so the scaffold can boot before optional external assets are present.

### Shipped model architecture

The shipped baseline is a policy-driven ensemble rather than a single thresholded classifier:

1. [`InsightFace`](https://github.com/deepinsight/insightface) detects faces and provides aligned crops plus detection confidence.
2. A custom `MiVOLOAgeEstimator` main model predicts:
   - scalar age estimate
   - calibrated `p_minor` score
   - age-bucket logits
3. An auxiliary [`DINOv2`](https://github.com/facebookresearch/dinov2) encoder predicts:
   - auxiliary `p_minor`
   - domain type (`real`, `ai_generated`, `cartoon`, `anime`, `three_d`, `edited`, `unknown`)
   - uncertainty score
4. The inference layer converts those signals into an age interval and conflict score.
5. A policy engine emits `safe`, `flagged`, or `uncertain` using:
   - calibrated `p_minor`
   - lower bound of the age interval
   - quality flags
   - face confidence and face area
   - disagreement between the main and auxiliary models
   - non-photographic domain detection

In the shipped checkpoint, `MiVOLOAgeEstimator` uses a `timm` `tf_efficientnet_b0` backbone with lightweight age, age-bucket, and minor-risk heads. The optional external MiVOLO V2 path is implemented, but it is not the default path used by the shipped submission metrics.

This matters because the challenge is safety-critical. The production decision is not “what is the argmax class,” it is “do we have enough evidence to safely approve this image?”

### Shipped training strategy

The shipped checkpoint is trained in two stages:

1. Main model training on `FairFace`
   - resize to `224x224`, light augmentation, ImageNet normalization
   - `tf_efficientnet_b0` backbone with three heads: age regression, age-bucket classification, and minor-risk
   - weighted multi-task loss:
     - `0.4 * SmoothL1(age)`
     - `0.3 * cross_entropy(age bucket)`
     - `0.3 * BCEWithLogits(minor label)`
   - hard-case upweighting for the `15-21` boundary region and rows with quality issues
   - best checkpoint selected by validation `minor_recall`, with validation loss as the tie-breaker
2. Auxiliary model training
   - frozen `DINOv2` encoder with a small MLP head
   - predicts auxiliary minor-risk, domain class, and uncertainty
   - trained with a weighted multi-task objective over minor label, domain label, and quality-derived uncertainty

After validation inference, the pipeline fits a temperature scaler on raw validation logits and exports `calibration.json`. The policy layer then uses the calibrated `p_minor` score, the age interval, conflict score, face quality, and domain cues to emit `safe`, `flagged`, or `uncertain`.

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

The repository now ships the recommended `FairFace` baseline checkpoint under `ml/training/outputs/exported`, so a clean clone can boot the API without retraining first. This is the default reviewer path.

```bash
cp .env.example .env
python ml/training/scripts/bootstrap_model_assets.py --config ml/training/configs/model_assets.yaml --emit-instructions --prepare-dirs
docker compose up --build
```

On first boot, the inference service may download public runtime assets such as the [`InsightFace buffalo_l` pack](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) and public [`DINOv2`](https://github.com/facebookresearch/dinov2) weights unless those caches are already warm.

If Docker fails during the first build with `no space left on device`, that is Docker disk exhaustion rather than an application startup bug. Free Docker build cache with `docker builder prune -af`, optionally free more unused images with `docker system prune -af`, or increase the Docker Desktop disk image size and retry.

Services:

- NestJS API: `http://localhost:3000`
- Python inference service: `http://localhost:8000`

Quick health check:

```bash
bash scripts/runpod_smoke_test.sh
```

Example single-image request:

```bash
curl -X POST http://127.0.0.1:3000/v1/age-safety/check \
  -F "file=@/absolute/path/to/sample.jpg"
```

### Reviewer sample workflow

The repo includes a small tracked unseen-image sample set under `reviewer_samples/`. With the stack running, execute:

```bash
python scripts/run_reviewer_samples.py
```

On Windows, if `python` is not on `PATH`, use:

```powershell
py -3 scripts/run_reviewer_samples.py
```

The script will:

- call the public health endpoint
- upload each available sample image to `POST /v1/age-safety/check`
- write one JSON response per image under `reviewer_samples/results/`
- run one batch request over all available samples

This is intended to make reviewer testing easy without having to assemble a sample set first.

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
bash scripts/runpod_train_recommended_submission.sh
bash scripts/runpod_fetch_robustness_datasets.sh
bash scripts/runpod_run_robustness_eval.sh
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

If you want to reproduce the rejected `UTKFace` ablation for comparison, stage `UTKFace` as well:

```bash
bash scripts/runpod_fetch_utkface_dataset.sh
```

That helper downloads the official aligned-and-cropped `UTKFace` archive, extracts it under `data/raw/utkface`, and reruns dataset validation. The current recommendation is not to use that checkpoint as the final shipped candidate unless you independently revalidate the safety tradeoff.

## Current Recommendation

Recommended final submission path:

```bash
cp .env.example .env
bash scripts/runpod_install_deps.sh
bash scripts/runpod_fetch_minimum_datasets.sh
bash scripts/runpod_train_recommended_submission.sh
bash scripts/runpod_fetch_robustness_datasets.sh
bash scripts/runpod_run_robustness_eval.sh
```

Current recommended final candidate metrics from the `FairFace` baseline:

- `minor_precision`: `0.8711`
- `minor_recall`: `0.9737`
- `minor_f1`: `0.9195`
- `minor_false_negative_rate`: `0.0263`
- `roc_auc`: `0.9959`
- `pr_auc`: `0.9855`

Validation metrics for the same shipped checkpoint:

- `minor_precision`: `0.8827`
- `minor_recall`: `0.9700`
- `minor_f1`: `0.9243`
- `minor_false_negative_rate`: `0.0300`
- `roc_auc`: `0.9953`
- `pr_auc`: `0.9845`

The shipped baseline weights used for local inference are included at:

- `ml/training/outputs/exported/main_best.pt`
- `ml/training/outputs/exported/aux_best.pt`
- `ml/training/outputs/exported/calibration.json`
- `ml/training/outputs/exported/policy.json`

Current robustness summary for the shipped baseline:

- `130,054` robustness rows
- `125,647` `uncertain`
- `4,407` `flagged`
- `0` `safe`

This is an intentional abstention posture, not a claim of strong cross-domain age accuracy. Under AI-generated and cartoon shift, the policy prefers `uncertain` over unsafe confidence.

Current subgroup findings from `reports/metrics/test_subgroup_metrics.csv`:

- gender:
  - `female` minor recall `0.9723`, minor false-negative rate `0.0277`
  - `male` minor recall `0.9748`, minor false-negative rate `0.0252`
- race:
  - highest false-negative rates in this shipped test split were `latino_hispanic` (`0.0450`) and `indian` (`0.0419`)
  - lowest false-negative rates were `southeast asian` (`0.0049`) and `east asian` (`0.0122`)

Important caveat:

- the shipped `FairFace` test split only contains labeled minors in the `0-12` bucket
- this means the demographic report is useful, but it does not by itself validate borderline `13-17` behavior
- the rejected `UTKFace` ablation was kept specifically because it exposed worse recall in the `13-17` slice and informed the final model-selection decision

`UTKFace` ablation summary:

- improved `minor_precision` to `0.9413`
- reduced `minor_recall` to `0.9567`
- increased `minor_false_negative_rate` to `0.0433`
- degraded the critical `13-17` slice, so it is rejected as the final shipped checkpoint

## Final Results

The final recommended submission model is the `FairFace` baseline, selected for safety rather than aggregate precision. It ships with:

- strong in-domain real-photo performance on the held-out test split
- conservative abstention on AI-generated and cartoon robustness inputs
- explicit calibration artifacts and policy thresholds
- demographic subgroup reporting over the shipped `FairFace` test split

The main tradeoff is deliberate:

- the system is optimized to reduce dangerous false negatives on minors
- under domain shift it prefers `uncertain` over overconfident `safe`
- the rejected `UTKFace` ablation showed why this mattered: higher precision, but worse recall in the `13-17` slice

### How to read the metrics

The most important metric in this repository is not raw accuracy. It is whether the system misses minors.

- `minor_recall`
  - Of all true minors, how many did the model flag as minor-risk at a binary `p_minor >= 0.5` decision threshold.
  - Higher is better.
- `minor_false_negative_rate`
  - Of all true minors, how many were missed by the binary classifier.
  - Lower is better.
  - This is the safety-critical metric for the challenge.
- `minor_precision`
  - Of all images flagged as minor-risk, how many were actually minors.
  - Higher is better, but less important than recall in this use case.
- `roc_auc` / `pr_auc`
  - Ranking-quality metrics for the continuous `p_minor` score.
  - They show the model separates minors from adults well, but they do not replace threshold or policy evaluation.
- `verdict_counts`
  - These are policy outputs, not raw classifier outputs.
  - A high `uncertain` count on robustness data is expected and intentional when the model sees domain shift.

For this submission, the key interpretation is:

- the shipped baseline performs strongly on real-photo test data
- the system remains conservative under AI-generated and cartoon shift
- the abstention policy is doing meaningful safety work on top of the classifier scores

To add AI-generated and cartoon robustness coverage after the baseline run:

```bash
bash scripts/runpod_fetch_robustness_datasets.sh
bash scripts/runpod_run_robustness_eval.sh
```

This stages:

- `DeepFakeFace` text-to-image and inpainting subsets from Hugging Face
- `iCartoonFace` detection data from the official Google Drive folder

and then evaluates the `robustness` split with the checkpoints already trained on the real-photo baseline.

If you want the external MiVOLO V2 inference path after training, set:

```bash
ENABLE_EXTERNAL_MIVOLO_HF=1
```

## Dataset Strategy

The project follows a data-first and failure-slice-driven process, but the final shipped submission is intentionally narrower than the full scaffold.

Shipped supervised training and evaluation:

- [`FairFace`](https://github.com/joojs/fairface)
  - this is the only dataset used to train the shipped baseline checkpoint
  - it also powers the shipped validation, test, and subgroup metrics

Evaluated ablation, not shipped:

- [`UTKFace`](https://susanqq.github.io/UTKFace/)
  - integrated into the pipeline and tested as a `UTKFace + FairFace` ablation
  - rejected for final deployment because it worsened minor recall in the critical `13-17` slice

Scaffolded but not used in the shipped metrics:

- [`APPA-REAL`](https://chalearnlap.cvc.uab.cat/dataset/26/description/)
- [`SFHQ`](https://github.com/SelfishGene/SFHQ-dataset)
- [`SFHQ-T2I`](https://github.com/SelfishGene/SFHQ-T2I-dataset)
- [`Generated Photos Synthetic Face Images Academic Dataset`](https://huggingface.co/datasets/GeneratedPhotos/Synthetic_Face_Images_Academic_Dataset)
- [`Anime Face Dataset`](https://github.com/bchao1/Anime-Face-Dataset)
- [`DigiFace-1M`](https://microsoft.github.io/DigiFace1M/)
- `TrueFace`, pending a cleaner provenance and licensing path

Shipped robustness evaluation:

- [`DeepFakeFace`](https://github.com/OpenRL-Lab/DeepFakeFace)
- [`iCartoonFace`](https://github.com/luxiangju-PersonAI/iCartoonFace)
- held-out real-photo rows from `FairFace`

The non-real sets are used to stress test domain shift and abstention behavior, not as the main source of exact age labels.

### Dataset references

Shipped supervised training and demographic evaluation:

- [`FairFace`](https://github.com/joojs/fairface)
  - balanced race, gender, and age labels
  - used as the shipped supervised baseline because it provides the cleanest subgroup reporting story

Evaluated ablation only:

- [`UTKFace`](https://susanqq.github.io/UTKFace/)
  - broad age coverage with simple filename-derived metadata
  - useful for ablations and hard-slice coverage, but the final checkpoint rejected it because of worse `13-17` recall

Scaffolded but not used in the shipped final metrics:

- [`APPA-REAL`](https://chalearnlap.cvc.uab.cat/dataset/26/description/)
  - real and apparent age estimation benchmark
  - scaffolded for future ambiguity-aware training, but not part of the shipped checkpoint

Shipped robustness and abstention evaluation:

- [`DeepFakeFace`](https://github.com/OpenRL-Lab/DeepFakeFace)
  - diffusion/deepfake-oriented synthetic face data
  - used to test abstention on AI-generated faces
- [`iCartoonFace`](https://github.com/luxiangju-PersonAI/iCartoonFace)
  - large cartoon-face dataset
  - used to test abstention on stylized/cartoon inputs
- [`FairFace`](https://github.com/joojs/fairface)
  - held-out real-photo rows included as the real reference domain in the shipped robustness report

Scaffolded robustness sources not used in the shipped robustness metrics:

- [`SFHQ`](https://github.com/SelfishGene/SFHQ-dataset)
- [`SFHQ-T2I`](https://github.com/SelfishGene/SFHQ-T2I-dataset)
- [`Generated Photos Synthetic Face Images Academic Dataset`](https://huggingface.co/datasets/GeneratedPhotos/Synthetic_Face_Images_Academic_Dataset)
- [`Anime Face Dataset`](https://github.com/bchao1/Anime-Face-Dataset)
- [`DigiFace-1M`](https://microsoft.github.io/DigiFace1M/)
- `TrueFace`
  - deliberately left as manual-review-only until provenance and licensing are clearer

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
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r ml/inference/requirements.txt
uvicorn ml.inference.app:app --host 0.0.0.0 --port 8000
```

The manual Python path is an alternative to Docker and has been verified on macOS arm64 with Python 3.12. `python -m venv .venv` can take a minute on a clean machine; let it finish before activating the environment.

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r ml/inference/requirements.txt
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

Example request:

```bash
curl -X POST http://127.0.0.1:3000/v1/age-safety/check \
  -F "file=@/absolute/path/to/sample.jpg"
```

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

This repository is a production-oriented scaffold that also ships a trained baseline checkpoint. The code is structured so you can:

- run the included `FairFace` baseline immediately,
- prepare and document datasets,
- train incrementally,
- calibrate thresholds,
- audit slices,
- export artifacts from a pod,
- and wire the resulting model into a public NestJS API.

## Model Assets

External asset bootstrap is documented and scripted for:

- shipped runtime dependencies:
  - [`InsightFace`](https://github.com/deepinsight/insightface) and the [`buffalo_l` pack](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip)
  - [`DINOv2` official repository](https://github.com/facebookresearch/dinov2)
- optional external inference path:
  - [`MiVOLO V2 Hugging Face weights`](https://huggingface.co/iitolstykh/mivolo_v2)
  - [`YOLO-Face-Person-Detector`](https://huggingface.co/iitolstykh/YOLO-Face-Person-Detector)
  - [`MiVOLO` official repository](https://github.com/WildChlamydia/MiVOLO)

See:

- [`ml/training/configs/model_assets.yaml`](ml/training/configs/model_assets.yaml)
- [`RUNPOD_CHECKLIST.md`](RUNPOD_CHECKLIST.md)

## Optional Packaging

The normal submission path is to use this repository directly: the shipped baseline weights are already present under `ml/training/outputs/exported`, so reviewers do not need an extra handoff archive to boot inference.

The packaging scripts are only needed if you want to export a separate standalone bundle from a RunPod training run:

```bash
bash scripts/runpod_export_artifacts.sh
INCLUDE_WEIGHTS=1 bash scripts/prepare_submission_bundle.sh
```

If you intentionally export a bundle without weights, the receiving machine can install the exported inference bundle with:

```bash
bash scripts/install_inference_bundle.sh <bundle.tar.gz-or-url>
```

## Documentation

- Dataset and provenance rationale: [`DATA_CARD.md`](DATA_CARD.md)
- System and policy rationale: [`DECISIONS.md`](DECISIONS.md)
- Assessment-question coverage in `DECISIONS.md`:
  - dataset choices and rejected alternatives
  - model architecture and training strategy
  - threshold decisions
  - evaluation harness, calibration, and failure analysis
  - bias analysis
  - real vs generated images
  - integration design for each platform touchpoint
  - production concerns
  - what I would do with more time
- Pod bootstrap checklist: [`RUNPOD_CHECKLIST.md`](RUNPOD_CHECKLIST.md)
