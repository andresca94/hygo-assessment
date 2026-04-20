# Decisions

## Product policy

This project uses a three-way safety policy instead of a hard binary classifier:

- `safe`
- `flagged`
- `uncertain`

This is deliberate. In a safety-critical setting, the cost of a false negative is higher than the cost of abstaining. The system should only return `safe` when adult evidence is strong, quality is acceptable, and the signals are consistent.

## Dataset choices

Primary supervision is built on:

- `UTKFace` for broad age coverage and a fast proof-of-concept baseline.
- `FairFace` for subgroup analysis and bias measurement.
- `APPA-REAL` for ambiguity-aware age supervision near visually difficult boundaries.

Robustness and abstention testing use:

- `SFHQ`
- `SFHQ-T2I`
- `Generated Photos Synthetic Face Images Academic Dataset`
- `Anime Face Dataset`
- `iCartoonFace`
- `DeepFakeFace`
- `DigiFace-1M`
- `TrueFace`, when access and licensing are validated

These non-real sources are intentionally not treated as the main source of exact age ground truth. Their role is to expose failure under domain shift and to justify conservative abstention.

## Model choices

Preferred architecture:

- `InsightFace` for detection and alignment.
- `MiVOLO` as the main age estimator.
- `DINOv2` as the auxiliary domain-robust encoder.

The codebase includes fallback implementations because model availability on first boot is often incomplete. That keeps the scaffold runnable on a pod before the final checkpoints are ready.

The production-oriented asset path is:

- optional MiVOLO V2 weights from Hugging Face for inference-time age estimation,
- official DINOv2 loading via PyTorch Hub with an optional local repo clone,
- InsightFace `buffalo_l` loaded from a local root to avoid unexpected runtime downloads on pods.

## Threshold policy

Suggested defaults:

- `safe` if `p_minor < 0.05`, quality is acceptable, there is no strong model conflict, and the adult interval is clearly above 18.
- `flagged` if `p_minor >= 0.40` or the age estimate is close to or below the legal boundary.
- `uncertain` otherwise.

Thresholds should be tuned on validation slices, not on global accuracy.

## Evaluation philosophy

The repository optimizes for:

- minor recall
- low minor false-negative rate
- calibrated risk scores
- slice-level reporting

Global accuracy is not the primary objective. The important analysis happens on:

- ages `16-21`
- low-quality images
- occlusion and blur
- profile views
- multi-face images
- real vs synthetic vs stylized content

## Production concerns

The design explicitly handles:

- no-face cases
- multiple faces
- domain drift
- low-confidence predictions
- monitoring and retraining loops

Future production improvements:

- stronger licensed hard-case collections
- domain-specific calibration
- provenance signals such as C2PA-aligned metadata
- active learning around repeated `uncertain` and confirmed review outcomes
