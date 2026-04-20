# Decisions

## Product policy

This project uses a three-way safety policy instead of a hard binary classifier:

- `safe`
- `flagged`
- `uncertain`

This is deliberate. In a safety-critical setting, the cost of a false negative is higher than the cost of abstaining. The system should only return `safe` when adult evidence is strong, quality is acceptable, and the signals are consistent.

## Dataset choices

The shipped checkpoint is trained on:

- [`FairFace`](https://github.com/joojs/fairface/) for the actual supervised baseline, held-out validation and test metrics, and demographic reporting.

Evaluated ablation, but not shipped:

- [`UTKFace`](https://susanqq.github.io/UTKFace/) for broader age coverage and a harder teenage slice, used only in the rejected `UTKFace + FairFace` comparison run.

Scaffolded but not used in the shipped metrics:

- [`APPA-REAL`](https://chalearnlap.cvc.uab.cat/dataset/26/description/) for future ambiguity-aware age supervision.
- [`SFHQ`](https://github.com/SelfishGene/SFHQ-dataset)
- [`SFHQ-T2I`](https://github.com/SelfishGene/SFHQ-T2I-dataset)
- [`Generated Photos Synthetic Face Images Academic Dataset`](https://huggingface.co/datasets/GeneratedPhotos/Synthetic_Face_Images_Academic_Dataset)
- [`Anime Face Dataset`](https://github.com/bchao1/Anime-Face-Dataset)
- [`DigiFace-1M`](https://microsoft.github.io/DigiFace1M/)
- `TrueFace`, when access and licensing are validated

The shipped robustness and abstention evaluation uses:

- [`DeepFakeFace`](https://github.com/OpenRL-Lab/DeepFakeFace)
- [`iCartoonFace`](https://github.com/luxiangju-PersonAI/iCartoonFace)
- held-out `FairFace` real-photo rows as the real reference domain

These non-real sources are intentionally not treated as the main source of exact age ground truth. Their role is to expose failure under domain shift and to justify conservative abstention.

For the current proof-of-concept run, the fastest reproducible trained baseline is `FairFace`. The pipeline already supports `UTKFace`, `APPA-REAL`, and the robustness datasets, but those are staged as the next iteration rather than being falsely claimed as already-trained sources.

On April 20, 2026, I also ran a `UTKFace + FairFace` ablation. That experiment improved precision but reduced minor recall and increased the false-negative rate in the dangerous `13-17` slice. Because this task is safety-critical, I rejected that checkpoint as the final shipped candidate and kept the `FairFace` baseline as the recommended deployment model.

## Model choices

Preferred architecture:

- [`InsightFace`](https://github.com/deepinsight/insightface) for detection and alignment.
- a custom `MiVOLOAgeEstimator` main model whose design is inspired by [`MiVOLO`](https://github.com/WildChlamydia/MiVOLO), but whose shipped checkpoint uses a `timm` `tf_efficientnet_b0` backbone with lightweight age, bucket, and minor-risk heads.
- [`DINOv2`](https://github.com/facebookresearch/dinov2) as the auxiliary domain-robust encoder.

The shipped architecture is intentionally not “detector -> classifier -> done.”

It is:

- face detection and aligned crop extraction with `InsightFace`
- primary age estimation and `p_minor` scoring with a `MiVOLO`-inspired head over `tf_efficientnet_b0`
- auxiliary domain and uncertainty scoring with `DINOv2`
- post-hoc temperature calibration of the main minor-risk score
- policy evaluation over age interval, conflict score, domain type, face confidence, face area, and quality flags

That extra policy layer is the main design choice of the system. It is what allows the final product to abstain under shift instead of pretending that every image belongs to the in-domain photographic distribution.

The codebase includes fallback implementations because model availability on first boot is often incomplete. That keeps the scaffold runnable on a pod before the final checkpoints are ready.

The production-oriented asset path is:

- optional MiVOLO V2 weights from Hugging Face for inference-time age estimation
- official DINOv2 loading via PyTorch Hub with an optional local repo clone
- InsightFace `buffalo_l` loaded from a local root to avoid unexpected runtime downloads on pods

## Threshold decisions

Suggested defaults for the shipped baseline:

- `safe` if `p_minor < 0.05`, quality is acceptable, there is no strong model conflict, and the adult interval is clearly above 18
- `flagged` if `p_minor >= 0.40` or the age estimate is close to or below the legal boundary
- `uncertain` otherwise

Thresholds should be tuned on validation slices, not on global accuracy. The explicit tradeoff is to accept more abstention and more false positives in exchange for reducing dangerous misses on minors.

How to interpret the shipped metrics:

- `minor_recall` is the core safety metric for the binary classifier
- `minor_false_negative_rate` is its inverse framing and is easier to reason about operationally
- `minor_precision` matters, but is secondary to avoiding dangerous misses
- `roc_auc` and `pr_auc` show the score quality is high enough that policy tuning is meaningful
- `verdict_counts` are policy outcomes, not direct classifier outputs

The `UTKFace` ablation made this tradeoff explicit: it improved `minor_precision` but worsened `minor_recall` and produced unsafe behavior in teenage slices unless the adult-safe gate was tightened so aggressively that `safe` throughput collapsed. That is the reason it remains an ablation rather than the final policy target.

## Bias analysis

Bias is evaluated primarily with `FairFace`, because it offers demographic attributes that make subgroup reporting feasible. The intended analysis breaks metrics down by:

- age bucket
- gender
- race
- domain type
- image quality slice

Known constraints:

- demographic labels in public datasets are imperfect and should not be treated as identity truth
- the final shipped checkpoint is intentionally the `FairFace` baseline because the `UTKFace` ablation degraded recall in the most safety-sensitive slice
- bias results should be reported as measured behavior, not as proof that the system is fair enough for fully autonomous enforcement
- in the shipped `FairFace` test split, labeled minors appear in the `0-12` bucket only, so the subgroup report does not by itself validate borderline `13-17` behavior

Current shipped-checkpoint subgroup findings:

- gender:
  - `female` minor recall `0.9723`, minor false-negative rate `0.0277`
  - `male` minor recall `0.9748`, minor false-negative rate `0.0252`
- race:
  - highest false-negative rates in the shipped test split were `latino_hispanic` (`0.0450`) and `indian` (`0.0419`)
  - lowest false-negative rates were `southeast asian` (`0.0049`) and `east asian` (`0.0122`)

These are not large enough subgroup counts to justify aggressive threshold changes by themselves, but they are large enough to justify explicit disclosure and continued monitoring.

## Real vs generated images

The repository treats real and generated images as a first-class domain-shift problem.

The strategy is:

- use `FairFace` as the shipped supervised training and evaluation set
- use the rejected `UTKFace + FairFace` ablation to understand borderline-age tradeoffs before shipping
- use synthetic, anime, cartoon, 3D, and edited datasets for robustness evaluation and abstention testing
- prefer `uncertain` when the domain signal is unstable or the auxiliary model conflicts with the main model

This is deliberate. Synthetic and stylized datasets often do not provide reliable legal-age labels, so forcing them into the main supervision pool would make the model look broader while actually reducing trustworthiness.

For the shipped baseline, the robustness split produced:

- `125,647` `uncertain`
- `4,407` `flagged`
- `0` `safe`

That should be read as a conservative abstention result, not as a claim that the model accurately estimates age on synthetic or cartoon images. In the shipped report, those robustness rows come primarily from `DeepFakeFace` and `iCartoonFace`, with held-out `FairFace` rows as the real-photo reference.

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

## Integration design

Recommended platform integration:

- profile photo upload: synchronous call, accept only `safe`, reject `flagged`, ask for replacement or manual review on `uncertain`
- training photo upload batch: asynchronous batch check with per-image results and a batch-level summary
- generated image delivery: check before the image is returned to the user, block on `flagged`, hold or discard on `uncertain` depending on product policy
- public gallery publish: run another check before publish, even if the image was screened earlier

Edge cases:

- no face detected: return `uncertain`
- multiple faces: evaluate every face and use the highest-risk result at the image level
- repeated `uncertain` or `flagged` outcomes: escalate to manual review, additional verification, or account-level friction depending on policy
- retraining: collect confirmed false positives, dangerous misses, repeated `uncertain` cases, and new domain-shift examples for the next dataset iteration

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

## What I'd do with more time

The next highest-value improvements would be:

- revisit `UTKFace` with stronger label audit, sample reweighting, or a borderline-age curriculum before promoting it into the shipped baseline
- add `APPA-REAL` into the trained baseline and rerun calibration
- add at least one synthetic robustness dataset such as `SFHQ-T2I` or `DeepFakeFace` into evaluation
- build a stronger borderline-age audit slice around `16-21`
- run a tighter demographic report from the completed evaluation artifacts and use it to retune thresholds
- verify the full NestJS-to-Python inference path under `docker-compose` with the exported bundle that will be submitted
