# Results and Visuals

This document collects the shipped metrics, the generated report artifacts, the dataset snapshot tables, and the recovered training-dynamics plots in one place. It complements:

- [README.md](README.md) for the high-level system story and setup path
- [METHODS_AND_METRICS.md](METHODS_AND_METRICS.md) for formulas, metric definitions, and figure walkthroughs
- [DATA_CARD.md](DATA_CARD.md) for data provenance, supervision policy, and limitations

## Metrics artifacts

The shipped metrics are stored in:

- [reports/metrics/val_metrics.json](reports/metrics/val_metrics.json)
- [reports/metrics/test_metrics.json](reports/metrics/test_metrics.json)
- [reports/metrics/robustness_metrics.json](reports/metrics/robustness_metrics.json)

Headline shipped numbers:

| Split | Rows | Precision | Recall | F1 | ROC AUC | PR AUC | Minor FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Val | 8,717 | 0.8827 | 0.9700 | 0.9243 | 0.9953 | 0.9845 | 0.0300 |
| Test | 8,722 | 0.8711 | 0.9737 | 0.9195 | 0.9959 | 0.9855 | 0.0263 |

These columns should not be read as a generic classifier scoreboard. In this moderation problem:

- `Recall` is the main safety metric because it answers how many true minors are actually caught.
- `Minor FNR` is the same risk framed operationally: how often a minor still slips through.
- `Precision` still matters because unnecessary escalations reduce product usability, but it is secondary to recall.
- `ROC AUC` and `PR AUC` tell us whether the score ranking is strong enough that calibration and policy tuning are worth doing.
- `Accuracy` is intentionally not the headline number because an adult-heavy split can look accurate while still hiding unsafe misses on minors.

The implementation is in [ml/training/scripts/evaluate.py](ml/training/scripts/evaluate.py), which also exports prediction tables, subgroup metrics, calibration files, confusion matrices, reliability diagrams, and failure-analysis CSVs.

For the actual formulas and the detailed interpretation of ROC, PR, confusion matrices, and robustness verdict plots, see [METHODS_AND_METRICS.md](METHODS_AND_METRICS.md).

## Visual artifact inventory

The generated report assets currently include:

- ROC curves
- precision-recall curves
- rendered confusion matrices
- score histograms
- reliability diagrams
- source-validation and split-summary tables
- manifest preview and source-count views
- trusted-label and age-bucket coverage charts
- main-model and auxiliary-model training dynamics
- split composition charts
- robustness verdict charts
- failure-reason charts
- representative domain galleries
- subgroup false-negative-rate charts

Regenerate them with:

```bash
python scripts/generate_report_visuals.py
```

If `data/raw` exists locally, the gallery generator will sample real dataset images. If not, it falls back to the tracked `reviewer_samples/` images so the documentation still shows the input regimes handled by the deployed API.

## Dataset snapshots

The shipped manifest and source-validation artifacts are rich enough to show the dataset as a concrete object rather than only describing it in prose.

| Source | Rows/images visible in artifacts | Domain | How it is used |
| --- | ---: | --- | --- |
| FairFace | 97,537 manifest rows | `real` | trusted supervision, validation, test, and real-photo robustness reference |
| DeepFakeFace | 60,000 manifest rows | `ai_generated` | robustness-only evaluation |
| iCartoonFace | 59,805 manifest rows | `cartoon` | robustness-only evaluation |
| UTKFace | 23,705 images in source validation | `real` | available for the rejected ablation, not part of the shipped manifest |

The first table below comes directly from [ml/training/outputs/manifests/source_validation.json](ml/training/outputs/manifests/source_validation.json). It matters because it distinguishes sources that were merely scaffolded in code from sources that were actually present and usable in the run that produced the shipped reports.

![Source validation table](reports/tables/source_validation_table.png)

The split summary makes the final manifest structure explicit. The key design choice is that `train`, `val`, and `test` are real-photo supervision, while `robustness` is a separate evaluation partition rather than hidden training data.

![Split summary table](reports/tables/split_summary_table.png)

The manifest preview shows what a row really carries forward into evaluation: provenance, domain, split, age bucket, minor label, supervision status, demographics, and quality tags. That is what makes the later slice metrics and failure-analysis files auditable.

![Manifest preview table](reports/tables/manifest_preview_table.png)

This source-count chart makes the shipped manifest composition visually obvious. `FairFace` dominates the trusted supervision, while `DeepFakeFace` and `iCartoonFace` exist to measure domain-shift behavior instead of inflating the age-supervision story.

![Manifest source counts](reports/charts/manifest_source_dataset_counts.png)

The trusted age-bucket chart exposes an important limitation of the shipped baseline. Trusted supervision is strongest for `0-12`, `21-25`, and `26+`; the system does not pretend it has equally strong clean labels in the sensitive teenage range.

![Trusted age-bucket counts](reports/charts/trusted_age_bucket_counts.png)

The label-status breakdown makes the same decision visible from another angle. A large share of the merged manifest is deliberately ambiguous or robustness-only, which is the mechanism that keeps the supervised claim conservative.

![Label status breakdown](reports/charts/label_status_breakdown.png)

The gallery below shows representative input regimes. In this repo snapshot the raw training datasets are not redistributed, so the gallery uses tracked reviewer samples; the generation script automatically swaps in real dataset examples when `data/raw` is available.

![Domain example gallery](reports/galleries/domain_example_gallery.png)

## Training dynamics

The repo now includes recovered epoch histories under [ml/training/outputs/history](ml/training/outputs/history). Those JSON logs come from the same training scripts that produced the shipped checkpoints, so the plots below are grounded in actual per-epoch records rather than reconstructed from memory.

The main-model plot shows a steady drop in training loss with a noisier validation curve, which is what a recall-first setup often looks like when the boundary region is explicitly upweighted. The important line is validation recall: it stays high and ultimately justifies the exported checkpoint.

![Main training dynamics](reports/charts/main_training_dynamics.png)

The auxiliary-model plot is calmer because the DINOv2 encoder is frozen and only the lightweight head is optimized. That is an intentional design choice: the auxiliary path is meant to be a stable disagreement and uncertainty signal, not a second aggressively adapted classifier.

![Aux training dynamics](reports/charts/aux_training_dynamics.png)

The failure-reason chart helps connect optimization back to policy outcomes. Many of the remaining bad rows are not cleanly wrong `safe` decisions; they are abstentions triggered by conflict or boundary ambiguity. That is a better residual failure mode for a moderation aid than silent adult approvals on minors.

![Test failure reasons](reports/charts/test_failure_reasons.png)

## Dataset quality and generalization

The generalization story in this repo is not “the model learned every domain equally well.” It is more conservative and more defensible:

1. Trusted exact-age supervision is kept clean.
   - The shipped `train`, `val`, and `test` splits are real-photo `FairFace` rows.
   - Ambiguous rows are downgraded instead of forced into trusted labels.
   - Non-real data is kept in robustness evaluation rather than treated as exact legal-age ground truth.
2. Cross-split leakage is actively reduced.
   - [reports/metrics/deduplication_report.json](reports/metrics/deduplication_report.json) shows `356` duplicate removals from the merged manifest.
3. The split structure is explicit.
   - [reports/metrics/split_summary.csv](reports/metrics/split_summary.csv) shows `69,849` train rows, `8,717` val rows, `8,722` test rows, and `130,054` robustness rows.
4. The training objective emphasizes hard cases.
   - The `15-21` boundary region is upweighted.
   - Quality-tagged rows receive extra weight.
5. The deployed decision rule is allowed to abstain.
   - That is why robustness rows mostly end up `uncertain` rather than being over-claimed as `safe`.

For the shipped robustness split, the policy outcomes are:

- `125,647` `uncertain`
- `4,407` `flagged`
- `0` `safe`

That should be interpreted as conservative abstention under shift, not as a claim that the system has solved legal-age prediction on synthetic or stylized domains.
