# Data Card

## Purpose

This dataset supports a conservative age-safety pipeline that tries to minimize dangerous false negatives for minors. The dataset is designed for model development, calibration, robustness evaluation, and failure analysis.

## Intended use

- age-safety triage for user-submitted profile photos
- batch screening of image sets before downstream training or publishing
- robustness evaluation under domain shift

## Not intended use

- legal age verification
- identity verification
- biometric identification
- fully automated moderation without human review policy for ambiguous cases

## Dataset hypothesis

The highest-risk false negatives are expected to concentrate in these slices:

- visually ambiguous ages `16-21`
- low-resolution or blurry images
- partially occluded faces
- extreme pose or profile views
- multi-face scenes
- synthetic, stylized, edited, cartoon, anime, or 3D-rendered faces

## Primary data sources

- `UTKFace`
- `FairFace`
- `APPA-REAL`

## Robustness-only sources

- `SFHQ`
- `SFHQ-T2I`
- `Generated Photos Synthetic Face Images Academic Dataset`
- `Anime Face Dataset`
- `iCartoonFace`
- `DeepFakeFace`
- `DigiFace-1M`
- `TrueFace`, subject to access and license review

## Provenance and licensing

Each source must be documented in the generated source manifest with:

- source dataset name
- original URL
- local acquisition path
- license summary
- manual restrictions or attribution notes

Rows without clear provenance or license context should be excluded from the trusted training set.

## Annotation schema

### Image-level fields

- `image_id`
- `source_dataset`
- `license_type`
- `domain_type`
- `num_faces`
- `has_detectable_face`
- `quality_tags`
- `split`

### Face-level fields

- `face_id`
- `image_id`
- `image_path`
- `bbox_x1`
- `bbox_y1`
- `bbox_x2`
- `bbox_y2`
- `face_size_ratio`
- `pose_tag`
- `occlusion_tag`
- `blur_tag`
- `age_value`
- `age_bucket`
- `minor_label`
- `label_confidence`
- `label_status`
- `gender`
- `race`

## Age buckets

- `0-12`
- `13-15`
- `16-17`
- `18-20`
- `21-25`
- `26+`

These buckets intentionally expose the legal and visual decision boundary instead of hiding it inside a single adult class.

## Cleaning rules

Records should be excluded or downgraded to weak supervision when:

- face height is below `64 px`
- blur is severe
- landmarks are unstable
- the face crop is visibly incorrect
- age labeling is ambiguous
- provenance or license context is unclear
- the item is a duplicate or near-duplicate

## Deduplication

Deduplication is performed with:

- exact file hash
- simple perceptual hash
- optional embedding-based review for near duplicates

Cross-split leakage is explicitly checked.

## Splits

The proof-of-concept target is intentionally small and auditable:

- `8k-20k` face crops for training
- `2k-5k` for validation and testing

Sampling should oversample hard slices instead of maximizing raw volume.

## Known limitations

- stylized and synthetic domains do not provide universally reliable exact age labels
- heavily edited social-media imagery is only partially covered
- mixed-source datasets such as `TrueFace` require an explicit provenance and split audit before promotion into trusted subsets
- age labels in public datasets are not legal ground truth
- demographic fields depend on source availability and should be treated cautiously

## Safety note

This dataset should be used to train a conservative moderation aid, not an authoritative age verification system.
