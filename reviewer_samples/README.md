## Reviewer Samples

This folder is intended for a small unseen manual-review set.

Place the provided test images here using the filenames listed in [sample_manifest.json](/Users/andrescarvajal/Documents/hygo-assessment/reviewer_samples/sample_manifest.json).

Then, with the stack running, execute:

```bash
python scripts/run_reviewer_samples.py
```

On Windows, if `python` is not on `PATH`, use:

```powershell
py -3 scripts/run_reviewer_samples.py
```

The script will:

- call `GET /v1/age-safety/health`
- upload each available sample to `POST /v1/age-safety/check`
- save one JSON response per image under `reviewer_samples/results`
- write a batch response to `reviewer_samples/results/batch_results.json`

Expected high-level behavior by category:

- `adult_face_*`: usually `safe`, sometimes `uncertain` if quality or boundary cues are weak
- `minor_face_*`: usually `flagged`
- `cartoon_face_*`: usually `uncertain`
- `anime_face_*`: usually `uncertain`
- `no_face_*`: should be `uncertain`
- `multi_face_*`: should use the highest-risk detected face as the image verdict
