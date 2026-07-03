# CLI Reference

Full command-line reference for NeuroTK. For a quick overview, see the [README](../README.md).

## neurotk validate

Scan directories of NIfTI images (and optional labels) for geometry, spacing, orientation, and annotation issues.

```sh
neurotk validate \
  --images imagesTr \
  --labels labelsTr \
  --out report.json \
  --html report.html \
  --max-samples 10 \
  --summary-only
```

| Option | Required | Description |
|--------|----------|-------------|
| `--images` | Yes | Directory of input NIfTI images |
| `--labels` | No | Directory of label NIfTI files (paired by filename) |
| `--out` | Yes | Output JSON report path |
| `--html` | No | Also write an HTML report |
| `--max-samples` | No | Limit number of images processed |
| `--summary-only` | No | Print text summary to stdout |

NeuroTK scans directories recursively for `.nii` / `.nii.gz` files. Image-label pairing requires exact filename matches.

**Expected directory layout:**
```
dataset/
  imagesTr/
    case_001.nii.gz
    case_002.nii.gz
  labelsTr/
    case_001.nii.gz
    case_002.nii.gz
```

---

## neurotk preprocess

Normalize orientation and resample voxel spacing. Original files are never modified — outputs are written to a new directory with a full audit trail in the JSON report.

```sh
neurotk preprocess \
  --images imagesTr \
  --labels labelsTr \
  --out preprocessed/ \
  --spacing 1.0 1.0 1.0 \
  --orientation RAS \
  --dry-run \
  --copy-metadata
```

| Option | Required | Description |
|--------|----------|-------------|
| `--images` | Yes | Directory of input NIfTI images |
| `--labels` | No | Directory of label NIfTI files |
| `--out` | Yes | Output directory for preprocessed files |
| `--spacing` | Yes | Target spacing as 3 floats (e.g., `1.0 1.0 1.0`) |
| `--orientation` | No | Target orientation code (default: `RAS`) |
| `--dry-run` | No | Preview preprocessing without writing outputs |
| `--copy-metadata` | No | Preserve metadata when applicable |

---

## neurotk infer

Run segmentation inference using MONAI bundles. Supports local bundles, Hugging Face repos, and batch processing.

```sh
# Single image with local bundle
neurotk infer --bundle-dir /path/to/bundle --input image.nii.gz --output-dir outputs/

# Default bundle from Hugging Face (UMNSHAMLAB/segresnet)
neurotk infer --input image.nii.gz --output-dir outputs/

# Explicit Hugging Face bundle
neurotk infer --bundle-dir hf:UMNSHAMLAB/segresnet --input image.nii.gz --output-dir outputs/

# Batch mode
neurotk infer --input-list images.txt --output-dir outputs/ --device cuda
```

| Option | Required | Description |
|--------|----------|-------------|
| `--bundle-dir` | No | MONAI bundle path, `org/model`, `hf:org/model`, or HF URL |
| `--input` | One of | Single NIfTI file or directory of NIfTI files |
| `--input-list` | One of | Text file with one image path per line |
| `--output-dir` | Yes | Output directory for predictions |
| `--device` | No | Device: `cuda`, `cuda:0`, `mps`, `cpu` |
| `--save-probs` | No | Save probability maps (`*_prob.nii.gz`) instead of segmentations |
| `--force` | No | Recompute even if outputs exist |
| `--skip-invalid-inputs` | No | Skip failing files (logged to `skipped_inputs.csv`) |
| `--labels-dir` | No | Labels directory for Dice computation during inference |
| `--reference-image` | No | Image whose affine/header are used for saved outputs |

**Notes:**
- Use exactly one of `--input` or `--input-list`.
- Default HF bundle: `UMNSHAMLAB/segresnet` (overridable via `NEUROTK_DEFAULT_BUNDLE` env var).
- Existing outputs are skipped by default; use `--force` to recompute.
- With `--skip-invalid-inputs`, invalid files are recorded in `outputs/skipped_inputs.csv`.
- If `--labels-dir` is omitted and `--input` is a directory, NeuroTK auto-detects sibling labels directories (e.g., `imagesTr` -> `labelsTr`).
- CPU inference prints a warning due to significantly slower runtime.

---

## neurotk dice

Compute Dice scores between prediction and label NIfTI files.

```sh
neurotk dice \
  --preds outputs/ \
  --labels-dir labels/ \
  --output dice_scores.csv
```

| Option | Required | Description |
|--------|----------|-------------|
| `--preds` | One of | Prediction NIfTI file or directory |
| `--preds-list` | One of | Text file with one prediction path per line |
| `--labels-dir` | Yes | Labels directory |
| `--output` | Yes | CSV output path |

---

## neurotk lesion-volume

Quantify lesion burden from segmentation predictions.

```sh
neurotk lesion-volume \
  --preds outputs/ \
  --output volumes.csv \
  --summary-output summary.csv \
  --histogram hist.png \
  --hist-bins 30
```

| Option | Required | Description |
|--------|----------|-------------|
| `--preds` | One of | Prediction NIfTI file or directory |
| `--preds-list` | One of | Text file with one prediction path per line |
| `--output` | Yes | CSV output path for per-case volumes |
| `--summary-output` | No | CSV output path for range summary |
| `--threshold` | No | Binarization threshold for probability maps (default: `0.5`) |
| `--histogram` | No | Save histogram image (PNG) |
| `--hist-bins` | No | Number of histogram bins (default: `30`) |

**Per-case CSV columns:** `image`, `lesion_voxels`, `voxel_volume_mm3`, `lesion_volume_mm3`, `lesion_volume_ml`

**Summary CSV columns:** `category` (range/overall), `metric`, `count`, `percent`, `value_ml`

---

## neurotk cohort-stats

Classify cases by lesion volume into true-negative, low, medium, and high groups.

```sh
neurotk cohort-stats \
  --labels labelsTr/ \
  --normal-csv normal_ct_flags.csv \
  --output cohort_classification.csv \
  --summary-output cohort_summary.csv \
  --tn-threshold-ml 0.2 \
  --low-max-ml 5.0 \
  --medium-max-ml 20.0
```

| Option | Required | Description |
|--------|----------|-------------|
| `--labels` | One of | Label NIfTI file or directory |
| `--labels-list` | One of | Text file with one label path per line |
| `--normal-csv` | Yes | CSV with normal CT flag (`image`/`id` + `normal_ct`/`normal`/`is_normal`) |
| `--output` | Yes | Per-case classification CSV |
| `--summary-output` | Yes | Cohort summary CSV |
| `--tn-threshold-ml` | No | True-negative threshold in mL (default: `0.2`) |
| `--low-max-ml` | No | Upper bound for TP-low group (default: `5.0`) |
| `--medium-max-ml` | No | Upper bound for TP-medium group (default: `20.0`) |

**Classification rules:**
- `true_negative`: `normal_ct == true` AND lesion volume <= `tn-threshold-ml`
- `true_positive`: all other cases, subdivided into `low`, `medium`, `high` by volume

---

## neurotk make-normal-csv

Generate normal-CT flags from label volumes, and optionally a MONAI-style datalist JSON for training.

```sh
neurotk make-normal-csv \
  --images imagesTr/ \
  --labels labelsTr/ \
  --output normal_ct_flags.csv \
  --threshold-ml 0.2 \
  --train-selection-json train_selection.json \
  --train-min-lesion-ml 1.0 \
  --num-folds 5
```

| Option | Required | Description |
|--------|----------|-------------|
| `--images` | No | Image directory (required when using `--train-selection-json`) |
| `--images-list` | No | Text file with one image path per line |
| `--labels` | One of | Label NIfTI file or directory |
| `--labels-list` | One of | Text file with one label path per line |
| `--output` | Yes | Output CSV path |
| `--threshold-ml` | No | Volume threshold for `normal_ct=true` (default: `0.2`) |
| `--train-selection-json` | No | Write MONAI datalist JSON for selected training cases |
| `--train-min-lesion-ml` | No | Min lesion volume for training inclusion (default: `1.0`) |
| `--num-folds` | No | Number of CV folds (default: `5`) |

**Train-selection JSON structure (MONAI-style):**
```json
{
  "description": "...",
  "labels": {"0": "background", "1": "lesion"},
  "training": [{"image": "...", "label": "...", "fold": 0}],
  "validation": [],
  "testing": [{"image": "..."}]
}
```
