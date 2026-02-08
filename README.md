[![DOI](https://zenodo.org/badge/1134680274.svg)](https://doi.org/10.5281/zenodo.18252017)


# NeuroTK: Dataset Validation for Neurology Brain Imaging

## Motivation
Neurology brain imaging datasets are heterogeneous and frequently contain inconsistencies. Geometry, spacing, orientation, and annotation issues occur commonly across CT and MRI collections. These problems often surface late in modeling, when remediation is costly and compromises reproducibility. NeuroTK surfaces issues early, explicitly, and reproducibly to support dataset hygiene prior to analysis.

## Scope
NeuroTK focuses on dataset quality assurance prior to downstream analysis. It provides dataset-level and file-level validation with structural and geometric consistency checks, and assessment of annotation presence and integrity.

- Dataset-level and file-level validation
- Structural and geometric consistency checks
- Annotation presence and integrity assessment

NeuroTK does not modify scientific data.

## Installation
```sh
pip install neurotk
```

## Quickstart
```sh
neurotk validate --images imagesTr --labels labelsTr --out report.json
```

For `validate`, NeuroTK scans directories recursively for `.nii`/`.nii.gz` files. Filenames must match exactly for image-label pairing.

```
dataset/
  imagesTr/
    case_001.nii.gz
    case_002.nii.gz
  labelsTr/
    case_001.nii.gz
    case_002.nii.gz
```

## CLI Reference

Validate:
```sh
neurotk validate \
  --images imagesTr \
  --labels labelsTr \
  --out report.json \
  --max-samples 10 \
  --html report.html \
  --summary-only
```

Key options:
- `--images` (required): directory of input NIfTI images.
- `--labels` (optional): directory of label NIfTI files.
- `--out` (required): output JSON report path.
- `--max-samples` (optional): limit number of images processed.
- `--html` (optional): write HTML report.
- `--summary-only` (optional): print text summary to stdout.

Preprocess:
```sh
neurotk preprocess \
  --images imagesTr \
  --labels labelsTr \
  --out preprocessed/ \
  --spacing 1.0 1.0 1.0 \
  --orientation RAS \
  --copy-metadata
```

Key options:
- `--images` (required): directory of input NIfTI images.
- `--labels` (optional): directory of label NIfTI files.
- `--out` (required): output directory for preprocessed files.
- `--spacing` (required): target spacing as 3 floats.
- `--orientation` (optional): target orientation (default `RAS`).
- `--dry-run` (optional): preview preprocessing without writing outputs.
- `--copy-metadata` (optional): preserve metadata when applicable.

## Inference (MONAI bundles)
NeuroTK can run inference from external MONAI bundles via the optional inference extras:

```sh
pip install neurotk[inference]
```

Single image:
```sh
neurotk infer \
  --bundle-dir /path/to/bundle \
  --input image.nii.gz \
  --output-dir outputs/
```

Default bundle (uses `NEUROTK_DEFAULT_BUNDLE` or `UMNSHAMLAB/segresnet`):
```sh
neurotk infer \
  --input image.nii.gz \
  --output-dir outputs/
```
Default HF bundle repo: `UMNSHAMLAB/segresnet`.

From Hugging Face (auto-download + cache full bundle):
```sh
neurotk infer \
  --bundle-dir hf:UMNSHAMLAB/segresnet \
  --input image.nii.gz \
  --output-dir outputs/
```

You can also pass a Hugging Face repo URL:
```sh
neurotk infer \
  --bundle-dir https://huggingface.co/UMNSHAMLAB/segresnet \
  --input image.nii.gz \
  --output-dir outputs/
```

Batch mode:
```sh
neurotk infer \
  --bundle-dir /path/to/bundle \
  --input-list images.txt \
  --output-dir outputs/
```

Key options:
- `--bundle-dir` (optional): local MONAI bundle path, `org/model`, `hf:org/model`, or HF URL.
- `--input` (optional): one NIfTI file or a directory of NIfTI files.
- `--input-list` (optional): text file with one image path per line.
- Use exactly one of `--input` or `--input-list`.
- `--output-dir` (required): output directory for predictions.
- `--device` (optional): inference device (for example `cuda`, `cuda:0`, `mps`, `cpu`).
- `--save-probs` (optional): save probability output (`*_prob.nii.gz`) instead of segmentation (`*_seg.nii.gz`).
- `--force` (optional): recompute outputs even if prediction files already exist.
- `--skip-invalid-inputs` (optional): continue inference by skipping files that fail (for example incompatible channels/dimensions).
- `--labels-dir` (optional): labels directory used to compute Dice during inference.
- `--reference-image` (optional): image whose affine/header are used for saved outputs.

Device selection:
```sh
# CUDA
neurotk infer --device cuda --input image.nii.gz --output-dir outputs/

# Apple Silicon
neurotk infer --device mps --input image.nii.gz --output-dir outputs/

# CPU
neurotk infer --device cpu --input image.nii.gz --output-dir outputs/
```

If inference runs on CPU (explicitly or via fallback), NeuroTK prints a warning because runtime may be significantly slower.

Dice during inference:
- `neurotk infer` computes Dice and writes `outputs/dice_scores.csv` only when labels are available.
- If `--labels-dir` is omitted and `--input` is a directory, NeuroTK auto-detects sibling labels directories such as `images -> labels` and `imagesTr -> labelsTr`.
- If labels are not present, Dice is skipped.
- If `--input` path does not exist, inference fails fast with a clear error.
- Existing prediction outputs are skipped by default; pass `--force` to recompute.
- With `--skip-invalid-inputs`, invalid files are skipped and recorded in `outputs/skipped_inputs.csv`.

Dice after inference:
```sh
neurotk dice \
  --preds outputs/ \
  --labels-dir labels/ \
  --output outputs/dice_scores.csv
```

Lesion volume from predictions:
```sh
neurotk lesion-volume \
  --preds outputs/ \
  --output outputs/lesion_volumes.csv \
  --summary-output outputs/lesion_volumes_summary.csv
```

With histogram:
```sh
neurotk lesion-volume \
  --preds outputs/ \
  --output outputs/lesion_volumes.csv \
  --histogram outputs/lesion_volume_hist.png \
  --hist-bins 30
```

Output columns:
- `image`
- `lesion_voxels`
- `voxel_volume_mm3`
- `lesion_volume_mm3`
- `lesion_volume_ml`

Summary CSV columns:
- `category` (`range` or `overall`)
- `metric` (range label or stat name)
- `count`
- `percent`
- `value_ml`

Included overall stats:
- `total_images`
- `min_ml`
- `p25_ml`
- `median_ml`
- `p75_ml`
- `max_ml`
- `mean_ml`

Key options:
- `--preds` (optional): one prediction NIfTI file or a directory of predictions.
- `--preds-list` (optional): text file with one prediction path per line.
- Use exactly one of `--preds` or `--preds-list`.
- `--labels-dir` (required): labels directory.
- `--output` (required): CSV output path for Dice/Hausdorff metrics.

Lesion volume options:
- `--preds` (optional): one prediction NIfTI file or a directory of predictions.
- `--preds-list` (optional): text file with one prediction path per line.
- `--output` (required): CSV output path for lesion volume report.
- `--summary-output` (optional): CSV output path for lesion-volume range summary.
- `--threshold` (optional): threshold for binarizing 3D probability maps (default `0.5`).
- `--histogram` (optional): path to save histogram image of lesion volumes (mL).
- `--hist-bins` (optional): number of histogram bins (default `30`).

Note: for full-bundle HF usage, the repo must contain a valid MONAI bundle layout (e.g., `configs/` with inference/evaluate config and `models/` checkpoints).

## Output
NeuroTK emits a JSON report containing a dataset-level summary, per-file diagnostics, and explicit listings of detected issues.
For validate+preprocess runs, the report includes a processed summary and preprocess traceability so original and processed
states are unambiguous.

```json
{
  "summary": {"scope": "original_inputs", "num_images": 100, "files_with_issues": 7},
  "summary_processed": {"scope": "processed_outputs", "num_images": 100},
  "files": {"case_001.nii.gz": {"issues": ["label_missing"]}}
}
```

### Validate vs preprocess semantics
- `summary` always reflects original inputs.
- `summary_processed` is present only for validate+preprocess runs and reflects outputs after preprocessing.
- `run_mode` indicates whether preprocessing was requested.

### Upgrading to v0.3.0
Reports now include explicit `scope` fields and preprocess traceability blocks. These additions are backward-compatible
for validation-only users.

## Web UI
The FastAPI app in `webapp/` is the primary landing page and execution interface. The older `site/` Next.js prototype
is deprecated and should not be used for deployment.

## Citation
If you use NeuroTK in your research, please cite it as follows:

```bibtex
@software{neurotk,
  title  = {NeuroTK: Dataset Validation for Neurology Brain Imaging},
  author = {Sakshi Rathi},
  year   = {2026},
  doi    = {10.5281/zenodo.18252017},
  url    = {https://github.com/SakshiRa/neurotk},
  note   = {Open-source toolkit for dataset validation and quality assurance in neurology brain imaging}
}
```
