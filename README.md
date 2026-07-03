[![DOI](https://zenodo.org/badge/1134680274.svg)](https://doi.org/10.5281/zenodo.18252017)
[![CI](https://github.com/SakshiRa/neurotk/actions/workflows/ci.yml/badge.svg)](https://github.com/SakshiRa/neurotk/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/neurotk)](https://pypi.org/project/neurotk/)
[![Python](https://img.shields.io/pypi/pyversions/neurotk)](https://pypi.org/project/neurotk/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

# NeuroTK

**Dataset validation and quality assurance for neurology brain imaging.**

NeuroTK inspects NIfTI datasets for geometry, spacing, orientation, and annotation issues — then reports them as structured JSON before they break your pipeline.

## Features

- **Validate** &mdash; scan image and label directories for spacing inconsistencies, orientation mismatches, missing annotations, and corrupt files
- **Preprocess** &mdash; deterministic orientation normalization and voxel resampling with full audit trail
- **Infer** &mdash; run MONAI bundle inference (local or Hugging Face) with automatic Dice scoring
- **Analyze** &mdash; lesion volume quantification, cohort stratification, and histogram generation
- **Report** &mdash; machine-readable JSON + human-readable HTML reports
- **Web UI** &mdash; browser-based interface for upload-and-validate workflows

## Installation

```sh
pip install neurotk                # core (validation, preprocessing, analysis)
pip install neurotk[inference]     # + MONAI inference support
```

Requires Python >= 3.8. No GPU needed for validation and preprocessing.

## Quick Start

**Validate a dataset:**
```sh
neurotk validate --images data/imagesTr --labels data/labelsTr --out report.json
```

**Standardize spacing and orientation:**
```sh
neurotk preprocess --images data/imagesTr --out preprocessed/ --spacing 1.0 1.0 1.0 --orientation RAS
```

**Run segmentation inference:**
```sh
neurotk infer --input data/imagesTr --output-dir predictions/ --device cuda
```

**Measure lesion volumes:**
```sh
neurotk lesion-volume --preds predictions/ --output volumes.csv --histogram hist.png
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `neurotk validate` | Check dataset geometry, spacing, orientation, annotations |
| `neurotk preprocess` | Normalize orientation and resample voxel spacing |
| `neurotk infer` | Run MONAI bundle inference (local or Hugging Face) |
| `neurotk dice` | Compute Dice scores between predictions and labels |
| `neurotk lesion-volume` | Quantify lesion burden per case with cohort statistics |
| `neurotk cohort-stats` | Classify cases by lesion volume ranges |
| `neurotk make-normal-csv` | Generate normal-CT flags from label volumes |

Run `neurotk <command> --help` for full option details, or see the [CLI Reference](docs/cli-reference.md).

## Output Format

NeuroTK emits structured JSON reports with dataset-level summaries and per-file diagnostics:

```json
{
  "run_mode": "validate",
  "summary": {
    "scope": "original_inputs",
    "num_images": 100,
    "files_with_issues": 7,
    "orientation_modal": "RAS",
    "spacing_min": [1.0, 1.0, 1.0],
    "spacing_max": [1.0, 1.0, 1.0]
  },
  "files": {
    "case_001.nii.gz": {
      "shape": [256, 256, 128],
      "spacing": [1.0, 1.0, 1.0],
      "orientation": "RAS",
      "issues": ["label_missing"]
    }
  }
}
```

Add `--html report.html` to generate a shareable visual report.

## Python API

All CLI commands are importable for programmatic use:

```python
from neurotk.validate import validate_dataset

report = validate_dataset(images_dir="data/images", labels_dir="data/labels")
print(f"Files with issues: {report['summary']['files_with_issues']}")
```

## Web UI

Launch the browser-based interface:

```sh
pip install neurotk
python -m neurotk.web.app
```

Or via Docker:

```sh
docker build -t neurotk .
docker run -p 8000:8000 neurotk
```

## Sample Data

The `sample_data/` directory contains two synthetic NIfTI image-label pairs for testing:

```
sample_data/
  images/   CASE_001.nii.gz, CASE_002.nii.gz
  labels/   CASE_001.nii.gz, CASE_002.nii.gz
```

```sh
neurotk validate --images sample_data/images --labels sample_data/labels --out report.json
```

## Support

- **Bug reports**: [file an issue](https://github.com/SakshiRa/neurotk/issues/new?template=bug_report.md) on GitHub
- **Feature requests**: [request a feature](https://github.com/SakshiRa/neurotk/issues/new?template=feature_request.md)
- **Contributing**: see [CONTRIBUTING.md](CONTRIBUTING.md)
- **Maintainer**: Sakshi Rathi (rathi036@umn.edu)

## Citation

```bibtex
@software{neurotk,
  title  = {NeuroTK: Dataset Validation for Neurology Brain Imaging},
  author = {Sakshi Rathi},
  year   = {2026},
  doi    = {10.5281/zenodo.18252017},
  url    = {https://github.com/SakshiRa/neurotk}
}
```

## License

[Apache 2.0](LICENSE)
