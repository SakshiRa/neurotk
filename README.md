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

Inputs are expected as flat directories of NIfTI files, and filenames must match exactly for image–label pairing.

```
dataset/
  imagesTr/
    case_001.nii.gz
    case_002.nii.gz
  labelsTr/
    case_001.nii.gz
    case_002.nii.gz
```

## Output
NeuroTK emits a JSON report containing a dataset-level summary, per-file diagnostics, and explicit listings of detected issues.

```json
{
  "summary": {"num_images": 100, "files_with_issues": 7},
  "files": {"case_001.nii.gz": {"issues": ["label_missing"]}}
}
```

## Citation
If you use NeuroTK in your research, please cite it as follows:

```bibtex
@software{neurotk,
  title  = {NeuroTK: Dataset Validation for Neurology Brain Imaging},
  author = {Sakshi Rathi},
  year   = {2026},
  url    = {https://github.com/SakshiRa/neurotk},
  note   = {Open-source toolkit for dataset validation and quality assurance in neurology brain imaging}
}
```
