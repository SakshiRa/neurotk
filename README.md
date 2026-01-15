NeuroTK: Dataset Validation for Neurology Brain Imaging

Overview
NeuroTK is an academic toolkit for dataset validation and quality assurance of neurology brain imaging data in NIfTI format. It exists to provide systematic QA prior to modeling, with a focus on reporting issues without modifying data.

What NeuroTK Does
- Validates NIfTI readability
- Checks geometry, spacing, orientation
- Detects missing labels
- Reports dataset-level inconsistencies
- Produces structured JSON reports

What NeuroTK Does NOT Do
- No preprocessing
- No training
- No inference
- No automatic fixing of data

Installation
pip install neurotk

Usage Example
neurotk validate --images imagesTr --labels labelsTr --out report.json
Expected input consists of directories containing NIfTI files, with label filenames matching image filenames.

Output
NeuroTK emits a JSON report containing a dataset-level summary and per-file diagnostics for images and optional labels.

Citation
If you use NeuroTK in your research, please cite it as follows.

@software{neurotk,
  title        = {NeuroTK: Dataset Validation for Neurology Brain Imaging},
  author       = {Sakshi Rathi},
  email        = {rathi036@umn.edu},
  year         = {2026},
  url          = {https://github.com/SakshiRa/neurotk},
  note         = {Open-source dataset validation toolkit for brain imaging}
}
