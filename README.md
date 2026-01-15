NeuroTK: Dataset Validation for Neurology Brain Imaging

Motivation
Neurology brain imaging datasets are heterogeneous and error-prone, with common issues in geometry, spacing, orientation, and annotations. These problems often surface late in the modeling pipeline, undermining reproducibility and interpretability. NeuroTK exists to surface these issues early, explicitly, and reproducibly.

Scope
NeuroTK provides dataset-level and file-level validation with structural and geometric consistency checks, and assessment of annotation presence and integrity. The scope is dataset quality assurance prior to downstream analysis.

What NeuroTK Provides
- Validation of NIfTI readability and dimensionality
- Inspection of voxel spacing, affine geometry, and orientation
- Detection of missing, mismatched, or empty labels
- Dataset-level consistency statistics (shape, spacing, orientation)
- Structured, machine-readable JSON reports suitable for archiving and review

Installation
pip install neurotk

Usage
neurotk validate --images imagesTr --labels labelsTr --out report.json
imagesTr and labelsTr are flat directories of NIfTI files, and filenames must match exactly for image–label pairing. Validation completes and reports issues rather than failing early.

Output
NeuroTK produces a dataset-level summary, per-file diagnostics for images and optional labels, and explicit listings of detected issues. The output is designed for reproducibility, auditing and dataset documentation, and inclusion in publications, benchmarks, or grant materials.

Citation
If you use NeuroTK in your research, please cite it as follows:

@software{neurotk,
  title  = {NeuroTK: Dataset Validation for Neurology Brain Imaging},
  author = {Sakshi Rathi},
  year   = {2026},
  url    = {https://github.com/SakshiRa/neurotk},
  note   = {Open-source toolkit for dataset validation and quality assurance in neurology brain imaging}
}
