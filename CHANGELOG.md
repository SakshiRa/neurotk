# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.3.3] - 2026-07-02

### Added
- `neurotk lesion-volume` command with per-case volumes, cohort summary CSV, and histogram generation.
- `neurotk cohort-stats` command for classifying cases by lesion volume ranges (TN, low, medium, high).
- `neurotk make-normal-csv` command for generating normal-CT flags from label volumes.
- MONAI-style train selection JSON output with inner-join image-label pairing and k-fold assignment.
- `--skip-invalid-inputs` mode for inference: invalid files are skipped and logged to `skipped_inputs.csv`.

### Changed
- Scientific rigor improvements to validation and reporting.

## [0.3.2] - 2026-06-15

### Added
- MONAI bundle compatibility layer (`monai_compat.py`) for robust bundle loading across MONAI versions.
- Pinned inference dependency ranges (`torch>=2.3,<2.10`, `monai>=1.3,<1.6`).

### Changed
- Existing inference outputs are skipped by default; use `--force` to recompute.

## [0.3.1] - 2026-05-20

### Added
- Inference CLI and webapp integration.
- Improved CLI robustness and warning handling for edge cases.

### Changed
- FastAPI is now the primary web interface; the older `site/` Next.js prototype is deprecated.

## [0.3.0] - 2026-01-24

### Added
- Validate+preprocess reports now include explicit original vs processed summaries with scope markers.
- Preprocess traceability: per-file verification fields (requested/applied/verified_by) and dataset-level effects summary.
- Label provenance fields in preprocess inputs (labels_provided, num_label_files_uploaded).
- Intensity value_range_hint to describe dynamic range without modality assumptions.

### Changed
- Validate+preprocess reports now include summary_processed computed from processed outputs.
- Preprocess reporting is machine-verifiable via original vs processed comparisons.

### Fixed
- Webapp downloads now persist generated reports after request completion.
- Labels_dir is no longer reported when labels are absent.

### Notes
- Report schema changes are additive; validation-only outputs remain unchanged.

## [0.2.1] - 2026-01-10

### Fixed
- Minor bug fixes and packaging corrections.

## [0.2.0] - 2025-12-15

### Added
- HTML report generation (`--html` flag).
- Web UI for interactive validation via file upload.
- Dockerfile for containerized deployment.

## [0.1.1] - 2025-11-20

### Fixed
- PyPI packaging and setuptools configuration.

## [0.1.0] - 2025-11-01

### Added
- Initial release.
- `neurotk validate` command for NIfTI dataset validation.
- `neurotk preprocess` command for orientation normalization and voxel resampling.
- Structured JSON report output.
- Sample data included for testing.
