# Changelog

## [0.3.3] - 2026-07-02

### Fixed
- Dice score now correctly returns 1.0 when both prediction and target are all-zeros (true negative case), rather than the previous epsilon-dominated value near 2.0.
- Hausdorff distance computation now skips and emits a `RuntimeWarning` when pred and label shapes differ, rather than silently resizing labels (which produced geometrically invalid metrics).
- Hausdorff distance now emits a `RuntimeWarning` when MONAI is unavailable, rather than returning None silently.
- Affine determinant check now flags negative determinants (`image_affine_negative_determinant`) indicating flipped/mirrored coordinate systems, not just zero determinants.
- NIfTI stem extraction is now case-insensitive for file extensions, fixing label pairing for mixed-case filenames (e.g. `CASE_001.NII.GZ`).
- CI workflow now runs the full pytest test suite across Python 3.8–3.11.

## [0.3.2] - 2026-03-15

### Added
- GitHub Actions workflow for automated PyPI publishing on version tags.
- Docker support for containerized deployment.

### Fixed
- Web app downloads now correctly persist generated reports.

## [0.3.1] - 2026-02-10

### Added
- Lesion volume histogram output (`--histogram`, `--hist-bins`).
- Cohort-level summary CSV for lesion volume ranges.

### Fixed
- CLI `--dry-run` flag now previews preprocessing without writing outputs.

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

