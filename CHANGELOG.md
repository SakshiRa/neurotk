# Changelog

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
