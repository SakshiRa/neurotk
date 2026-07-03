"""Per-file and dataset-level validation logic for NIfTI datasets."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .io import load_nifti
from .utils import nifti_stem, orientation_codes, safe_stats, spacing_from_header, to_list


def _image_info_template(path: Path) -> Dict[str, object]:
    return {
        "path": str(path),
        "readable": False,
        "dimensionality": None,
        "shape": None,
        "spacing": None,
        "orientation": None,
        "affine_determinant": None,
        "dtype": None,
        "has_nan": None,
        "has_inf": None,
        "stats": None,
    }


def _label_info_template(path: Optional[Path]) -> Dict[str, object]:
    return {
        "path": str(path) if path else None,
        "present": path is not None,
        "readable": False,
        "dimensionality": None,
        "shape": None,
        "shape_matches_image": None,
        "integer_valued": None,
        "unique_values": None,
        "empty": None,
    }


def validate_image(path: Path) -> Tuple[Dict[str, object], List[str]]:
    """Validate a single NIfTI image file.

    Checks readability, dimensionality, voxel spacing, orientation,
    affine matrix integrity, and presence of NaN/Inf values.

    :param path: Path to a NIfTI file (.nii or .nii.gz).
    :returns: Tuple of (info dict, list of issue strings).
              Issues are empty when the file passes all checks.
    """
    info = _image_info_template(path)
    issues: List[str] = []
    try:
        img, data, dtype = load_nifti(path)
    except Exception as exc:
        info["readable"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
        issues.append("image_not_readable")
        return info, issues

    info["readable"] = True
    info["dtype"] = str(dtype)
    info["dimensionality"] = int(img.ndim)
    info["shape"] = to_list(img.shape)

    if img.ndim != 3:
        issues.append("image_dimensionality_not_3")

    spacing = spacing_from_header(img)
    info["spacing"] = to_list(spacing)
    if spacing is None:
        issues.append("image_spacing_missing")

    orientation = orientation_codes(img.affine)
    info["orientation"] = to_list(orientation)
    if orientation is None:
        issues.append("image_orientation_missing")

    try:
        det = float(np.linalg.det(img.affine))
    except Exception as exc:
        det = None
        info["affine_error"] = f"{type(exc).__name__}: {exc}"
    info["affine_determinant"] = det
    if det is None or det == 0.0:
        issues.append("image_affine_singular")

    try:
        info["has_nan"] = bool(np.isnan(data).any())
        info["has_inf"] = bool(np.isinf(data).any())
    except Exception as exc:
        info["nan_inf_error"] = f"{type(exc).__name__}: {exc}"
        issues.append("image_nan_inf_check_failed")

    if info.get("has_nan"):
        issues.append("image_contains_nan")
    if info.get("has_inf"):
        issues.append("image_contains_inf")

    try:
        info["stats"] = safe_stats(data)
        if info["stats"]["min"] is None:
            issues.append("image_stats_no_finite")
    except Exception as exc:
        info["stats_error"] = f"{type(exc).__name__}: {exc}"
        issues.append("image_stats_failed")

    return info, issues


def validate_label(
    path: Optional[Path], image_shape: Optional[Tuple[int, ...]]
) -> Tuple[Dict[str, object], List[str]]:
    """Validate a single NIfTI label file.

    Checks readability, dimensionality, shape consistency with the
    paired image, integer-valued voxels, and non-empty mask.

    :param path: Path to a NIfTI label file, or None if no label exists.
    :param image_shape: Shape of the paired image for shape-match check,
                        or None to skip the check.
    :returns: Tuple of (info dict, list of issue strings).
              Issues are empty when the file passes all checks.
    """
    info = _label_info_template(path)
    issues: List[str] = []
    if path is None:
        issues.append("label_missing")
        return info, issues

    try:
        img, data, _dtype = load_nifti(path)
    except Exception as exc:
        info["readable"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
        issues.append("label_not_readable")
        return info, issues

    info["readable"] = True
    info["dimensionality"] = int(img.ndim)
    info["shape"] = to_list(img.shape)

    if img.ndim != 3:
        issues.append("label_dimensionality_not_3")

    if image_shape is not None:
        matches = tuple(img.shape) == tuple(image_shape)
        info["shape_matches_image"] = matches
        if not matches:
            issues.append("label_shape_mismatch")

    try:
        finite = np.isfinite(data)
        if not np.any(finite):
            info["integer_valued"] = False
            issues.append("label_no_finite_values")
        else:
            vals = data[finite]
            info["integer_valued"] = bool(np.all(vals == np.round(vals)))
            if not info["integer_valued"]:
                issues.append("label_not_integer_valued")
    except Exception as exc:
        info["integer_error"] = f"{type(exc).__name__}: {exc}"
        issues.append("label_integer_check_failed")

    try:
        unique_vals = np.unique(data)
        info["unique_values"] = [int(v) for v in unique_vals.tolist()]
        info["empty"] = bool(np.all(unique_vals == 0))
        if info["empty"]:
            issues.append("label_empty_mask")
    except Exception as exc:
        info["unique_error"] = f"{type(exc).__name__}: {exc}"
        issues.append("label_unique_values_failed")

    return info, issues


def _list_nifti(directory: Path) -> List[Path]:
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
    )


def validate_dataset(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    max_samples: int | None = None,
) -> Dict[str, object]:
    """Validate a directory of NIfTI images (and optionally labels).

    Runs per-file checks on every image and paired label, then aggregates
    a cohort-level summary covering spacing consistency, orientation
    uniformity, missing annotations, and file-level issues.

    :param images_dir: Path to directory containing NIfTI image files.
    :param labels_dir: Path to directory containing NIfTI label files,
                       or None to skip label validation.
    :param max_samples: Limit validation to the first N images (useful
                        for quick sanity checks on large cohorts).
    :returns: Report dict with keys ``summary``, ``files``, and ``meta``.
              ``summary["files_with_issues"]`` is 0 when all files pass.

    Example::

        from neurotk.validate import validate_dataset

        report = validate_dataset("data/images", "data/labels")
        print(f"Files with issues: {report['summary']['files_with_issues']}")
    """
    from .report import build_summary

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir) if labels_dir is not None else None

    image_files = _list_nifti(images_dir)
    if max_samples is not None:
        image_files = image_files[:max(max_samples, 0)]

    label_files: List[Path] = []
    if labels_dir is not None:
        label_files = _list_nifti(labels_dir)
    label_index = {nifti_stem(p.name): p for p in label_files}

    files_report: Dict[str, Dict[str, object]] = {}
    shapes: List[Tuple[int, int, int]] = []
    spacings: List[Tuple[float, float, float]] = []
    orientations: List[Tuple[str, str, str]] = []
    missing_labels: List[str] = []

    for image_path in image_files:
        image_info, image_issues = validate_image(image_path)

        shape = image_info.get("shape")
        image_shape: Optional[Tuple[int, ...]] = None
        if isinstance(shape, list) and len(shape) == 3:
            image_shape = tuple(shape)
            shapes.append(image_shape)
        spacing = image_info.get("spacing")
        if isinstance(spacing, list) and len(spacing) == 3:
            spacings.append(tuple(float(x) for x in spacing))
        orientation = image_info.get("orientation")
        if isinstance(orientation, list) and len(orientation) == 3:
            orientations.append(tuple(str(x) for x in orientation))

        label_info: Optional[Dict[str, object]] = None
        label_issues: List[str] = []
        if labels_dir is not None:
            label_path = label_index.get(nifti_stem(image_path.name))
            if label_path is None:
                missing_labels.append(image_path.name)
            label_info, label_issues = validate_label(label_path, image_shape)

        files_report[image_path.name] = {
            "image": image_info,
            "label": label_info,
            "issues": image_issues + label_issues,
        }

    label_stems = {nifti_stem(p.name) for p in label_files}
    image_stems = {nifti_stem(p.name) for p in image_files}
    missing_images = sorted(label_stems - image_stems)
    files_with_issues = sum(1 for v in files_report.values() if v.get("issues"))

    summary = build_summary(
        image_count=len(image_files),
        label_count=len(label_files),
        missing_labels=missing_labels,
        missing_images=list(missing_images),
        shapes=shapes,
        spacings=spacings,
        orientations=orientations,
        files_with_issues=files_with_issues,
    )

    return {
        "summary": {"scope": "original_inputs", **summary},
        "files": files_report,
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
