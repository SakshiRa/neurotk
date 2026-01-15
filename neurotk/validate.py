"""Per-file validation logic for NIfTI datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .io import load_nifti
from .utils import orientation_codes, safe_stats, spacing_from_header, to_list


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
    """Validate a single image file."""
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
    """Validate a single label file."""
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
