"""Utility helpers for validation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np


def nifti_stem(path_name: str) -> str:
    """Return filename stem without NIfTI extensions."""
    if path_name.endswith(".nii.gz"):
        return path_name[:-7]
    if path_name.endswith(".nii"):
        return path_name[:-4]
    return path_name.rsplit(".", 1)[0]


def orientation_codes(affine: np.ndarray) -> Optional[Tuple[str, str, str]]:
    """Return orientation codes from affine, or None on failure."""
    try:
        codes = nib.orientations.aff2axcodes(affine)
    except Exception:
        return None
    return codes


def safe_stats(data: np.ndarray) -> Dict[str, Optional[float]]:
    """Compute stats on finite values; return None when unavailable."""
    finite = np.isfinite(data)
    if not np.any(finite):
        return {"min": None, "max": None, "mean": None, "std": None}
    vals = data[finite]
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def spacing_from_header(img: nib.Nifti1Image) -> Optional[Tuple[float, float, float]]:
    """Extract voxel spacing for the first three axes."""
    zooms = img.header.get_zooms()
    if len(zooms) < 3:
        return None
    return (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def to_list(values: Optional[Iterable]) -> Optional[List]:
    """Convert iterable to list, preserving None."""
    if values is None:
        return None
    return list(values)
