"""Preprocessing transforms for orientation and spacing."""

from __future__ import annotations

from typing import Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage


def affine_spacing(affine: np.ndarray) -> Tuple[float, float, float]:
    """Return voxel spacing derived from affine."""
    spacing = nib.affines.voxel_sizes(affine)
    return (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def reorient_to(
    data: np.ndarray, affine: np.ndarray, target: Tuple[str, str, str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorient data and affine to target axis codes."""
    if data.ndim != 3:
        raise ValueError("Reorientation expects 3D data")
    orig_ornt = nib.orientations.io_orientation(affine)
    target_ornt = nib.orientations.axcodes2ornt(target)
    transform = nib.orientations.ornt_transform(orig_ornt, target_ornt)
    reoriented = nib.orientations.apply_orientation(data, transform)
    new_affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape)
    return reoriented, new_affine


def resample_to_spacing(
    data: np.ndarray,
    affine: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order: int,
    mode: str = "nearest",
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample data to target spacing."""
    if data.ndim != 3:
        raise ValueError("Resampling expects 3D data")
    zoom_factors = tuple(cs / ts for cs, ts in zip(current_spacing, target_spacing))
    if any(z <= 0 for z in zoom_factors):
        raise ValueError("Invalid zoom factors for resampling")
    resampled = ndimage.zoom(data, zoom=zoom_factors, order=order, mode=mode)
    direction = affine[:3, :3] / np.asarray(current_spacing)
    new_affine = affine.copy()
    new_affine[:3, :3] = direction * np.asarray(target_spacing)
    return resampled, new_affine
