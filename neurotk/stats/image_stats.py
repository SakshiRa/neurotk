"""Dataset-level image statistics."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Optional
from pathlib import Path

import numpy as np

from neurotk.io import load_nifti
from neurotk.utils import spacing_from_header, nifti_stem


_PERCENTILES = (0.5, 10.0, 90.0, 99.5)
_ROUND = 6


def _round_float(value: float) -> float:
    return float(np.round(value, _ROUND))


def _safe_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _scalar_stats(values: Sequence[float]) -> Dict[str, object]:
    arr = _safe_array(values)
    if arr.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
            "percentiles": {
                "p0_5": None,
                "p10": None,
                "p90": None,
                "p99_5": None,
            },
        }

    percentiles = np.percentile(arr, _PERCENTILES)
    return {
        "min": _round_float(float(np.min(arr))),
        "max": _round_float(float(np.max(arr))),
        "mean": _round_float(float(np.mean(arr))),
        "median": _round_float(float(np.median(arr))),
        "stdev": _round_float(float(np.std(arr))),
        "percentiles": {
            "p0_5": _round_float(float(percentiles[0])),
            "p10": _round_float(float(percentiles[1])),
            "p90": _round_float(float(percentiles[2])),
            "p99_5": _round_float(float(percentiles[3])),
        },
    }


def _intensity_hint(stats: Dict[str, object]) -> str:
    percentiles = stats.get("percentiles", {})
    p0_5 = percentiles.get("p0_5")
    p99_5 = percentiles.get("p99_5")
    min_v = stats.get("min")
    max_v = stats.get("max")
    if all(v is not None for v in (p0_5, p99_5, min_v, max_v)):
        if min_v >= 0.0 and max_v <= 1.0 and p99_5 <= 1.0:
            return "binary_or_normalized"
        if (p99_5 - p0_5) > 100:
            return "wide_dynamic_range"
    return "unknown"


def _vector_stats(values: Sequence[Sequence[float]]) -> Dict[str, object]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
            "percentiles": {
                "p0_5": None,
                "p10": None,
                "p90": None,
                "p99_5": None,
            },
        }

    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    means = np.mean(arr, axis=0)
    medians = np.median(arr, axis=0)
    stdevs = np.std(arr, axis=0)
    percentiles = np.percentile(arr, _PERCENTILES, axis=0)

    def _vec(values: np.ndarray) -> List[float]:
        return [_round_float(float(v)) for v in values.tolist()]

    return {
        "min": _vec(mins),
        "max": _vec(maxs),
        "mean": _vec(means),
        "median": _vec(medians),
        "stdev": _vec(stdevs),
        "percentiles": {
            "p0_5": _vec(percentiles[0]),
            "p10": _vec(percentiles[1]),
            "p90": _vec(percentiles[2]),
            "p99_5": _vec(percentiles[3]),
        },
    }


def compute_image_stats(image_files: Sequence[object]) -> Dict[str, object]:
    shapes: List[Tuple[int, int, int]] = []
    spacings: List[Tuple[float, float, float]] = []
    size_mm: List[Tuple[float, float, float]] = []
    channels: List[int] = []
    intensities: List[np.ndarray] = []

    for path_obj in image_files:
        try:
            path = str(path_obj)
            img, data, _dtype = load_nifti(path_obj)
        except Exception:
            continue

        if img.ndim < 3:
            continue

        shape = img.shape[:3]
        shapes.append((int(shape[0]), int(shape[1]), int(shape[2])))

        spacing = spacing_from_header(img)
        if spacing is None:
            continue

        spacings.append((float(spacing[0]), float(spacing[1]), float(spacing[2])))
        size_mm.append(
            (
                float(shape[0]) * float(spacing[0]),
                float(shape[1]) * float(spacing[1]),
                float(shape[2]) * float(spacing[2]),
            )
        )

        channel_count = 1
        if img.ndim > 3:
            channel_count = int(np.prod(img.shape[3:]))
        channels.append(channel_count)

        finite = np.isfinite(data)
        if np.any(finite):
            intensities.append(data[finite].ravel())

    intensity_values: np.ndarray
    if intensities:
        intensity_values = np.concatenate(intensities, axis=0)
    else:
        intensity_values = np.asarray([], dtype=float)

    intensity_stats = _scalar_stats(intensity_values)
    intensity_stats["value_range_hint"] = _intensity_hint(intensity_stats)
    return {
        "shape": _vector_stats(shapes),
        "channels": _scalar_stats(channels),
        "spacing": _vector_stats(spacings),
        "size_mm": _vector_stats(size_mm),
        "intensity": intensity_stats,
    }

def compute_label_stats(
    image_files: Sequence[object],
    label_index: Dict[str, object],
) -> Dict[str, object]:
    label_ids: List[int] = []
    cases: List[Tuple[object, object]] = []

    for image_path in image_files:
        stem = nifti_stem(Path(str(image_path)).name)
        label_path = label_index.get(stem)
        if label_path is None:
            continue
        try:
            label_img, label_data, _ = load_nifti(label_path)
        except Exception:
            continue
        if label_img.ndim != 3:
            continue
        cases.append((image_path, label_path))
        unique_vals = np.unique(label_data)
        label_ids.extend(int(v) for v in unique_vals.tolist())

    if not cases:
        return {}

    unique_labels = sorted(set(label_ids))
    per_label_percentages: Dict[int, List[float]] = {lbl: [] for lbl in unique_labels}
    per_label_intensities: Dict[int, List[np.ndarray]] = {lbl: [] for lbl in unique_labels}
    foreground_intensities: List[np.ndarray] = []

    for image_path, label_path in cases:
        try:
            img, img_data, _ = load_nifti(image_path)
            label_img, label_data, _ = load_nifti(label_path)
        except Exception:
            continue
        if img.ndim != 3 or label_img.ndim != 3:
            continue
        if img.shape[:3] != label_img.shape[:3]:
            continue

        total_voxels = float(np.prod(img.shape[:3]))
        if total_voxels <= 0:
            continue

        for lbl in unique_labels:
            mask = label_data == lbl
            pct = float(np.sum(mask)) / total_voxels * 100.0
            per_label_percentages[lbl].append(pct)
            if np.any(mask):
                per_label_intensities[lbl].append(img_data[mask].ravel())
        fg_mask = label_data > 0
        if np.any(fg_mask):
            foreground_intensities.append(img_data[fg_mask].ravel())

    per_label: Dict[str, object] = {}
    for lbl in unique_labels:
        pct_stats = _scalar_stats(per_label_percentages[lbl])
        if per_label_intensities[lbl]:
            intensities = np.concatenate(per_label_intensities[lbl], axis=0)
        else:
            intensities = np.asarray([], dtype=float)
        intensity_stats = _scalar_stats(intensities)
        per_label[str(lbl)] = {
            "foreground_percentage": pct_stats,
            "image_intensity": intensity_stats,
        }

    if foreground_intensities:
        fg_intensity_vals = np.concatenate(foreground_intensities, axis=0)
    else:
        fg_intensity_vals = np.asarray([], dtype=float)

    fg_stats = _scalar_stats(fg_intensity_vals)
    fg_stats["value_range_hint"] = _intensity_hint(fg_stats)
    return {
        "image_foreground_stats": {
            "intensity": fg_stats,
        },
        "label_stats": {
            "labels": unique_labels,
            "per_label": per_label,
        },
    }


def build_stats_summary(
    image_files: Sequence[object],
    label_index: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    stats = compute_image_stats(image_files)
    summary: Dict[str, object] = {
        "n_cases": len(image_files),
        "image_stats": stats,
    }
    if label_index:
        label_stats = compute_label_stats(image_files, label_index)
        summary.update(label_stats)
    return summary
