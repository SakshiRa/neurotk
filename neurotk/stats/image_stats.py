"""Dataset-level image statistics."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from neurotk.io import load_nifti
from neurotk.utils import spacing_from_header


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

    return {
        "shape": _vector_stats(shapes),
        "channels": _scalar_stats(channels),
        "spacing": _vector_stats(spacings),
        "size_mm": _vector_stats(size_mm),
        "intensity": _scalar_stats(intensity_values),
    }


def build_stats_summary(image_files: Sequence[object]) -> Dict[str, object]:
    stats = compute_image_stats(image_files)
    return {
        "n_cases": len(image_files),
        "image_stats": stats,
    }
