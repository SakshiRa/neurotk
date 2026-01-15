"""Dataset-level aggregation and report formatting."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _modal(values: Iterable[Tuple]) -> Optional[Tuple]:
    items = list(values)
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def build_summary(
    image_count: int,
    label_count: int,
    missing_labels: List[str],
    missing_images: List[str],
    shapes: List[Tuple[int, int, int]],
    spacings: List[Tuple[float, float, float]],
    orientations: List[Tuple[str, str, str]],
    files_with_issues: int,
) -> Dict[str, object]:
    modal_shape = _modal(shapes)
    shape_deviation_count = 0
    if modal_shape is not None:
        shape_deviation_count = sum(1 for s in shapes if s != modal_shape)

    spacing_mean = None
    spacing_std = None
    if spacings:
        spacing_arr = np.array(spacings, dtype=float)
        spacing_mean = spacing_arr.mean(axis=0).tolist()
        spacing_std = spacing_arr.std(axis=0).tolist()

    orientation_modal = _modal(orientations)
    orientation_deviation_count = 0
    orientation_counts = None
    if orientations:
        orientation_counts = {
            ",".join(k): v for k, v in Counter(orientations).items()
        }
        if orientation_modal is not None:
            orientation_deviation_count = sum(
                1 for o in orientations if o != orientation_modal
            )

    return {
        "num_images": image_count,
        "num_labels": label_count,
        "missing_label_images": missing_labels,
        "missing_image_labels": missing_images,
        "modal_shape": list(modal_shape) if modal_shape else None,
        "shape_deviation_count": shape_deviation_count,
        "spacing_mean": spacing_mean,
        "spacing_std": spacing_std,
        "orientation_modal": list(orientation_modal) if orientation_modal else None,
        "orientation_counts": orientation_counts,
        "orientation_deviation_count": orientation_deviation_count,
        "files_with_issues": files_with_issues,
    }
