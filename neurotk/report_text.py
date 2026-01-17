"""Text rendering utilities for validation summaries."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple


def _format_shape(shape: Optional[List[object]]) -> str:
    if not shape:
        return "unknown"
    return "(" + ", ".join(str(v) for v in shape) + ")"


def _format_spacing(mean: Optional[List[float]], std: Optional[List[float]]) -> str:
    if not mean or not std:
        return "unknown"
    parts = []
    for m, s in zip(mean, std):
        parts.append(f"{m:.3f}±{s:.3f}")
    return "[" + ", ".join(parts) + "]"


def _orientation_consistency(
    modal: Optional[List[str]], deviation: Optional[int]
) -> str:
    if not modal or deviation is None:
        return "unknown"
    if deviation == 0:
        return "consistent (" + ", ".join(modal) + ")"
    return "mixed (modal: " + ", ".join(modal) + ")"


def render_summary_text(report: Dict[str, object]) -> str:
    summary = report.get("summary", {})
    warnings: List[str] = report.get("warnings", [])  # type: ignore[assignment]
    meta: Dict[str, object] = report.get("meta", {})  # type: ignore[assignment]

    timestamp = meta.get("timestamp")
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()
    version = meta.get("version", "unknown")

    num_images = summary.get("num_images", "unknown")
    num_labels = summary.get("num_labels", "unknown")
    files_with_issues = summary.get("files_with_issues", "unknown")
    modal_shape = summary.get("modal_shape")
    spacing_mean = summary.get("spacing_mean")
    spacing_std = summary.get("spacing_std")
    missing_labels = summary.get("missing_label_images")
    missing_label_count = len(missing_labels) if isinstance(missing_labels, list) else "unknown"

    orientation = _orientation_consistency(
        summary.get("orientation_modal"),
        summary.get("orientation_deviation_count"),
    )

    top_warnings = sorted(warnings)[:5]

    lines = [
        "-" * 40,
        "NeuroTK Dataset Validation Summary",
        "-" * 40,
        f"Version           : {version}",
        f"Timestamp         : {timestamp}",
        f"Images            : {num_images}",
        f"Labels            : {num_labels}",
        f"Files w/ issues   : {files_with_issues}",
        f"Orientation       : {orientation}",
        f"Modal shape       : {_format_shape(modal_shape)}",
        f"Spacing (mean±sd) : {_format_spacing(spacing_mean, spacing_std)}",
        f"Missing labels    : {missing_label_count}",
        "",
        "Warnings:",
    ]

    if top_warnings:
        for warning in top_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- none")

    lines.append("-" * 40)
    return "\n".join(lines)
