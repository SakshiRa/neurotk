"""Command-line interface for NeuroTK."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .report import build_summary
from .report_html import write_html_report
from .report_text import render_summary_text
from .stats.image_stats import build_stats_summary
from .utils import nifti_stem
from .preprocess import preprocess_dataset
from .validate import validate_image, validate_label
from . import __version__


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def list_nifti_files(directory: Path) -> List[Path]:
    return sorted([p for p in directory.rglob("*") if p.is_file() and _is_nifti(p)])


def _build_label_index(label_files: List[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in label_files:
        index[nifti_stem(path.name)] = path
    return index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="neurotk")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--images", required=True, type=Path)
    validate_parser.add_argument("--labels", required=False, type=Path)
    validate_parser.add_argument("--out", required=True, type=Path)
    validate_parser.add_argument("--max-samples", required=False, type=int, default=None)
    validate_parser.add_argument("--html", required=False, type=Path)
    validate_parser.add_argument("--summary-only", action="store_true")

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--images", required=True, type=Path)
    preprocess_parser.add_argument("--labels", required=False, type=Path)
    preprocess_parser.add_argument("--out", required=True, type=Path)
    preprocess_parser.add_argument("--spacing", required=True, type=float, nargs=3)
    preprocess_parser.add_argument("--orientation", default="RAS")
    preprocess_parser.add_argument("--dry-run", action="store_true")
    preprocess_parser.add_argument("--copy-metadata", action="store_true")

    return parser.parse_args()


def _run_validate(args: argparse.Namespace) -> int:
    images_dir: Path = args.images
    labels_dir: Optional[Path] = args.labels

    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    image_files = list_nifti_files(images_dir)
    if args.max_samples is not None:
        image_files = image_files[: max(args.max_samples, 0)]

    label_files: List[Path] = []
    if labels_dir is not None:
        if not labels_dir.exists() or not labels_dir.is_dir():
            raise SystemExit(f"Labels directory not found: {labels_dir}")
        label_files = list_nifti_files(labels_dir)

    label_index = _build_label_index(label_files)

    files_report: Dict[str, Dict[str, object]] = {}
    warnings: List[str] = []
    shapes: List[Tuple[int, int, int]] = []
    spacings: List[Tuple[float, float, float]] = []
    orientations: List[Tuple[str, str, str]] = []
    missing_labels: List[str] = []

    for image_path in image_files:
        image_info, image_issues = validate_image(image_path)

        shape = image_info.get("shape")
        image_shape = None
        if isinstance(shape, list) and len(shape) == 3:
            image_shape = tuple(shape)
            shapes.append(image_shape)
        spacing = image_info.get("spacing")
        if isinstance(spacing, list) and len(spacing) == 3:
            spacings.append(tuple(float(x) for x in spacing))
        orientation = image_info.get("orientation")
        if isinstance(orientation, list) and len(orientation) == 3:
            orientations.append(tuple(str(x) for x in orientation))

        label_info = None
        label_issues: List[str] = []
        if labels_dir is not None:
            stem = nifti_stem(image_path.name)
            label_path = label_index.get(stem)
            if label_path is None:
                missing_labels.append(image_path.name)
            label_info, label_issues = validate_label(
                label_path, image_shape
            )

        issues = image_issues + label_issues
        files_report[image_path.name] = {
            "image": image_info,
            "label": label_info,
            "issues": issues,
        }

    label_stems = {nifti_stem(p.name) for p in label_files}
    image_stems = {nifti_stem(p.name) for p in image_files}
    missing_images = sorted(list(label_stems - image_stems))

    files_with_issues = sum(
        1 for v in files_report.values() if v.get("issues")
    )

    summary = build_summary(
        image_count=len(image_files),
        label_count=len(label_files),
        missing_labels=missing_labels,
        missing_images=missing_images,
        shapes=shapes,
        spacings=spacings,
        orientations=orientations,
        files_with_issues=files_with_issues,
    )

    stats_summary = build_stats_summary(image_files)

    report = {
        "summary": summary,
        "stats_summary": stats_summary,
        "files": files_report,
        "warnings": warnings,
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": __version__,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.html is not None:
        try:
            write_html_report(report, args.html)
        except Exception as exc:
            print(
                f"Warning: HTML report generation failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if args.summary_only:
        try:
            print(render_summary_text(report))
        except Exception as exc:
            print(
                f"Warning: summary rendering failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    print("Validation complete")
    return 0


def _run_preprocess(args: argparse.Namespace) -> int:
    preprocess_dataset(
        images_dir=args.images,
        labels_dir=args.labels,
        out_dir=args.out,
        spacing=tuple(args.spacing),
        orientation=args.orientation,
        dry_run=args.dry_run,
        copy_metadata=args.copy_metadata,
    )
    print("Preprocess complete")
    return 0


def run() -> int:
    args = _parse_args()
    if args.command == "validate":
        return _run_validate(args)
    if args.command == "preprocess":
        return _run_preprocess(args)
    raise SystemExit(f"Unknown command: {args.command}")


def main() -> None:
    """CLI entrypoint."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
