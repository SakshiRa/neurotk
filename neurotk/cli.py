"""Command-line interface for dataset validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .report import build_summary
from .utils import nifti_stem
from .validate import validate_image, validate_label


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
    parser = argparse.ArgumentParser(prog="neurotk validate")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=False, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-samples", required=False, type=int, default=None)
    return parser.parse_args()


def run() -> int:
    args = _parse_args()
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

    report = {"summary": summary, "files": files_report, "warnings": warnings}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Validation complete")
    return 0


def main() -> None:
    """CLI entrypoint."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
