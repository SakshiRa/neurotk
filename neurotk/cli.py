"""Command-line interface for NeuroTK."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib

from .report import build_summary
from .report_html import write_html_report
from .report_text import render_summary_text
from .stats.image_stats import build_stats_summary
from .utils import nifti_stem, orientation_codes, spacing_from_header
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


def _parse_spacing(value: Optional[str]) -> Optional[List[float]]:
    if not value:
        return None
    parts = value.split()
    if len(parts) != 3:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    if value.lower() in {"1", "true", "yes"}:
        return True
    if value.lower() in {"0", "false", "no"}:
        return False
    return None


def _processed_info(path: Path) -> Dict[str, object]:
    img = nib.load(str(path))
    spacing = spacing_from_header(img)
    orientation = orientation_codes(img.affine)
    return {
        "shape": list(img.shape[:3]),
        "spacing": list(spacing) if spacing is not None else None,
        "orientation": "".join(orientation) if orientation is not None else None,
    }


def _build_summary_from_dir(
    images_dir: Path, labels_dir: Optional[Path]
) -> Dict[str, object]:
    image_files = list_nifti_files(images_dir)
    label_files: List[Path] = []
    if labels_dir is not None and labels_dir.exists():
        label_files = list_nifti_files(labels_dir)
    label_index = _build_label_index(label_files)

    shapes: List[Tuple[int, int, int]] = []
    spacings: List[Tuple[float, float, float]] = []
    orientations: List[Tuple[str, str, str]] = []
    missing_labels: List[str] = []
    files_report: Dict[str, Dict[str, object]] = {}

    for image_path in image_files:
        try:
            img = nib.load(str(image_path))
            spacing = spacing_from_header(img)
            orientation = orientation_codes(img.affine)
        except Exception:
            continue

        shape = img.shape[:3]
        shapes.append(tuple(int(x) for x in shape))
        if spacing is not None:
            spacings.append(tuple(float(x) for x in spacing))
        if orientation is not None:
            orientations.append(tuple(str(x) for x in orientation))

        label_info = None
        if labels_dir is not None:
            stem = nifti_stem(image_path.name)
            label_path = label_index.get(stem)
            if label_path is None:
                missing_labels.append(image_path.name)
            label_info = {
                "present": label_path is not None,
            }

        files_report[image_path.name] = {
            "label": label_info,
        }

    label_stems = {nifti_stem(p.name) for p in label_files}
    image_stems = {nifti_stem(p.name) for p in image_files}
    missing_images = sorted(list(label_stems - image_stems))

    files_with_issues = 0
    if labels_dir is not None:
        files_with_issues = len(missing_labels)

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
    return summary


def _add_preprocess_info(
    report: Dict[str, object],
    image_files: List[Path],
    preprocess_dir: Path,
    spacing_env: Optional[str],
    orientation_env: Optional[str],
    copy_metadata_env: Optional[str],
    labels_provided: bool,
) -> None:
    images_dir = preprocess_dir / "images"
    labels_dir = preprocess_dir / "labels"
    if not images_dir.exists():
        return

    processed_images = {
        nifti_stem(p.name): p for p in list_nifti_files(images_dir)
    }

    processed_labels = {}
    if labels_dir.exists():
        processed_labels = {
            nifti_stem(p.name): p for p in list_nifti_files(labels_dir)
        }

    orientation_changed_any = False
    spacing_changed_any = False
    n_files_modified = 0
    n_files_requested = 0
    n_files_applied = 0
    n_files_noop = 0

    for image_path in image_files:
        entry = report["files"].get(image_path.name)
        if entry is None:
            continue
        original_info = entry.get("image") or {}
        original_orientation = original_info.get("orientation")
        original_block = {
            "shape": original_info.get("shape"),
            "spacing": original_info.get("spacing"),
            "orientation": (
                "".join(original_orientation) if isinstance(original_orientation, list) else None
            ),
        }
        stem = nifti_stem(image_path.name)
        processed_path = processed_images.get(stem)
        processed_block = {"shape": None, "spacing": None, "orientation": None}
        if processed_path is not None:
            try:
                processed_block = _processed_info(processed_path)
            except Exception:
                pass

        label_processed = labels_provided and stem in processed_labels

        n_files_requested += 1
        orientation_changed = original_block.get("orientation") != processed_block.get("orientation")
        spacing_changed = original_block.get("spacing") != processed_block.get("spacing")
        shape_changed = original_block.get("shape") != processed_block.get("shape")
        applied = orientation_changed or spacing_changed or shape_changed
        if applied:
            n_files_modified += 1
            n_files_applied += 1
            if orientation_changed:
                orientation_changed_any = True
            if spacing_changed:
                spacing_changed_any = True
        else:
            n_files_noop += 1

        preprocess_block = {
            "requested": True,
            "applied": applied,
            "label_processed": label_processed,
            "original": original_block,
            "processed": processed_block,
            "verified_by": {
                "orientation_changed": orientation_changed,
                "spacing_changed": spacing_changed,
                "shape_changed": shape_changed,
            },
        }
        if not applied:
            preprocess_block["noop_reason"] = "inputs already matched target parameters"

        entry["preprocess"] = preprocess_block

    summary_processed = _build_summary_from_dir(
        images_dir, labels_dir if labels_provided else None
    )
    report["summary_processed"] = {
        "scope": "processed_outputs",
        **summary_processed,
    }
    report["run_mode"] = "validate+preprocess"
    report["preprocess"] = {
        "enabled": True,
        "inputs": {
            "labels_provided": labels_provided,
        },
        "parameters": {
            "target_spacing": _parse_spacing(spacing_env),
            "target_orientation": orientation_env,
            "copy_metadata": _parse_bool(copy_metadata_env),
        },
        "outputs": {
            "images_dir": "images",
            "labels_dir": "labels" if labels_provided and labels_dir.exists() else None,
        },
        "effects_summary": {
            "n_files_requested": n_files_requested,
            "n_files_applied": n_files_applied,
            "n_files_noop": n_files_noop,
            "orientation_changed": orientation_changed_any,
            "spacing_changed": spacing_changed_any,
            "n_files_modified": n_files_modified,
        },
    }

    if n_files_requested > 0 and n_files_applied == 0:
        report["preprocess"]["valid"] = False
        report["preprocess"]["error"] = "Preprocess requested but no effects detected"


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

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument(
        "--bundle-dir",
        required=False,
        default=None,
        type=str,
        help=(
            "Local MONAI bundle path or HF repo (org/model, hf:org/model, or https://huggingface.co/org/model). "
            "Defaults to NEUROTK_DEFAULT_BUNDLE or UMNSHAMLAB/segresnet."
        ),
    )
    infer_parser.add_argument("--input", default=None, type=Path)
    infer_parser.add_argument("--input-list", default=None, type=Path)
    infer_parser.add_argument("--output-dir", required=True, type=Path)
    infer_parser.add_argument("--device", default=None)
    infer_parser.add_argument("--save-probs", action="store_true")
    infer_parser.add_argument("--labels-dir", default=None, type=Path)
    infer_parser.add_argument("--reference-image", default=None, type=Path)

    dice_parser = subparsers.add_parser("dice")
    dice_parser.add_argument("--preds", default=None, type=Path)
    dice_parser.add_argument("--preds-list", default=None, type=Path)
    dice_parser.add_argument("--labels-dir", required=True, type=Path)
    dice_parser.add_argument("--output", required=True, type=Path)

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

    stats_summary = build_stats_summary(image_files, label_index if labels_dir is not None else None)

    report = {
        "summary": {
            "scope": "original_inputs",
            **summary,
        },
        "stats_summary": stats_summary,
        "files": files_report,
        "warnings": warnings,
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": __version__,
        },
    }

    preprocess_dir_env = os.environ.get("NEUROTK_PREPROCESS_OUTPUT")
    if preprocess_dir_env:
        preprocess_dir = Path(preprocess_dir_env)
        if preprocess_dir.exists():
            labels_env = os.environ.get("NEUROTK_LABELS_PROVIDED")
            labels_provided = labels_dir is not None
            if labels_env is not None:
                labels_provided = labels_env.lower() in {"1", "true", "yes"}
            _add_preprocess_info(
                report,
                image_files,
                preprocess_dir,
                os.environ.get("NEUROTK_PREPROCESS_SPACING"),
                os.environ.get("NEUROTK_PREPROCESS_ORIENTATION"),
                os.environ.get("NEUROTK_PREPROCESS_COPY_METADATA"),
                labels_provided,
            )
            if "preprocess" in report:
                labels_uploaded = os.environ.get("NEUROTK_LABELS_UPLOADED")
                try:
                    uploaded_count = int(labels_uploaded) if labels_uploaded else None
                except ValueError:
                    uploaded_count = None
                report["preprocess"]["inputs"]["num_label_files_uploaded"] = uploaded_count
    else:
        report["run_mode"] = "validate"

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


def _run_infer(args: argparse.Namespace) -> int:
    try:
        from .inference.runner import run_inference
    except ImportError as exc:
        raise SystemExit(
            "Inference dependencies are missing. Install with `pip install neurotk[inference]`."
        ) from exc

    try:
        default_bundle = os.environ.get("NEUROTK_DEFAULT_BUNDLE", "UMNSHAMLAB/segresnet")
        bundle_dir = args.bundle_dir or default_bundle
        run_inference(
            bundle_dir=bundle_dir,
            input_path=args.input,
            input_list=args.input_list,
            output_dir=args.output_dir,
            device=args.device,
            save_probs=args.save_probs,
            labels_dir=args.labels_dir,
            reference_image=args.reference_image,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("Inference complete")
    return 0


def _run_dice(args: argparse.Namespace) -> int:
    try:
        from .inference.runner import run_dice
    except ImportError as exc:
        raise SystemExit(
            "Inference dependencies are missing. Install with `pip install neurotk[inference]`."
        ) from exc

    try:
        run_dice(
            preds_path=args.preds,
            preds_list=args.preds_list,
            labels_dir=args.labels_dir,
            output_csv=args.output,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print("Dice computation complete")
    return 0


def run() -> int:
    args = _parse_args()
    if args.command == "validate":
        return _run_validate(args)
    if args.command == "preprocess":
        return _run_preprocess(args)
    if args.command == "infer":
        return _run_infer(args)
    if args.command == "dice":
        return _run_dice(args)
    raise SystemExit(f"Unknown command: {args.command}")


def main() -> None:
    """CLI entrypoint."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
