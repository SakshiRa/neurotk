"""Dataset preprocessing logic."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from .io import load_nifti
from .transforms import affine_spacing, reorient_to, resample_to_spacing
from .utils import nifti_stem, orientation_codes, spacing_from_header, to_list
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


def _parse_orientation(orientation: str) -> Tuple[str, str, str]:
    orientation = orientation.strip().upper()
    if len(orientation) != 3:
        raise ValueError("Orientation must be a 3-letter code")
    return (orientation[0], orientation[1], orientation[2])


def _spacing_tuple(spacing: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (float(spacing[0]), float(spacing[1]), float(spacing[2]))


def _spacing_close(
    spacing: Tuple[float, float, float], target: Tuple[float, float, float]
) -> bool:
    return bool(np.allclose(spacing, target, rtol=1e-5, atol=1e-5))


def _anisotropic(spacing: Tuple[float, float, float]) -> bool:
    minimum = min(spacing)
    if minimum <= 0:
        return False
    return max(spacing) / minimum >= 3.0


def _yaml_config(args: Dict[str, object]) -> str:
    lines = [
        "cli_args:",
        f"  images: {args['images']}",
        f"  labels: {args['labels']}",
        f"  out: {args['out']}",
        f"  spacing: [{args['spacing'][0]}, {args['spacing'][1]}, {args['spacing'][2]}]",
        f"  orientation: {args['orientation']}",
        f"  dry_run: {str(args['dry_run']).lower()}",
        f"  copy_metadata: {str(args['copy_metadata']).lower()}",
        f"neurotk_version: {args['version']}",
        f"timestamp: {args['timestamp']}",
    ]
    return "\n".join(lines) + "\n"


def _save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    header: Optional[nib.Nifti1Header],
    path: Path,
) -> None:
    img = nib.Nifti1Image(data, affine, header=header)
    img.update_header()
    nib.save(img, str(path))


def _process_image(
    path: Path,
    target_spacing: Tuple[float, float, float],
    target_orientation: Tuple[str, str, str],
    copy_metadata: bool,
    order: int,
    is_label: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], List[str]]:
    img, data, dtype = load_nifti(path)
    info: Dict[str, object] = {
        "path": str(path),
        "original_spacing": to_list(spacing_from_header(img)),
        "original_orientation": to_list(orientation_codes(img.affine)),
        "new_spacing": None,
        "new_orientation": None,
        "interpolation": "nearest" if is_label else "linear",
        "transforms": [],
    }
    warnings: List[str] = []

    if info["original_orientation"] is None:
        warnings.append("orientation_missing")

    if info["original_spacing"] is None:
        warnings.append("spacing_missing")

    data_out = data
    affine_out = img.affine
    if info["original_orientation"] is not None:
        original_orientation = tuple(info["original_orientation"])
        if original_orientation != target_orientation:
            data_out, affine_out = reorient_to(
                data_out, affine_out, target_orientation
            )
            info["transforms"].append("reorient")

    current_spacing = affine_spacing(affine_out)
    if _anisotropic(current_spacing):
        warnings.append("input_spacing_anisotropic")

    if _anisotropic(target_spacing):
        warnings.append("target_spacing_anisotropic")

    if not _spacing_close(current_spacing, target_spacing):
        data_out, affine_out = resample_to_spacing(
            data_out, affine_out, current_spacing, target_spacing, order
        )
        info["transforms"].append("resample")

    if is_label:
        if np.issubdtype(dtype, np.integer):
            data_out = np.rint(data_out).astype(dtype)
        else:
            data_out = np.rint(data_out).astype(np.int32)

    info["new_spacing"] = to_list(affine_spacing(affine_out))
    info["new_orientation"] = to_list(orientation_codes(affine_out))

    return data_out, affine_out, info, warnings


def preprocess_dataset(
    images_dir: Path,
    labels_dir: Optional[Path],
    out_dir: Path,
    spacing: Tuple[float, float, float],
    orientation: str = "RAS",
    dry_run: bool = False,
    copy_metadata: bool = False,
) -> Dict[str, object]:
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")

    image_files = list_nifti_files(images_dir)
    label_files: List[Path] = []
    if labels_dir is not None:
        if not labels_dir.exists() or not labels_dir.is_dir():
            raise SystemExit(f"Labels directory not found: {labels_dir}")
        label_files = list_nifti_files(labels_dir)

    label_index = _build_label_index(label_files)
    target_orientation = _parse_orientation(orientation)
    target_spacing = _spacing_tuple(spacing)

    files_report: Dict[str, Dict[str, object]] = {}
    processed: List[str] = []

    images_out_dir = out_dir / "images"
    labels_out_dir = out_dir / "labels"
    if not dry_run:
        images_out_dir.mkdir(parents=True, exist_ok=True)
        if labels_dir is not None:
            labels_out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_files:
        entry: Dict[str, object] = {
            "image": None,
            "label": None,
            "warnings": [],
            "errors": [],
        }
        try:
            data_out, affine_out, info, warnings = _process_image(
                image_path,
                target_spacing,
                target_orientation,
                copy_metadata,
                order=1,
                is_label=False,
            )
            entry["image"] = info
            entry["warnings"].extend(warnings)
            if not dry_run:
                header = None
                if copy_metadata:
                    header = nib.load(str(image_path)).header.copy()
                _save_nifti(
                    data_out,
                    affine_out,
                    header,
                    images_out_dir / image_path.name,
                )
            processed.append(image_path.name)
        except Exception as exc:
            entry["errors"].append(f"image_error: {type(exc).__name__}: {exc}")
            files_report[image_path.name] = entry
            continue

        if labels_dir is not None:
            stem = nifti_stem(image_path.name)
            label_path = label_index.get(stem)
            if label_path is None:
                entry["warnings"].append("label_missing")
            else:
                try:
                    label_data, label_affine, label_info, label_warnings = _process_image(
                        label_path,
                        target_spacing,
                        target_orientation,
                        copy_metadata,
                        order=0,
                        is_label=True,
                    )
                    entry["label"] = label_info
                    entry["warnings"].extend(label_warnings)
                    if not dry_run:
                        header = None
                        if copy_metadata:
                            header = nib.load(str(label_path)).header.copy()
                        _save_nifti(
                            label_data,
                            label_affine,
                            header,
                            labels_out_dir / label_path.name,
                        )
                except Exception as exc:
                    entry["errors"].append(
                        f"label_error: {type(exc).__name__}: {exc}"
                    )

        files_report[image_path.name] = entry

    report = {
        "processed_files": processed,
        "files": files_report,
    }

    config = {
        "images": str(images_dir),
        "labels": str(labels_dir) if labels_dir is not None else "null",
        "out": str(out_dir),
        "spacing": [target_spacing[0], target_spacing[1], target_spacing[2]],
        "orientation": "".join(target_orientation),
        "dry_run": dry_run,
        "copy_metadata": copy_metadata,
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "preprocess_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        config_path = out_dir / "preprocess_config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            f.write(_yaml_config(config))
    else:
        report["config"] = config

    return report
