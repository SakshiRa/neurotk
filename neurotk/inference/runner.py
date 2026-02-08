from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import nibabel as nib
import numpy as np
import torch
from monai.transforms import Resize
from tqdm import tqdm

from .config import resolve_bundle_dir
from .io_utils import load_nifti, save_nifti, to_uint8_mask
from .metrics import compute_metrics
from .predictor import BundlePredictor


def _read_input_list(path: Path) -> List[Path]:
    items: List[Path] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(Path(line))
    return items


def _is_nifti(path: Path) -> bool:
    return path.name.lower().endswith((".nii", ".nii.gz"))


def _resolve_inputs(input_path: Optional[Path], input_list: Optional[Path]) -> List[Path]:
    if input_path and input_list:
        raise ValueError("Provide either --input or --input_list, not both")
    if input_list:
        return _read_input_list(input_list)
    if input_path:
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")
        if input_path.is_dir():
            files = [
                p for p in input_path.iterdir() if p.is_file() and _is_nifti(p)
            ]
            return sorted(files)
        return [input_path]
    raise ValueError("Provide --input or --input_list")


def run_dice(
    *,
    preds_path: Optional[Path],
    preds_list: Optional[Path],
    labels_dir: Path,
    output_csv: Path,
) -> None:
    if labels_dir is None or not labels_dir.exists():
        raise ValueError("Provide a valid --labels-dir")
    preds = _resolve_inputs(preds_path, preds_list)
    if not preds:
        raise ValueError("No prediction files found")

    metrics = []
    for pred_path in tqdm(preds, desc="dice"):
        label_path = labels_dir / pred_path.name
        if not label_path.exists():
            alt_name = pred_path.name.replace("_seg.nii.gz", ".nii.gz").replace("_seg.nii", ".nii")
            label_path = labels_dir / alt_name
        if not label_path.exists():
            continue
        pred_np, _ = load_nifti(pred_path)
        label_np, _ = load_nifti(label_path)
        pred_mask = torch.as_tensor(pred_np > 0.5)
        label_mask = torch.as_tensor(label_np > 0.5)
        if pred_mask.shape != label_mask.shape:
            resize = Resize(spatial_size=pred_mask.shape, mode="nearest")
            label_mask = resize(label_mask.unsqueeze(0).unsqueeze(0))[0, 0]
        d, h = compute_metrics(pred_mask, label_mask)
        metrics.append((pred_path.name, d, h))

    if not metrics:
        raise ValueError("No matching label files found for predictions")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8") as f:
        f.write("image,dice,hausdorff95\n")
        for name, d, h in metrics:
            h_str = "" if h is None else f"{h:.4f}"
            f.write(f"{name},{d:.4f},{h_str}\n")
        mean_dice = sum(m[1] for m in metrics) / len(metrics)
        f.write(f"mean_dice,{mean_dice:.4f},\n")
    print("image,dice,hausdorff95")
    for name, d, h in metrics:
        h_str = "" if h is None else f"{h:.4f}"
        print(f"{name},{d:.4f},{h_str}")
    mean_dice = sum(m[1] for m in metrics) / len(metrics)
    print(f"mean_dice,{mean_dice:.4f}")


def _to_binary_mask(arr: np.ndarray, threshold: float) -> np.ndarray:
    if arr.ndim == 3:
        return arr > threshold
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            return arr[0] > threshold
        labels = np.argmax(arr, axis=0)
        return labels > 0
    raise ValueError(f"Unsupported prediction array shape for volume computation: {arr.shape}")


def _volume_range_summary_rows(values_ml: List[float]):
    ranges = [
        ("0 mL", lambda x: x == 0.0),
        (">0 to <1 mL", lambda x: 0.0 < x < 1.0),
        ("1 to <5 mL", lambda x: 1.0 <= x < 5.0),
        ("5 to <10 mL", lambda x: 5.0 <= x < 10.0),
        ("10 to <20 mL", lambda x: 10.0 <= x < 20.0),
        ("20 to <50 mL", lambda x: 20.0 <= x < 50.0),
        ("50+ mL", lambda x: x >= 50.0),
    ]
    total = len(values_ml)
    rows = []
    for label, predicate in ranges:
        count = sum(1 for value in values_ml if predicate(value))
        percent = (100.0 * count / total) if total else 0.0
        rows.append((label, count, percent))
    return rows


def _volume_overall_stats(values_ml: List[float]):
    if not values_ml:
        return {
            "total_images": 0.0,
            "min_ml": 0.0,
            "p25_ml": 0.0,
            "median_ml": 0.0,
            "p75_ml": 0.0,
            "max_ml": 0.0,
            "mean_ml": 0.0,
        }
    arr = np.asarray(values_ml, dtype=np.float64)
    return {
        "total_images": float(arr.size),
        "min_ml": float(np.min(arr)),
        "p25_ml": float(np.quantile(arr, 0.25)),
        "median_ml": float(np.quantile(arr, 0.50)),
        "p75_ml": float(np.quantile(arr, 0.75)),
        "max_ml": float(np.max(arr)),
        "mean_ml": float(np.mean(arr)),
    }


def run_lesion_volume(
    *,
    preds_path: Optional[Path],
    preds_list: Optional[Path],
    output_csv: Path,
    threshold: float = 0.5,
    histogram_path: Optional[Path] = None,
    hist_bins: int = 30,
    summary_csv: Optional[Path] = None,
) -> None:
    preds = _resolve_inputs(preds_path, preds_list)
    if not preds:
        raise ValueError("No prediction files found")

    rows = []
    for pred_path in tqdm(preds, desc="lesion-volume"):
        img = nib.load(os.fspath(pred_path))
        arr = img.get_fdata(dtype=np.float32)
        mask = _to_binary_mask(arr, threshold)
        lesion_voxels = int(np.count_nonzero(mask))
        spacing = img.header.get_zooms()[:3]
        voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
        lesion_volume_mm3 = float(lesion_voxels * voxel_volume_mm3)
        lesion_volume_ml = lesion_volume_mm3 / 1000.0
        rows.append(
            (
                pred_path.name,
                lesion_voxels,
                voxel_volume_mm3,
                lesion_volume_mm3,
                lesion_volume_ml,
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8") as f:
        f.write("image,lesion_voxels,voxel_volume_mm3,lesion_volume_mm3,lesion_volume_ml\n")
        for image_name, voxels, voxel_mm3, vol_mm3, vol_ml in rows:
            f.write(f"{image_name},{voxels},{voxel_mm3:.6f},{vol_mm3:.6f},{vol_ml:.6f}\n")

    if summary_csv is None:
        summary_csv = output_csv.with_name(f"{output_csv.stem}_summary.csv")
    summary_rows = _volume_range_summary_rows([r[4] for r in rows])
    overall_stats = _volume_overall_stats([r[4] for r in rows])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8") as f:
        f.write("category,metric,count,percent,value_ml\n")
        for label, count, percent in summary_rows:
            f.write(f"range,{label},{count},{percent:.2f},\n")
        for metric, value in overall_stats.items():
            if metric == "total_images":
                f.write(f"overall,{metric},{int(value)},,\n")
            else:
                f.write(f"overall,{metric},,,{value:.6f}\n")

    if histogram_path is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        histogram_path.parent.mkdir(parents=True, exist_ok=True)
        vals_ml = [r[4] for r in rows]
        fig = plt.figure(figsize=(8, 5))
        plt.hist(vals_ml, bins=max(int(hist_bins), 1), edgecolor="black", alpha=0.85)
        plt.xlabel("Lesion Volume (mL)")
        plt.ylabel("Count")
        plt.title("Lesion Volume Distribution")
        plt.tight_layout()
        fig.savefig(histogram_path, dpi=180)
        plt.close(fig)

    print("image,lesion_voxels,voxel_volume_mm3,lesion_volume_mm3,lesion_volume_ml")
    for image_name, voxels, voxel_mm3, vol_mm3, vol_ml in rows:
        print(f"{image_name},{voxels},{voxel_mm3:.6f},{vol_mm3:.6f},{vol_ml:.6f}")
    print(f"Saved lesion volume summary: {summary_csv}")


def _infer_output_path(output_dir: Path, image_path: Path, suffix: str) -> Path:
    base = image_path.name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    return output_dir / f"{base}{suffix}.nii.gz"


def _prepare_pred(pred: torch.Tensor, save_probs: bool) -> np.ndarray:
    if pred.ndim >= 4:
        if save_probs:
            return pred.float().cpu().numpy()
        if pred.shape[0] == 1:
            return pred.squeeze(0).cpu().numpy()
        return torch.argmax(pred, dim=0).cpu().numpy()
    return pred.cpu().numpy()


def _effective_device_name(device: Optional[str]) -> str:
    if device is not None:
        try:
            return str(torch.device(device))
        except Exception:
            return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def run_inference(
    *,
    bundle_dir: Union[str, Path],
    input_path: Optional[Path],
    input_list: Optional[Path],
    output_dir: Path,
    device: Optional[str],
    save_probs: bool,
    force: bool,
    skip_invalid_inputs: bool,
    labels_dir: Optional[Path],
    reference_image: Optional[Path],
) -> None:
    bundle_dir = Path(resolve_bundle_dir(os.fspath(bundle_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_device = _effective_device_name(device)
    if effective_device == "cpu":
        print(
            "Warning: running inference on CPU. This may be slow; use --device cuda or --device mps if available.",
            file=sys.stderr,
        )
    if labels_dir is not None and (not labels_dir.exists() or not labels_dir.is_dir()):
        labels_dir = None

    predictor = BundlePredictor(bundle_dir=os.fspath(bundle_dir), device=device)
    inputs = _resolve_inputs(input_path, input_list)

    metrics = []
    dice_report_path = output_dir / "dice_scores.csv"
    skipped_invalid_path = output_dir / "skipped_inputs.csv"
    skipped_existing = 0
    skipped_invalid = []
    for image_path in tqdm(inputs, desc="inference"):
        out_path = _infer_output_path(output_dir, image_path, "_prob" if save_probs else "_seg")
        if not force and out_path.exists():
            skipped_existing += 1
            continue
        try:
            pred, meta = predictor.predict_volume(os.fspath(image_path))
            pred_arr = _prepare_pred(pred, save_probs)
            if not save_probs:
                if pred_arr.dtype != np.uint8:
                    pred_arr = (pred_arr > 0.5).astype(np.uint8)
                else:
                    pred_arr = to_uint8_mask(pred_arr)
            affine = None
            header = None
            if isinstance(meta, dict):
                if "affine" in meta and meta["affine"] is not None:
                    affine = meta["affine"]
                elif "original_affine" in meta and meta["original_affine"] is not None:
                    affine = meta["original_affine"]
            if reference_image is not None:
                ref = nib.load(os.fspath(reference_image))
                affine = ref.affine
                header = ref.header
            if affine is None:
                img = nib.load(os.fspath(image_path))
                affine = img.affine
                header = img.header
            save_nifti(pred_arr, affine, out_path, header=header)

            if labels_dir:
                label_path = _infer_output_path(labels_dir, image_path, "")
                if not label_path.exists():
                    label_path = _infer_output_path(labels_dir, image_path, "_seg")
                if label_path.exists():
                    label_np, _ = load_nifti(label_path)
                    pred_mask = torch.as_tensor(pred_arr > 0.5)
                    label_mask = torch.as_tensor(label_np > 0.5)
                    if pred_mask.shape != label_mask.shape:
                        resize = Resize(spatial_size=pred_mask.shape, mode="nearest")
                        label_mask = resize(label_mask.unsqueeze(0).unsqueeze(0))[0, 0]
                    d, h = compute_metrics(pred_mask, label_mask)
                    metrics.append((image_path.name, d, h))
        except Exception as exc:
            if not skip_invalid_inputs:
                raise
            skipped_invalid.append((image_path.name, f"{type(exc).__name__}: {exc}"))
            print(
                f"Skipping invalid input {image_path.name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue

    if metrics:
        with dice_report_path.open("w", encoding="utf-8") as f:
            f.write("image,dice,hausdorff95\n")
            for name, d, h in metrics:
                h_str = "" if h is None else f"{h:.4f}"
                f.write(f"{name},{d:.4f},{h_str}\n")
            mean_dice = sum(m[1] for m in metrics) / len(metrics)
            f.write(f"mean_dice,{mean_dice:.4f},\n")
        print("image,dice,hausdorff95")
        for name, d, h in metrics:
            h_str = "" if h is None else f"{h:.4f}"
            print(f"{name},{d:.4f},{h_str}")
        mean_dice = sum(m[1] for m in metrics) / len(metrics)
        print(f"mean_dice,{mean_dice:.4f}")
    elif dice_report_path.exists():
        dice_report_path.unlink()

    if skipped_existing > 0:
        print(
            f"Skipped {skipped_existing} input(s) with existing outputs. Use --force to recompute.",
            file=sys.stderr,
        )

    if skipped_invalid:
        with skipped_invalid_path.open("w", encoding="utf-8") as f:
            f.write("image,error\n")
            for image_name, error in skipped_invalid:
                safe_error = error.replace("\n", " ").replace("\r", " ").replace(",", ";")
                f.write(f"{image_name},{safe_error}\n")
        print(
            f"Skipped {len(skipped_invalid)} invalid input(s). Details: {skipped_invalid_path}",
            file=sys.stderr,
        )
