from __future__ import annotations

import csv
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


def _nifti_stem(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    return name


def _to_bool(value: str) -> Optional[bool]:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _read_normal_ct_map(normal_csv: Path) -> dict:
    if not normal_csv.exists():
        raise ValueError(f"Normal CT CSV not found: {normal_csv}")
    with normal_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Normal CT CSV must include a header row")
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        id_col = None
        for cand in ("image", "filename", "id", "case", "subject"):
            if cand in cols:
                id_col = cols[cand]
                break
        if id_col is None:
            id_col = reader.fieldnames[0]
        normal_col = None
        for cand in ("normal_ct", "normal", "is_normal"):
            if cand in cols:
                normal_col = cols[cand]
                break
        if normal_col is None:
            raise ValueError(
                "Normal CT CSV must include one of columns: normal_ct, normal, is_normal"
            )

        out = {}
        for row in reader:
            raw_id = (row.get(id_col) or "").strip()
            raw_flag = (row.get(normal_col) or "").strip()
            if not raw_id:
                continue
            is_normal = _to_bool(raw_flag)
            if is_normal is None:
                raise ValueError(f"Invalid normal-CT value '{raw_flag}' for row id '{raw_id}'")
            out[raw_id] = is_normal
            out[_nifti_stem(raw_id)] = is_normal
        return out


def _label_to_binary(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr > 0.5
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0] > 0.5
    raise ValueError(f"Unsupported label shape for lesion volume computation: {arr.shape}")


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


def run_cohort_selection_stats(
    *,
    labels_path: Optional[Path],
    labels_list: Optional[Path],
    normal_csv: Path,
    output_csv: Path,
    summary_csv: Path,
    tn_threshold_ml: float = 0.2,
    low_max_ml: float = 5.0,
    medium_max_ml: float = 20.0,
) -> None:
    labels = _resolve_inputs(labels_path, labels_list)
    if not labels:
        raise ValueError("No label files found")
    normal_map = _read_normal_ct_map(normal_csv)

    rows = []
    for label_path in tqdm(labels, desc="cohort-stats"):
        img = nib.load(os.fspath(label_path))
        arr = img.get_fdata(dtype=np.float32)
        mask = _label_to_binary(arr)
        spacing = img.header.get_zooms()[:3]
        voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
        lesion_voxels = int(np.count_nonzero(mask))
        lesion_volume_ml = (lesion_voxels * voxel_volume_mm3) / 1000.0

        name = label_path.name
        stem = _nifti_stem(name)
        if name in normal_map:
            is_normal_ct = normal_map[name]
        elif stem in normal_map:
            is_normal_ct = normal_map[stem]
        else:
            raise ValueError(
                f"No normal-CT record found for label '{name}'. Add row to {normal_csv}."
            )

        if is_normal_ct and lesion_volume_ml <= tn_threshold_ml:
            cls = "true_negative"
            subgroup = ""
        else:
            cls = "true_positive"
            if lesion_volume_ml < low_max_ml:
                subgroup = "low"
            elif lesion_volume_ml < medium_max_ml:
                subgroup = "medium"
            else:
                subgroup = "high"

        rows.append((name, stem, is_normal_ct, lesion_voxels, lesion_volume_ml, cls, subgroup))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8") as f:
        f.write(
            "image,case_id,normal_ct,lesion_voxels,lesion_volume_ml,classification,tp_volume_subgroup\n"
        )
        for name, stem, is_normal, voxels, vol_ml, cls, subgroup in rows:
            f.write(
                f"{name},{stem},{str(is_normal).lower()},{voxels},{vol_ml:.6f},{cls},{subgroup}\n"
            )

    total = len(rows)
    tn = sum(1 for r in rows if r[5] == "true_negative")
    tp = sum(1 for r in rows if r[5] == "true_positive")
    tp_low = sum(1 for r in rows if r[5] == "true_positive" and r[6] == "low")
    tp_med = sum(1 for r in rows if r[5] == "true_positive" and r[6] == "medium")
    tp_high = sum(1 for r in rows if r[5] == "true_positive" and r[6] == "high")

    metrics = [
        ("total", total),
        ("true_negative", tn),
        ("true_positive", tp),
        ("tp_low", tp_low),
        ("tp_medium", tp_med),
        ("tp_high", tp_high),
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8") as f:
        f.write("metric,count,percent\n")
        for metric, count in metrics:
            pct = (100.0 * count / total) if total else 0.0
            f.write(f"{metric},{count},{pct:.2f}\n")

    print("metric,count,percent")
    for metric, count in metrics:
        pct = (100.0 * count / total) if total else 0.0
        print(f"{metric},{count},{pct:.2f}")
    print(f"Saved cohort classification: {output_csv}")
    print(f"Saved cohort summary: {summary_csv}")


def run_make_normal_ct_flags(
    *,
    labels_path: Optional[Path],
    labels_list: Optional[Path],
    output_csv: Path,
    normal_threshold_ml: float = 0.2,
) -> None:
    labels = _resolve_inputs(labels_path, labels_list)
    if not labels:
        raise ValueError("No label files found")

    rows = []
    for label_path in tqdm(labels, desc="normal-ct-flags"):
        img = nib.load(os.fspath(label_path))
        arr = img.get_fdata(dtype=np.float32)
        mask = _label_to_binary(arr)
        spacing = img.header.get_zooms()[:3]
        voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
        lesion_voxels = int(np.count_nonzero(mask))
        lesion_volume_ml = (lesion_voxels * voxel_volume_mm3) / 1000.0
        normal_ct = lesion_volume_ml <= normal_threshold_ml
        rows.append((label_path.name, _nifti_stem(label_path.name), normal_ct, lesion_volume_ml))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8") as f:
        f.write("image,case_id,normal_ct,lesion_volume_ml\n")
        for image, case_id, normal_ct, vol_ml in rows:
            f.write(f"{image},{case_id},{str(normal_ct).lower()},{vol_ml:.6f}\n")

    n_true = sum(1 for _, _, normal_ct, _ in rows if normal_ct)
    n_false = len(rows) - n_true
    print("metric,count")
    print(f"total,{len(rows)}")
    print(f"normal_ct_true,{n_true}")
    print(f"normal_ct_false,{n_false}")
    print(f"Saved normal CT flags: {output_csv}")


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
