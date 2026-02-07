from __future__ import annotations

import os
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


def run_inference(
    *,
    bundle_dir: Union[str, Path],
    input_path: Optional[Path],
    input_list: Optional[Path],
    output_dir: Path,
    device: Optional[str],
    save_probs: bool,
    labels_dir: Optional[Path],
    reference_image: Optional[Path],
) -> None:
    bundle_dir = Path(resolve_bundle_dir(os.fspath(bundle_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    if labels_dir is not None and (not labels_dir.exists() or not labels_dir.is_dir()):
        labels_dir = None

    predictor = BundlePredictor(bundle_dir=os.fspath(bundle_dir), device=device)
    inputs = _resolve_inputs(input_path, input_list)

    metrics = []
    dice_report_path = output_dir / "dice_scores.csv"
    for image_path in tqdm(inputs, desc="inference"):
        pred, meta = predictor.predict_volume(os.fspath(image_path))
        out_path = _infer_output_path(output_dir, image_path, "_prob" if save_probs else "_seg")
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
