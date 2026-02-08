from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from neurotk.inference.runner import run_lesion_volume


def _write_nifti(path: Path, data: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> None:
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4, dtype=np.float32))
    hdr = img.header
    if data.ndim == 3:
        hdr.set_zooms(tuple(float(x) for x in spacing))
    elif data.ndim == 4:
        hdr.set_zooms(tuple(float(x) for x in spacing) + (1.0,))
    nib.save(img, str(path))


def test_run_lesion_volume_computes_mm3_and_ml(tmp_path: Path) -> None:
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    pred_path = preds_dir / "case_001_seg.nii.gz"

    data = np.zeros((2, 2, 2), dtype=np.float32)
    data[0, 0, 0] = 1
    data[1, 1, 1] = 1
    _write_nifti(pred_path, data, spacing=(2.0, 2.0, 2.0))

    out_csv = tmp_path / "lesion_volumes.csv"
    run_lesion_volume(
        preds_path=preds_dir,
        preds_list=None,
        output_csv=out_csv,
        threshold=0.5,
    )

    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "image,lesion_voxels,voxel_volume_mm3,lesion_volume_mm3,lesion_volume_ml"
    assert lines[1].startswith("case_001_seg.nii.gz,2,8.000000,16.000000,0.016000")
    summary_csv = tmp_path / "lesion_volumes_summary.csv"
    assert summary_csv.exists()
    summary_lines = summary_csv.read_text(encoding="utf-8").strip().splitlines()
    assert summary_lines[0] == "category,metric,count,percent,value_ml"
    # 0.016 mL falls in >0 to <1 mL
    assert any(line.startswith("range,>0 to <1 mL,1,100.00,") for line in summary_lines)
    assert any(line.startswith("overall,total_images,1,,") for line in summary_lines)
    assert any(line.startswith("overall,median_ml,,,0.016000") for line in summary_lines)


def test_run_lesion_volume_4d_probs_uses_argmax(tmp_path: Path) -> None:
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    pred_path = preds_dir / "case_prob.nii.gz"

    # Shape [C, D, H, W] with 2 classes.
    data = np.zeros((2, 2, 1, 1), dtype=np.float32)
    # class 1 wins in exactly one voxel -> lesion voxels = 1
    data[1, 0, 0, 0] = 0.9
    _write_nifti(pred_path, data, spacing=(1.0, 1.0, 1.0))

    out_csv = tmp_path / "lesion_volumes.csv"
    run_lesion_volume(
        preds_path=preds_dir,
        preds_list=None,
        output_csv=out_csv,
        threshold=0.5,
    )

    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[1].startswith("case_prob.nii.gz,1,1.000000,1.000000,0.001000")


def test_run_lesion_volume_writes_histogram(tmp_path: Path) -> None:
    preds_dir = tmp_path / "preds"
    preds_dir.mkdir()
    pred_path = preds_dir / "case_001_seg.nii.gz"

    data = np.zeros((2, 2, 2), dtype=np.float32)
    data[0, 0, 0] = 1
    data[1, 1, 1] = 1
    _write_nifti(pred_path, data, spacing=(1.0, 1.0, 1.0))

    out_csv = tmp_path / "lesion_volumes.csv"
    hist_png = tmp_path / "lesion_volume_hist.png"
    run_lesion_volume(
        preds_path=preds_dir,
        preds_list=None,
        output_csv=out_csv,
        threshold=0.5,
        histogram_path=hist_png,
        hist_bins=8,
    )

    assert hist_png.exists()
    assert hist_png.stat().st_size > 0
