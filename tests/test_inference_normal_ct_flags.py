from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from neurotk.inference.runner import run_make_normal_ct_flags


def _write_label(path: Path, data: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> None:
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4, dtype=np.float32))
    img.header.set_zooms(tuple(float(x) for x in spacing))
    nib.save(img, str(path))


def test_run_make_normal_ct_flags_from_labels(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    # 0 mL -> true
    _write_label(labels_dir / "case_a.nii.gz", np.zeros((2, 2, 2), dtype=np.float32), spacing=(1.0, 1.0, 1.0))

    # 1 mL -> false for threshold 0.2
    arr = np.zeros((2, 2, 2), dtype=np.float32)
    arr[0, 0, 0] = 1.0
    _write_label(labels_dir / "case_b.nii.gz", arr, spacing=(10.0, 10.0, 10.0))

    out_csv = tmp_path / "normal_ct_flags.csv"
    run_make_normal_ct_flags(
        labels_path=labels_dir,
        labels_list=None,
        output_csv=out_csv,
        normal_threshold_ml=0.2,
    )

    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "image,case_id,normal_ct,lesion_volume_ml"
    assert any("case_a.nii.gz,case_a,true,0.000000" in line for line in lines)
    assert any("case_b.nii.gz,case_b,false,1.000000" in line for line in lines)
