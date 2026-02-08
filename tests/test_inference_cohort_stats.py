from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from neurotk.inference.runner import run_cohort_selection_stats


def _write_label(path: Path, data: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> None:
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4, dtype=np.float32))
    img.header.set_zooms(tuple(float(x) for x in spacing))
    nib.save(img, str(path))


def test_run_cohort_selection_stats_outputs_classification_and_summary(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    # case_a: normal CT + 0 mL => true_negative
    _write_label(labels_dir / "case_a.nii.gz", np.zeros((2, 2, 2), dtype=np.float32), spacing=(1.0, 1.0, 1.0))

    # case_b: normal CT + 1 mL => true_positive, low
    arr_b = np.zeros((2, 2, 2), dtype=np.float32)
    arr_b[0, 0, 0] = 1.0
    _write_label(labels_dir / "case_b.nii.gz", arr_b, spacing=(10.0, 10.0, 10.0))

    # case_c: abnormal CT + 0 mL => true_positive, low
    _write_label(labels_dir / "case_c.nii.gz", np.zeros((2, 2, 2), dtype=np.float32), spacing=(1.0, 1.0, 1.0))

    # case_d: abnormal CT + 30 mL => true_positive, high
    arr_d = np.ones((3, 5, 2), dtype=np.float32)  # 30 voxels
    _write_label(labels_dir / "case_d.nii.gz", arr_d, spacing=(10.0, 10.0, 10.0))

    normal_csv = tmp_path / "normal.csv"
    normal_csv.write_text(
        "image,normal_ct\n"
        "case_a.nii.gz,true\n"
        "case_b.nii.gz,true\n"
        "case_c.nii.gz,false\n"
        "case_d.nii.gz,false\n",
        encoding="utf-8",
    )

    out_csv = tmp_path / "cohort.csv"
    summary_csv = tmp_path / "cohort_summary.csv"
    run_cohort_selection_stats(
        labels_path=labels_dir,
        labels_list=None,
        normal_csv=normal_csv,
        output_csv=out_csv,
        summary_csv=summary_csv,
        tn_threshold_ml=0.2,
        low_max_ml=5.0,
        medium_max_ml=20.0,
    )

    out_lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert out_lines[0] == "image,case_id,normal_ct,lesion_voxels,lesion_volume_ml,classification,tp_volume_subgroup"
    assert any("case_a.nii.gz,case_a,true,0,0.000000,true_negative," in line for line in out_lines)
    assert any("case_b.nii.gz,case_b,true,1,1.000000,true_positive,low" in line for line in out_lines)
    assert any("case_c.nii.gz,case_c,false,0,0.000000,true_positive,low" in line for line in out_lines)
    assert any("case_d.nii.gz,case_d,false,30,30.000000,true_positive,high" in line for line in out_lines)

    summary_lines = summary_csv.read_text(encoding="utf-8").strip().splitlines()
    assert summary_lines[0] == "metric,count,percent"
    assert "total,4,100.00" in summary_lines
    assert "true_negative,1,25.00" in summary_lines
    assert "true_positive,3,75.00" in summary_lines
    assert "tp_low,2,50.00" in summary_lines
    assert "tp_medium,0,0.00" in summary_lines
    assert "tp_high,1,25.00" in summary_lines
