from __future__ import annotations

from pathlib import Path
import json

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
        images_path=None,
        images_list=None,
        output_csv=out_csv,
        normal_threshold_ml=0.2,
    )

    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "image,case_id,normal_ct,lesion_volume_ml"
    assert any("case_a.nii.gz,case_a,true,0.000000" in line for line in lines)
    assert any("case_b.nii.gz,case_b,false,1.000000" in line for line in lines)


def test_run_make_normal_ct_flags_writes_monai_style_train_selection_json(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    # 0.5 mL (below train threshold 1.0) -> excluded
    arr_a = np.zeros((1, 1, 1), dtype=np.float32)
    arr_a[0, 0, 0] = 1.0
    _write_label(images_dir / "case_a.nii.gz", np.zeros((1, 1, 1), dtype=np.float32))
    _write_label(labels_dir / "case_a_seg.nii.gz", arr_a, spacing=(10.0, 10.0, 5.0))

    # 2.0 mL (above train threshold 1.0) -> included
    arr_b = np.zeros((1, 1, 2), dtype=np.float32)
    arr_b[0, 0, :] = 1.0
    _write_label(images_dir / "case_b.nii.gz", np.zeros((1, 1, 2), dtype=np.float32))
    _write_label(labels_dir / "case_b_seg.nii.gz", arr_b, spacing=(10.0, 10.0, 10.0))

    out_csv = tmp_path / "normal_ct_flags.csv"
    out_json = tmp_path / "train_selection.json"
    run_make_normal_ct_flags(
        labels_path=labels_dir,
        labels_list=None,
        images_path=images_dir,
        images_list=None,
        output_csv=out_csv,
        normal_threshold_ml=0.2,
        train_selection_json=out_json,
        train_min_lesion_ml=1.0,
        num_folds=5,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["description"] == "TBI dataset"
    assert payload["labels"] == {"0": "background", "1": "lesion"}
    assert payload["numTotalMatchedImages"] == 2
    assert payload["numSelectedByLesionThreshold"] == 1
    assert payload["numTraining"] == 1
    assert payload["numValidation"] == 0
    assert payload["numTesting"] == 1
    assert isinstance(payload["training"], list)
    assert payload["validation"] == []
    assert isinstance(payload["testing"], list)
    assert payload["training"][0]["image"].endswith("case_b.nii.gz")
    assert payload["training"][0]["label"].endswith("case_b_seg.nii.gz")
    assert payload["training"][0]["fold"] == 0
    assert payload["testing"][0]["image"].endswith("case_b.nii.gz")


def test_run_make_normal_ct_flags_train_selection_uses_inner_join(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    # Matched pair
    _write_label(images_dir / "case_a.nii.gz", np.zeros((1, 1, 1), dtype=np.float32))
    arr_a = np.zeros((1, 1, 1), dtype=np.float32)
    arr_a[0, 0, 0] = 1.0
    _write_label(labels_dir / "case_a_seg.nii.gz", arr_a, spacing=(10.0, 10.0, 10.0))

    # Image without label
    _write_label(images_dir / "case_img_only.nii.gz", np.zeros((1, 1, 1), dtype=np.float32))

    # Label without image
    arr_lbl_only = np.zeros((1, 1, 1), dtype=np.float32)
    arr_lbl_only[0, 0, 0] = 1.0
    _write_label(labels_dir / "case_lbl_only_seg.nii.gz", arr_lbl_only, spacing=(10.0, 10.0, 10.0))

    out_csv = tmp_path / "normal_ct_flags.csv"
    out_json = tmp_path / "train_selection.json"
    run_make_normal_ct_flags(
        labels_path=labels_dir,
        labels_list=None,
        images_path=images_dir,
        images_list=None,
        output_csv=out_csv,
        normal_threshold_ml=0.2,
        train_selection_json=out_json,
        train_min_lesion_ml=0.1,
        num_folds=5,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["numTotalMatchedImages"] == 1
    assert payload["numSelectedByLesionThreshold"] == 1
    assert len(payload["training"]) == 1
    assert payload["training"][0]["image"].endswith("case_a.nii.gz")
    assert payload["training"][0]["label"].endswith("case_a_seg.nii.gz")
