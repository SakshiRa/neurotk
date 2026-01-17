"""Preprocess tests."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from neurotk.preprocess import preprocess_dataset


def test_preprocess_writes_outputs(sample_dataset: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    report = preprocess_dataset(
        images_dir=sample_dataset / "images",
        labels_dir=sample_dataset / "labels",
        out_dir=out_dir,
        spacing=(1.0, 1.0, 1.0),
        orientation="RAS",
        dry_run=False,
        copy_metadata=False,
    )

    assert out_dir.exists()
    assert (out_dir / "images" / "CASE_000.nii.gz").exists()
    assert (out_dir / "labels" / "CASE_000.nii.gz").exists()
    assert (out_dir / "preprocess_report.json").exists()
    assert (out_dir / "preprocess_config.yaml").exists()
    assert "processed_files" in report


def test_preprocess_orientation_and_spacing(oriented_dataset: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    preprocess_dataset(
        images_dir=oriented_dataset / "images",
        labels_dir=oriented_dataset / "labels",
        out_dir=out_dir,
        spacing=(1.0, 1.0, 1.0),
        orientation="RAS",
        dry_run=False,
        copy_metadata=False,
    )

    img = nib.load(str(out_dir / "images" / "CASE_000.nii.gz"))
    codes = nib.orientations.aff2axcodes(img.affine)
    assert codes == ("R", "A", "S")

    spacing = img.header.get_zooms()[:3]
    assert np.allclose(spacing, (1.0, 1.0, 1.0), atol=1e-3)


def test_preprocess_labels_integer(oriented_dataset: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    preprocess_dataset(
        images_dir=oriented_dataset / "images",
        labels_dir=oriented_dataset / "labels",
        out_dir=out_dir,
        spacing=(1.0, 1.0, 1.0),
        orientation="RAS",
        dry_run=False,
        copy_metadata=False,
    )

    label_img = nib.load(str(out_dir / "labels" / "CASE_000.nii.gz"))
    data = label_img.get_fdata()
    assert np.allclose(data, np.round(data))


def test_preprocess_dry_run(sample_dataset: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    preprocess_dataset(
        images_dir=sample_dataset / "images",
        labels_dir=sample_dataset / "labels",
        out_dir=out_dir,
        spacing=(1.0, 1.0, 1.0),
        orientation="RAS",
        dry_run=True,
        copy_metadata=False,
    )

    assert not out_dir.exists()
