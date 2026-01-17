"""Pytest fixtures for NeuroTK."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import pytest


def _write_nifti(path: Path, data: np.ndarray, spacing: Tuple[float, float, float]) -> None:
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


def _write_oriented_nifti(
    path: Path, data: np.ndarray, spacing: Tuple[float, float, float]
) -> None:
    affine = np.diag([-spacing[0], spacing[1], spacing[2], 1.0])
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


@pytest.fixture()
def sample_dataset(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    rng = np.random.RandomState(0)
    for idx in range(2):
        data = rng.randn(16, 16, 8)
        label = (data > 0).astype(np.int16)
        _write_nifti(images / f"CASE_{idx:03d}.nii.gz", data, (2.0, 2.0, 2.0))
        _write_nifti(labels / f"CASE_{idx:03d}.nii.gz", label, (2.0, 2.0, 2.0))

    return tmp_path


@pytest.fixture()
def dataset_with_missing_label(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    rng = np.random.RandomState(1)
    for idx in range(2):
        data = rng.randn(16, 16, 8)
        label = (data > 0).astype(np.int16)
        _write_nifti(images / f"CASE_{idx:03d}.nii.gz", data, (1.5, 1.5, 2.5))
        if idx == 0:
            _write_nifti(labels / f"CASE_{idx:03d}.nii.gz", label, (1.5, 1.5, 2.5))

    return tmp_path


@pytest.fixture()
def dataset_with_corrupt_file(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir()

    rng = np.random.RandomState(2)
    data = rng.randn(16, 16, 8)
    _write_nifti(images / "CASE_000.nii.gz", data, (2.0, 2.0, 2.0))

    corrupt = images / "CORRUPT.nii.gz"
    corrupt.write_bytes(b"not a nifti")

    return tmp_path


@pytest.fixture()
def oriented_dataset(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    rng = np.random.RandomState(3)
    data = rng.randn(12, 10, 6)
    label = (data > 0).astype(np.int16)
    _write_oriented_nifti(images / "CASE_000.nii.gz", data, (2.0, 2.0, 2.0))
    _write_oriented_nifti(labels / "CASE_000.nii.gz", label, (2.0, 2.0, 2.0))

    return tmp_path
