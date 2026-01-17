"""Validation CLI tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib

from neurotk import cli


def test_validate_creates_report(sample_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    args = _validate_args(sample_dataset, out)

    rc = cli._run_validate(args)
    assert rc == 0
    assert out.exists()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "summary" in data
    assert "files" in data
    assert "warnings" in data

    summary = data["summary"]
    assert summary.get("num_images") == 2
    assert summary.get("num_labels") == 2
    assert summary.get("files_with_issues") is not None

    files = data["files"]
    assert "CASE_000.nii.gz" in files
    assert files["CASE_000.nii.gz"]["image"]["spacing"] is not None
    assert files["CASE_000.nii.gz"]["image"]["orientation"] is not None


def test_validate_missing_labels(dataset_with_missing_label: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    args = _validate_args(dataset_with_missing_label, out)

    rc = cli._run_validate(args)
    assert rc == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    summary = report["summary"]
    missing = summary.get("missing_label_images")
    assert isinstance(missing, list)
    assert len(missing) == 1


def test_validate_corrupt_nifti(dataset_with_corrupt_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    args = _validate_args(dataset_with_corrupt_file, out, labels=False)

    rc = cli._run_validate(args)
    assert rc == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    files = report["files"]
    corrupt = files.get("CORRUPT.nii.gz")
    assert corrupt is not None
    assert "image_not_readable" in corrupt.get("issues", [])


def test_validate_help() -> None:
    result = _run_help(["validate", "--help"])
    assert result == 0


def _validate_args(dataset: Path, out: Path, labels: bool = True) -> object:
    labels_dir = dataset / "labels" if labels else None
    return type("Args", (), {
        "images": dataset / "images",
        "labels": labels_dir,
        "out": out,
        "max_samples": None,
        "html": None,
        "summary_only": False,
    })()


def _run_help(args: list) -> int:
    import subprocess

    cmd = [sys.executable, "-m", "neurotk.cli", *args]
    return subprocess.call(cmd)
