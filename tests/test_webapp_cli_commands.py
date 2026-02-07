"""Webapp command wiring tests."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import UploadFile
from starlette.requests import Request

import webapp.main as webmain


def _request() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/run", "headers": []})


def _upload(name: str, data: bytes = b"nifti-bytes") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(data))


def _configure_dirs(tmp_path: Path, monkeypatch) -> None:
    outputs = tmp_path / "outputs"
    tmp_runs = tmp_path / "tmp_runs"
    outputs.mkdir()
    tmp_runs.mkdir()
    monkeypatch.setattr(webmain, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(webmain, "TMP_ROOT", tmp_runs)


def test_run_validate_invokes_validate_with_labels(tmp_path: Path, monkeypatch) -> None:
    _configure_dirs(tmp_path, monkeypatch)
    calls: list[tuple[list[str], dict | None]] = []

    def fake_run_cli(command: list[str], env=None) -> None:
        calls.append((command, env))
        out_idx = command.index("--out") + 1
        html_idx = command.index("--html") + 1
        Path(command[out_idx]).write_text("{}", encoding="utf-8")
        Path(command[html_idx]).write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(webmain, "_run_cli", fake_run_cli)

    response = webmain.run(
        request=_request(),
        mode="validate",
        images_files=[_upload("img_001.nii.gz")],
        labels_files=[_upload("img_001.nii.gz")],
    )

    assert response.status_code == 200
    assert len(calls) == 1
    command, env = calls[0]
    assert env is None
    assert command[:2] == ["neurotk", "validate"]
    assert "--images" in command
    assert "--labels" in command
    assert "--out" in command
    assert "--html" in command


def test_run_preprocess_invokes_preprocess_then_validate_with_env(tmp_path: Path, monkeypatch) -> None:
    _configure_dirs(tmp_path, monkeypatch)
    calls: list[tuple[list[str], dict | None]] = []

    def fake_run_cli(command: list[str], env=None) -> None:
        calls.append((command, env))
        if command[1] == "validate":
            out_idx = command.index("--out") + 1
            html_idx = command.index("--html") + 1
            Path(command[out_idx]).write_text("{}", encoding="utf-8")
            Path(command[html_idx]).write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(webmain, "_run_cli", fake_run_cli)

    response = webmain.run(
        request=_request(),
        mode="preprocess",
        spacing="1 1 1",
        orientation="RAS",
        images_files=[_upload("img_001.nii.gz")],
        labels_files=[_upload("img_001.nii.gz")],
    )

    assert response.status_code == 200
    assert len(calls) == 2

    preprocess_cmd, preprocess_env = calls[0]
    assert preprocess_cmd[:2] == ["neurotk", "preprocess"]
    assert "--labels" in preprocess_cmd
    assert preprocess_env is None

    validate_cmd, validate_env = calls[1]
    assert validate_cmd[:2] == ["neurotk", "validate"]
    assert "--labels" in validate_cmd
    assert validate_env is not None
    assert validate_env["NEUROTK_PREPROCESS_SPACING"] == "1 1 1"
    assert validate_env["NEUROTK_PREPROCESS_ORIENTATION"] == "RAS"
    assert validate_env["NEUROTK_PREPROCESS_COPY_METADATA"] == "false"
    assert validate_env["NEUROTK_LABELS_PROVIDED"] == "true"
    assert validate_env["NEUROTK_LABELS_UPLOADED"] == "1"


def test_run_validate_without_labels_omits_label_flag(tmp_path: Path, monkeypatch) -> None:
    _configure_dirs(tmp_path, monkeypatch)
    calls: list[tuple[list[str], dict | None]] = []

    def fake_run_cli(command: list[str], env=None) -> None:
        calls.append((command, env))
        out_idx = command.index("--out") + 1
        html_idx = command.index("--html") + 1
        Path(command[out_idx]).write_text("{}", encoding="utf-8")
        Path(command[html_idx]).write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(webmain, "_run_cli", fake_run_cli)

    response = webmain.run(
        request=_request(),
        mode="validate",
        images_files=[_upload("img_001.nii.gz")],
        labels_files=[],
    )

    assert response.status_code == 200
    assert len(calls) == 1
    command, _ = calls[0]
    assert command[:2] == ["neurotk", "validate"]
    assert "--labels" not in command
