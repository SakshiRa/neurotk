"""HTML report tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_html_report_created(sample_dataset: Path, tmp_path: Path) -> None:
    json_out = tmp_path / "report.json"
    html_out = tmp_path / "report.html"
    cmd = [
        sys.executable,
        "-m",
        "neurotk.cli",
        "validate",
        "--images",
        str(sample_dataset / "images"),
        "--labels",
        str(sample_dataset / "labels"),
        "--out",
        str(json_out),
        "--html",
        str(html_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert html_out.exists()

    html = html_out.read_text(encoding="utf-8")
    assert "Dataset Summary" in html
    assert "File-Level Summary" in html
    assert "<table" in html
    assert "NeuroTK" in html
    assert "<script" not in html


def test_preprocess_help() -> None:
    result = subprocess.call([sys.executable, "-m", "neurotk.cli", "preprocess", "--help"])
    assert result == 0
