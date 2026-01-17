"""Summary-only output tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_summary_only_output(sample_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
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
        str(out),
        "--summary-only",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0

    stdout = result.stdout
    assert "NeuroTK Dataset Validation Summary" in stdout
    assert "Images" in stdout
    assert "Warnings" in stdout
    assert "Validation complete" in stdout

    report = json.loads(out.read_text(encoding="utf-8"))
    assert "summary" in report
    assert "files" in report
