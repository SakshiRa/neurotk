"""Regression tests for validation-only reports."""

from __future__ import annotations

import json
from pathlib import Path


def _load_fixture(name: str) -> dict:
    path = Path(__file__).parent / "fixtures" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_validate_only_report_schema() -> None:
    report = _load_fixture("validate_only.json")

    assert report.get("run_mode") == "validate", "run_mode should be validate"
    assert report["summary"].get("scope") == "original_inputs"
    assert "summary_processed" not in report, "summary_processed must be absent for validate-only"
    assert "preprocess" not in report, "preprocess block must be absent for validate-only"
