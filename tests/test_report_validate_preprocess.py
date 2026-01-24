"""Regression tests for validate+preprocess reports."""

from __future__ import annotations

import json
from pathlib import Path


def _load_fixture(name: str) -> dict:
    path = Path(__file__).parent / "fixtures" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_validate_preprocess_summary_scopes() -> None:
    report = _load_fixture("validate_preprocess.json")

    assert report.get("run_mode") == "validate+preprocess"
    assert report["summary"].get("scope") == "original_inputs"
    assert report["summary_processed"].get("scope") == "processed_outputs"

    original_orientation = report["summary"].get("orientation_modal")
    processed_orientation = report["summary_processed"].get("orientation_modal")
    assert original_orientation != processed_orientation, "original vs processed orientation must differ"

    effects = report["preprocess"]["effects_summary"]
    assert effects["n_files_applied"] > 0


def test_preprocess_verification_per_file() -> None:
    report = _load_fixture("validate_preprocess.json")

    for name, entry in report["files"].items():
        preprocess = entry.get("preprocess")
        assert preprocess is not None, f"missing preprocess block for {name}"
        assert preprocess.get("requested") is True, f"requested missing for {name}"
        applied = preprocess.get("applied")
        noop_reason = preprocess.get("noop_reason")
        assert applied or noop_reason, f"applied false without noop_reason for {name}"
        if applied:
            verified = preprocess.get("verified_by", {})
            assert any(
                verified.get(k) for k in ("orientation_changed", "spacing_changed", "shape_changed")
            ), f"applied true without verified_by change for {name}"


def test_label_provenance_without_labels() -> None:
    report = _load_fixture("validate_preprocess.json")

    preprocess = report.get("preprocess")
    inputs = preprocess.get("inputs", {})
    assert inputs.get("labels_provided") is False
    assert inputs.get("num_label_files_uploaded") == 0

    outputs = preprocess.get("outputs", {})
    assert outputs.get("labels_dir") in (None, "null"), "labels_dir must be null when labels absent"

    assert report["summary"].get("num_labels") == 0
    assert "label_stats" not in report.get("stats_summary", {})

    for name, entry in report["files"].items():
        preprocess_block = entry.get("preprocess", {})
        assert preprocess_block.get("label_processed") is False, f"label_processed must be false for {name}"
