from __future__ import annotations

from pathlib import Path

import torch

from neurotk.inference import runner


def test_run_inference_skips_when_output_exists(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "case_001.nii.gz"
    input_path.write_text("placeholder", encoding="utf-8")

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "case_001_seg.nii.gz").write_text("existing", encoding="utf-8")

    calls = {"predict": 0}

    class DummyPredictor:
        def __init__(self, bundle_dir: str, device: str | None = None):
            self.bundle_dir = bundle_dir
            self.device = device

        def predict_volume(self, image_path: str):
            calls["predict"] += 1
            return torch.zeros((1, 2, 2, 2)), {"affine": torch.eye(4).numpy()}

    monkeypatch.setattr(runner, "BundlePredictor", DummyPredictor)
    monkeypatch.setattr(runner, "resolve_bundle_dir", lambda value: value)
    monkeypatch.setattr(runner, "save_nifti", lambda *args, **kwargs: None)

    runner.run_inference(
        bundle_dir=tmp_path,
        input_path=input_path,
        input_list=None,
        output_dir=output_dir,
        device="cpu",
        save_probs=False,
        force=False,
        skip_invalid_inputs=False,
        labels_dir=None,
        reference_image=None,
    )

    assert calls["predict"] == 0


def test_run_inference_force_recomputes_existing_output(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "case_001.nii.gz"
    input_path.write_text("placeholder", encoding="utf-8")

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "case_001_seg.nii.gz").write_text("existing", encoding="utf-8")

    calls = {"predict": 0, "save": 0}

    class DummyPredictor:
        def __init__(self, bundle_dir: str, device: str | None = None):
            self.bundle_dir = bundle_dir
            self.device = device

        def predict_volume(self, image_path: str):
            calls["predict"] += 1
            return torch.zeros((1, 2, 2, 2)), {"affine": torch.eye(4).numpy()}

    monkeypatch.setattr(runner, "BundlePredictor", DummyPredictor)
    monkeypatch.setattr(runner, "resolve_bundle_dir", lambda value: value)
    monkeypatch.setattr(runner, "save_nifti", lambda *args, **kwargs: calls.__setitem__("save", calls["save"] + 1))

    runner.run_inference(
        bundle_dir=tmp_path,
        input_path=input_path,
        input_list=None,
        output_dir=output_dir,
        device="cpu",
        save_probs=False,
        force=True,
        skip_invalid_inputs=False,
        labels_dir=None,
        reference_image=None,
    )

    assert calls["predict"] == 1
    assert calls["save"] == 1


def test_run_inference_skips_invalid_inputs_and_writes_report(monkeypatch, tmp_path: Path) -> None:
    good = tmp_path / "good.nii.gz"
    bad = tmp_path / "bad.nii.gz"
    good.write_text("placeholder", encoding="utf-8")
    bad.write_text("placeholder", encoding="utf-8")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    calls = {"save": 0}

    class DummyPredictor:
        def __init__(self, bundle_dir: str, device: str | None = None):
            self.bundle_dir = bundle_dir
            self.device = device

        def predict_volume(self, image_path: str):
            if image_path.endswith("bad.nii.gz"):
                raise RuntimeError("expected input to have 1 channels, but got 2 channels instead")
            return torch.zeros((1, 2, 2, 2)), {"affine": torch.eye(4).numpy()}

    monkeypatch.setattr(runner, "BundlePredictor", DummyPredictor)
    monkeypatch.setattr(runner, "resolve_bundle_dir", lambda value: value)
    monkeypatch.setattr(runner, "save_nifti", lambda *args, **kwargs: calls.__setitem__("save", calls["save"] + 1))

    runner.run_inference(
        bundle_dir=tmp_path,
        input_path=tmp_path,
        input_list=None,
        output_dir=output_dir,
        device="cpu",
        save_probs=False,
        force=True,
        skip_invalid_inputs=True,
        labels_dir=None,
        reference_image=None,
    )

    assert calls["save"] == 1
    skipped = output_dir / "skipped_inputs.csv"
    assert skipped.exists()
    text = skipped.read_text(encoding="utf-8")
    assert "bad.nii.gz" in text
