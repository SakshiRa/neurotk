"""CLI method wiring tests for all subcommands."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurotk import cli


def test_run_preprocess_invokes_preprocess_dataset(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_preprocess_dataset(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "preprocess_dataset", fake_preprocess_dataset)

    args = SimpleNamespace(
        images=tmp_path / "images",
        labels=tmp_path / "labels",
        out=tmp_path / "out",
        spacing=[1.0, 1.5, 2.0],
        orientation="RAS",
        dry_run=True,
        copy_metadata=False,
    )

    rc = cli._run_preprocess(args)
    assert rc == 0
    assert captured["images_dir"] == args.images
    assert captured["labels_dir"] == args.labels
    assert captured["out_dir"] == args.out
    assert captured["spacing"] == (1.0, 1.5, 2.0)
    assert captured["orientation"] == "RAS"
    assert captured["dry_run"] is True
    assert captured["copy_metadata"] is False


def test_run_infer_invokes_runner_with_default_bundle(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run_inference(**kwargs):
        captured.update(kwargs)

    fake_runner = SimpleNamespace(run_inference=fake_run_inference, run_dice=lambda **_: None)
    monkeypatch.setitem(sys.modules, "neurotk.inference.runner", fake_runner)
    monkeypatch.setenv("NEUROTK_DEFAULT_BUNDLE", "ORG/default-bundle")

    args = SimpleNamespace(
        bundle_dir=None,
        input=tmp_path / "image.nii.gz",
        input_list=None,
        output_dir=tmp_path / "preds",
        device="cpu",
        save_probs=True,
        labels_dir=tmp_path / "labels",
        reference_image=tmp_path / "ref.nii.gz",
    )

    rc = cli._run_infer(args)
    assert rc == 0
    assert captured["bundle_dir"] == "ORG/default-bundle"
    assert captured["input_path"] == args.input
    assert captured["input_list"] is None
    assert captured["output_dir"] == args.output_dir
    assert captured["device"] == "cpu"
    assert captured["save_probs"] is True
    assert captured["labels_dir"] == args.labels_dir
    assert captured["reference_image"] == args.reference_image


def test_run_infer_wraps_value_error_as_system_exit(monkeypatch, tmp_path: Path) -> None:
    def fake_run_inference(**kwargs):
        raise ValueError("bad infer args")

    fake_runner = SimpleNamespace(run_inference=fake_run_inference, run_dice=lambda **_: None)
    monkeypatch.setitem(sys.modules, "neurotk.inference.runner", fake_runner)

    args = SimpleNamespace(
        bundle_dir="bundle",
        input=tmp_path / "image.nii.gz",
        input_list=None,
        output_dir=tmp_path / "preds",
        device=None,
        save_probs=False,
        labels_dir=None,
        reference_image=None,
    )

    with pytest.raises(SystemExit, match="bad infer args"):
        cli._run_infer(args)


def test_run_dice_invokes_runner(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run_dice(**kwargs):
        captured.update(kwargs)

    fake_runner = SimpleNamespace(run_inference=lambda **_: None, run_dice=fake_run_dice)
    monkeypatch.setitem(sys.modules, "neurotk.inference.runner", fake_runner)

    args = SimpleNamespace(
        preds=tmp_path / "preds",
        preds_list=tmp_path / "preds.txt",
        labels_dir=tmp_path / "labels",
        output=tmp_path / "dice.csv",
    )

    rc = cli._run_dice(args)
    assert rc == 0
    assert captured["preds_path"] == args.preds
    assert captured["preds_list"] == args.preds_list
    assert captured["labels_dir"] == args.labels_dir
    assert captured["output_csv"] == args.output


def test_run_dispatches_all_subcommands(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_run_validate", lambda _args: 11)
    monkeypatch.setattr(cli, "_run_preprocess", lambda _args: 22)
    monkeypatch.setattr(cli, "_run_infer", lambda _args: 33)
    monkeypatch.setattr(cli, "_run_dice", lambda _args: 44)

    monkeypatch.setattr(cli, "_parse_args", lambda: SimpleNamespace(command="validate"))
    assert cli.run() == 11

    monkeypatch.setattr(cli, "_parse_args", lambda: SimpleNamespace(command="preprocess"))
    assert cli.run() == 22

    monkeypatch.setattr(cli, "_parse_args", lambda: SimpleNamespace(command="infer"))
    assert cli.run() == 33

    monkeypatch.setattr(cli, "_parse_args", lambda: SimpleNamespace(command="dice"))
    assert cli.run() == 44


def test_run_unknown_command_raises(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_parse_args", lambda: SimpleNamespace(command="unknown"))
    with pytest.raises(SystemExit, match="Unknown command: unknown"):
        cli.run()
