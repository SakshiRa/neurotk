from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from neurotk.inference.monai_compat import (
    install_transform_compat,
    make_dice_helper_compat,
    prepare_bundle_import_compat,
)


def test_make_dice_helper_compat_maps_threshold_to_sigmoid() -> None:
    class NewDiceHelper:
        def __init__(self, *, sigmoid=False, ignore_empty=True):
            self.sigmoid = sigmoid
            self.ignore_empty = ignore_empty

    compat = make_dice_helper_compat(NewDiceHelper)
    inst = compat(threshold=True, ignore_empty=False)
    assert isinstance(inst, NewDiceHelper)
    assert inst.sigmoid is True
    assert inst.ignore_empty is False


def test_make_dice_helper_compat_maps_sigmoid_to_threshold() -> None:
    class OldDiceHelper:
        def __init__(self, *, threshold=False, ignore_empty=True):
            self.threshold = threshold
            self.ignore_empty = ignore_empty

    compat = make_dice_helper_compat(OldDiceHelper)
    inst = compat(sigmoid=True, ignore_empty=False)
    assert isinstance(inst, OldDiceHelper)
    assert inst.threshold is True
    assert inst.ignore_empty is False


def test_install_transform_compat_adds_typo_alias() -> None:
    transforms = SimpleNamespace(RandScaleIntensityFixedMeanD=object)
    alias = install_transform_compat(transforms)
    assert alias == ("RandScaleIntensityFixedMeand", "RandScaleIntensityFixedMeanD")
    assert hasattr(transforms, "RandScaleIntensityFixedMeand")


def test_prepare_bundle_import_compat_allows_segmenter_import_with_new_dice_signature(
    tmp_path: Path, monkeypatch
) -> None:
    segmenter_code = (
        "from monai.metrics import DiceHelper\n"
        "class Segmenter:\n"
        "    def __init__(self, config_file=None, config_dict=None):\n"
        "        self.acc = DiceHelper(threshold=True, ignore_empty=True)\n"
    )
    seg_path = tmp_path / "segmenter.py"
    seg_path.write_text(segmenter_code, encoding="utf-8")

    monai_mod = ModuleType("monai")
    monai_metrics = ModuleType("monai.metrics")
    monai_transforms = ModuleType("monai.transforms")

    class NewDiceHelper:
        def __init__(self, *, sigmoid=False, ignore_empty=True):
            self.sigmoid = sigmoid
            self.ignore_empty = ignore_empty

    monai_metrics.DiceHelper = NewDiceHelper
    monai_transforms.RandScaleIntensityFixedMeanD = object
    monai_mod.metrics = monai_metrics
    monai_mod.transforms = monai_transforms

    monkeypatch.setitem(sys.modules, "monai", monai_mod)
    monkeypatch.setitem(sys.modules, "monai.metrics", monai_metrics)
    monkeypatch.setitem(sys.modules, "monai.transforms", monai_transforms)

    prepare_bundle_import_compat()

    spec = importlib.util.spec_from_file_location("segmenter", seg_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seg = module.Segmenter(config_file="x", config_dict={})
    assert isinstance(seg.acc, NewDiceHelper)
    assert seg.acc.sigmoid is True
