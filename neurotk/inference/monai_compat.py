from __future__ import annotations

import inspect
import sys
from typing import Dict, Optional, Sequence, Tuple

from packaging.version import Version, InvalidVersion


TESTED_MONAI_MIN = Version("1.3.0")
TESTED_MONAI_MAX = Version("1.6.0")


def _safe_version_parse(raw: Optional[str]) -> Optional[Version]:
    if not raw:
        return None
    try:
        return Version(raw)
    except InvalidVersion:
        return None


def _safe_module_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "not-installed"
    return str(getattr(module, "__version__", "unknown"))


def runtime_versions() -> Dict[str, str]:
    return {
        "monai": _safe_module_version("monai"),
        "torch": _safe_module_version("torch"),
        "ignite": _safe_module_version("ignite"),
    }


def report_runtime_versions(*, stream=None) -> None:
    if stream is None:
        stream = sys.stderr
    versions = runtime_versions()
    print(
        "Inference runtime versions: "
        f"monai={versions['monai']}, torch={versions['torch']}, ignite={versions['ignite']}",
        file=stream,
    )

    monai_v = _safe_version_parse(versions["monai"])
    if monai_v is None or not (TESTED_MONAI_MIN <= monai_v < TESTED_MONAI_MAX):
        print(
            "Warning: MONAI version is outside NeuroTK tested range "
            f"[{TESTED_MONAI_MIN}, {TESTED_MONAI_MAX}). "
            "Bundle import compatibility shims will be applied where possible.",
            file=stream,
        )


def _dice_helper_params(dice_helper_cls) -> Sequence[str]:
    try:
        return tuple(inspect.signature(dice_helper_cls.__init__).parameters.keys())
    except (TypeError, ValueError):
        return tuple()


def make_dice_helper_compat(dice_helper_cls):
    params = set(_dice_helper_params(dice_helper_cls))
    supports_threshold = "threshold" in params
    supports_sigmoid = "sigmoid" in params
    supports_activate = "activate" in params

    class _NeuroTKCompatDiceHelper:
        __name__ = "_NeuroTKCompatDiceHelper"

        def __new__(cls, *args, **kwargs):
            mapped = dict(kwargs)

            if "threshold" in mapped and not supports_threshold:
                threshold_value = mapped.pop("threshold")
                if supports_sigmoid and "sigmoid" not in mapped:
                    mapped["sigmoid"] = threshold_value
                elif supports_activate and "activate" not in mapped:
                    mapped["activate"] = threshold_value

            if "sigmoid" in mapped and not supports_sigmoid and supports_threshold:
                mapped["threshold"] = mapped.pop("sigmoid")

            return dice_helper_cls(*args, **mapped)

    return _NeuroTKCompatDiceHelper


def install_dice_helper_compat(monai_metrics_module) -> bool:
    dice_helper = getattr(monai_metrics_module, "DiceHelper", None)
    if dice_helper is None:
        return False
    if getattr(dice_helper, "__name__", "") == "_NeuroTKCompatDiceHelper":
        return False
    setattr(monai_metrics_module, "DiceHelper", make_dice_helper_compat(dice_helper))
    return True


def install_transform_compat(monai_transforms_module) -> Optional[Tuple[str, str]]:
    target = "RandScaleIntensityFixedMeand"
    if hasattr(monai_transforms_module, target):
        return None

    candidates = (
        "RandScaleIntensityFixedMeanD",
        "RandScaleIntensityFixedMeanDict",
        "RandScaleIntensityFixedMean",
    )
    for candidate in candidates:
        if hasattr(monai_transforms_module, candidate):
            setattr(monai_transforms_module, target, getattr(monai_transforms_module, candidate))
            return target, candidate

    raise ValueError(
        "MONAI transform compatibility error: bundle import expects "
        "`RandScaleIntensityFixedMeand`, but no compatible transform was found in installed MONAI. "
        "Install a supported MONAI release (recommended: >=1.3,<1.6)."
    )


def prepare_bundle_import_compat() -> Dict[str, object]:
    from monai import metrics as monai_metrics
    from monai import transforms as monai_transforms

    dice_patched = install_dice_helper_compat(monai_metrics)
    alias = install_transform_compat(monai_transforms)
    return {"dice_helper_patched": dice_patched, "transform_alias": alias}
