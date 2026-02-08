from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
from monai.bundle import ConfigParser, load
from monai.inferers import SlidingWindowInferer
from monai.transforms import Compose, SaveImaged

from .io_utils import save_nifti
from .monai_compat import prepare_bundle_import_compat, report_runtime_versions


class BundlePredictor:
    def __init__(self, bundle_dir: str, device: Optional[str] = None, ckpt_path: Optional[str] = None) -> None:
        self.bundle_dir = os.path.abspath(bundle_dir)
        self.device = device
        self.ckpt_path = ckpt_path
        self._segmenter = None
        self.parser = ConfigParser()
        self.config_file = self._resolve_config_file()
        self.parser.read_config(self.config_file)
        meta_path = os.path.join(self.bundle_dir, "configs", "metadata.json")
        if os.path.exists(meta_path):
            self.parser.read_meta(meta_path)
        self._net = None
        self._pre = None
        self._post = None
        self._inferer = None
        report_runtime_versions()
        self._load_from_bundle()

    def _resolve_config_file(self) -> str:
        cfg_dir = os.path.join(self.bundle_dir, "configs")
        for name in ["inference.yaml", "inference.json", "evaluate.yaml", "evaluate.json"]:
            path = os.path.join(cfg_dir, name)
            if os.path.exists(path):
                return path
        hp = os.path.join(cfg_dir, "hyper_parameters.yaml")
        if os.path.exists(hp):
            return hp
        raise FileNotFoundError("No inference/evaluate config found in bundle configs/")

    def _get_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_from_bundle(self) -> None:
        # Auto3DSeg bundle with scripts/segmenter.py + hyper_parameters.yaml
        if os.path.basename(self.config_file) == "hyper_parameters.yaml":
            scripts_dir = os.path.join(self.bundle_dir, "scripts")
            segmenter_py = os.path.join(scripts_dir, "segmenter.py")
            if os.path.exists(segmenter_py):
                import sys

                prepare_bundle_import_compat()
                sys.path.insert(0, scripts_dir)
                try:
                    from segmenter import Segmenter  # type: ignore
                except ImportError as exc:
                    raise ValueError(
                        "Failed to import bundle Segmenter due to MONAI API mismatch. "
                        "Install a supported MONAI version (recommended: >=1.3,<1.6) "
                        "or use a bundle compatible with your MONAI release."
                    ) from exc

                overrides = {
                    "infer#enabled": True,
                    "bundle_root": self.bundle_dir,
                    "ckpt_path": self.ckpt_path or os.path.join(self.bundle_dir, "model"),
                }
                dataset_json = os.path.join(os.path.dirname(self.bundle_dir), "dataset.json")
                if os.path.exists(dataset_json):
                    overrides["data_list_file_path"] = dataset_json
                overrides["data_file_base_dir"] = os.path.dirname(self.bundle_dir)
                try:
                    self._segmenter = Segmenter(config_file=self.config_file, config_dict=overrides)
                except TypeError as exc:
                    raise ValueError(
                        "Bundle Segmenter initialization failed due to MONAI API mismatch "
                        "(for example DiceHelper signature drift). "
                        "Install a supported MONAI version (recommended: >=1.3,<1.6)."
                    ) from exc
                return

        device = self._get_device()
        net_id = "network"
        if net_id not in self.parser:
            net_id = "network_def"
        net = self.parser.get_parsed_content(net_id, instantiate=True)
        net = net.to(device)
        net.eval()

        ckpt_path = self.ckpt_path
        if "checkpoint" in self.parser and self.ckpt_path is None:
            ckpt_path = self.parser.get_parsed_content("checkpoint", instantiate=False)
        if ckpt_path is None:
            for cand in ["models/best_metric_model.pt", "models/model_final.pt", "models/model.pt"]:
                p = os.path.join(self.bundle_dir, cand)
                if os.path.exists(p):
                    ckpt_path = p
                    break
        if ckpt_path is not None and not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(os.path.join(self.bundle_dir, ckpt_path))
        if ckpt_path is None:
            raise FileNotFoundError("No checkpoint found in bundle models/")

        model_file = ckpt_path if os.path.isabs(ckpt_path) else os.path.relpath(ckpt_path, self.bundle_dir)
        model_dict = load(
            name=".",
            bundle_dir=self.bundle_dir,
            model_file=model_file,
            workflow_type="inference",
        )
        if isinstance(model_dict, dict) and "state_dict" in model_dict:
            state = model_dict["state_dict"]
        elif isinstance(model_dict, dict) and "model" in model_dict:
            state = model_dict["model"]
        else:
            state = model_dict
        try:
            net.load_state_dict(state, strict=False)
        except RuntimeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
            net_cfg = cfg.get("network") if isinstance(cfg, dict) else None
            if isinstance(net_cfg, dict):
                alt_parser = ConfigParser({"network": net_cfg})
                net = alt_parser.get_parsed_content("network", instantiate=True)
                net = net.to(device)
                net.load_state_dict(state, strict=False)
            else:
                raise

        pre_id = "preprocessing"
        post_id = "postprocessing"
        inferer_id = "inferer"
        self._pre = self.parser.get_parsed_content(pre_id, instantiate=True) if pre_id in self.parser else None
        self._post = self.parser.get_parsed_content(post_id, instantiate=True) if post_id in self.parser else None
        if isinstance(self._post, Compose):
            self._post.transforms = [t for t in self._post.transforms if not isinstance(t, SaveImaged)]
        self._inferer = (
            self.parser.get_parsed_content(inferer_id, instantiate=True)
            if inferer_id in self.parser
            else SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.25)
        )
        self._net = net
        self.device = str(device)

    def predict_volume(self, image_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self._segmenter is not None:
            try:
                self._segmenter.config["data_file_base_dir"] = os.path.dirname(image_path)
            except Exception:
                pass
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Using a non-tuple sequence for multidimensional indexing is deprecated.*",
                    category=UserWarning,
                    module=r"monai\.inferers\.utils",
                )
                pred = self._segmenter.infer_image({"image": image_path})
            meta = {}
            if hasattr(pred, "meta"):
                meta = pred.meta
            return pred, meta

        data = {"image": image_path}
        if self._pre is not None:
            try:
                data = self._pre(data)
            except Exception:
                data = {"image": image_path, "label": image_path}
                data = self._pre(data)
        img = data["image"]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        img = img.unsqueeze(0).to(self._get_device())
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Using a non-tuple sequence for multidimensional indexing is deprecated.*",
                    category=UserWarning,
                    module=r"monai\.inferers\.utils",
                )
                pred = self._inferer(inputs=img, network=self._net)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        pred = pred.detach().cpu()
        data["pred"] = pred[0]
        if self._post is not None:
            try:
                data = self._post(data)
            except Exception:
                data["label"] = data.get("label", data["image"])
                data = self._post(data)
        meta = {}
        if isinstance(data, dict) and "pred_meta_dict" in data:
            meta = data["pred_meta_dict"]
        return data["pred"], meta

    def save_output(self, pred: torch.Tensor, meta: Dict[str, Any], out_path: str) -> None:
        import numpy as np

        affine = None
        if isinstance(meta, dict):
            affine = meta.get("affine") or meta.get("original_affine") or meta.get("qform") or meta.get("sform")
        if affine is None:
            affine = np.eye(4, dtype=np.float32)
        data = pred.detach().cpu().numpy()
        save_nifti(data, affine, out_path)
