from __future__ import annotations

import os
import re
from typing import Optional
from urllib.parse import urlparse


_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _parse_hf_repo_id(value: str) -> Optional[str]:
    if value.startswith("hf:"):
        repo_id = value[3:].strip("/")
        return repo_id or None

    if value.startswith("https:/huggingface.co/") and not value.startswith("https://huggingface.co/"):
        value = value.replace("https:/", "https://", 1)
    if value.startswith("https://huggingface.co/"):
        parsed = urlparse(value)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    if _REPO_ID_RE.match(value):
        return value
    return None


def _download_hf_repo(repo_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required for HF bundle downloads. "
            "Install with `pip install neurotk[inference]`."
        ) from exc

    return snapshot_download(repo_id=repo_id, repo_type="model")




def resolve_bundle_dir(bundle_dir: str) -> str:
    bundle_dir = os.path.abspath(bundle_dir) if os.path.exists(bundle_dir) else bundle_dir
    if os.path.isdir(bundle_dir):
        return bundle_dir

    repo_id = _parse_hf_repo_id(bundle_dir)
    if repo_id is not None:
        return _download_hf_repo(repo_id)

    raise FileNotFoundError(f"bundle_dir not found: {bundle_dir}")
