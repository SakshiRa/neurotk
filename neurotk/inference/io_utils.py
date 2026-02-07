from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np


PathLike = Union[str, os.PathLike]


def load_nifti(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(os.fspath(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img.affine


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    out_path: PathLike,
    header: Optional[nib.Nifti1Header] = None,
) -> None:
    out_path = os.fspath(out_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if header is not None:
        hdr = header.copy()
        img = nib.Nifti1Image(data, affine, header=hdr)
        img.set_qform(affine, code=int(hdr["qform_code"]) if hdr["qform_code"] > 0 else 1)
        img.set_sform(affine, code=int(hdr["sform_code"]) if hdr["sform_code"] > 0 else 1)
    else:
        img = nib.Nifti1Image(data, affine)
    nib.save(img, out_path)


def to_uint8_mask(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8:
        return data
    return data.astype(np.uint8, copy=False)
