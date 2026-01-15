"""NIfTI input helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def load_nifti(path: Path) -> Tuple[nib.Nifti1Image, np.ndarray, np.dtype]:
    """Load a NIfTI file and return image, float data, and original dtype."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float64)
    dtype = np.asarray(img.dataobj).dtype
    return img, data, dtype
