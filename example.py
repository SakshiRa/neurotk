"""
NeuroTK end-to-end example using the included sample data.

Demonstrates:
  1. Validating a dataset via the Python API
  2. Inspecting the cohort-level summary
  3. Standardizing spacing and orientation
  4. Checking the preprocessing report

Run from the repository root:
    uv run python example.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent
IMAGES_DIR = REPO_ROOT / "sample_data" / "images"
LABELS_DIR = REPO_ROOT / "sample_data" / "labels"

# ── 1. Validate ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1 — Dataset Validation")
print("=" * 60)

from neurotk.validate import validate_dataset

report = validate_dataset(
    images_dir=IMAGES_DIR,
    labels_dir=LABELS_DIR,
)

summary = report["summary"]
print(f"  Images found        : {summary['num_images']}")
print(f"  Labels found        : {summary['num_labels']}")
print(f"  Files with issues   : {summary['files_with_issues']}")
print(f"  Modal shape         : {summary['modal_shape']}")
print(f"  Mean spacing (mm)   : {[round(s, 3) for s in summary['spacing_mean']]}")
print(f"  Orientation         : {summary['orientation_modal']}")

if summary["files_with_issues"] == 0:
    print("\n  All files passed validation.")
else:
    print("\n  Issues found:")
    for fname, fdata in report["files"].items():
        if fdata["issues"]:
            print(f"    {fname}: {fdata['issues']}")

# ── 2. Per-file detail ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2 — Per-file detail")
print("=" * 60)

for fname, fdata in report["files"].items():
    img = fdata["image"]
    lbl = fdata["label"]
    print(f"\n  {fname}")
    print(f"    shape    : {img['shape']}")
    print(f"    spacing  : {[round(s, 3) for s in img['spacing']]}")
    print(f"    orient   : {img['orientation']}")
    if lbl:
        print(f"    label    : present={lbl['present']}, "
              f"unique={lbl.get('unique_values')}")

# ── 3. Preprocess ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3 — Dataset Standardization")
print("=" * 60)

from neurotk.preprocess import preprocess_dataset

with tempfile.TemporaryDirectory() as tmp:
    out_dir = Path(tmp) / "preprocessed"

    pp_report = preprocess_dataset(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        out_dir=out_dir,
        spacing=(1.0, 1.0, 1.0),
        orientation="RAS",
    )

    print(f"  Processed {len(pp_report['processed_files'])} files "
          f"→ {out_dir}")

    for fname, fdata in pp_report["files"].items():
        img_info = fdata.get("image") or {}
        transforms = img_info.get("transforms", [])
        print(f"  {fname}: transforms applied = {transforms or 'none'}")

    # show the JSON report structure
    report_path = out_dir / "preprocess_report.json"
    if report_path.exists():
        with report_path.open() as f:
            pp_json = json.load(f)
        print(f"\n  preprocess_report.json keys: {list(pp_json.keys())}")

print("\nDone.")
