"""Generate NeuroTK architecture figure for JORS submission."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(8, 13))
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis("off")

# ── colours ──────────────────────────────────────────────────────────────────
C_ENTRY   = "#2C3E50"   # dark blue-grey  — entry points
C_INPUT   = "#5D6D7E"   # mid-grey        — input
C_VALID   = "#1A6B8A"   # teal            — validation
C_REPORT  = "#2E86AB"   # lighter teal    — reports
C_PREP    = "#E67E22"   # orange          — optional preprocessing
C_INFER   = "#7D3C98"   # purple          — optional inference
C_EVAL    = "#1E8449"   # green           — evaluation
C_OUT     = "#117A65"   # dark green      — outputs
WHITE     = "#FFFFFF"
LIGHT     = "#F2F3F4"

def box(ax, x, y, w, h, label, sublabel=None, color=C_VALID, fontsize=10,
        radius=0.3, text_color=WHITE):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad=0.05,rounding_size={radius}",
                           facecolor=color, edgecolor="white", linewidth=1.5,
                           zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.18, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
        ax.text(x, y - 0.25, sublabel, ha="center", va="center",
                fontsize=7.5, color=text_color, style="italic", zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)

def arrow(ax, x, y_start, y_end, color="#555555"):
    ax.annotate("", xy=(x, y_end + 0.05), xytext=(x, y_start - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
                zorder=2)

def optional_label(ax, x, y, text="optional"):
    ax.text(x + 0.1, y, text, ha="left", va="center",
            fontsize=7.5, color="#888888", style="italic")

# ── title ─────────────────────────────────────────────────────────────────────
ax.text(5, 15.4, "NeuroTK", ha="center", va="center",
        fontsize=18, fontweight="bold", color=C_ENTRY)
ax.text(5, 14.95, "Dataset Validation & Standardization Toolkit",
        ha="center", va="center", fontsize=9, color="#555555")

# ── entry points ──────────────────────────────────────────────────────────────
y_entry = 14.2
for i, (label, cx) in enumerate([("CLI", 2.8), ("Python API", 5.0), ("Web UI", 7.2)]):
    box(ax, cx, y_entry, 2.8 if label == "Python API" else 2.2, 0.65,
        label, color=C_ENTRY, fontsize=9.5)

# bracket line from entry points down
ax.plot([2.8, 7.2], [y_entry - 0.33, y_entry - 0.33],
        color="#555555", lw=1.2, zorder=2)
ax.annotate("", xy=(5, y_entry - 0.33 - 0.35), xytext=(5, y_entry - 0.33),
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.5), zorder=2)

# ── NIfTI input ───────────────────────────────────────────────────────────────
y_input = 13.1
box(ax, 5, y_input, 5.5, 0.65, "NIfTI Dataset",
    sublabel="images/ + labels/ directories",
    color=C_INPUT, fontsize=10)

arrow(ax, 5, y_input - 0.33, 12.05)

# ── validation ────────────────────────────────────────────────────────────────
y_valid = 11.7
box(ax, 5, y_valid, 5.5, 0.75, "Dataset Validation",
    sublabel="geometry · spacing · orientation · labels · metadata",
    color=C_VALID, fontsize=10)

arrow(ax, 5, y_valid - 0.38, 10.72)

# ── QC report ─────────────────────────────────────────────────────────────────
y_rep = 10.38
box(ax, 5, y_rep, 5.5, 0.62, "Validation Report  (JSON / HTML)",
    color=C_REPORT, fontsize=9.5)

arrow(ax, 5, y_rep - 0.31, 9.42)

# ── optional preprocessing ────────────────────────────────────────────────────
y_prep = 9.07
box(ax, 5, y_prep, 5.5, 0.65,
    "Optional Standardization",
    sublabel="orientation normalization · voxel spacing resampling",
    color=C_PREP, fontsize=10)
optional_label(ax, 7.75, y_prep)

arrow(ax, 5, y_prep - 0.33, 8.12)

# ── processing log ────────────────────────────────────────────────────────────
y_log = 7.78
box(ax, 5, y_log, 5.5, 0.62,
    "Processing Log  (JSON — transformations applied)",
    color=C_PREP, fontsize=9)
optional_label(ax, 7.75, y_log)

arrow(ax, 5, y_log - 0.31, 6.82)

# ── optional inference ────────────────────────────────────────────────────────
y_infer = 6.47
box(ax, 5, y_infer, 5.5, 0.65,
    "Optional MONAI Bundle Inference",
    sublabel="auto-download from Hugging Face Hub · --skip-invalid-inputs",
    color=C_INFER, fontsize=10)
optional_label(ax, 7.75, y_infer)

arrow(ax, 5, y_infer - 0.33, 5.52)

# ── evaluation ────────────────────────────────────────────────────────────────
y_eval = 5.17
box(ax, 5, y_eval, 5.5, 0.65,
    "Evaluation & Quantification",
    sublabel="Dice · lesion volumes (mL) · cohort statistics · histograms",
    color=C_EVAL, fontsize=10)
optional_label(ax, 7.75, y_eval)

arrow(ax, 5, y_eval - 0.33, 4.22)

# ── outputs ───────────────────────────────────────────────────────────────────
y_out = 3.87
for cx, label in [(2.5, "volumes.csv"), (5.0, "summary.csv"), (7.5, "hist.png")]:
    box(ax, cx, y_out, 2.6, 0.58, label, color=C_OUT, fontsize=9)

# connecting line for outputs
ax.plot([2.5, 7.5], [y_out + 0.29, y_out + 0.29],
        color="#555555", lw=1.0, linestyle="--", zorder=2)

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (C_ENTRY, "User Interface (CLI / API / Web)"),
    (C_VALID, "Core — Validation"),
    (C_REPORT, "Output Artifact"),
    (C_PREP,  "Optional — Preprocessing"),
    (C_INFER, "Optional — Inference"),
    (C_EVAL,  "Optional — Evaluation"),
]
for i, (color, label) in enumerate(legend_items):
    bx = 0.45 + (i % 3) * 3.35
    by = 2.6 - (i // 3) * 0.52
    rect = FancyBboxPatch((bx, by - 0.17), 0.38, 0.34,
                           boxstyle="round,pad=0.02,rounding_size=0.05",
                           facecolor=color, edgecolor="white", lw=1, zorder=3)
    ax.add_patch(rect)
    ax.text(bx + 0.52, by, label, va="center", fontsize=7.5, color="#333333")

ax.text(5, 1.55, "Figure 1. NeuroTK architecture and processing pipeline.",
        ha="center", va="center", fontsize=8.5, color="#555555", style="italic")

plt.tight_layout(pad=0.2)
plt.savefig("neurotk_architecture.png", dpi=300, bbox_inches="tight",
            facecolor="white")
print("Saved: neurotk_architecture.png")
