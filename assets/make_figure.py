"""Generate NeuroTK workflow diagram for paper figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis("off")

def box(ax, x, y, w, h, label, sublabel=None, color="#4C72B0"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.92
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
            ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.25, sublabel,
                ha="center", va="center", fontsize=7.5, color="#dde4f0")

def arrow(ax, x1, x2, y=2.5):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color="#555555",
                                lw=1.8, mutation_scale=16))

# Input
box(ax, 0.2, 1.5, 1.8, 2.0, "NIfTI Dataset", ".nii / .nii.gz", color="#5A7FA8")

arrow(ax, 2.0, 2.4)

# Validate
box(ax, 2.4, 1.5, 1.8, 2.0, "Validate", "geometry · spacing\norientation · labels", color="#2E7D32")

# outputs from validate
ax.annotate("", xy=(3.3, 1.5), xytext=(3.3, 1.0),
            arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.4))
ax.text(3.3, 0.75, "JSON / HTML report", ha="center", fontsize=8, color="#444")

arrow(ax, 4.2, 4.6)

# Preprocess (optional)
box(ax, 4.6, 1.5, 1.9, 2.0, "Preprocess\n(optional)", "orient · resample\naudit trail", color="#1565C0")

arrow(ax, 6.5, 6.9)

# Infer
box(ax, 6.9, 1.5, 1.8, 2.0, "Infer", "MONAI bundle\nHuggingFace", color="#6A1B9A")

arrow(ax, 8.7, 9.1)

# Analyse
box(ax, 9.1, 1.5, 2.6, 2.0, "Analyse", "Dice · lesion volume\ncohort stats · histogram", color="#C62828")

# Title
ax.text(6.0, 4.6, "NeuroTK Workflow", ha="center", va="center",
        fontsize=13, fontweight="bold", color="#222")

plt.tight_layout()
plt.savefig("assets/workflow.png", dpi=180, bbox_inches="tight",
            facecolor="white", edgecolor="none")
print("Saved assets/workflow.png")
