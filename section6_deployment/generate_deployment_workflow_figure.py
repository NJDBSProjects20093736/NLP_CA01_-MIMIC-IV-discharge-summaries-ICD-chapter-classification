
"""
Deployment — Human-in-the-loop workflow diagram 
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED = os.path.join(PROJECT_ROOT, "shared")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from paths import *  
ensure_dirs()

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(FIGURES_DIR, "fig6_deployment_workflow.png")


def box(ax, xy, text, fc="#E8F4FC", ec="#2B6CB0", w=2.35, h=0.72):
    """Draw a rounded box with centred label."""
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.6, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, color="#1A365D")
    return patch


def arrow(ax, start, end, color="#4A5568"):
    """Draw a directed arrow between two points."""
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.5, color=color, shrinkA=4, shrinkB=4,
    ))


def main():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Main horizontal workflow (left → right)
    y = 2.6
    steps = [
        (1.4, "1. EHR export\n(discharge summary)"),
        (3.5, "2. Load pipeline\nbest_model.pkl"),
        (5.6, "3. Predict\nICD chapter"),
        (7.7, "4. Clinical coder\nreview"),
        (9.8, "5. Final ICD code\n(official rules)"),
    ]
    for i, (x, label) in enumerate(steps):
        # Final step highlighted green — human makes the official decision
        fc, ec = ("#D4EDDA", "#276749") if i == 4 else ("#E8F4FC", "#2B6CB0")
        box(ax, (x, y), label, fc=fc, ec=ec)

    for i in range(len(steps) - 1):
        arrow(ax, (steps[i][0] + 1.18, y), (steps[i + 1][0] - 1.18, y))

    # Safety branch: low-confidence predictions must be reviewed
    box(ax, (5.6, 4.15), "Flag low-confidence /\nweak chapters\n(mandatory review)",
        fc="#FFF3CD", ec="#B7791F", w=2.5, h=0.85)
    arrow(ax, (5.6, 3.55), (5.6, 3.72), color="#B7791F")
    arrow(ax, (6.35, 4.15), (7.2, 3.05), color="#B7791F")

    ax.text(5.5, 1.35,
        "Model output = decision support only — human coder makes the final coding decision",
        ha="center", fontsize=10, style="italic", color="#4A5568",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7FAFC", edgecolor="#CBD5E0"))
    ax.set_title("Figure 6.1 — Proposed human-in-the-loop deployment workflow",
        fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
