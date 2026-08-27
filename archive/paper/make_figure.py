"""
make_figure.py
Fig. 1 for the ICAC 2026 combined paper: four-component pipeline architecture.

Greyscale and line weight only, no colour dependence, so it stays legible as a
printed single-column IEEE figure. Box titles and body text are drawn
separately so neither overruns its container.

Output: reports/paper/fig1_architecture.png (600 dpi)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fig1_architecture.png")

FILL_INTAKE = "#EAEAEA"
FILL_COMP   = "#FFFFFF"
FILL_OUT    = "#CFCFCF"
EDGE        = "#1A1A1A"

TITLE_SIZE = 6.0
BODY_SIZE  = 5.4
NOTE_SIZE  = 4.8


def box(ax, x, y, w, h, title, body=None, fill=FILL_COMP, bold_title=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor=fill, edgecolor=EDGE, linewidth=0.8, zorder=2))
    if body is None:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=TITLE_SIZE, zorder=3, linespacing=1.5,
                fontweight="bold" if bold_title else "normal")
        return
    ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center",
            fontsize=TITLE_SIZE, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + h * 0.32, body, ha="center", va="center",
            fontsize=BODY_SIZE, zorder=3, linespacing=1.5)


def arrow(ax, p1, p2, dashed=False):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=6.5,
        linewidth=0.8, color=EDGE, zorder=1,
        linestyle=(0, (2.2, 1.8)) if dashed else "solid",
        shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(3.42, 3.35))   # IEEE single-column width
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

L, R, W = 0.05, 0.53, 0.38          # left/right column x, box width

# ── intake ───────────────────────────────────────────────────────────────────
box(ax, L, 0.885, W, 0.085, "SMS report\n(6 language styles)", fill=FILL_INTAKE)
box(ax, R, 0.885, W, 0.085, "Voice call\n(Si / Ta / En)", fill=FILL_INTAKE)

# ── components ───────────────────────────────────────────────────────────────
box(ax, L, 0.665, W, 0.155, "§V  SMS Triage",
    "guarded LLM\nextraction\nMPDS priority")
box(ax, R, 0.665, W, 0.155, "§VI  Clarification",
    "relevance-gated\nNER\nturn-by-turn policy")
box(ax, R, 0.435, W, 0.145, "§IV  Urgency",
    "91 acoustic\n+ 20 text\nHigh / Med / Low")
box(ax, 0.14, 0.215, 0.72, 0.115, "§VII  Incident-History Matching",
    "merge  /  separate  /  defer")
box(ax, 0.22, 0.045, 0.56, 0.075, "Ranked dispatch queue",
    fill=FILL_OUT, bold_title=True)

cxL, cxR = L + W / 2, R + W / 2

# ── flow ─────────────────────────────────────────────────────────────────────
arrow(ax, (cxL, 0.885), (cxL, 0.820))          # SMS in
arrow(ax, (cxR, 0.885), (cxR, 0.820))          # voice in
arrow(ax, (cxR, 0.665), (cxR, 0.580))          # dialogue -> urgency
arrow(ax, (cxL, 0.665), (cxL, 0.330))          # SMS record -> correlation
arrow(ax, (cxR, 0.435), (cxR, 0.330))          # urgency record -> correlation
arrow(ax, (0.50, 0.215), (0.50, 0.120))        # correlation -> queue

# transcript reused by the urgency component instead of re-transcribing
ax.plot([R + W, 0.955], [0.742, 0.742], color=EDGE, lw=0.8, zorder=1)
ax.plot([0.955, 0.955], [0.742, 0.508], color=EDGE, lw=0.8, zorder=1)
arrow(ax, (0.955, 0.508), (R + W, 0.508))
ax.text(0.968, 0.625, "transcript reused", fontsize=NOTE_SIZE, rotation=90,
        ha="left", va="center", style="italic")

# urgency label orders the dispatch queue
ax.plot([R, 0.475], [0.470, 0.470], color=EDGE, lw=0.8,
        linestyle=(0, (2.2, 1.8)), zorder=1)
ax.plot([0.475, 0.475], [0.470, 0.335], color=EDGE, lw=0.8,
        linestyle=(0, (2.2, 1.8)), zorder=1)
ax.text(0.468, 0.402, "urgency", fontsize=NOTE_SIZE, rotation=90,
        ha="right", va="center", style="italic")

fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
