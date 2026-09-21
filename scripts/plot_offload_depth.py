#!/usr/bin/env python3
"""Validation offload across speculation depth, per far-uncore ratio.

  ./plot_offload_depth.py [summary_csv] [out_png]

Left: absolute qps, with each ratio's in-run exact search as a dashed reference. Right: the same
rows divided by that exact search. The right panel is the clean comparison -- both numbers come from
one process on one day, so the run-to-run drift between sessions cancels.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
# Ordinal blue ramp (steps 250 / 450 / 650): darker = slower far memory. Step 250 sits near 2:1 on
# the surface, so every series also carries its own marker shape and a direct label.
SERIES = [  # ratio, colour, marker, label
    (24, "#86b6ef", "o", "r=24 (2.4 GHz, unthrottled)"),
    (16, "#2a78d6", "s", "r=16 (1.6 GHz)"),
    (8, "#104281", "D", "r=8 (800 MHz, floor)"),
]

src = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/data/offload_100m_0916_1659_summary.csv")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else ".claude/assets/spec100m_offload_depth.png")

rows = list(csv.DictReader(l for l in src.read_text().splitlines() if not l.startswith("#")))
qps = defaultdict(dict)
exact = {}
for r in rows:
    qps[int(r["ratio"])][int(r["k"])] = float(r["qps"])
    exact[int(r["ratio"])] = float(r["exact_qps"])
KS = sorted({int(r["k"]) for r in rows})
xs = range(len(KS))

fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)

for ratio, colour, marker, label in SERIES:
    ys = [qps[ratio][k] for k in KS]
    ax_abs.axhline(exact[ratio] / 1e3, color=colour, linestyle="--", linewidth=1.2, zorder=2)
    ax_abs.plot(xs, [y / 1e3 for y in ys], color=colour, marker=marker, linewidth=2, markersize=8,
                solid_joinstyle="round", solid_capstyle="round", zorder=4, label=label)

    rel = [y / exact[ratio] for y in ys]
    ax_rel.plot(xs, rel, color=colour, marker=marker, linewidth=2, markersize=8,
                solid_joinstyle="round", solid_capstyle="round", zorder=4)
    ax_rel.annotate(f"r={ratio}", (xs[-1], rel[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=10, color=INK)

ax_abs.plot([], [], color=INK_MUTED, linestyle="--", linewidth=1.2, label="exact search, same run")

ax_rel.axhline(1.0, color=INK, linestyle="--", linewidth=1.2, zorder=3)
ax_rel.annotate("exact search = 1.0", (xs[0], 1.0), xytext=(0, -14), textcoords="offset points",
                ha="left", fontsize=9, color=INK_MUTED)
best = max((qps[8][k] / exact[8], i) for i, k in enumerate(KS))
ax_rel.annotate(f"{best[0]:.2f}×", (best[1], best[0]), xytext=(10, 2), textcoords="offset points",
                ha="left", va="bottom", fontsize=10, color=INK, fontweight="bold")

for ax, ylabel, title in (
    (ax_abs, "queries per second (thousands)", "Throughput"),
    (ax_rel, "speedup over exact search", "Relative to the exact search in the same run"),
):
    ax.set_facecolor(SURFACE)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"k={k}" for k in KS], fontsize=10, color=INK_MUTED)
    ax.set_xlabel("speculation depth", fontsize=10, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_MUTED)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=INK_MUTED, alpha=0.15, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_alpha(0.4)
ax_abs.set_ylim(bottom=0)
ax_rel.set_xlim(-0.3, len(KS) - 1 + 0.45)

fig.legend(*ax_abs.get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False,
           fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Validation offload vs speculation depth — SIFT100M, far cores at 800 MHz",
             fontsize=12, color=INK, x=0.01, ha="left", y=1.07)
fig.tight_layout()
dst.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(dst, dpi=160, bbox_inches="tight", facecolor=SURFACE)
print(dst)
