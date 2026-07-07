"""Visualize lp_olmoearth_pastis.csv: one subplot per feature set, test_miou on y,
points colored by head_mode, shared y-axis across subplots.

    python visualize_lp_results.py                      # -> lp_olmoearth_pastis.png
    python visualize_lp_results.py --csv other.csv --out other.png
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Categorical palette (dataviz skill, fixed order -- color follows the head_mode entity,
# assigned in order, never cycled). Any head_mode beyond these falls back to grey.
HEAD_COLORS = {
    "lp_pa2pa_bu": "#2a78d6",   # blue
    "lp_pa2px":    "#1baf7a",   # aqua
    "anyup":       "#eda100",   # yellow
    "anyup_t1":    "#008300",   # green
    "anyup_t2":    "#4a3aa7",   # violet
    "anyup_t1_ens": "#e34948",  # red
    "anyup_t2_ens": "#e87ba4",  # magenta
}
FALLBACK = "#8a8a86"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="lp_olmoearth_pastis.csv")
    ap.add_argument("--out", default="lp_olmoearth_pastis.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    features = sorted(df["features"].unique())
    # Legend order: fixed palette order first, then any extras, restricted to what's present.
    present = set(df["head_mode"].unique())
    heads = [h for h in HEAD_COLORS if h in present] + sorted(present - set(HEAD_COLORS))

    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 1, 4.2), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, feat in zip(axes, features):
        sub = df[df["features"] == feat]
        for h in heads:
            pts = sub[sub["head_mode"] == h]
            if pts.empty:
                continue
            ax.scatter([h] * len(pts), pts["test_miou"],
                       color=HEAD_COLORS.get(h, FALLBACK), s=60, zorder=3,
                       edgecolor="white", linewidth=0.6)
        ax.set_title(feat, fontsize=9)
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels(heads, rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", color="#e6e6e3", linewidth=0.8, zorder=0)  # recessive grid
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("test mIoU")
    # One legend for the whole figure (identity is never color-alone).
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=7,
                          markerfacecolor=HEAD_COLORS.get(h, FALLBACK),
                          markeredgecolor="white", label=h) for h in heads]
    fig.legend(handles=handles, title="head_mode", loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.suptitle("LP head test mIoU by feature set", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.99, 1))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}  ({n} feature sets, {len(heads)} head modes)")


if __name__ == "__main__":
    main()
