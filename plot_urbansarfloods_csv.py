"""Plot per-class F1 (NF / FO / FU) from the UrbanSARFloods LP sweep CSV (concat head only).

Three subplots (one per class F1). Configs ordered by ascending patch_size, then ascending
tile_size.

    python plot_urbansarfloods_csv.py [--csv results/urbansarfloods_lp.csv] [--out results/urbansarfloods_f1.png]
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["NF", "FO"] # "FU"
HEAD = "concat"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/urbansarfloods_lp.csv")
    ap.add_argument("--out", default="results/urbansarfloods_f1.png")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv)) if r["head"] == HEAD]
    # config key = (patch_size, tile_size); order ascending patch then tile
    configs = sorted({(int(r["patch_size"]), int(r["tile_size"])) for r in rows},
                     key=lambda pt: (pt[0], pt[1]))
    labels = [f"ps{ps}\nt{ts}" for ps, ts in configs]

    f1 = {}
    for r in rows:
        key = (int(r["patch_size"]), int(r["tile_size"]))
        f1[key] = {cls: (float(r[f"{cls}_f1"]) if r.get(f"{cls}_f1", "") not in ("", None)
                         else np.nan) for cls in CLASSES}

    x = np.arange(len(configs))
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(15, 5), sharex=True, sharey=True)
    for ax, cls in zip(axes, CLASSES):
        vals = [f1[c].get(cls, np.nan) for c in configs]
        bars = ax.bar(x, vals, 0.6, color="#1f77b4")
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_title(f"{cls} F1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0.7, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlabel("config (ascending patch, ascending tile)")
    axes[0].set_ylabel("F1")
    fig.suptitle(f"UrbanSARFloods LP ({HEAD} head): per-class F1 by (patch, tile)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=130)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
