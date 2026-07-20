"""Visualize one patch from the S1S2 landslide reference dataset.

Shows the Sentinel-2 RGB composite (B04/B03/B02) before (PRE1) and after
(POST1) the landslide event, plus the landslide mask.

Usage:
    python visualize_s1s2_landslide.py [--split train|val|test] [--idx N] [--out PNG]
"""
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "/home/timz/scratch/s1s2_landslide_reference_data/reference_data"


def load_rgb(f: h5py.File, when: str, idx: int) -> np.ndarray:
    """Stack S2 bands 4,3,2 (R,G,B) for one patch and stretch for display."""
    rgb = np.stack(
        [f[f"S2_{when}_B{b}"][idx, 0] for b in ("04", "03", "02")], axis=-1
    )
    # Percentile stretch per image so reflectance values map to visible range.
    lo, hi = np.nanpercentile(rgb, (2, 98))
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--idx", type=int, default=0, help="patch index within the split")
    ap.add_argument("--out", default="s1s2_landslide_example.png")
    args = ap.parse_args()

    with h5py.File(f"{DATA_DIR}/{args.split}_s1s2a.h5", "r") as f:
        n = f["None_None_MASK"].shape[0]
        if not 0 <= args.idx < n:
            raise SystemExit(f"--idx must be in [0, {n}) for split {args.split}")
        pre = load_rgb(f, "PRE1", args.idx)
        post = load_rgb(f, "POST1", args.idx)
        mask = f["None_None_MASK"][args.idx, 0]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(pre)
    axes[0].set_title("S2 pre-event RGB (B04/B03/B02)")
    axes[1].imshow(post)
    axes[1].set_title("S2 post-event RGB (B04/B03/B02)")
    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Landslide mask ({100 * mask.mean():.1f}% positive)")
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(f"{args.split}_s1s2a.h5  patch {args.idx}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
