"""Visualize cached OlmoEarth features vs. raw S2 input for PASTIS test images.

Two figures per run:
  1. features_meanpool.png  -- for the first N test images, a row of [raw RGB | one PCA-RGB
     panel per feature dir], features mean-pooled over time. Each feature's true (gH,gW,D) is
     printed and shown as its subplot title; all panels rendered at the same display size
     regardless of native resolution.
  2. features_pertimestep_<cfg>_img<idx>.png -- for each feature dir and each image, a 2xT
     grid: top row the T raw RGB inputs, bottom row the T per-timestep PCA-RGB feature maps.

Feature -> RGB is per-image PCA: flatten (H*W, D), take the top 3 principal components,
reshape to (H,W,3), min-max normalize. Raw S2 RGB is bands [3,2,1] (R,G,B) with a per-image
percentile stretch for display.

    source env_olmo.sh    # (or any env with torch + matplotlib)
    python visualize_features.py                 # first 4 test images, all features/ dirs
    python visualize_features.py --n_images 4 --split test
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")        # headless cluster: never try an interactive backend (it hangs)

import numpy as np
import torch
import matplotlib.pyplot as plt

RGB_BANDS = [3, 2, 1]        # B04/B03/B02 = R,G,B in the 13-band stack
DISPLAY_PX = 128             # all feature panels rendered at this size (nearest, keeps blocks)


def raw_rgb(s2_t: torch.Tensor) -> np.ndarray:
    """(13,64,64) one timestep -> (64,64,3) uint8-ish float in [0,1], percentile-stretched."""
    rgb = s2_t[RGB_BANDS].float().numpy().transpose(1, 2, 0)   # (H,W,3)
    lo = np.percentile(rgb, 2, axis=(0, 1))
    hi = np.percentile(rgb, 98, axis=(0, 1))
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def pca_rgb(feat_hw_d: torch.Tensor) -> np.ndarray:
    """(H,W,D) -> (H,W,3) via per-image PCA: top-3 principal components as RGB, min-max norm.

    Top-3 dirs come from the DxD covariance eigendecomposition (eigh), not full SVD of the
    (H*W, D) data matrix: for D=768 the covariance is 768x768 regardless of H*W, so this is
    much faster than svd of a 4096x768 matrix (the ps1 64x64 grids)."""
    H, W, D = feat_hw_d.shape
    x = feat_hw_d.reshape(H * W, D).float().numpy()
    x = x - x.mean(0, keepdims=True)
    cov = (x.T @ x) / max(H * W - 1, 1)          # (D,D)
    # eigh returns eigenvalues ascending; take the top 3 eigenvectors as principal dirs.
    _, evecs = np.linalg.eigh(cov)
    dirs = evecs[:, -3:][:, ::-1]                # (D,3), descending variance
    proj = (x @ dirs).reshape(H, W, 3)
    lo = proj.min((0, 1), keepdims=True)
    hi = proj.max((0, 1), keepdims=True)
    return (proj - lo) / (hi - lo + 1e-6)


def pca_rgb_joint(feat_t_hw_d: torch.Tensor) -> np.ndarray:
    """(T,H,W,D) -> (T,H,W,3): ONE PCA fit over all T timesteps so colors are consistent
    across time (same principal directions + same min-max scaling for every t). Contrast
    with pca_rgb, which fits per-image and would make colors flicker across timesteps."""
    T, H, W, D = feat_t_hw_d.shape
    x = feat_t_hw_d.reshape(T * H * W, D).float().numpy()
    x = x - x.mean(0, keepdims=True)
    cov = (x.T @ x) / max(T * H * W - 1, 1)      # (D,D) over ALL timesteps
    _, evecs = np.linalg.eigh(cov)
    dirs = evecs[:, -3:][:, ::-1]                # (D,3), descending variance
    proj = (x @ dirs).reshape(T, H, W, 3)
    lo = proj.min((0, 1, 2), keepdims=True)      # shared scaling across all t
    hi = proj.max((0, 1, 2), keepdims=True)
    return (proj - lo) / (hi - lo + 1e-6)


def nearest_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Upscale (H,W,3) to (size,size,3) with nearest-neighbor (preserves patch blocks)."""
    H, W = img.shape[:2]
    yi = (np.arange(size) * H // size).clip(0, H - 1)
    xi = (np.arange(size) * W // size).clip(0, W - 1)
    return img[yi][:, xi]


def feature_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.glob("oe_*") if (d / f"pastis_r_test").exists()
                  or any(d.glob("pastis_r_*")))


def fig1_meanpool(root: Path, split: str, n_images: int, out: Path) -> None:
    """Row per image: [raw RGB | mean-pooled PCA-RGB per feature dir]."""
    dirs = feature_dirs(root)
    ncols = 1 + len(dirs)
    fig, axes = plt.subplots(n_images, ncols, figsize=(2.1 * ncols, 2.3 * n_images),
                             squeeze=False)
    for r in range(n_images):
        s2 = torch.load(root.parent / "data" / "pastis_olmoearth" / f"pastis_r_{split}"
                        / "s2_images" / f"{r}.pt")            # (T,13,64,64)
        ax = axes[r][0]
        ax.imshow(raw_rgb(s2.mean(0)))                        # time-averaged raw RGB
        ax.set_title("raw RGB (mean T)" if r == 0 else "", fontsize=8)
        ax.set_ylabel(f"test #{r}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        for c, d in enumerate(dirs, start=1):
            feat = torch.load(d / f"pastis_r_{split}" / f"{r}.pt")   # (T,gH,gW,D)
            gH, gW, D = feat.shape[1], feat.shape[2], feat.shape[3]
            rgb = pca_rgb(feat.mean(0))                       # mean over T -> (gH,gW,D)
            axes[r][c].imshow(nearest_resize(rgb, DISPLAY_PX))
            if r == 0:
                axes[r][c].set_title(f"{d.name}\n({gH}x{gW}x{D})", fontsize=7)
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
            if r == 0:
                print(f"{d.name}: feature (mean-T) {gH}x{gW}x{D}")

    fig.suptitle("Mean-pooled features (per-image PCA -> RGB) vs. raw input", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig2_pertimestep(root: Path, split: str, img_idx: int, d: Path, out: Path) -> None:
    """2xT grid for one (feature dir, image): raw RGB per t (top) + PCA-RGB feature per t."""
    s2 = torch.load(root.parent / "data" / "pastis_olmoearth" / f"pastis_r_{split}"
                    / "s2_images" / f"{img_idx}.pt")          # (T,13,64,64)
    feat = torch.load(d / f"pastis_r_{split}" / f"{img_idx}.pt")   # (T,gH,gW,D)
    T = feat.shape[0]
    gH, gW, D = feat.shape[1], feat.shape[2], feat.shape[3]

    feat_rgb = pca_rgb_joint(feat)               # (T,gH,gW,3), one PCA fit across all T
    fig, axes = plt.subplots(2, T, figsize=(1.35 * T, 3.0), squeeze=False)
    for t in range(T):
        axes[0][t].imshow(raw_rgb(s2[t]))
        axes[0][t].set_title(f"t{t}", fontsize=7)
        axes[1][t].imshow(nearest_resize(feat_rgb[t], DISPLAY_PX))
        for row in (0, 1):
            axes[row][t].set_xticks([]); axes[row][t].set_yticks([])
    axes[0][0].set_ylabel("raw RGB", fontsize=8)
    axes[1][0].set_ylabel("feat PCA", fontsize=8)
    fig.suptitle(f"{d.name}  test #{img_idx}  feature {gH}x{gW}x{D}, T={T}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features_root", default="features")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_images", type=int, default=4)
    ap.add_argument("--out_dir", default="feature_viz")
    args = ap.parse_args()

    root = Path(args.features_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = feature_dirs(root)
    print(f"{len(dirs)} feature dirs: {[d.name for d in dirs]}")

    fig1_meanpool(root, args.split, args.n_images, out_dir / "features_meanpool.png")

    # Per-timestep figures: only the first test image (img0), joint PCA across its 12 steps.
    for d in dirs:
        fig2_pertimestep(root, args.split, 0, d,
                         out_dir / f"pertimestep_{d.name}_img0.png")


if __name__ == "__main__":
    main()
