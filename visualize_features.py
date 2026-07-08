"""Visualize cached OlmoEarth features vs. raw S2 input for PASTIS test images.

Two figures per run:
  1. features_meanpool.png  -- for the first N test images, a row of [raw RGB | one PCA-RGB
     panel per feature dir], features mean-pooled over time. Each feature's true (gH,gW,D) is
     printed and shown as its subplot title; all panels rendered at the same display size
     regardless of native resolution.
  2. pertimestep_img<idx>.png -- one combined grid per image: rows are the T timesteps, columns
     are [raw RGB | one per feature dir, ordered ascending (ps, tile)]. Each feature column uses
     its own joint PCA across time (colors consistent down a column, not across columns).

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


def pca_rgb_shared(fit: torch.Tensor, apply_to: list[torch.Tensor]) -> list[np.ndarray]:
    """Fit PCA (top-3 dirs + min-max scaling) on `fit` (H,W,D), then apply that SAME basis to
    every (H,W,D) map in `apply_to`, returning a list of (H,W,3). Sharing the basis makes the
    panels directly comparable (e.g. low-res vs AnyUp-upsampled): identical feature vectors get
    identical colors, so differences you see are real, not a per-panel recoloring."""
    Hf, Wf, D = fit.shape
    xf = fit.reshape(Hf * Wf, D).float().numpy()
    mean = xf.mean(0, keepdims=True)
    cov = ((xf - mean).T @ (xf - mean)) / max(Hf * Wf - 1, 1)   # (D,D) from the fit map
    _, evecs = np.linalg.eigh(cov)                              # ascending eigenvalues
    dirs = evecs[:, -3:][:, ::-1]                               # (D,3), descending variance
    proj_fit = (xf - mean) @ dirs
    lo, hi = proj_fit.min(0), proj_fit.max(0)                   # shared scaling from the fit map
    out = []
    for m in apply_to:
        H, W, _ = m.shape
        x = m.reshape(H * W, D).float().numpy()
        proj = ((x - mean) @ dirs).reshape(H, W, 3)
        out.append(np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1))
    return out


def nearest_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Upscale (H,W,3) to (size,size,3) with nearest-neighbor (preserves patch blocks)."""
    H, W = img.shape[:2]
    yi = (np.arange(size) * H // size).clip(0, H - 1)
    xi = (np.arange(size) * W // size).clip(0, W - 1)
    return img[yi][:, xi]


def _ps_tile(name: str) -> tuple[int, int]:
    """Parse (patch_size, tile_size) from a dir name like oe_base_s2_ps1_tile32.
    Missing/unparseable tokens sort last (inf) so odd names don't jump the order."""
    ps = tile = float("inf")
    for tok in name.split("_"):
        if tok.startswith("ps") and tok[2:].isdigit():
            ps = int(tok[2:])
        elif tok.startswith("tile") and tok[4:].isdigit():
            tile = int(tok[4:])
    return ps, tile


def feature_dirs(root: Path) -> list[Path]:
    # Order by (patch_size, tile_size) ascending -- numeric, so tile8 precedes tile32,
    # not the lexical order sorted() would give (tile1, tile32, tile64, tile8).
    return sorted((d for d in root.glob("oe_*") if (d / f"pastis_r_test").exists()
                   or any(d.glob("pastis_r_*"))),
                  key=lambda d: _ps_tile(d.name))


def has_samples(d: Path, split: str, indices) -> bool:
    """True iff `d` has every per-sample .pt this run will read for `split`.

    Extraction runs may still be in flight, so a feature dir can exist with only
    some samples written. Missing any needed index means we skip the whole dir
    (a partial column/figure would misalign the grid) with a warning."""
    missing = [i for i in indices if not (d / f"pastis_r_{split}" / f"{i}.pt").exists()]
    if missing:
        shown = ", ".join(str(i) for i in missing[:5]) + ("..." if len(missing) > 5 else "")
        print(f"WARNING: skipping {d.name}: missing {len(missing)} {split} file(s) "
              f"(idx {shown}) -- extraction likely still running")
        return False
    return True


def fig1_meanpool(root: Path, split: str, n_images: int, out: Path) -> None:
    """Row per image: [raw RGB | mean-pooled PCA-RGB per feature dir]."""
    # Only keep dirs that have all n_images samples; warn+skip the incomplete ones
    # up front so the subplot grid width matches what we actually render.
    dirs = [d for d in feature_dirs(root) if has_samples(d, split, range(n_images))]
    if not dirs:
        print("WARNING: no feature dirs with complete samples; skipping features_meanpool.png")
        return
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


def fig2_pertimestep(root: Path, split: str, img_idx: int, dirs: list[Path], out: Path) -> None:
    """One combined figure for an image: rows = the T timesteps, columns = [raw RGB | one per
    feature dir]. Grid is T x (1 + len(dirs)); `dirs` is already ordered ascending (ps, tile).

    Each feature column uses its OWN joint PCA (pca_rgb_joint) across that dir's T steps, so
    colors are consistent down the column (over time) but NOT comparable across columns -- each
    dir has its own basis, since grids/resolutions differ. Dirs missing this image's file are
    dropped with a warning (keeps the grid width honest during in-flight extraction)."""
    s2 = torch.load(root.parent / "data" / "pastis_olmoearth" / f"pastis_r_{split}"
                    / "s2_images" / f"{img_idx}.pt")          # (T,13,64,64)
    T = s2.shape[0]

    # Load each dir's per-timestep PCA up front; skip+warn any missing this image.
    cols = []   # (dir, feat_rgb (T,gH,gW,3), (gH,gW,D))
    for d in dirs:
        fp = d / f"pastis_r_{split}" / f"{img_idx}.pt"
        if not fp.exists():
            print(f"WARNING: pertimestep img{img_idx}: {d.name} missing {fp.name}, dropping column")
            continue
        feat = torch.load(fp)                                 # (T,gH,gW,D)
        cols.append((d, pca_rgb_joint(feat), feat.shape[1:]))
    if not cols:
        print(f"WARNING: pertimestep img{img_idx}: no feature dirs available; skipping")
        return

    ncols = 1 + len(cols)
    fig, axes = plt.subplots(T, ncols, figsize=(1.9 * ncols, 1.9 * T), squeeze=False)
    for t in range(T):
        axes[t][0].imshow(raw_rgb(s2[t]))
        axes[t][0].set_ylabel(f"t{t}", fontsize=8)
        axes[t][0].set_xticks([]); axes[t][0].set_yticks([])
        if t == 0:
            axes[t][0].set_title("raw RGB", fontsize=8)
        for c, (d, feat_rgb, (gH, gW, D)) in enumerate(cols, start=1):
            axes[t][c].imshow(nearest_resize(feat_rgb[t], DISPLAY_PX))
            axes[t][c].set_xticks([]); axes[t][c].set_yticks([])
            if t == 0:
                axes[t][c].set_title(f"{d.name}\n({gH}x{gW}x{D})", fontsize=7)

    fig.suptitle(f"Per-timestep features (per-dir joint PCA) -- test #{img_idx}, T={T}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig3_anyup(root: Path, split: str, n_images: int, d: Path, out: Path,
               ref_dir: Path | None = None, pca_mode: str = "all") -> None:
    """AnyUp guided upsampling for one feature dir (e.g. ps4 tile64, 16x16 grid).

    Per image, a row of 3-4: [raw RGB | low-res mean-pooled features | AnyUp-upsampled to 64x64
    | (optional) native reference `ref_dir` (ps1 tile1, a real 64x64x768 grid)].

    pca_mode controls how the feature panels are colored -- the reference is INDEPENDENTLY
    extracted from lr, so sharing its basis is a judgement call, hence three variants:
      "all"         -- one PCA basis (fit on lr) applied to lr, AnyUp, AND the reference. All
                       feature panels directly comparable in color.
      "lr_anyup"    -- shared basis for lr+AnyUp (what AnyUp changed is comparable); the
                       reference gets its OWN per-image PCA (honest to its independence).
      "independent" -- every feature panel gets its own per-image PCA (max contrast each,
                       no cross-panel color meaning).

    Guidance is the time-averaged (3,64,64) RGB, matching the plain 'anyup' head. AnyUp is the
    frozen pretrained upsampler; we import it lazily via lp_on_cached_features' shared modules so
    the RGB normalization is bit-identical to training."""
    assert pca_mode in ("all", "lr_anyup", "independent"), f"bad pca_mode {pca_mode}"
    # Import here (not at module top): pulls torch.hub AnyUp + olmo bootstrap, which are heavy
    # and only needed for --anyup. Keeps the default LP-only run light.
    from finetune_olmoearth_pastis import AnyUpUpsampleProbe, _load_rgb_guidance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anyup = AnyUpUpsampleProbe(num_classes=1).anyup.to(device).eval()   # frozen upsampler only

    ncols = 4 if ref_dir is not None else 3
    fig, axes = plt.subplots(n_images, ncols, figsize=(ncols * 2.3, 2.3 * n_images),
                             squeeze=False)
    for r in range(n_images):
        feat_path = d / f"pastis_r_{split}" / f"{r}.pt"
        if not feat_path.exists():
            print(f"WARNING: skipping anyup for {d.name} img{r}: {feat_path.name} missing")
            for c in range(ncols):
                axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
            continue
        feat = torch.load(feat_path)                       # (T,gH,gW,D)
        lr = feat.float().mean(0)                          # (gH,gW,D) time-averaged
        gH, gW, D = lr.shape

        s2 = torch.load(root.parent / "data" / "pastis_olmoearth" / f"pastis_r_{split}"
                        / "s2_images" / f"{r}.pt")         # (T,13,64,64)
        rgb_guide = _load_rgb_guidance(split, r, temporal=False)   # (3,64,64), normalized
        out_hw = (s2.shape[-2], s2.shape[-1])              # (64,64)

        with torch.no_grad():
            f_in = lr.permute(2, 0, 1).unsqueeze(0).to(device)     # (1,D,gH,gW)
            g_in = rgb_guide.unsqueeze(0).to(device)               # (1,3,64,64)
            hr = anyup(g_in, f_in, output_size=out_hw)             # (1,D,64,64)
        hr = hr.squeeze(0).permute(1, 2, 0).cpu()                  # (64,64,D)

        # Native per-pixel reference (ps1 tile1): mean-pool over T -> (64,64,D). Only usable
        # under the SHARED basis if it has the same D; otherwise show blank + warn.
        ref = None
        if ref_dir is not None:
            ref_path = ref_dir / f"pastis_r_{split}" / f"{r}.pt"
            if ref_path.exists():
                cand = torch.load(ref_path).float().mean(0)        # (rH,rW,D)
                if cand.shape[-1] == D:
                    ref = cand
                else:
                    print(f"WARNING: {ref_dir.name} img{r} D={cand.shape[-1]} != {D}; "
                          f"can't share PCA basis, skipping ref panel")
            else:
                print(f"WARNING: ref {ref_dir.name} img{r} missing {ref_path.name}; blank panel")

        # Color the feature panels per pca_mode. lr+AnyUp always share a basis in "all" and
        # "lr_anyup" (that comparison is the point of the figure); the reference and the
        # "independent" case use per-image PCA (pca_rgb).
        if pca_mode == "all" and ref is not None:
            lr_rgb, hr_rgb, ref_rgb = pca_rgb_shared(lr, [lr, hr, ref])
        elif pca_mode == "independent":
            lr_rgb, hr_rgb = pca_rgb(lr), pca_rgb(hr)
            ref_rgb = pca_rgb(ref) if ref is not None else None
        else:  # "lr_anyup", or "all" with no reference -> share lr+AnyUp, ref independent
            lr_rgb, hr_rgb = pca_rgb_shared(lr, [lr, hr])
            ref_rgb = pca_rgb(ref) if ref is not None else None

        panels = [raw_rgb(s2.mean(0)), nearest_resize(lr_rgb, DISPLAY_PX),
                  nearest_resize(hr_rgb, DISPLAY_PX)]
        titles = ["raw RGB (mean T)", f"low-res ({gH}x{gW}x{D})", f"AnyUp -> 64x64x{D}"]
        if ref_dir is not None:
            if ref is not None:
                panels.append(nearest_resize(ref_rgb, DISPLAY_PX))
                titles.append(f"{ref_dir.name}\n(native {ref.shape[0]}x{ref.shape[1]}x{D})")
            else:
                panels.append(np.zeros((DISPLAY_PX, DISPLAY_PX, 3)))   # missing -> blank
                titles.append(f"{ref_dir.name}\n(missing)")
        for c in range(ncols):
            axes[r][c].imshow(panels[c])
            if r == 0:
                axes[r][c].set_title(titles[c], fontsize=8)
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        axes[r][0].set_ylabel(f"test #{r}", fontsize=9)

    ref_note = f" vs {ref_dir.name}" if ref_dir is not None else ""
    mode_note = {"all": "shared PCA (lr+AnyUp+ref)",
                 "lr_anyup": "shared PCA (lr+AnyUp), ref independent",
                 "independent": "independent PCA per panel"}[pca_mode]
    fig.suptitle(f"AnyUp guided upsampling: {d.name}{ref_note} -- {mode_note}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features_root", default="~/projects/aip-gpleiss/timz/features")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_images", type=int, default=4)
    ap.add_argument("--out_dir", default="feature_viz")
    ap.add_argument("--anyup", action="store_true",
                    help="also render AnyUp guided upsampling of --anyup_dir (needs GPU + torch.hub)")
    ap.add_argument("--anyup_dir", default="oe_base_s2_ps4_tile64",
                    help="feature dir name to AnyUp-upsample (default the 16x16 ps4 tile64 grid)")
    ap.add_argument("--anyup_ref_dir", default="oe_base_s2_ps1_tile64",
                    help="native per-pixel feature dir shown as the rightmost reference column "
                         "(same D required to share the PCA basis); '' to omit")
    args = ap.parse_args()

    root = Path(args.features_root).expanduser().resolve()   # expanduser: default uses ~
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = feature_dirs(root)
    print(f"{len(dirs)} feature dirs: {[d.name for d in dirs]}")

    fig1_meanpool(root, args.split, args.n_images, out_dir / "features_meanpool.png")

    # Per-timestep figure: one combined grid per image (rows=timesteps, cols=raw RGB + each dir).
    for r in range(args.n_images):
        fig2_pertimestep(root, args.split, r, dirs,
                         out_dir / f"pertimestep_img{r}.png")

    if args.anyup:
        ad = root / args.anyup_dir
        if not ad.exists() or not any(ad.glob("pastis_r_*")):
            print(f"WARNING: --anyup_dir {args.anyup_dir} not found under {root}; skipping AnyUp")
        else:
            ref = root / args.anyup_ref_dir if args.anyup_ref_dir else None
            if ref is not None and not (ref.exists() and any(ref.glob("pastis_r_*"))):
                print(f"WARNING: --anyup_ref_dir {args.anyup_ref_dir} not found under {root}; "
                      f"omitting reference column")
                ref = None
            # Three PCA-coloring variants -- see fig3_anyup docstring for what each means.
            for mode in ("all", "lr_anyup", "independent"):
                fig3_anyup(root, args.split, args.n_images, ad,
                           out_dir / f"anyup_{args.anyup_dir}_pca-{mode}.png",
                           ref_dir=ref, pca_mode=mode)


if __name__ == "__main__":
    main()

# python -u visualize_features.py