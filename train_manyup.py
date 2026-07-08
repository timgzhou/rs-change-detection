"""Train mAnyUp: a guided feature upsampler for OlmoEarth (multimodal guidance).

Adapts wimmerth/anyup to our remote-sensing setting. AnyUp upsamples a low-res feature map to
a high-res one, guided by a co-located RGB image. We keep AnyUp's model + Cosine_MSE loss, but
replace its self-supervised crop trick with REAL low-res / high-res feature pairs from our cache:

  input  (LR feats): oe_base_s2_ps4_tile64   -> (T, 16, 16, 768)   cheap, coarse
  target (HR feats): oe_base_s2_ps1_tile1    -> (T, 64, 64, 768)   expensive, per-pixel  (swappable)
  guidance:          full Sentinel-2 image   -> (13, 64, 64)       high-res, multi-band

For v1 we mean-pool over T on both features and guidance -> a single (D, gH, gW) map each, matching
the plain 'anyup' head in lp_on_cached_features. The only architectural change vs AnyUp is the
guidance input channel count: 13 (S2 bands) instead of 3 (RGB). Trained from scratch (13!=3 makes
the pretrained RGB weights unusable anyway).

Losses (subset of AnyUp's three):
  anyup_hr   -- upsample LR -> compare to HR target. The objective.
  anyup_down -- area-downsample the prediction back to the LR grid, match the LR input feats.
  (anyup_reg is intentionally omitted for now.)

    source env_olmo.sh
    python train_manyup.py --help
    # typical (from a GPU node):
    python train_manyup.py --epochs 20 --batch_size 8 --stage_to_tmpdir

The AnyUp repo is imported from its clone; set --anyup_repo if it moved.
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")        # headless cluster
import matplotlib.pyplot as plt

# ----- locate the cloned AnyUp repo and import its model + loss (reuse, don't reimplement) -----
DEFAULT_ANYUP_REPO = "/scratch/timz/anyup"


def _import_anyup(repo: str):
    sys.path.insert(0, repo)
    from anyup.model import AnyUp          # noqa: E402
    from anyup.loss import Cosine_MSE      # noqa: E402
    return AnyUp, Cosine_MSE


DATA_ROOT = Path("/scratch/timz/rs-change-detection/data/pastis_olmoearth")
FEATURES_ROOT = Path("/home/timz/projects/aip-gpleiss/timz/features")
S2_BANDS = 13


# --------------------------------------------------------------------------------------------- #
# Dataset: pair (LR feats, HR feats, S2 guidance) per sample, skipping any index missing in any
# of the three sources (robust to partially-extracted feature dirs and to swapping the HR target).
# --------------------------------------------------------------------------------------------- #
class PairedFeatureDataset(Dataset):
    def __init__(self, lr_dir: Path, hr_dir: Path, s2_dir: Path, split: str):
        self.lr_dir = lr_dir / f"pastis_r_{split}"
        self.hr_dir = hr_dir / f"pastis_r_{split}"
        self.s2_dir = s2_dir / f"pastis_r_{split}" / "s2_images"

        def indices(d: Path):
            return {int(p.stem) for p in d.glob("*.pt")}

        # Train only on indices present in ALL three sources.
        common = indices(self.lr_dir) & indices(self.hr_dir) & indices(self.s2_dir)
        self.ids = sorted(common)
        if not self.ids:
            raise RuntimeError(f"no common {split} samples across\n  {self.lr_dir}\n  {self.hr_dir}"
                               f"\n  {self.s2_dir}")
        n_lr, n_hr, n_s2 = len(indices(self.lr_dir)), len(indices(self.hr_dir)), len(indices(self.s2_dir))
        print(f"[{split}] LR={n_lr} HR={n_hr} S2={n_s2} -> {len(self.ids)} common samples")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        # features: (T, gH, gW, D) fp16 -> mean over T -> (D, gH, gW) fp32
        lr = torch.load(self.lr_dir / f"{idx}.pt").float().mean(0).permute(2, 0, 1)   # (D, gh, gw)
        hr = torch.load(self.hr_dir / f"{idx}.pt").float().mean(0).permute(2, 0, 1)   # (D, GH, GW)
        # guidance: (T, 13, 64, 64) fp32 -> mean over T -> (13, 64, 64)
        s2 = torch.load(self.s2_dir / f"{idx}.pt").float().mean(0)                    # (13, 64, 64)
        s2 = _norm_guidance(s2)
        return lr, hr, s2


def _norm_guidance(s2: torch.Tensor) -> torch.Tensor:
    """Per-image, per-band min-max normalize the S2 guidance to [0,1]. No ImageNet stats -- with
    13 bands and a from-scratch guidance encoder, a simple [0,1] scaling is the natural choice
    (the encoder learns its own band statistics)."""
    C = s2.shape[0]
    flat = s2.reshape(C, -1)
    lo = flat.min(1).values.view(C, 1, 1)
    hi = flat.max(1).values.view(C, 1, 1)
    return (s2 - lo) / (hi - lo + 1e-6)


# --------------------------------------------------------------------------------------------- #
def stage_to_tmpdir(dirs: list[Path]) -> list[Path]:
    """Copy feature/data dirs into $SLURM_TMPDIR (node-local SSD) if set and they fit, returning
    the new paths. Cached features + S2 images are large and moved to GPU every step, so reading
    them from fast local disk instead of Lustre is a big speedup. No-op (returns originals) if
    SLURM_TMPDIR is unset or the data doesn't fit in the available space."""
    tmp = os.environ.get("SLURM_TMPDIR")
    if not tmp:
        print("SLURM_TMPDIR unset -> reading features in place (no staging).")
        return dirs
    tmp = Path(tmp)

    def dir_bytes(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    total = sum(dir_bytes(d) for d in dirs)
    free = shutil.disk_usage(tmp).free
    if total > free * 0.95:
        print(f"staging: need {total/1e9:.1f} GB but only {free/1e9:.1f} GB free in {tmp} "
              f"-> reading in place.")
        return dirs

    staged = []
    for d in dirs:
        dst = tmp / d.name
        if dst.exists():
            print(f"staging: {dst} already present, reusing.")
        else:
            t0 = time.time()
            shutil.copytree(d, dst)
            print(f"staged {d} -> {dst} ({dir_bytes(d)/1e9:.1f} GB, {time.time()-t0:.0f}s)")
        staged.append(dst)
    return staged


def _pca_rgb_shared(fit_chw: torch.Tensor, maps_chw: list[torch.Tensor]) -> list[np.ndarray]:
    """Fit PCA (top-3 dirs + min-max) on `fit_chw` (C,H,W), apply the SAME basis to each map in
    `maps_chw` -> list of (H,W,3). Shared basis so lr/manyup/hr panels are color-comparable."""
    C = fit_chw.shape[0]
    xf = fit_chw.reshape(C, -1).T.float().cpu().numpy()      # (H*W, C)
    mean = xf.mean(0, keepdims=True)
    cov = ((xf - mean).T @ (xf - mean)) / max(xf.shape[0] - 1, 1)
    _, evecs = np.linalg.eigh(cov)
    dirs = evecs[:, -3:][:, ::-1]                            # (C,3)
    proj_fit = (xf - mean) @ dirs
    lo, hi = proj_fit.min(0), proj_fit.max(0)
    out = []
    for m in maps_chw:
        H, W = m.shape[-2:]
        x = m.reshape(C, -1).T.float().cpu().numpy()
        proj = ((x - mean) @ dirs).reshape(H, W, 3)
        out.append(np.clip((proj - lo) / (hi - lo + 1e-6), 0, 1))
    return out


def _raw_rgb(s2_chw: torch.Tensor) -> np.ndarray:
    """(13,64,64) -> (64,64,3) percentile-stretched RGB (bands 3,2,1 = B04/B03/B02)."""
    rgb = s2_chw[[3, 2, 1]].float().cpu().numpy().transpose(1, 2, 0)
    lo = np.percentile(rgb, 2, (0, 1)); hi = np.percentile(rgb, 98, (0, 1))
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


@torch.no_grad()
def save_epoch_viz(model, proj_head, sample, GH, GW, device, out_path, epoch) -> None:
    """4-panel viz of one held-out TEST sample: raw RGB | LR feats | mAnyUp output | HR target.
    Feature panels share one PCA basis (fit on HR) so they're directly comparable. mAnyUp panel
    is the projected output (proj_head applied) -- i.e. what the HR loss actually sees."""
    lr, hr, s2 = sample
    lr_b = lr.unsqueeze(0).to(device); s2_b = s2.unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        pred = model(s2_b, lr_b, (GH, GW))
        if proj_head is not None:
            pred = proj_head(pred)
    if was_training:
        model.train()
    pred = pred.squeeze(0).float()                           # (D,GH,GW)

    # Shared PCA basis fit on the HR target; lr upsampled (nearest) only for display sizing.
    lr_disp = F.interpolate(lr.unsqueeze(0).float(), size=(GH, GW), mode="nearest").squeeze(0)
    lr_rgb, mu_rgb, hr_rgb = _pca_rgb_shared(hr, [lr_disp, pred.cpu(), hr])

    fig, axes = plt.subplots(1, 4, figsize=(4 * 2.6, 2.8))
    panels = [_raw_rgb(s2), lr_rgb, mu_rgb, hr_rgb]
    titles = ["raw S2 RGB", f"LR feats ({lr.shape[-2]}x{lr.shape[-1]})",
              "mAnyUp (proj)", f"HR target ({GH}x{GW})"]
    for ax, p, t in zip(axes, panels, titles):
        ax.imshow(p); ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"epoch {epoch} -- test sample (shared PCA on HR)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved viz {out_path}")


def _warmup_cosine(optimizer, total_steps, warmup_frac, lr, lr_min):
    """LambdaLR: linear warmup 0->1 over the first warmup_frac of steps, then cosine decay to
    lr_min/lr. Stepped per batch. total_steps = epochs * batches_per_epoch."""
    import math
    warmup_steps = max(1, int(total_steps * warmup_frac))
    floor = lr_min / lr if lr > 0 else 0.0

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps                       # linear 0 -> 1
        # cosine 1 -> floor over the remaining steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def _bilinear_baseline(loader, criterion, GH, GW, device, n_batches=20) -> float:
    """Mean HR Cosine_MSE loss when the LR feats are upsampled by plain bilinear interpolation
    (no guidance, no model). This is the bar the learned upsampler must clear. Averaged over the
    first n_batches for a stable, cheap estimate."""
    tot, cnt = 0.0, 0
    for bi, (lr, hr, _s2) in enumerate(loader):
        lr, hr = lr.to(device), hr.to(device)
        up = F.interpolate(lr.float(), size=(GH, GW), mode="bilinear", align_corners=False)
        tot += criterion(up, hr)["total"].item()
        cnt += 1
        if bi + 1 >= n_batches:
            break
    return tot / max(cnt, 1)


# --------------------------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lr_cfg", default="oe_base_s2_ps4_tile64",
                   help="low-res INPUT feature dir (cheap, coarse)")
    p.add_argument("--hr_cfg", default="oe_base_s2_ps1_tile64",
                   help="high-res TARGET feature dir (swap to oe_base_s2_ps1_tile64 when extracted)")
    p.add_argument("--split", default="train")
    p.add_argument("--features_root", default=str(FEATURES_ROOT))
    p.add_argument("--data_root", default=str(DATA_ROOT))
    p.add_argument("--anyup_repo", default=DEFAULT_ANYUP_REPO)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="AnyUp uses 2e-4, but our loss surface is flatter (LR/HR feats start "
                        "already aligned), so a higher LR converges faster")
    p.add_argument("--lr_min", type=float, default=1e-6,
                   help="floor the cosine decay reaches at the end of training")
    p.add_argument("--warmup_frac", type=float, default=0.05,
                   help="fraction of total steps for linear LR warmup before cosine decay")
    p.add_argument("--baseline_batches", type=int, default=20,
                   help="batches to average the bilinear baseline over")
    p.add_argument("--qk_dim", type=int, default=128)
    p.add_argument("--down_reg", type=float, default=0.1,     # AnyUp downsampling_regularization default
                   help="weight of anyup_down loss (0 to disable)")
    p.add_argument("--proj_head", action=argparse.BooleanOptionalAction, default=True,
                   help="learned 1x1 conv projecting upsampled ps4-space features into the HR "
                        "target's space before anyup_hr; --no-proj_head to A/B against stock AnyUp")
    p.add_argument("--linear_baseline", action=argparse.BooleanOptionalAction, default=True,
                   help="co-train a bilinear+linear-head baseline to attribute mAnyUp's gain to "
                        "guidance-driven upsampling vs. a plain linear ps4->ps1 map")
    p.add_argument("--stage_to_tmpdir", action="store_true",
                   help="copy feature/S2 dirs to $SLURM_TMPDIR for faster reads")
    p.add_argument("--out_dir", default="checkpoints/manyup")
    p.add_argument("--ckpt_every", type=int, default=5, help="save every N epochs")
    p.add_argument("--sanity", action="store_true", help="one batch then exit")
    args = p.parse_args()

    AnyUp, Cosine_MSE = _import_anyup(args.anyup_repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    froot = Path(args.features_root)
    lr_dir, hr_dir = froot / args.lr_cfg, froot / args.hr_cfg
    s2_root = Path(args.data_root)

    # Optionally stage the big feature dirs + S2 images to node-local disk.
    if args.stage_to_tmpdir:
        lr_dir, hr_dir = stage_to_tmpdir([lr_dir, hr_dir])
        # S2 images live under data_root/pastis_r_<split>/s2_images; stage the split dir.
        s2_split = s2_root / f"pastis_r_{args.split}"
        (s2_split_staged,) = stage_to_tmpdir([s2_split])
        s2_root = s2_split_staged.parent   # so PairedFeatureDataset finds pastis_r_<split>/s2_images

    ds = PairedFeatureDataset(lr_dir, hr_dir, s2_root, args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Held-out TEST sample for the per-epoch viz. Feature dirs (possibly staged) contain the test
    # split too; S2 test images come from the ORIGINAL data_root (only train S2 was staged).
    viz_sample = None
    try:
        test_ds = PairedFeatureDataset(lr_dir, hr_dir, Path(args.data_root), "test")
        viz_sample = test_ds[0]   # first common test sample: (lr, hr, s2)
    except RuntimeError as e:
        print(f"viz disabled: no usable test sample ({e})")

    # peek one sample for shapes
    lr0, hr0, _ = ds[0]
    D, gh, gw = lr0.shape
    _, GH, GW = hr0.shape
    print(f"LR feats {tuple(lr0.shape)}  ->  HR target {tuple(hr0.shape)}  (upsample {gh}x{gw} -> {GH}x{GW})")
    assert lr0.shape[0] == hr0.shape[0], "LR and HR feature dims (D) must match to share the upsampler"

    # 13-channel guidance is the ONLY architectural change vs stock AnyUp. Train from scratch.
    model = AnyUp(input_dim=S2_BANDS, qk_dim=args.qk_dim).to(device).train()

    # Optional pixel-wise linear head (1x1 conv, D->D). AnyUp pools VALUES from the LR feats, so
    # it assumes LR and HR share a feature space -- true when they're one backbone at two
    # resolutions. Our LR (ps4) and HR (ps1) targets are DIFFERENT patch sizes, so OlmoEarth may
    # produce systematically different features; forcing anyup_hr to match them directly is then
    # ill-posed. The head lets the upsampled ps4-space map be linearly PROJECTED into ps1-space:
    # we only require it can PREDICT the HR target, not equal it. anyup_down stays in ps4-space
    # (on the raw, pre-projection upsampled map), so the fidelity anchor is unaffected.
    proj_head = nn.Conv2d(D, D, kernel_size=1).to(device) if args.proj_head else None
    params = list(model.parameters()) + (list(proj_head.parameters()) if proj_head else [])
    print(f"mAnyUp params: {sum(p.numel() for p in model.parameters())}"
          + (f" + proj_head {sum(p.numel() for p in proj_head.parameters())}" if proj_head else ""))

    criterion = Cosine_MSE()
    opt = torch.optim.AdamW(params, lr=args.lr)

    # Warmup + cosine LR schedule, stepped PER BATCH. total_steps drives the cosine period; the
    # first warmup_frac of steps ramp linearly 0 -> lr, then cosine-decay lr -> lr_min. Both the
    # mAnyUp optimizer and the baseline's share the SAME schedule so the attribution comparison
    # (mAnyUp vs bilinear+linear head) stays fair -- otherwise one would train under a decaying LR
    # and the other a constant one.
    total_steps = args.epochs * len(loader)
    def make_sched(o):
        return _warmup_cosine(o, total_steps, args.warmup_frac, args.lr, args.lr_min)
    sched = make_sched(opt)

    # ---- baseline 1: NAIVE bilinear upsample of LR feats (no guidance, no learning). One-shot.
    baseline = _bilinear_baseline(loader, criterion, GH, GW, device, n_batches=args.baseline_batches)
    print(f"[baseline] bilinear (no learning) HR loss = {baseline:.4f}")

    # ---- baseline 2 (optional): bilinear upsample + a LEARNED 1x1 conv (D->D), co-trained on the
    # same data. This isolates how much of mAnyUp's gain comes from the guidance-driven upsampling
    # vs. just a linear ps4->ps1 projection. If mAnyUp (with proj_head) barely beats THIS, the win
    # is mostly the linear map, not the guidance. Trained in lockstep below with its own optimizer.
    lin_head = nn.Conv2d(D, D, kernel_size=1).to(device) if args.linear_baseline else None
    lin_opt = torch.optim.AdamW(lin_head.parameters(), lr=args.lr) if lin_head else None
    lin_sched = make_sched(lin_opt) if lin_opt else None
    if lin_head:
        print(f"[baseline] co-training bilinear+linear head ({sum(p.numel() for p in lin_head.parameters())} params)")

    for epoch in range(args.epochs):
        model.train()
        running = {"hr": 0.0, "down": 0.0, "lin": 0.0}
        for bi, (lr, hr, s2) in enumerate(loader):
            lr, hr, s2 = lr.to(device), hr.to(device), s2.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred = model(s2, lr, (GH, GW))            # guidance, LR feats, target grid (ps4-space)
                # HR loss sees the projection (ps4-space -> ps1-space); identity if no head.
                pred_hr = proj_head(pred) if proj_head is not None else pred
                loss_hr = criterion(pred_hr, hr)["total"]
                loss = loss_hr
                loss_down = torch.tensor(0.0, device=device)
                if args.down_reg > 0:
                    # down-reg stays in ps4-space: downsample the RAW (pre-projection) upsampled
                    # map and match the LR input feats -- the anchor is native to the upsampler.
                    down = F.interpolate(pred.float(), size=(gh, gw), mode="area")
                    loss_down = criterion(down, lr)["total"] * args.down_reg
                    loss = loss + loss_down

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()                                       # warmup+cosine, per batch

            # Co-train the bilinear+linear baseline on the same batch (independent optimizer, no
            # guidance, no learned upsampling -- just a linear map on bilinearly-upsampled LR).
            loss_lin = torch.tensor(0.0, device=device)
            if lin_head is not None:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    up = F.interpolate(lr.float(), size=(GH, GW), mode="bilinear", align_corners=False)
                    loss_lin = criterion(lin_head(up), hr)["total"]
                lin_opt.zero_grad()
                loss_lin.backward()
                lin_opt.step()
                lin_sched.step()                               # same schedule -> fair comparison

            running["hr"] += loss_hr.item()
            running["down"] += float(loss_down)
            running["lin"] += float(loss_lin)
            if bi % 50 == 0:
                print(f"epoch {epoch} batch {bi}/{len(loader)}  lr={sched.get_last_lr()[0]:.2e}  "
                      f"hr={loss_hr.item():.4f} down={float(loss_down):.4f}"
                      + (f" lin={float(loss_lin):.4f}" if lin_head is not None else ""))
            if args.sanity:
                print("sanity: one batch done, exiting.")
                return

        n = len(loader)
        mean_hr, mean_lin = running["hr"] / n, running["lin"] / n
        vs = baseline - mean_hr
        lin_str = ""
        if lin_head is not None:
            # The honest attribution: how much mAnyUp beats a learned linear map (not just raw
            # bilinear). If this gap is ~0, the guidance-driven upsampling isn't adding much.
            lin_str = (f"  | lin_head={mean_lin:.4f}  (mAnyUp vs lin_head: {mean_lin - mean_hr:+.4f})")
        print(f"== epoch {epoch} done  hr={mean_hr:.4f}  down={running['down']/n:.4f}  "
              f"| bilinear={baseline:.4f}  (mAnyUp {'beats' if vs > 0 else 'WORSE than'} "
              f"bilinear by {vs:+.4f}){lin_str}")

        if viz_sample is not None:
            viz_dir = out_dir / "viz"; viz_dir.mkdir(exist_ok=True)
            save_epoch_viz(model, proj_head, viz_sample, GH, GW, device,
                           viz_dir / f"test0_ep{epoch:03d}.png", epoch)

        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            ckpt = out_dir / f"manyup_{args.lr_cfg}_to_{args.hr_cfg}_ep{epoch}.pth"
            torch.save({"model": model.state_dict(),
                        "proj_head": proj_head.state_dict() if proj_head else None,
                        "args": vars(args), "epoch": epoch,
                        "input_dim": S2_BANDS, "qk_dim": args.qk_dim}, ckpt)
            print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
