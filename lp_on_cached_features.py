"""Train a segmentation head on CACHED OlmoEarth features (no encoder).

Pairs with extract_olmoearth_features.py: that script writes per-sample
features/<cfg>/pastis_r_<split>/<idx>.pt of shape (T, gH, gW, D). Here we load those and
fit a head directly, so there is no encoder forward pass -- head iteration is fast and
needs no GPU for the linear-probe heads.

Heads consume (B, T, gH, gW, D), mean over T, then probe (see the heads section):
  - lp_pa2pa_bu: 1x1 conv on tokens -> bilinear-upsample the predictions to label res.
  - lp_pa2px:    1x1 conv D->C*p^2 -> unfold sub-pixels (the live BackboneWithHead seg head).
The mIoU gap between them measures within-patch spatial structure vs. patch-level semantics.
Extra heads (lp_per_t, AnyUp-on-cache, ...) drop into the build_cached_head registry; AnyUp
variants would additionally load cached RGB guidance and are out of scope for v1.

Runs in the OlmoEarth venv (for segmentation_metrics); via salloc or even CPU:
    source env_olmo.sh
    python -u lp_on_cached_features.py --features oe_base_s2s1_ps4_tile64 --head_mode lp_pa2px
"""
import os
import sys

# Bootstrap before importing olmoearth_pretrain (only segmentation_metrics is needed; no
# model is ever loaded here). Mirrors finetune_olmoearth_pastis.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "olmo_shims"))
import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.metrics import segmentation_metrics

# AnyUp upsample+probe and the RGB-guidance loader are shared with the live finetune path
# (single source of truth). Importing the module is cheap; AnyUp (torch.hub) only loads when
# an AnyUpUpsampleProbe is actually constructed (i.e. only for the anyup heads).
from finetune_olmoearth_pastis import AnyUpUpsampleProbe, _load_rgb_guidance

# Scheduler knobs: same values as olmoearth_pretrain.evals.finetune.constants, inlined so
# this script stays decoupled from the finetune training path.
SCHEDULER_FACTOR = 0.2
SCHEDULER_PATIENCE = 2
SCHEDULER_MIN_LR = 1e-6
SCHEDULER_COOLDOWN = 0

NUM_CLASSES = 20            # PASTIS: 20 classes (class 19 = void -> ignore via -1 labels)
IGNORE_LABEL = -1
LABEL_SIZE = 64            # PASTIS label resolution

# Guidance mode each head needs from the dataset: "none" (LP heads), "mean" ((3,64,64)
# time-averaged RGB, plain anyup), or "temporal" ((T,3,64,64) per-timestep RGB, anyup_t1/t2).
HEAD_GUIDANCE = {
    "lp_pa2pa_bu": "none",
    "lp_pa2px": "none",
    "anyup": "mean",
    "anyup_t2": "temporal",
    "anyup_t1": "temporal",
}


# ----------------------------- data -----------------------------
class CachedFeatureDataset(torch.utils.data.Dataset):
    """Loads cached (T, gH, gW, D) features, the matching (64,64) label, and -- for AnyUp
    heads -- the RGB guidance image.

    Labels come from the original prep (targets.pt is a stacked (N,64,64) tensor); features
    are keyed by the same contiguous index used at extraction time.

    guidance: "none" -> rgb is an empty tensor (LP heads ignore it); "mean" -> (3,64,64)
    time-averaged RGB; "temporal" -> (T,3,64,64) per-timestep RGB. Guidance is built by
    finetune_olmoearth_pastis._load_rgb_guidance, which reads s2_images from that module's
    DATA_SPLITS global -- main() points it at args.data_splits before loaders are built."""

    def __init__(self, features_dir: Path, data_splits: Path, split: str,
                 guidance: str = "none", max_ram_gb: float = 32.0):
        self.feat_dir = features_dir / f"pastis_r_{split}"
        self.labels = torch.load(data_splits / f"pastis_r_{split}" / "targets.pt")
        self.split = split
        self.guidance = guidance
        self.n = len(list(self.feat_dir.glob("*.pt")))
        if self.n != len(self.labels):
            raise ValueError(
                f"{split}: {self.n} feature files != {len(self.labels)} labels. "
                "Re-run extraction or check the features/<cfg> path.")

        # Speed path: torch.load-ing N small .pt files from Lustre EVERY epoch is the
        # bottleneck (the GPU sits idle waiting on disk). If the whole split fits in a RAM
        # budget, read it ONCE into a single fp16 tensor and index that instead -- epochs
        # then run at compute speed. Otherwise fall back to per-sample disk loading (still
        # works, just slower; pair with --num_workers). Decision is logged verbosely.
        self._feats = None          # set iff preloaded
        self._rgb = None
        self._maybe_preload(max_ram_gb)

    def _est_gb(self, per_sample_feat_elems: int) -> float:
        """Estimated RAM for the preloaded fp16 feature tensor (+ fp16 guidance if needed)."""
        feat_gb = self.n * per_sample_feat_elems * 2 / 1e9          # fp16 = 2 bytes
        rgb_gb = 0.0
        if self.guidance != "none":
            # mean: (3,64,64); temporal: (T,3,64,64) -- T read from a sample below
            rgb_gb = self._rgb_elems * self.n * 2 / 1e9
        return feat_gb + rgb_gb

    def _maybe_preload(self, max_ram_gb: float) -> None:
        if self.n == 0:
            return
        probe = torch.load(self.feat_dir / "0.pt")                 # (T, gH, gW, D), fp16 on disk
        feat_elems = probe.numel()
        T = probe.shape[0]
        self._rgb_elems = (T * 3 * 64 * 64) if self.guidance == "temporal" else (3 * 64 * 64)
        est = self._est_gb(feat_elems)
        if est > max_ram_gb:
            print(f"[{self.split}] preload SKIPPED: est {est:.1f} GB > --max_ram_gb "
                  f"{max_ram_gb:.1f} GB. Falling back to per-sample disk loading "
                  f"({self.n} files/epoch); raise --max_ram_gb or --num_workers to speed up.")
            return
        print(f"[{self.split}] preloading {self.n} samples (~{est:.1f} GB fp16) into RAM "
              f"once; epochs will run at compute speed...")
        # Keep features in fp16 to halve RAM; cast to float per-batch in __getitem__.
        self._feats = torch.empty((self.n, *probe.shape), dtype=torch.float16)
        rgb_buf = (torch.empty((self.n, *self._rgb_shape(T)), dtype=torch.float16)
                   if self.guidance != "none" else None)
        for i in tqdm(range(self.n), desc=f"preload {self.split}", leave=False):
            self._feats[i] = torch.load(self.feat_dir / f"{i}.pt")
            if rgb_buf is not None:
                rgb_buf[i] = _load_rgb_guidance(
                    self.split, i, temporal=self.guidance == "temporal").half()
        self._rgb = rgb_buf
        print(f"[{self.split}] preload done.")

    def _rgb_shape(self, T: int):
        return (T, 3, 64, 64) if self.guidance == "temporal" else (3, 64, 64)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        label = self.labels[idx].long()                           # (64,64)
        if self._feats is not None:                                # RAM path
            feat = self._feats[idx].float()                        # (T, gH, gW, D)
            if self.guidance == "none":
                rgb = torch.empty(0)
            else:
                rgb = self._rgb[idx].float()
            return feat, label, rgb
        # disk fallback
        feat = torch.load(self.feat_dir / f"{idx}.pt").float()    # (T, gH, gW, D)
        if self.guidance == "none":
            rgb = torch.empty(0)
        else:
            rgb = _load_rgb_guidance(self.split, idx, temporal=self.guidance == "temporal")
        return feat, label, rgb


# ----------------------------- heads -----------------------------
# Two linear-probe heads that probe WHAT a patch token encodes. The mIoU gap between them
# measures whether tokens carry within-patch spatial structure or only patch-level semantics
# -- a direct motivation for a learned, guidance-driven upsampler (e.g. AnyUp).
#
# Both first mean-pool over time -> (B, gH, gW, D).
#   lp_pa2pa_bu (patch->patch, then bilinear-upsample the PREDICTIONS):
#       1x1 conv D->C on the (gH,gW) token grid, then bilinear-upsample the C-channel logits
#       to the label resolution. Assumes a token says "this patch is class X"; pixel detail
#       comes purely from the spatial-smoothness inductive bias of bilinear upsampling.
#   lp_pa2px (patch->pixel):
#       1x1 conv D->C*patch_size^2, then rearrange the extra channels into sub-pixels
#       (b (c i j) gh gw) -> (b c (gh i) (gw j)). Assumes a token encodes the within-patch
#       spatial layout. This mirrors the live BackboneWithHead seg head exactly
#       (finetune_olmoearth_pastis.py:313).
#
# Dropped for now: lp_bu_px2px (bilinear-upsample FEATURES then per-pixel linear probe). With
# a single linear probe it equals lp_pa2pa_bu -- 1x1 conv and bilinear upsample are both linear
# and act on disjoint axes (channel vs spatial), so they commute. Only meaningful with a
# nonlinear (MLP) per-pixel probe; revisit then.


class LPPatchToPatchBU(nn.Module):
    """lp_pa2pa_bu: 1x1 conv on tokens -> bilinear-upsample the logits to label res."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 label_size: int = LABEL_SIZE):
        super().__init__()
        self.probe = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.label_size = label_size

    def forward(self, feats: torch.Tensor, rgb=None) -> torch.Tensor:   # (B,T,gH,gW,D)
        x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()  # (B, D, gH, gW)
        logits = self.probe(x)                                  # (B, C, gH, gW)
        if logits.shape[-2:] != (self.label_size, self.label_size):
            logits = F.interpolate(logits, size=(self.label_size, self.label_size),
                                   mode="bilinear", align_corners=True)
        return logits


class LPPatchToPixel(nn.Module):
    """lp_pa2px: 1x1 conv D->C*patch_size^2, then unfold the extra channels into sub-pixels."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 label_size: int = LABEL_SIZE):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.label_size = label_size
        self.probe = nn.Conv2d(embed_dim, num_classes * patch_size * patch_size, kernel_size=1)

    def forward(self, feats: torch.Tensor, rgb=None) -> torch.Tensor:   # (B,T,gH,gW,D)
        x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()  # (B, D, gH, gW)
        logits = self.probe(x)                                  # (B, C*p*p, gH, gW)
        p = self.patch_size
        logits = rearrange(logits, "b (c i j) gh gw -> b c (gh i) (gw j)",
                           c=self.num_classes, i=p, j=p)         # (B, C, gH*p, gW*p)
        if logits.shape[-2:] != (self.label_size, self.label_size):
            logits = F.interpolate(logits, size=(self.label_size, self.label_size),
                                   mode="bilinear", align_corners=True)
        return logits


# ---- AnyUp heads on cached features ----
# Same upsample+probe logic as the live finetune AnyUp heads (via the shared
# AnyUpUpsampleProbe), but features come from the cache (T,gH,gW,D) instead of the encoder.
# This is exactly why the cache keeps per-timestep features: the three variants differ only
# in how they feed the cached (T,gH,gW,D) and the RGB guidance into AnyUpUpsampleProbe:
#   anyup    : mean over T -> single (B,D,gH,gW); single mean RGB (B,3,64,64).      [guidance=mean]
#   anyup_t2 : mean over T -> single (B,D,gH,gW); per-timestep RGB (B,T,3,64,64).   [guidance=temporal]
#   anyup_t1 : per-timestep features (list of T (B,D,gH,gW)); per-timestep RGB.     [guidance=temporal]

class CachedAnyUp(nn.Module):
    """anyup: cached features mean-pooled over T, single mean RGB guidance."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 label_size: int = LABEL_SIZE):
        super().__init__()
        self.up = AnyUpUpsampleProbe(num_classes)
        self.label_size = label_size

    def _feats_2d(self, feats):                          # (B,T,gH,gW,D) -> (B,D,gH,gW)
        return feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()

    def forward(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        out = (self.label_size, self.label_size)
        return self.up(self._feats_2d(feats), rgb, out_size=out)


class CachedAnyUpT2(CachedAnyUp):
    """anyup_t2: shared time-pooled features, per-timestep RGB guidance (rgb is (B,T,3,64,64))."""
    # forward identical to CachedAnyUp: AnyUpUpsampleProbe loops T off the 5-D rgb, reusing
    # the single feature map for every timestep.


class CachedAnyUpT1(CachedAnyUp):
    """anyup_t1: per-timestep features AND per-timestep RGB (heaviest)."""

    def forward(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        out = (self.label_size, self.label_size)
        # list of T (B,D,gH,gW), one per cached timestep
        feats_per_t = [feats[:, t].permute(0, 3, 1, 2).contiguous() for t in range(feats.shape[1])]
        return self.up(feats_per_t, rgb, out_size=out)


def build_cached_head(name: str, embed_dim: int, num_classes: int, patch_size: int) -> nn.Module:
    HEADS = {
        "lp_pa2pa_bu": LPPatchToPatchBU,
        "lp_pa2px": LPPatchToPixel,
        "anyup": CachedAnyUp,
        "anyup_t2": CachedAnyUpT2,
        "anyup_t1": CachedAnyUpT1,
    }
    if name not in HEADS:
        raise ValueError(f"head_mode={name!r} not in {list(HEADS)} (cached-feature heads)")
    return HEADS[name](embed_dim, num_classes, patch_size)


# ----------------------------- train / eval -----------------------------
def _run_head(head, feats, rgb, device):
    """Forward a batch through any cached head. rgb is an empty tensor for LP heads."""
    rgb = None if rgb.numel() == 0 else rgb.to(device)
    return head(feats.to(device), rgb)


@torch.no_grad()
def evaluate(head, loader, device):
    head.eval()
    preds, labels = [], []
    for feats, label, rgb in loader:
        logits = _run_head(head, feats, rgb, device)
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(label)
    return segmentation_metrics(torch.cat(preds), torch.cat(labels),
                                num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL)


def main() -> None:
    p = argparse.ArgumentParser(description="LP on cached OlmoEarth features.")
    p.add_argument("--features", required=True,
                   help="extraction config folder name under --out_root, e.g. oe_base_s2s1_ps4_tile64")
    p.add_argument("--out_root", default="features")
    p.add_argument("--data_splits", default="data/pastis_olmoearth")
    p.add_argument("--head_mode", default="lp_pa2px",
                   choices=list(HEAD_GUIDANCE))
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers for the DISK-FALLBACK path; ignored when a split "
                        "is preloaded into RAM (then workers=0, since indexing RAM is instant "
                        "and forking would copy the big tensor into every worker).")
    p.add_argument("--max_ram_gb", type=float, default=32.0,
                   help="per-split RAM budget for preloading features into memory; splits "
                        "estimated above this fall back to per-sample disk loading.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = Path(args.out_root) / args.features
    data_splits = Path(args.data_splits)

    meta_path = feat_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing; run extract_olmoearth_features.py first.")
    meta = json.loads(meta_path.read_text())
    embed_dim = meta["embed_dim"]
    patch_size = meta["patch_size"]
    guidance = HEAD_GUIDANCE[args.head_mode]
    print(f"Features: {args.features} | shape {meta['feature_shape']} | "
          f"patch_size {patch_size} | head {args.head_mode} | guidance {guidance}")

    # _load_rgb_guidance reads s2_images from finetune_olmoearth_pastis.DATA_SPLITS (a module
    # global). Point it at our data_splits so AnyUp guidance comes from the right place.
    if guidance != "none":
        import finetune_olmoearth_pastis as fmod
        fmod.DATA_SPLITS = args.data_splits

    def loader(split, shuffle):
        ds = CachedFeatureDataset(feat_dir, data_splits, split, guidance=guidance,
                                  max_ram_gb=args.max_ram_gb)
        preloaded = ds._feats is not None
        # Preloaded: index RAM in-process (workers would duplicate the tensor). Disk fallback:
        # use workers + pin_memory + persistent_workers to overlap reads with GPU compute.
        workers = 0 if preloaded else args.num_workers
        use_cuda = device.type == "cuda"
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=workers,
            pin_memory=use_cuda,
            persistent_workers=workers > 0,
        )

    train_loader = loader("train", True)
    val_loader = loader("valid", False)
    test_loader = loader("test", False)

    head = build_cached_head(args.head_mode, embed_dim, NUM_CLASSES, patch_size).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(opt, mode="max", factor=SCHEDULER_FACTOR,
                                  patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR,
                                  cooldown=SCHEDULER_COOLDOWN)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    best_val_miou = float("-inf")

    for epoch in range(args.epochs):
        head.train()
        last_loss = float("nan")
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False)
        for feats, label, rgb in pbar:
            logits = _run_head(head, feats, rgb, device)
            loss = loss_fn(logits, label.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
            pbar.set_postfix(loss=f"{last_loss:.4f}")

        val = evaluate(head, val_loader, device)
        scheduler.step(val.primary)
        print(f"epoch {epoch+1}/{args.epochs} | train_loss {last_loss:.4f} | "
              f"val miou {val.primary:.4f} | {val.metrics}")
        if val.primary > best_val_miou:
            best_val_miou = val.primary
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    test = evaluate(head, test_loader, device)
    print(f"\nBEST val miou {best_val_miou:.4f}")
    print(f"TEST {test.metrics}")


if __name__ == "__main__":
    main()
