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
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.metrics import segmentation_metrics

# AnyUp upsample+probe and the RGB-guidance loader are shared with the live finetune path
# (single source of truth). Importing the module is cheap; AnyUp (torch.hub) only loads when
# an AnyUpUpsampleProbe is actually constructed (i.e. only for the anyup heads).
from finetune_olmoearth_pastis import AnyUpUpsampleProbe, _load_rgb_guidance

# Cosine annealing decays the LR from args.lr to SCHEDULER_MIN_LR over args.epochs.
SCHEDULER_MIN_LR = 1e-6

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
    "anyup_t2_ens": "temporal",
    "anyup_t1_ens": "temporal",
}

# Whether the head collapses time via feats.mean(dim=1) as its first op. When True the dataset
# pre-reduces the cached (T,gH,gW,D) to (1,gH,gW,D) ONCE at preload, so we don't float-cast and
# ship the full T=12 tensor across PCIe every step only for the head to average it away (a 12x
# cut in RAM + host->device traffic; the head's mean over a singleton T is then a no-op). Only
# anyup_t1 needs the per-timestep features, so it opts out.
HEAD_REDUCES_TIME = {
    "lp_pa2pa_bu": True,
    "lp_pa2px": True,
    "anyup": True,
    "anyup_t2": True,
    "anyup_t1": False,
    # t2_ens shares one time-pooled feature map (T comes from the 5-D per-t RGB, so the ensemble
    # still gets its T probes); t1_ens needs the real per-timestep features.
    "anyup_t2_ens": True,
    "anyup_t1_ens": False,
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
                 guidance: str = "none", max_ram_gb: float = 32.0,
                 reduce_time: bool = False):
        self.feat_dir = features_dir / f"pastis_r_{split}"
        self.labels = torch.load(data_splits / f"pastis_r_{split}" / "targets.pt")
        self.split = split
        self.guidance = guidance
        # If the head mean-pools over time, collapse (T,gH,gW,D)->(1,gH,gW,D) once here so we
        # never float-cast / ship the full T tensor per step. Keeps a singleton T so heads that
        # do feats.mean(dim=1) / feats.shape[1] stay correct unchanged.
        self.reduce_time = reduce_time
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
        T = probe.shape[0]
        # Shape actually stored per sample: (1,gH,gW,D) when the head averages over time.
        feat_shape = (1, *probe.shape[1:]) if self.reduce_time else tuple(probe.shape)
        feat_elems = int(torch.tensor(feat_shape).prod())
        self._rgb_elems = (T * 3 * 64 * 64) if self.guidance == "temporal" else (3 * 64 * 64)
        est = self._est_gb(feat_elems)
        if est > max_ram_gb:
            print(f"[{self.split}] preload SKIPPED: est {est:.1f} GB > --max_ram_gb "
                  f"{max_ram_gb:.1f} GB. Falling back to per-sample disk loading "
                  f"({self.n} files/epoch); raise --max_ram_gb or --num_workers to speed up.")
            return
        print(f"[{self.split}] preloading {self.n} samples (~{est:.1f} GB fp16) into RAM "
              f"once{' [time-reduced]' if self.reduce_time else ''}; "
              f"epochs will run at compute speed...")
        # Keep features in fp16 to halve RAM; cast to float per-batch in __getitem__.
        self._feats = torch.empty((self.n, *feat_shape), dtype=torch.float16)
        rgb_buf = (torch.empty((self.n, *self._rgb_shape(T)), dtype=torch.float16)
                   if self.guidance != "none" else None)
        for i in tqdm(range(self.n), desc=f"preload {self.split}", leave=False):
            feat = torch.load(self.feat_dir / f"{i}.pt")           # (T,gH,gW,D) fp16
            # Mean over T in fp32 for accuracy, then store fp16; keep a singleton T dim.
            self._feats[i] = feat.float().mean(dim=0, keepdim=True).half() if self.reduce_time else feat
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
        if self.reduce_time:
            feat = feat.mean(dim=0, keepdim=True)                 # (1, gH, gW, D)
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
    """anyup: cached features mean-pooled over T, single mean RGB guidance.

    ensemble=False (default): the T AnyUp-upsampled maps are mean-pooled, then ONE probe.
    ensemble=True: each timestep's upsampled map gets its OWN probe and the T logits are
    averaged pre-softmax (a temporal ensemble). anyup (single-timestep) ignores the flag."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 label_size: int = LABEL_SIZE, ensemble: bool = False):
        super().__init__()
        self.up = AnyUpUpsampleProbe(num_classes, ensemble=ensemble)
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
    # (class, extra kwargs). The _ens variants reuse the same wrapper but give each timestep its
    # own probe and average the per-timestep logits instead of mean-pooling features (see
    # AnyUpUpsampleProbe.ensemble). t1_ens = per-timestep features; t2_ens = shared time-pooled
    # features -- both with per-timestep hr probes.
    HEADS = {
        "lp_pa2pa_bu": (LPPatchToPatchBU, {}),
        "lp_pa2px": (LPPatchToPixel, {}),
        "anyup": (CachedAnyUp, {}),
        "anyup_t2": (CachedAnyUpT2, {}),
        "anyup_t1": (CachedAnyUpT1, {}),
        "anyup_t2_ens": (CachedAnyUpT2, {"ensemble": True}),
        "anyup_t1_ens": (CachedAnyUpT1, {"ensemble": True}),
    }
    if name not in HEADS:
        raise ValueError(f"head_mode={name!r} not in {list(HEADS)} (cached-feature heads)")
    cls, kwargs = HEADS[name]
    return cls(embed_dim, num_classes, patch_size, **kwargs)


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


RESULT_COLUMNS = [
    "timestamp", "features", "head_mode", "epochs", "lr", "batch_size", "seed",
    "test_miou", "test_overall_acc", "avg_epoch_sec",
]


def append_result(csv_path: Path, row: dict) -> None:
    """Append one run's args + test metrics to csv_path, writing the header if the file is
    new. Fixed RESULT_COLUMNS so concurrent jobs (one per head/feature set) all share one
    schema."""
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="LP on cached OlmoEarth features.")
    p.add_argument("--features", required=True,
                   help="extraction config folder name under --out_root, e.g. oe_base_s2s1_ps4_tile64")
    p.add_argument("--out_root", default="features")
    p.add_argument("--data_splits", default="data/pastis_olmoearth")
    p.add_argument("--head_mode", default="lp_pa2px",
                   choices=list(HEAD_GUIDANCE))
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers for the DISK-FALLBACK path; ignored when a split "
                        "is preloaded into RAM (then workers=0, since indexing RAM is instant "
                        "and forking would copy the big tensor into every worker).")
    p.add_argument("--max_ram_gb", type=float, default=32.0,
                   help="per-split RAM budget for preloading features into memory; splits "
                        "estimated above this fall back to per-sample disk loading.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results_csv", default="lp_olmoearth_pastis.csv",
                   help="append run args + test metrics to this CSV (created with a header "
                        "if absent).")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    # AnyUp runs its (attention-heavy) upsample in fp32; TF32 lets the L40s tensor cores do
    # those matmuls ~2x faster at negligible precision cost. Frozen AnyUp + tiny probe means
    # the slight TF32 rounding is immaterial to results. Big win for the anyup* heads, which
    # call AnyUp T times per sample.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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

    reduce_time = HEAD_REDUCES_TIME[args.head_mode]

    def loader(split, shuffle):
        ds = CachedFeatureDataset(feat_dir, data_splits, split, guidance=guidance,
                                  max_ram_gb=args.max_ram_gb, reduce_time=reduce_time)
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

    # AnyUp heads lazily create their real probe (Conv2d(embed_dim, C)) on the FIRST forward
    # (AnyUpUpsampleProbe starts with a 1x1x1 placeholder). Run a no-grad dry pass now so the
    # real probe exists before AdamW captures head.parameters() -- otherwise the only trainable
    # module is never optimized and the AnyUp heads don't learn. Mirrors the live finetune path
    # (finetune_olmoearth_pastis.py dry pass before the optimizer). LP heads build their probe
    # in __init__, so this pass is a harmless no-op for them.
    with torch.no_grad():
        feats0, _label0, rgb0 = next(iter(train_loader))
        _run_head(head, feats0, rgb0, device)
    head = head.to(device)  # re-move in case _init_probe created the probe on a fresh device

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=SCHEDULER_MIN_LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    best_val_miou = float("-inf")

    epoch_times = []   # wall time (train + val) per epoch, for the average below
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
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
        scheduler.step()
        epoch_times.append(time.perf_counter() - epoch_start)
        print(f"epoch {epoch+1}/{args.epochs} | train_loss {last_loss:.4f} | "
              f"val miou {val.primary:.4f} | {epoch_times[-1]:.1f}s | {val.metrics}")
        if val.primary > best_val_miou:
            best_val_miou = val.primary
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else float("nan")
    head.load_state_dict(best_state)
    test = evaluate(head, test_loader, device)
    print(f"\nBEST val miou {best_val_miou:.4f}")
    print(f"TEST {test.metrics}")
    print(f"Avg epoch time: {avg_epoch_time:.1f}s over {len(epoch_times)} epochs")

    # Keep the CSV readable: ints/strings as-is, metrics+time as 2-decimal floats. lr is the
    # one value .2f would mangle (1e-3 -> 0.00), so log it with %g (compact, full precision).
    append_result(Path(args.results_csv), {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "features": args.features,
        "head_mode": args.head_mode,
        "epochs": int(args.epochs),
        "lr": f"{args.lr:g}",
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "test_miou": f"{test.metrics['miou']:.2f}",
        "test_overall_acc": f"{test.metrics['overall_acc']:.2f}",
        "avg_epoch_sec": f"{avg_epoch_time:.2f}",
    })
    print(f"Appended result to {args.results_csv}")


if __name__ == "__main__":
    main()
