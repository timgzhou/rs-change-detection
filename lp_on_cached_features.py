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
    python -u lp_on_cached_features.py --features oe_base_s2_ps4_tile64 --head_mode lp_pa2px
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
    "lp_pa2px_ens": "none",     # temporal ensemble of per-t pa2px probes (no guidance)
    "anyup": "mean",
    "anyup_t2": "temporal",
    "anyup_t1": "temporal",
    "anyup_t2_ens": "temporal",
    "anyup_t1_ens": "temporal",
    # mAnyUp: our trained upsampler with FULL 13-band S2 guidance (time-averaged), matching how
    # train_manyup.py fed it. "mean13" -> (13,64,64), distinct from anyup's 3-band "mean".
    "manyup": "mean13",
}

# Whether the head collapses time via feats.mean(dim=1) as its first op. When True the dataset
# pre-reduces the cached (T,gH,gW,D) to (1,gH,gW,D) ONCE at preload, so we don't float-cast and
# ship the full T=12 tensor across PCIe every step only for the head to average it away (a 12x
# cut in RAM + host->device traffic; the head's mean over a singleton T is then a no-op). Only
# anyup_t1 needs the per-timestep features, so it opts out.
HEAD_REDUCES_TIME = {
    "lp_pa2pa_bu": True,
    "lp_pa2px": True,
    "lp_pa2px_ens": False,      # ensemble needs the full per-timestep feature map
    "anyup": True,
    "anyup_t2": True,
    "anyup_t1": False,
    # t2_ens shares one time-pooled feature map (T comes from the 5-D per-t RGB, so the ensemble
    # still gets its T probes); t1_ens needs the real per-timestep features.
    "anyup_t2_ens": True,
    "anyup_t1_ens": False,
    "manyup": True,          # mAnyUp mean-pools T on the LR feats before upsampling
}


S2_BANDS = 13   # full Sentinel-2 L2A stack used as mAnyUp guidance


def _load_s2_guidance(split: str, idx: int) -> torch.Tensor:
    """Full 13-band S2 guidance for mAnyUp: (T,13,64,64) -> mean over T -> (13,64,64), per-band
    min-max normalized to [0,1]. Matches train_manyup._norm_guidance so the frozen mAnyUp sees
    exactly the guidance distribution it trained on. Reads from finetune_olmoearth_pastis's
    DATA_SPLITS (a module global the caller points at our data_splits)."""
    import finetune_olmoearth_pastis as fmod
    s2 = torch.load(Path(fmod.DATA_SPLITS) / f"pastis_r_{split}" / "s2_images" / f"{idx}.pt")
    s2 = s2.float().mean(0)                                   # (13,64,64)
    flat = s2.reshape(s2.shape[0], -1)
    lo = flat.min(1).values.view(-1, 1, 1)
    hi = flat.max(1).values.view(-1, 1, 1)
    return (s2 - lo) / (hi - lo + 1e-6)


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
                rgb_buf[i] = self._load_guidance(i).half()
        self._rgb = rgb_buf
        print(f"[{self.split}] preload done.")

    def _load_guidance(self, idx: int) -> torch.Tensor:
        """Guidance tensor for one sample, per self.guidance mode. mean13 -> full 13-band S2;
        mean/temporal -> 3-band RGB via the shared finetune loader."""
        if self.guidance == "mean13":
            return _load_s2_guidance(self.split, idx)
        return _load_rgb_guidance(self.split, idx, temporal=self.guidance == "temporal")

    def _rgb_shape(self, T: int):
        if self.guidance == "mean13":
            return (S2_BANDS, 64, 64)
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
            rgb = self._load_guidance(idx)
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

    def features(self, feats: torch.Tensor, rgb=None) -> torch.Tensor:
        """Per-pixel feature map at label res, for KNN: mean-T token grid bilinear-upsampled
        (D,gH,gW)->(D,label,label). The probe-free counterpart of forward()."""
        x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()  # (B,D,gH,gW)
        if x.shape[-2:] != (self.label_size, self.label_size):
            x = F.interpolate(x, size=(self.label_size, self.label_size),
                              mode="bilinear", align_corners=True)
        return x                                                # (B,D,label,label)

    def forward(self, feats: torch.Tensor, rgb=None) -> torch.Tensor:   # (B,T,gH,gW,D)
        x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()  # (B, D, gH, gW)
        logits = self.probe(x)                                  # (B, C, gH, gW)
        if logits.shape[-2:] != (self.label_size, self.label_size):
            logits = F.interpolate(logits, size=(self.label_size, self.label_size),
                                   mode="bilinear", align_corners=True)
        return logits


class LPPatchToPixel(nn.Module):
    """lp_pa2px: 1x1 conv D->C*patch_size^2, then unfold the extra channels into sub-pixels.

    ensemble=False (default): mean-pool over T, then ONE probe (the original lp_pa2px).
    ensemble=True (lp_pa2px_ens): keep the full T feature map, fit an INDEPENDENT probe per
    timestep, and average the T per-pixel logits (pre-softmax) -- a temporal ensemble, mirroring
    the anyup *_ens variants but on the raw sub-pixel LP head. The per-t probes are created
    lazily on the first forward (T is a runtime dim); pass reduce_time=False for this head so the
    dataset keeps all T timesteps."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 label_size: int = LABEL_SIZE, ensemble: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.label_size = label_size
        self.ensemble = ensemble
        out_ch = num_classes * patch_size * patch_size
        if ensemble:
            self.probe = None            # T independent probes, built lazily (T known at forward)
            self._out_ch = out_ch
        else:
            self.probe = nn.Conv2d(embed_dim, out_ch, kernel_size=1)

    def _init_probes(self, T: int, device) -> None:
        self.probe = nn.ModuleList(
            [nn.Conv2d(self.embed_dim, self._out_ch, 1) for _ in range(T)]).to(device)

    def _unfold(self, logits):
        """(B, C*p*p, gH, gW) -> (B, C, label, label) via sub-pixel unfold + resize if needed."""
        p = self.patch_size
        logits = rearrange(logits, "b (c i j) gh gw -> b c (gh i) (gw j)",
                           c=self.num_classes, i=p, j=p)
        if logits.shape[-2:] != (self.label_size, self.label_size):
            logits = F.interpolate(logits, size=(self.label_size, self.label_size),
                                   mode="bilinear", align_corners=True)
        return logits

    def features(self, feats: torch.Tensor, rgb=None) -> torch.Tensor:
        """Per-pixel features at label res for KNN. pa2px's sub-pixel unfold is a PROBE-space
        trick (it needs the class dim), so for feature-space KNN we fall back to the same
        bilinear-upsampled token grid as pa2pa_bu -- KNN on raw features doesn't use the p^2
        sub-pixel channels."""
        x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()  # (B,D,gH,gW)
        if x.shape[-2:] != (self.label_size, self.label_size):
            x = F.interpolate(x, size=(self.label_size, self.label_size),
                              mode="bilinear", align_corners=True)
        return x

    def forward(self, feats: torch.Tensor, rgb=None) -> torch.Tensor:   # (B,T,gH,gW,D)
        if not self.ensemble:
            x = feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()  # (B, D, gH, gW)
            return self._unfold(self.probe(x))

        # Ensemble: per-timestep probe on that timestep's feature map, average the logits.
        T = feats.shape[1]
        if self.probe is None:
            self._init_probes(T, feats.device)
        acc = None
        for t in range(T):
            x_t = feats[:, t].permute(0, 3, 1, 2).contiguous()  # (B, D, gH, gW)
            logit_t = self._unfold(self.probe[t](x_t))          # (B, C, label, label)
            acc = logit_t if acc is None else acc + logit_t
        return acc / T                                          # mean of per-t per-pixel logits


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

    def _upsampled(self, feats_per_t, rgb) -> torch.Tensor:
        """Run AnyUp's per-timestep upsampling and mean-pool the T maps -> (B,D,64,64), WITHOUT
        the probe. Mirrors AnyUpUpsampleProbe.forward's non-ensemble accumulation so KNN reads
        exactly the features the LP probe would see. `feats_per_t` is a single (B,D,gH,gW) tensor
        (reused for all t) or a list of T of them; rgb is (B,3,64,64) or (B,T,3,64,64)."""
        out = (self.label_size, self.label_size)
        per_t_feats = isinstance(feats_per_t, (list, tuple))
        per_t_rgb = rgb.dim() == 5
        T = len(feats_per_t) if per_t_feats else (rgb.shape[1] if per_t_rgb else 1)
        shared_f = None if per_t_feats else feats_per_t.float()
        acc = None
        for t in range(T):
            f = feats_per_t[t].float() if per_t_feats else shared_f
            g = (rgb[:, t] if per_t_rgb else rgb).float()
            hr_t = self.up.anyup(g, f, output_size=out)      # (B,D,64,64)
            acc = hr_t if acc is None else acc + hr_t
        return acc / T

    def _feed(self, feats, rgb):
        """Per-subclass (feats_per_t, rgb) feed. Base: single time-pooled map."""
        return self._feats_2d(feats), rgb

    @torch.no_grad()
    def features(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Frozen AnyUp-upsampled per-pixel feature map (B,D,64,64) for KNN (pre-probe)."""
        f, g = self._feed(feats, rgb)
        return self._upsampled(f, g)

    def forward(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        out = (self.label_size, self.label_size)
        return self.up(self._feats_2d(feats), rgb, out_size=out)


class CachedAnyUpT2(CachedAnyUp):
    """anyup_t2: shared time-pooled features, per-timestep RGB guidance (rgb is (B,T,3,64,64))."""
    # forward identical to CachedAnyUp: AnyUpUpsampleProbe loops T off the 5-D rgb, reusing
    # the single feature map for every timestep.


class CachedAnyUpT1(CachedAnyUp):
    """anyup_t1: per-timestep features AND per-timestep RGB (heaviest)."""

    def _feats_per_t(self, feats):
        # list of T (B,D,gH,gW), one per cached timestep
        return [feats[:, t].permute(0, 3, 1, 2).contiguous() for t in range(feats.shape[1])]

    def _feed(self, feats, rgb):                          # per-t features + per-t rgb
        return self._feats_per_t(feats), rgb

    def forward(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        out = (self.label_size, self.label_size)
        return self.up(self._feats_per_t(feats), rgb, out_size=out)


class CachedManyUp(nn.Module):
    """mAnyUp head: FROZEN trained upsampler (+ optional projector) with a trainable LP probe.

    Pipeline: cached LR feats (mean-T) -> [frozen mAnyUp upsample to 64x64] -> [frozen projector
    if use_proj] -> trainable 1x1 probe -> bilinear to label size. Only the probe trains -- this
    is linear-probing on top of frozen mAnyUp-upsampled features. Guidance is the 13-band S2
    ('mean13'). The checkpoint (from train_manyup.py) carries model + optional proj_head weights,
    input_dim (13), and qk_dim."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 ckpt_path: str, use_proj: bool = True, label_size: int = LABEL_SIZE):
        super().__init__()
        import sys
        sys.path.insert(0, "/scratch/timz/anyup")
        from anyup.model import AnyUp

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.up = AnyUp(input_dim=ck.get("input_dim", S2_BANDS), qk_dim=ck.get("qk_dim", 128))
        self.up.load_state_dict(ck["model"])
        self.proj = None
        if use_proj and ck.get("proj_head") is not None:
            self.proj = nn.Conv2d(embed_dim, embed_dim, 1)
            self.proj.load_state_dict(ck["proj_head"])
        # Freeze the whole upsampling pipeline; only the probe below is trained.
        for m in (self.up, self.proj):
            if m is not None:
                for prm in m.parameters():
                    prm.requires_grad = False
        self.probe = nn.Conv2d(embed_dim, num_classes, 1)     # the ONLY trainable module (LP)
        self.label_size = label_size
        self.ckpt_path = ckpt_path

    def _feats_2d(self, feats):                               # (B,T,gH,gW,D) -> (B,D,gH,gW)
        return feats.mean(dim=1).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def features(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Frozen mAnyUp-upsampled (+projected) per-pixel feature map (B,D,64,64). Shared by
        forward()'s LP probe and by KNN eval -- both read the SAME frozen features."""
        hr = self.up(rgb, self._feats_2d(feats), (self.label_size, self.label_size))
        if self.proj is not None:
            hr = self.proj(hr)
        return hr                                            # (B,D,64,64)

    def forward(self, feats: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        return self.probe(self.features(feats, rgb))         # (B,C,64,64) at label size already


def build_cached_head(name: str, embed_dim: int, num_classes: int, patch_size: int,
                      manyup_ckpt: str = None, manyup_use_proj: bool = True) -> nn.Module:
    # (class, extra kwargs). The _ens variants reuse the same wrapper but give each timestep its
    # own probe and average the per-timestep logits instead of mean-pooling features (see
    # AnyUpUpsampleProbe.ensemble). t1_ens = per-timestep features; t2_ens = shared time-pooled
    # features -- both with per-timestep hr probes.
    HEADS = {
        "lp_pa2pa_bu": (LPPatchToPatchBU, {}),
        "lp_pa2px": (LPPatchToPixel, {}),
        "lp_pa2px_ens": (LPPatchToPixel, {"ensemble": True}),
        "anyup": (CachedAnyUp, {}),
        "anyup_t2": (CachedAnyUpT2, {}),
        "anyup_t1": (CachedAnyUpT1, {}),
        "anyup_t2_ens": (CachedAnyUpT2, {"ensemble": True}),
        "anyup_t1_ens": (CachedAnyUpT1, {"ensemble": True}),
    }
    if name == "manyup":
        if not manyup_ckpt:
            raise ValueError("head_mode=manyup requires a checkpoint (--manyup discovery or --manyup_ckpt)")
        return CachedManyUp(embed_dim, num_classes, patch_size,
                            ckpt_path=manyup_ckpt, use_proj=manyup_use_proj)
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


# ----------------------------- KNN eval -----------------------------
@torch.no_grad()
def _collect_pixels(head, loader, device, max_pixels=None, seed=0):
    """Run head.features() over a loader, flatten to per-pixel (N,D) features + (N,) labels,
    dropping ignore-label pixels. If max_pixels is set, randomly subsample to that many (the KNN
    reference set is bounded this way). L2-normalizes features so dot product == cosine sim."""
    feats_all, labels_all = [], []
    for feats, label, rgb in loader:
        rgb_d = None if rgb.numel() == 0 else rgb.to(device)
        f = head.features(feats.to(device), rgb_d)              # (B,D,H,W)
        B, D, H, W = f.shape
        f = f.permute(0, 2, 3, 1).reshape(-1, D)                # (B*H*W, D)
        lab = label.reshape(-1)                                 # (B*H*W,)
        keep = lab != IGNORE_LABEL
        feats_all.append(F.normalize(f[keep].float(), dim=1).cpu())
        labels_all.append(lab[keep])
    X = torch.cat(feats_all); y = torch.cat(labels_all)
    if max_pixels is not None and X.shape[0] > max_pixels:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(X.shape[0], generator=g)[:max_pixels]
        X, y = X[idx], y[idx]
    return X, y


@torch.no_grad()
def knn_evaluate(head, train_loader, test_loader, device, k=20,
                 ref_pixels=2_000_000, query_chunk=8192, seed=0):
    """Non-parametric KNN segmentation on frozen head.features(). Builds an L2-normalized
    reference set from (subsampled) TRAIN pixels, then for each TEST pixel takes a majority vote
    over its k nearest reference features (cosine similarity). No training. Returns the same
    segmentation_metrics dict as evaluate() for apples-to-apples comparison with LP."""
    head.eval()
    print(f"KNN: building reference from train (<= {ref_pixels} pixels)...")
    Xr, yr = _collect_pixels(head, train_loader, device, max_pixels=ref_pixels, seed=seed)
    Xr = Xr.to(device); yr = yr.to(device)
    print(f"KNN: reference {Xr.shape[0]} pixels x {Xr.shape[1]}-d; k={k}. Scoring test...")

    preds, labels = [], []
    for feats, label, rgb in test_loader:
        rgb_d = None if rgb.numel() == 0 else rgb.to(device)
        f = head.features(feats.to(device), rgb_d)             # (B,D,H,W)
        B, D, H, W = f.shape
        q = F.normalize(f.permute(0, 2, 3, 1).reshape(-1, D).float(), dim=1)  # (Bq,D)
        out = torch.empty(q.shape[0], dtype=torch.long)
        # Chunk queries so the (chunk x ref) similarity matrix fits in VRAM.
        for s in range(0, q.shape[0], query_chunk):
            qc = q[s:s + query_chunk].to(device)               # (c,D)
            sim = qc @ Xr.T                                     # (c, Nref) cosine
            nn_idx = sim.topk(k, dim=1).indices                # (c, k)
            votes = yr[nn_idx]                                 # (c, k) labels
            # majority vote per row via bincount over class ids
            maj = torch.stack([torch.bincount(v, minlength=NUM_CLASSES).argmax() for v in votes])
            out[s:s + query_chunk] = maj.cpu()
        preds.append(out.reshape(B, H, W))
        labels.append(label)
    return segmentation_metrics(torch.cat(preds), torch.cat(labels),
                                num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL)


RESULT_COLUMNS = [
    "timestamp", "features", "head_mode", "eval_kind", "manyup_ckpt", "manyup_use_proj",
    "epochs", "lr", "knn_k", "batch_size", "seed",
    "test_miou", "test_overall_acc", "avg_epoch_sec",
]


def append_result(csv_path: Path, row: dict) -> None:
    """Append one run's args + test metrics to csv_path, writing the header if the file is new.
    If an EXISTING file has an older header (fewer columns, e.g. from before manyup_* were added),
    honor that file's header so rows stay column-aligned; keys not in it are dropped and missing
    ones filled blank (extrasaction='ignore', restval='')."""
    fieldnames = RESULT_COLUMNS
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            header = next(csv.reader(f), None)
        if header:
            fieldnames = header          # match the file already on disk
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        if new_file:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="LP on cached OlmoEarth features.")
    p.add_argument("--features", required=True,
                   help="extraction config folder name under --out_root, e.g. oe_base_s2s1_ps4_tile64")
    p.add_argument("--out_root", default="~/projects/aip-gpleiss/timz/features")
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
    # --- mAnyUp options (head_mode=manyup) ---
    p.add_argument("--manyup", action="store_true",
                   help="head_mode=manyup + auto-discover all mAnyUp checkpoints trained to "
                        "upsample --features (the LR config) and LP over each (one CSV row per).")
    p.add_argument("--manyup_ckpt", default=None,
                   help="explicit mAnyUp checkpoint to LP (instead of auto-discovery)")
    p.add_argument("--manyup_root", default="checkpoints/manyup",
                   help="root scanned for <features>__to__*/*.pth mAnyUp checkpoints")
    p.add_argument("--manyup_use_proj", action=argparse.BooleanOptionalAction, default=True,
                   help="include the trained projector in the frozen mAnyUp pipeline "
                        "(--no-manyup_use_proj to probe the raw upsampled ps4-space features)")
    # --- KNN eval (instead of LP): non-parametric, no training ---
    p.add_argument("--knn", action="store_true",
                   help="evaluate features by KNN vote (no probe training) instead of LP. Works "
                        "for lp_pa2pa_bu/lp_pa2px (raw features) and manyup (upsampled features).")
    p.add_argument("--knn_k", type=int, default=20, help="neighbors per KNN query")
    p.add_argument("--knn_ref_pixels", type=int, default=2_000_000,
                   help="max train pixels in the KNN reference set (subsampled)")
    args = p.parse_args()

    # --manyup / --manyup_ckpt implies head_mode=manyup (convenience so you don't pass both).
    if args.manyup or args.manyup_ckpt:
        args.head_mode = "manyup"

    torch.manual_seed(args.seed)
    # AnyUp runs its (attention-heavy) upsample in fp32; TF32 lets the L40s tensor cores do
    # those matmuls ~2x faster at negligible precision cost. Frozen AnyUp + tiny probe means
    # the slight TF32 rounding is immaterial to results. Big win for the anyup* heads, which
    # call AnyUp T times per sample.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = Path(args.out_root).expanduser() / args.features   # expanduser: default uses ~
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

    # --- mAnyUp: discover the checkpoints to LP over. --manyup scans for models trained to
    # upsample THIS --features (the LR config): checkpoints/<features>__to__*/*.pth, latest epoch
    # per LR->HR pair. Each becomes its own LP run + CSV row. --manyup_ckpt runs a single explicit
    # one. For non-manyup heads this is a single [None] -> one normal run.
    if args.head_mode == "manyup":
        if args.manyup_ckpt:
            ckpts = [Path(args.manyup_ckpt)]
        else:
            ckpts = _discover_manyup_ckpts(Path(args.manyup_root), args.features)
            if not ckpts:
                raise FileNotFoundError(
                    f"no mAnyUp checkpoints for LR={args.features} under {args.manyup_root} "
                    f"(expected {args.features}__to__*/*.pth). Train one with train_manyup.sh.")
            print(f"mAnyUp: {len(ckpts)} checkpoint(s) to LP over:")
            for c in ckpts:
                print(f"  {c}")
    else:
        ckpts = [None]

    for ckpt in ckpts:
        run_one(args, device, embed_dim, patch_size, train_loader, val_loader, test_loader,
                manyup_ckpt=str(ckpt) if ckpt is not None else None)


def _discover_manyup_ckpts(manyup_root: Path, lr_features: str) -> list:
    """Find the latest-epoch checkpoint for each mAnyUp model that upsamples `lr_features`.
    Layout (from train_manyup.sh): <manyup_root>/<lr>__to__<hr>/manyup_..._ep<N>.pth."""
    pairs = sorted(manyup_root.glob(f"{lr_features}__to__*"))
    latest = []
    for d in pairs:
        cks = list(d.glob("*.pth"))
        if not cks:
            continue
        # pick highest ep<N> (fall back to mtime if names don't parse)
        def epoch_of(p: Path) -> int:
            stem = p.stem
            return int(stem.split("_ep")[-1]) if "_ep" in stem else -1
        latest.append(max(cks, key=lambda p: (epoch_of(p), p.stat().st_mtime)))
    return latest


def run_one(args, device, embed_dim, patch_size, train_loader, val_loader, test_loader,
            manyup_ckpt=None) -> None:
    """One LP training run (build head -> train -> eval -> log). Loaders are shared across
    mAnyUp checkpoints (same LR features + guidance), so only the head differs per run."""
    tag = f"{'KNN' if args.knn else 'LP'} run"
    if manyup_ckpt:
        print(f"\n===== mAnyUp {tag}: {manyup_ckpt} (proj={args.manyup_use_proj}) =====")
    head = build_cached_head(args.head_mode, embed_dim, NUM_CLASSES, patch_size,
                             manyup_ckpt=manyup_ckpt,
                             manyup_use_proj=args.manyup_use_proj).to(device)

    # KNN: non-parametric, no training. Extract frozen features, vote over train neighbors, log.
    if args.knn:
        if not hasattr(head, "features"):
            raise ValueError(f"head_mode={args.head_mode!r} has no features() for KNN "
                             f"(supported: lp_pa2pa_bu, lp_pa2px, anyup*, manyup)")
        t0 = time.perf_counter()
        test = knn_evaluate(head, train_loader, test_loader, device,
                            k=args.knn_k, ref_pixels=args.knn_ref_pixels, seed=args.seed)
        elapsed = time.perf_counter() - t0
        print(f"KNN TEST {test.metrics}  ({elapsed:.1f}s)")
        _log_result(args, manyup_ckpt, test, avg_epoch_time=elapsed, eval_kind="knn")
        return

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

    _log_result(args, manyup_ckpt, test, avg_epoch_time, eval_kind="lp")


def _log_result(args, manyup_ckpt, test, avg_epoch_time, eval_kind="lp") -> None:
    """Append one run's test metrics + provenance to the results CSV. Shared by the LP and KNN
    paths. eval_kind distinguishes them; epochs/lr are blanked for KNN (not applicable)."""
    # Keep the CSV readable: ints/strings as-is, metrics+time as 2-decimal floats. lr is the
    # one value .2f would mangle (1e-3 -> 0.00), so log it with %g (compact, full precision).
    append_result(Path(args.results_csv), {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "features": args.features,
        "head_mode": args.head_mode,
        "eval_kind": eval_kind,
        # mAnyUp provenance: which upsampler checkpoint (+ whether its projector was used) so
        # looped runs are distinguishable in the CSV. Empty for non-manyup heads.
        "manyup_ckpt": Path(manyup_ckpt).name if manyup_ckpt else "",
        "manyup_use_proj": (args.manyup_use_proj if manyup_ckpt else ""),
        "epochs": ("" if eval_kind == "knn" else int(args.epochs)),
        "lr": ("" if eval_kind == "knn" else f"{args.lr:g}"),
        "knn_k": (args.knn_k if eval_kind == "knn" else ""),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "test_miou": f"{test.metrics['miou']:.2f}",
        "test_overall_acc": f"{test.metrics['overall_acc']:.2f}",
        "avg_epoch_sec": f"{avg_epoch_time:.2f}",
    })
    print(f"Appended {eval_kind} result to {args.results_csv}")


if __name__ == "__main__":
    main()
