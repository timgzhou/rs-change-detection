"""Fine-tune OlmoEarth on PASTIS semantic segmentation.

Reuses OlmoEarth's own eval pipeline (PASTISRDataset, BackboneWithHead, eval_seg)
so band-mapping / temporal aggregation / pooling / seg-head all match their
benchmark. The training loop is lifted from olmoearth_pretrain.evals.finetune.train
.run_finetune_eval, minus the olmo_core.Trainer / wandb wrapper (we only need the
inner loop + the freeze-then-unfreeze warmup).

Runs in the SEPARATE OlmoEarth venv (torch 2.7.x), not ./env:
    source env_olmo.sh
    python -u prepare_pastis_olmoearth.py   # one-time, builds data/pastis_olmoearth/
    python -u finetune_olmoearth_pastis.py
"""
import os
import sys
# Bootstrap MUST run before any olmoearth_pretrain import: it stubs hdf5plugin +
# unused dataset/model siblings and loads h5py early to dodge the cluster HDF5/rasterio
# ABI clash. See olmo_shims/olmo_bootstrap.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "olmo_shims"))
import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()  # MUST run before any olmoearth_pretrain import

import math
from pathlib import Path
from typing import cast

import torch 
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Load from the FULL package (not olmoearth_pretrain_minimal): the eval wrapper's
# get_eval_wrapper dispatches on isinstance(encoder, FlexiVitBase), and only the full
# package's encoder class matches it. The full package exposes OLMOEARTH_V1_BASE
# (no V1_1 variant).
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.evals.datasets.configs import DATASET_TO_CONFIG, TaskType
from olmoearth_pretrain.evals.datasets.pastis_dataset import PASTISRDataset
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn
from olmoearth_pretrain.evals.finetune.constants import (
    FREEZE_EPOCH_FRACTION,
    SCHEDULER_COOLDOWN,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    UNFREEZE_LR_FACTOR,
)
from olmoearth_pretrain.evals.eval_wrapper import get_eval_wrapper
from olmoearth_pretrain.evals.metrics import segmentation_metrics
from olmoearth_pretrain.evals.finetune.model import (
    BackboneWithHead,
    set_backbone_trainable,
    snapshot_state_dict,
    to_device,
)
from olmoearth_pretrain.nn.flexi_vit import PoolingType

from config import Config, load_config, to_dict

# ---- runtime config (module globals; helpers below read these at call time) ----
# These are populated from a Config by _apply_config() in main(). The defaults here
# mirror Config()'s defaults so the module is importable/usable without a YAML (e.g.
# visualize_olmoearth_pastis.py imports DATA_SPLITS / INPUT_MODALITIES / etc.).
DATA_SPLITS = "data/pastis_olmoearth"   # output of prepare_pastis_olmoearth.py
MODEL_ID = ModelID.OLMOEARTH_V1_BASE
DATASET = "pastis"                       # 64x64 config (matches resize_to_64=True)
INPUT_MODALITIES = ["sentinel2_l2a", "sentinel1"]
POOLING_TYPE = PoolingType.MEAN          # evaluator_callback default
EPOCHS = 64
BATCH_SIZE = 32
NUM_WORKERS = 0     # 0 avoids DataLoader-worker fork crashing on the h5py/HDF5 ABI;
                    # PASTIS reads small .pt files so this isn't a bottleneck.
LR = 1e-3
SEED = 0
# Head mode drives everything: "lp" | "anyup" | "anyup_t2" | "anyup_t1".
# anyup_t2 = shared time-pooled features, per-timestep RGB guidance, mean-pool outputs.
# anyup_t1 = per-timestep features AND per-timestep guidance, mean-pool outputs.
HEAD = "lp"
FREEZE_BACKBONE = False   # if True, encoder stays frozen all epochs (only head trains)
PATCH_SIZE = 4            # token grid = 64/PATCH_SIZE per side (OlmoEarth LP eval uses 4)
CKPT_PATH = "checkpoints/olmoearth_pastis_lp_best.pt"


def _is_anyup(head=None):
    return (head or HEAD).startswith("anyup")


def _is_temporal_anyup(head=None):
    return (head or HEAD) in ("anyup_t1", "anyup_t2")


def _apply_config(cfg: Config) -> None:
    """Populate the module globals from a Config. Helpers read these names at call
    time, so setting them here (before building loaders/model) is sufficient."""
    g = globals()
    g["DATA_SPLITS"] = cfg.data_splits
    g["DATASET"] = cfg.dataset
    g["INPUT_MODALITIES"] = cfg.input_modalities
    g["MODEL_ID"] = getattr(ModelID, cfg.model_id_name)
    g["EPOCHS"] = cfg.epochs
    g["BATCH_SIZE"] = cfg.batch_size
    g["NUM_WORKERS"] = cfg.num_workers
    g["LR"] = cfg.lr
    g["SEED"] = cfg.seed
    g["HEAD"] = cfg.head
    g["FREEZE_BACKBONE"] = cfg.freeze_backbone
    g["PATCH_SIZE"] = cfg.patch_size
    g["CKPT_PATH"] = cfg.ckpt_path

# ImageNet normalization for AnyUp's RGB guidance image.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _norm_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """(...,3,H,W) raw -> [0,1] (per-image min/max) -> ImageNet-normalized, as AnyUp expects."""
    flat = rgb.reshape(*rgb.shape[:-3], 3, -1)
    lo = flat.amin(-1).reshape(*rgb.shape[:-3], 3, 1, 1)
    hi = flat.amax(-1).reshape(*rgb.shape[:-3], 3, 1, 1)
    rgb = (rgb - lo) / (hi - lo + 1e-6)
    # (3,1,1) so it broadcasts to both (3,H,W) and (...,3,H,W) without adding a dim.
    return (rgb - _IMAGENET_MEAN.view(3, 1, 1)) / _IMAGENET_STD.view(3, 1, 1)


def _load_rgb_guidance(split: str, idx: int, temporal: bool = False) -> torch.Tensor:
    """Raw S2 -> RGB (B04/B03/B02 = idx 3/2/1 in the 13-band L1C stack), normalized.
    temporal=False -> time-averaged (3,64,64); temporal=True -> per-timestep (T,3,64,64)."""
    s2 = torch.load(Path(DATA_SPLITS) / f"pastis_r_{split}" / "s2_images" / f"{idx}.pt").float()
    if temporal:
        rgb = s2[:, [3, 2, 1]]                # (T,3,64,64)
    else:
        rgb = s2.mean(0)[[3, 2, 1]]           # (3,64,64)
    return _norm_rgb(rgb)


class _GuidancePASTIS(torch.utils.data.Dataset):
    """Wraps PASTISRDataset to also yield AnyUp's RGB guidance. temporal=True yields a
    per-timestep guidance stack (T,3,64,64); else a single time-averaged (3,64,64)."""

    def __init__(self, split: str, temporal: bool):
        self.ds = PASTISRDataset(
            path_to_splits=Path(DATA_SPLITS), split=split, partition="default",
            norm_stats_from_pretrained=True, input_modalities=INPUT_MODALITIES,
        )
        self.split = split
        self.temporal = temporal

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        masked, label = self.ds[i]
        return masked, label, _load_rgb_guidance(self.split, i, self.temporal)


def _guidance_collate(batch):
    samples = [(m, l) for m, l, _ in batch]
    masked, label = eval_collate_fn(samples)
    rgb = torch.stack([r for _, _, r in batch])   # (B,3,64,64) or (B,T,3,64,64)
    return masked, label, rgb


class AnyUpUpsampleProbe(nn.Module):
    """The encoder-independent half of every AnyUp head: a FROZEN AnyUp upsampler plus a
    lazily-sized 1x1 probe. Given per-timestep feature map(s) and RGB guidance, it runs
    AnyUp (once or per-timestep), mean-pools the upsampled maps over time, and probes.

    Shared by the live AnyUp heads (which get features from the OlmoEarth encoder) and the
    cached-feature heads (which read (T,gH,gW,D) from disk) so the AnyUp+probe path is one
    source of truth. AnyUp is the pretrained multi-backbone model, kept frozen (we rely on
    its RGB guidance). The probe's in_dim (= D) is only known at first forward.
    """

    def __init__(self, num_classes: int, ensemble: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        # ensemble=False: mean-pool the T upsampled feature maps, then one shared probe (the
        # original behaviour). ensemble=True: give EACH timestep its own probe and mean the T
        # per-timestep LOGITS (pre-softmax logit averaging) -- a genuine temporal ensemble that
        # is NOT algebraically equal to mean-then-probe because the probes differ per timestep.
        self.ensemble = ensemble
        self.anyup = cast(nn.Module, torch.hub.load(
            "wimmerth/anyup", "anyup_multi_backbone", use_natten=False, pretrained=True))
        for p in self.anyup.parameters():       # frozen guidance upsampler
            p.requires_grad = False
        self.probe = nn.Conv2d(1, 1, 1)         # placeholder; real in_dim (+ T) on first forward
        self._inited = False

    def _init_probe(self, dim: int, device: torch.device, n_probes: int = 1) -> None:
        # n_probes: 1 shared probe (default) or T independent probes for the ensemble variant.
        if self.ensemble:
            self.probe = nn.ModuleList(
                [nn.Conv2d(dim, self.num_classes, kernel_size=1) for _ in range(n_probes)]
            ).to(device)
        else:
            self.probe = nn.Conv2d(dim, self.num_classes, kernel_size=1).to(device)
        self._inited = True

    def forward(self, feats_per_t, rgb, out_size):
        """feats_per_t: either a single (B,D,gH,gW) tensor reused for every timestep, or a
        list/tuple of T such tensors (one per timestep). rgb: (B,3,H,W) reused for all t,
        or (B,T,3,H,W) per-timestep. AnyUp runs in fp32. Returns (B, num_classes, *out_size).
        """
        per_t_feats = isinstance(feats_per_t, (list, tuple))
        per_t_rgb = rgb.dim() == 5
        T = (len(feats_per_t) if per_t_feats
             else rgb.shape[1] if per_t_rgb else 1)
        # Cast the shared feature map ONCE (t2/anyup reuse a single feats across all T, so
        # re-.float()ing it each iteration is pure waste). Per-t feature lists (t1) still
        # cast inside the loop since each timestep is a distinct tensor.
        shared_f = None if per_t_feats else feats_per_t.float()
        if not self._inited:
            probe_dim = (feats_per_t[0] if per_t_feats else feats_per_t).shape[1]
            probe_dev = (feats_per_t[0] if per_t_feats else feats_per_t).device
            self._init_probe(probe_dim, probe_dev, n_probes=T)
        acc = None
        for t in range(T):
            f = feats_per_t[t].float() if per_t_feats else shared_f
            g = (rgb[:, t] if per_t_rgb else rgb).float()
            hr_t = self.anyup(g, f, output_size=out_size)           # (B,D,*out_size)
            if self.ensemble:
                # probe each timestep's upsampled features with its OWN probe, then average the
                # T logits (pre-softmax). One probe per t makes this a real ensemble.
                logit_t = self.probe[t](hr_t)                       # (B,C,*out_size)
                acc = logit_t if acc is None else acc + logit_t
            else:
                acc = hr_t if acc is None else acc + hr_t           # accumulate features
        if self.ensemble:
            return acc / T                                          # mean of per-t logits
        return self.probe(acc / T)                                  # one probe on mean features


class AnyUpHead(nn.Module):
    """OlmoEarth encoder -> (B,gH,gW,D) feature map -> AnyUp upsample to (B,D,64,64) using
    an RGB guidance image -> 1x1 conv (D -> num_classes). Returns (B,C,64,64).

    The AnyUp+probe half lives in AnyUpUpsampleProbe; this class only supplies the encoder
    features (single, time-pooled) and a single mean RGB guidance.
    """

    def __init__(self, encoder: nn.Module, patch_size: int, num_classes: int,
                 task_type: TaskType, pooling_type) -> None:
        super().__init__()
        # Register the encoder as a real submodule so .to(device)/state_dict recurse
        # into it (the EvalWrapper is a plain object and would not be moved otherwise).
        self.encoder = encoder
        self.wrapper = get_eval_wrapper(
            encoder, task_type=task_type, patch_size=patch_size,
            pooling_type=pooling_type, concat_features=False, use_pooled_tokens=False,
        )
        self.up = AnyUpUpsampleProbe(num_classes)

    @property
    def backbone(self) -> nn.Module:
        return self.encoder  # the OlmoEarth encoder (for freeze/unfreeze)

    def forward(self, masked, label, rgb, is_train: bool = True):
        dev = next(self.wrapper.parameters()).device
        emb, label = self.wrapper(masked, label, is_train=is_train)  # (B,gH,gW,D)
        feats = cast(torch.Tensor, emb).permute(0, 3, 1, 2).contiguous()  # (B,D,gH,gW)
        logits = self.up(feats, rgb.to(dev), out_size=label.shape[-2:])
        return logits, label


class AnyUpHeadT2(AnyUpHead):
    """Shared time-pooled features (B,gH,gW,D), but AnyUp is run once per timestep using
    that month's RGB guidance; the T upsampled feature maps are mean-pooled, then probed.
    rgb is (B,T,3,64,64). ~T x the AnyUp cost of the plain head."""

    def forward(self, masked, label, rgb, is_train: bool = True):
        dev = next(self.wrapper.parameters()).device
        emb, label = self.wrapper(masked, label, is_train=is_train)
        feats = cast(torch.Tensor, emb).permute(0, 3, 1, 2).contiguous()  # (B,D,gH,gW)
        logits = self.up(feats, rgb.to(dev), out_size=label.shape[-2:])    # per-t rgb
        return logits, label


class AnyUpHeadT1(AnyUpHead):
    """Per-timestep features AND per-timestep guidance: for each t, pool only that
    timestep's tokens -> (B,gH,gW,D), AnyUp with that month's RGB, mean-pool over T, probe.
    rgb is (B,T,3,64,64). Heaviest variant (per-t feature pooling + per-t AnyUp)."""

    def forward(self, masked, label, rgb, is_train: bool = True):
        dev = next(self.encoder.parameters()).device
        label = label.to(dev)
        # raw tokens (B,gH,gW,T,BandSets,D), before any temporal pooling
        tam = self.encoder(masked, patch_size=self.wrapper.patch_size,
                           fast_pass=True)["tokens_and_masks"]
        T = rgb.shape[1]
        feats_per_t = [pool_per_timestep(tam, t, POOLING_TYPE).permute(0, 3, 1, 2).contiguous()
                       for t in range(T)]
        logits = self.up(feats_per_t, rgb.to(dev), out_size=label.shape[-2:])
        return logits, label


def pool_per_timestep(tam, t, pooling_type):
    """Pool only timestep t of a TokensAndMasks -> (B, gH, gW, D), reusing the model's
    own spatial/bandset/modality reduction (a time-mean over a single sliced step is a
    no-op). Shared by AnyUpHeadT1 and the offline feature extractor so both produce
    identical per-timestep features."""
    repl = {}
    for m in tam.modalities:
        mn = tam.get_masked_modality_name(m)
        repl[m] = getattr(tam, m)[:, :, :, t:t + 1]
        repl[mn] = getattr(tam, mn)[:, :, :, t:t + 1]
    return tam._replace(**repl).pool_unmasked_tokens(pooling_type, spatial_pooling=True)


def build_head(head: str, encoder, patch_size, task_config):
    """Construct the segmentation head for the given mode."""
    if head == "lp":
        return BackboneWithHead(
            model=encoder, task_type=task_config.task_type, patch_size=patch_size,
            pooling_type=POOLING_TYPE, num_classes=task_config.num_classes,
            use_pooled_tokens=False,
        )
    cls = {"anyup": AnyUpHead, "anyup_t2": AnyUpHeadT2, "anyup_t1": AnyUpHeadT1}[head]
    return cls(encoder, patch_size, task_config.num_classes,
               task_config.task_type, POOLING_TYPE)


def make_loader(split: str, shuffle: bool) -> DataLoader:
    if _is_anyup():
        # temporal heads need per-timestep guidance (T,3,64,64); plain anyup a single (3,64,64)
        return DataLoader(_GuidancePASTIS(split, temporal=_is_temporal_anyup()),
                          batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                          shuffle=shuffle, collate_fn=_guidance_collate)
    ds = PASTISRDataset(
        path_to_splits=Path(DATA_SPLITS),
        split=split,
        partition="default",   # use all labels (no low-label-fraction subset)
        norm_stats_from_pretrained=True,
        input_modalities=INPUT_MODALITIES,
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=shuffle,
        collate_fn=eval_collate_fn,
    )


def _forward_logits(ft, batch, device, task_config, patch_size):
    """Unified forward -> (B, num_classes, 64, 64) for both heads.

    LP batch is (masked, label); AnyUp batch is (masked, label, rgb).
    """
    if _is_anyup():
        masked, label, rgb = batch
        label = label.to(device)
        logits, label = ft(to_device(masked, device), label, rgb, is_train=ft.training)
        return logits, label  # already (B, C, H, W)
    masked, label = batch
    label = label.to(device)
    logits, label = ft(to_device(masked, device), label)          # (B, h, w, C*p*p)
    h, w = logits.shape[1], logits.shape[2]
    logits = rearrange(logits, "b h w (c i j) -> b c (h i) (w j)",
                       h=h, w=w, c=task_config.num_classes, i=patch_size, j=patch_size)
    if logits.shape[-2:] != label.shape[-2:]:
        logits = F.interpolate(logits.float(), size=label.shape[-2:],
                               mode="bilinear", align_corners=True)
    return logits, label


@torch.no_grad()
def _evaluate(ft, loader, device, task_config, patch_size):
    """Mirror eval_seg but via _forward_logits so it works for both heads."""
    ft.eval()
    preds, labels = [], []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, label = _forward_logits(ft, batch, device, task_config, patch_size)
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(label.cpu())
    return segmentation_metrics(torch.cat(preds), torch.cat(labels),
                                num_classes=task_config.num_classes, ignore_label=-1)


def main(cfg: Config) -> None:
    _apply_config(cfg)
    print(f"Run: {cfg.run_name}")
    print(f"Config: {to_dict(cfg)}")
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_config = DATASET_TO_CONFIG[DATASET]
    assert task_config.task_type == TaskType.SEGMENTATION

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("valid", shuffle=False)
    test_loader = make_loader("test", shuffle=False)

    model = load_model_from_id(MODEL_ID, load_weights=True)
    # BackboneWithHead/get_eval_wrapper want the ENCODER (a FlexiVitBase), not the
    # full LatentMIM (matches evaluator_callback, which passes model.encoder).
    encoder = cast(nn.Module, model.encoder if hasattr(model, "encoder") else model)
    # patch_size comes from config (default 4, OlmoEarth's LP eval protocol); encoder's
    # own patch_size is None (FlexiViT takes it per-call).
    patch_size = PATCH_SIZE
    print(f"Using patch_size={patch_size}, pooling={POOLING_TYPE}, num_classes={task_config.num_classes}")

    print(f"Head: {HEAD}")
    ft = build_head(HEAD, encoder, patch_size, task_config).to(device)

    # Dry pass to lazy-init the head (probe in_dim depends on model embedding size).
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _forward_logits(ft, next(iter(train_loader)), device, task_config, patch_size)

    # Freeze schedule. FREEZE_BACKBONE=True -> encoder stays frozen for ALL epochs
    # (linear-probe-on-frozen-features). Otherwise the freeze-then-unfreeze warmup.
    if FREEZE_BACKBONE:
        freeze_epochs = EPOCHS + 1   # unfreeze condition (epoch >= freeze_epochs) never fires
    else:
        freeze_epochs = math.ceil(FREEZE_EPOCH_FRACTION * EPOCHS) if EPOCHS > 0 else 0
    backbone_unfrozen = freeze_epochs == 0
    if not backbone_unfrozen:
        set_backbone_trainable(ft.backbone, False)
        print(f"Backbone frozen for {'ALL' if FREEZE_BACKBONE else f'first {freeze_epochs}'} epoch(s).")

    current_lr = LR
    opt = torch.optim.AdamW(ft.parameters(), lr=current_lr)
    scheduler = ReduceLROnPlateau(
        opt, mode="max", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR, cooldown=SCHEDULER_COOLDOWN,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_state = snapshot_state_dict(ft)
    best_val_miou = float("-inf")

    for epoch in range(EPOCHS):
        if not backbone_unfrozen and epoch >= freeze_epochs:
            set_backbone_trainable(ft.backbone, True)
            backbone_unfrozen = True
            current_lr = LR * UNFREEZE_LR_FACTOR
            for group in opt.param_groups:
                group["lr"] = current_lr
            print(f"Backbone unfrozen at epoch {epoch}; lr -> {current_lr:.3e}")

        ft.train()
        last_loss = float("nan")
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in pbar:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, label = _forward_logits(ft, batch, device, task_config, patch_size)
                loss = loss_fn(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
            pbar.set_postfix(loss=f"{last_loss:.4f}")

        val = _evaluate(ft, val_loader, device, task_config, patch_size)
        scheduler.step(val.primary)
        print(f"epoch {epoch+1}/{EPOCHS} | train_loss {last_loss:.4f} | "
              f"val miou {val.primary:.4f} | {val.metrics}")
        if val.primary > best_val_miou:
            best_val_miou = val.primary
            best_state = snapshot_state_dict(ft)

    # Restore best and evaluate on test.
    ft.load_state_dict(best_state)
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save(best_state, CKPT_PATH)
    test = _evaluate(ft, test_loader, device, task_config, patch_size)
    print(f"\nBEST val miou {best_val_miou:.4f}")
    print(f"TEST {test.metrics}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finetune OlmoEarth on PASTIS.")
    parser.add_argument("--config", default=None,
                        help="Optional YAML overriding configs/defaults.yaml.")
    parser.add_argument("--set", nargs="*", default=[], metavar="key=value",
                        help="Override config fields, e.g. --set model_size=base "
                             "modalities=sentinel2_l2a,sentinel1 head_mode=anyup_t1 freeze_backbone=true")
    args = parser.parse_args()
    main(load_config(args.config, getattr(args, "set")))
