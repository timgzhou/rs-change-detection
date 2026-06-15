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
# Head: False -> patch linear probe (BackboneWithHead); True -> AnyUp pixel head.
USE_ANYUP = False
HEAD = "lp"
CKPT_PATH = "checkpoints/olmoearth_pastis_lp_best.pt"


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
    g["USE_ANYUP"] = cfg.use_anyup
    g["HEAD"] = cfg.head
    g["CKPT_PATH"] = cfg.ckpt_path

# ImageNet normalization for AnyUp's RGB guidance image.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _load_rgb_guidance(split: str, idx: int) -> torch.Tensor:
    """Raw S2 -> time-averaged RGB (B04/B03/B02 = idx 3/2/1 in the 13-band L1C stack),
    scaled to [0,1] then ImageNet-normalized. Returns (3, 64, 64)."""
    s2 = torch.load(Path(DATA_SPLITS) / f"pastis_r_{split}" / "s2_images" / f"{idx}.pt")
    rgb = s2.float().mean(0)[[3, 2, 1]]               # (3, 64, 64)
    rgb = (rgb - rgb.amin()) / (rgb.amax() - rgb.amin() + 1e-6)
    rgb = (rgb - _IMAGENET_MEAN[0]) / _IMAGENET_STD[0]
    return rgb


class _GuidancePASTIS(torch.utils.data.Dataset):
    """Wraps PASTISRDataset to also yield the RGB guidance image AnyUp needs."""

    def __init__(self, split: str):
        self.ds = PASTISRDataset(
            path_to_splits=Path(DATA_SPLITS), split=split, partition="default",
            norm_stats_from_pretrained=True, input_modalities=INPUT_MODALITIES,
        )
        self.split = split

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        masked, label = self.ds[i]
        return masked, label, _load_rgb_guidance(self.split, i)


def _guidance_collate(batch):
    samples = [(m, l) for m, l, _ in batch]
    masked, label = eval_collate_fn(samples)
    rgb = torch.stack([r for _, _, r in batch])
    return masked, label, rgb


class AnyUpHead(nn.Module):
    """OlmoEarth encoder -> (B,8,8,D) feature map -> AnyUp upsample to (B,D,64,64)
    using an RGB guidance image -> 1x1 conv (D -> num_classes). Returns (B,C,64,64).

    AnyUp is the pretrained multi-backbone model, kept frozen (we rely on RGB bands).
    The probe (1x1 conv) is lazily sized on the first forward (D depends on model size).
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
        self.num_classes = num_classes
        self.anyup = cast(nn.Module, torch.hub.load(
            "wimmerth/anyup", "anyup_multi_backbone", use_natten=False, pretrained=True))
        for p in self.anyup.parameters():       # frozen guidance upsampler
            p.requires_grad = False
        self.probe = nn.Conv2d(1, 1, 1)         # placeholder; real in_dim on first forward
        self._inited = False

    @property
    def backbone(self) -> nn.Module:
        return self.encoder  # the OlmoEarth encoder (for freeze/unfreeze)

    def _init_probe(self, dim: int, device: torch.device) -> None:
        self.probe = nn.Conv2d(dim, self.num_classes, kernel_size=1).to(device)
        self._inited = True

    def forward(self, masked, label, rgb, is_train: bool = True):
        dev = next(self.wrapper.parameters()).device
        emb, label = self.wrapper(masked, label, is_train=is_train)  # (B,8,8,D)
        emb = cast(torch.Tensor, emb)
        feats = emb.permute(0, 3, 1, 2).contiguous()                # (B,D,8,8)
        if not self._inited:
            self._init_probe(feats.shape[1], dev)
        # AnyUp runs in fp32; guidance + features upsampled to label resolution.
        hr = self.anyup(rgb.float().to(dev), feats.float(),
                        output_size=label.shape[-2:])               # (B,D,64,64)
        return self.probe(hr), label


def make_loader(split: str, shuffle: bool) -> DataLoader:
    if USE_ANYUP:
        return DataLoader(_GuidancePASTIS(split), batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS, shuffle=shuffle, collate_fn=_guidance_collate)
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
    if USE_ANYUP:
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
    # encoder.patch_size is None for OlmoEarth; fall back to 8 (the value used in the
    # OlmoEarth README/eval for S2 @10m). 64x64 tiles / 8 -> 8x8 token grid.
    patch_size = getattr(encoder, "patch_size", None) or 8
    print(f"Using patch_size={patch_size}, pooling={POOLING_TYPE}, num_classes={task_config.num_classes}")

    print(f"Head: {HEAD} ({'AnyUp pixel' if USE_ANYUP else 'patch linear probe'})")
    if USE_ANYUP:
        ft = AnyUpHead(encoder, patch_size, task_config.num_classes,
                       task_config.task_type, POOLING_TYPE).to(device)
    else:
        ft = BackboneWithHead(
            model=encoder,
            task_type=task_config.task_type,
            patch_size=patch_size,
            pooling_type=POOLING_TYPE,
            num_classes=task_config.num_classes,
            use_pooled_tokens=False,
        ).to(device)

    # Dry pass to lazy-init the head (probe in_dim depends on model embedding size).
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        _forward_logits(ft, next(iter(train_loader)), device, task_config, patch_size)

    # Freeze-then-unfreeze warmup (verbatim policy from run_finetune_eval).
    freeze_epochs = math.ceil(FREEZE_EPOCH_FRACTION * EPOCHS) if EPOCHS > 0 else 0
    backbone_unfrozen = freeze_epochs == 0
    if not backbone_unfrozen:
        set_backbone_trainable(ft.backbone, False)
        print(f"Backbone frozen for first {freeze_epochs} epoch(s).")

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
    parser.add_argument("--config", default="configs/base_s2s1_anyup.yaml",
                        help="Path to a YAML run config (see config.py / configs/).")
    args = parser.parse_args()
    main(load_config(args.config))
