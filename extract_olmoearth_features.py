"""Extract & cache OlmoEarth encoder features for PASTIS (frozen backbone).

With the backbone frozen, its output is identical across epochs and across head
variants (lp / anyup / anyup_t1 / anyup_t2). Computing it once and caching to disk lets
head experiments (see lp_on_cached_features.py) skip the encoder entirely.

What we cache, per sample: the encoder's PER-TIMESTEP spatial token map, i.e.
    feats[t] = pool over bandsets and across modalities (S2+S1) of timestep t's tokens
        -> (T, gH, gW, D)   with gH = gW = input_size / patch_size, D = embed dim.
Time is kept (not mean-pooled) so temporal heads (anyup_t1/t2) can use it; lp/anyup just
mean over T. This reuses pool_per_timestep() from finetune_olmoearth_pastis.py so the
cached features are bit-for-bit the same reduction the live heads use.

Spatial handling: we operate on the existing 64x64 PASTIS tiles (no resize). input_res is
FIXED at BASE_GSD = 10m (the true physical pixel size). patch_size is the resolution knob
(token grid = 64/patch_size). Smaller patch_size => finer grid => quadratically more tokens
=> quadratically more attention compute; to bound peak memory/compute we optionally split
each 64x64 sample into tile_size x tile_size sub-tiles, encode each independently, and
stitch the token grids back together.

  NOTE: transformer attention is global, so independent sub-tiles do NOT cross-attend.
  tile_size < 64 therefore yields an APPROXIMATION of the full-image features (accepted
  for the compute savings). tile_size == 64 is the exact, non-approximated reference.

Runs in the OlmoEarth venv (torch 2.7.x), via salloc:
    source env_olmo.sh
    python -u extract_olmoearth_features.py --model_size base --patch_size 4 --tile_size 64
"""
import os
import sys

# Bootstrap MUST run before any olmoearth_pretrain import (HDF5/rasterio ABI). Same as
# finetune_olmoearth_pastis.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "olmo_shims"))
import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()

import argparse
import json
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.evals.datasets.pastis_dataset import PASTISRDataset
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn
from olmoearth_pretrain.evals.finetune.model import to_device
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.data.constants import BASE_GSD

from config import MODEL_SIZE_TO_ID, ALLOWED_MODALITIES
from finetune_olmoearth_pastis import pool_per_timestep

POOLING_TYPE = PoolingType.MEAN          # matches finetune default
IMAGE_SIZE = 64                          # PASTIS tiles are 64x64 (resize_to_64=True)
SPLITS = ("train", "valid", "test")


def cfg_name(model_size: str, modalities: list[str], patch_size: int, tile_size: int) -> str:
    """Stable folder name encoding the extraction params, so different settings never
    collide on disk, e.g. oe_base_s2s1_ps4_tile64."""
    mods = "".join({"sentinel2_l2a": "s2", "sentinel1": "s1"}[m] for m in modalities)
    return f"oe_{model_size}_{mods}_ps{patch_size}_tile{tile_size}"


def make_loader(split: str, data_splits: str, modalities: list[str],
                batch_size: int, num_workers: int) -> DataLoader:
    """PASTIS loader (non-anyup branch of finetune's make_loader). shuffle is always
    False here: we write per-sample files keyed by dataset index, so order must be stable."""
    ds = PASTISRDataset(
        path_to_splits=Path(data_splits),
        split=split,
        partition="default",            # all labels, contiguous indices 0..N-1
        norm_stats_from_pretrained=True,
        input_modalities=modalities,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, collate_fn=eval_collate_fn)


def _slice_tile(masked, h0: int, h1: int, w0: int, w1: int):
    """Slice a spatial sub-tile [h0:h1, w0:w1] out of a batched MaskedOlmoEarthSample.
    Spatial modalities and their masks are [B, H, W, T, ...]; non-spatial fields
    (timestamps, latlon) pass through unchanged."""
    repl = {}
    for field in masked._fields:
        val = getattr(masked, field)
        if val is None or field in ("timestamps", "latlon", "latlon_mask"):
            continue
        if val.dim() >= 3:              # [B, H, W, ...]
            repl[field] = val[:, h0:h1, w0:w1]
    return masked._replace(**repl)


@torch.no_grad()
def encode_batch(encoder, masked, patch_size: int, tile_size: int, device) -> torch.Tensor:
    """Encode one batch -> (B, T, gH, gW, D), tiling spatially if tile_size < IMAGE_SIZE.

    For each tile we run the encoder and pool per timestep (pool_per_timestep), then place
    the tile's (B, tg, tg, T, D) token block into the full (B, gH, gW, T, D) grid and
    finally move time to dim 1."""
    masked = to_device(masked, device)
    tg = tile_size // patch_size                       # tokens per tile side
    n = IMAGE_SIZE // tile_size                         # tiles per side
    grid = IMAGE_SIZE // patch_size                     # full token grid side

    full = None  # lazily sized (B, gH, gW, T, D) once we know B, T, D
    for ti in range(n):
        for tj in range(n):
            tile = _slice_tile(masked, ti * tile_size, (ti + 1) * tile_size,
                               tj * tile_size, (tj + 1) * tile_size)
            tam = encoder(tile, patch_size=patch_size, input_res=BASE_GSD,
                          fast_pass=True)["tokens_and_masks"]
            # stack per-timestep pooled maps -> (B, tg, tg, T, D)
            per_t = [pool_per_timestep(tam, t, POOLING_TYPE)
                     for t in range(_num_timesteps(tam))]
            block = torch.stack(per_t, dim=-2)          # (B, tg, tg, T, D)
            if full is None:
                B, _, _, T, D = block.shape
                full = block.new_zeros((B, grid, grid, T, D))
            full[:, ti * tg:(ti + 1) * tg, tj * tg:(tj + 1) * tg] = block
    return full.permute(0, 3, 1, 2, 4).contiguous()     # (B, T, gH, gW, D)


def _num_timesteps(tam) -> int:
    """T from the first spatial modality token tensor (B, gH, gW, T, BandSets, D)."""
    for m in tam.modalities:
        return getattr(tam, m).shape[3]
    raise ValueError("no modalities in TokensAndMasks")


def _dir_size_bytes(path: Path) -> int:
    """Total bytes of all files under path (recursive)."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(nbytes: int) -> str:
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def extract_split(encoder, split: str, out_dir: Path, args, device) -> int:
    """Write one .pt per sample: features/<cfg>/pastis_r_<split>/<idx>.pt -> (T,gH,gW,D) fp16.
    Idempotent: if the folder already has the expected file count, skip the split."""
    split_dir = out_dir / f"pastis_r_{split}"
    split_dir.mkdir(parents=True, exist_ok=True)

    loader = make_loader(split, args.data_splits, args.modalities,
                         args.batch_size, args.num_workers)
    n_samples = len(loader.dataset)
    existing = len(list(split_dir.glob("*.pt")))
    if existing == n_samples:
        print(f"[{split}] {n_samples} files already present, skipping.")
        return n_samples

    idx = 0
    for batch in tqdm(loader, desc=f"extract {split}"):
        masked, _label = batch
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            feats = encode_batch(encoder, masked, args.patch_size, args.tile_size, device)
        feats = feats.half().cpu()                       # (B, T, gH, gW, D)
        for b in range(feats.shape[0]):
            torch.save(feats[b].clone(), split_dir / f"{idx}.pt")
            idx += 1
    assert idx == n_samples, f"wrote {idx} != {n_samples} for {split}"
    return n_samples


def main() -> None:
    p = argparse.ArgumentParser(description="Cache OlmoEarth per-timestep features for PASTIS.")
    p.add_argument("--model_size", default="base", choices=list(MODEL_SIZE_TO_ID))
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--tile_size", type=int, default=IMAGE_SIZE,
                   help=f"spatial sub-tile size (<= {IMAGE_SIZE}); {IMAGE_SIZE} = no tiling (exact)")
    p.add_argument("--modalities", default="sentinel2_l2a", # sentinel1
                   help="comma-separated; subset of " + ",".join(ALLOWED_MODALITIES))
    p.add_argument("--data_splits", default="data/pastis_olmoearth")
    p.add_argument("--out_root", default="features")
    p.add_argument("--splits", default=",".join(SPLITS),
                   help="comma-separated subset of train,valid,test")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    args.modalities = [m for m in args.modalities.split(",") if m]
    bad = [m for m in args.modalities if m not in ALLOWED_MODALITIES]
    if bad:
        raise ValueError(f"modalities {bad} not in {list(ALLOWED_MODALITIES)}")
    splits = [s for s in args.splits.split(",") if s]
    if IMAGE_SIZE % args.tile_size != 0:
        raise ValueError(f"tile_size {args.tile_size} must divide {IMAGE_SIZE}")
    if args.tile_size % args.patch_size != 0:
        raise ValueError(f"tile_size {args.tile_size} must be divisible by patch_size {args.patch_size}")

    name = cfg_name(args.model_size, args.modalities, args.patch_size, args.tile_size)
    out_dir = Path(args.out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extraction config: {name}")
    print(f"  patch_size={args.patch_size} tile_size={args.tile_size} input_res={BASE_GSD} "
          f"grid={IMAGE_SIZE // args.patch_size} modalities={args.modalities}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_id(getattr(ModelID, MODEL_SIZE_TO_ID[args.model_size]),
                               load_weights=True)
    encoder = cast(nn.Module, model.encoder if hasattr(model, "encoder") else model)
    encoder = encoder.to(device).eval()
    for prm in encoder.parameters():
        prm.requires_grad = False

    counts = {}
    grid = IMAGE_SIZE // args.patch_size
    embed_dim = None
    for split in splits:
        counts[split] = extract_split(encoder, split, out_dir, args, device)
        if embed_dim is None:
            # peek one file to record D / T for meta
            sample = torch.load(out_dir / f"pastis_r_{split}" / "0.pt")
            T_dim, embed_dim = sample.shape[0], sample.shape[-1]

    meta = {
        "model_size": args.model_size,
        "model_id": MODEL_SIZE_TO_ID[args.model_size],
        "patch_size": args.patch_size,
        "tile_size": args.tile_size,
        "input_res": BASE_GSD,
        "image_size": IMAGE_SIZE,
        "grid": grid,
        "modalities": args.modalities,
        "pooling": str(POOLING_TYPE),
        "timesteps": T_dim,
        "embed_dim": embed_dim,
        "dtype": "float16",
        "counts": counts,
        "feature_shape": [T_dim, grid, grid, embed_dim],
        "exact": args.tile_size == IMAGE_SIZE,
        "size_bytes": _dir_size_bytes(out_dir),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_dir}/meta.json: {meta}")
    print(f"Total size of {out_dir}: {_human(meta['size_bytes'])}")


if __name__ == "__main__":
    main()
