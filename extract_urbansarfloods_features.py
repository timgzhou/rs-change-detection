"""Extract & cache OlmoEarth features for the tiled UrbanSARFloods dataset.

Reads the 64x64 sub-tiles from prep_urbansarfloods_tiles.py, runs the FROZEN OlmoEarth
encoder (patch_size=4, input_res=20 -- the data is 20 m), and caches PER-TIMESTEP features
so downstream heads can use pre/post separately (e.g. concat pre+post for change detection).

Per tile we save:
    features/<cfg>/<split>/<idx>.pt = {
        "feat":  (T=2, gH=16, gW=16, D) float16,   # per-timestep pooled tokens (t0=pre, t1=post)
        "label": (64, 64) int8,                    # {0,1,2}
    }
T is kept (NOT mean-pooled): t0/t1 are the two S1 intensity dates. gH=gW=64/patch_size.

Reuses:
  - sar8_to_olmoearth_sample (urbansarfloods_dataset): identical band mapping/normalization
    as the on-the-fly dataset, so cached features match a live forward pass.
  - pool_per_timestep (finetune_olmoearth_pastis): the same per-timestep spatial pooling used
    everywhere else.

Runs in the OlmoEarth venv, on GPU (salloc):
    source env_olmo.sh
    python -u extract_urbansarfloods_features.py --splits train,valid
"""
import os
import sys

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

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn
from olmoearth_pretrain.evals.finetune.model import to_device
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.data.normalize import Normalizer, Strategy

from config import MODEL_SIZE_TO_ID
from finetune_olmoearth_pastis import pool_per_timestep
from urbansarfloods_dataset import sar8_to_olmoearth_sample

POOLING_TYPE = PoolingType.MEAN
INPUT_RES = 20                    # UrbanSARFloods is 20 m (SLC-derived); NOT the 10 m default
DATE_FALLBACK = ""                # date comes from each tile's stored "date"


class TiledSARDataset(torch.utils.data.Dataset):
    """Yields (MaskedOlmoEarthSample, label, idx) from the prepped 64x64 .pt tiles."""

    def __init__(self, tiles_dir: Path, split: str, normalizer):
        self.dir = tiles_dir / split
        self.n = len(list(self.dir.glob("*.pt")))
        self.normalizer = normalizer

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rec = torch.load(self.dir / f"{i}.pt")
        sar8 = rec["sar"].float().numpy()                 # (8,64,64)
        masked = sar8_to_olmoearth_sample(sar8, rec.get("date", ""), self.normalizer)
        label = rec["label"].long()                       # (64,64)
        return masked, label, i


def _collate(batch):
    samples = [(m, l) for m, l, _ in batch]
    masked, label = eval_collate_fn(samples)
    idxs = [i for _, _, i in batch]
    return masked, label, idxs


@torch.no_grad()
def extract_split(encoder, split, tiles_dir, out_dir, args, device, normalizer) -> int:
    split_out = out_dir / split
    split_out.mkdir(parents=True, exist_ok=True)
    ds = TiledSARDataset(tiles_dir, split, normalizer)
    if ds.n == 0:
        print(f"[{split}] no tiles found in {tiles_dir / split}; run prep first.")
        return 0
    existing = len(list(split_out.glob("*.pt")))
    if existing == ds.n:
        print(f"[{split}] {ds.n} feature files already present, skipping.")
        return ds.n

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, collate_fn=_collate)
    n_sanitized = 0
    # The encoder uses bf16 autocast on GPU (it casts activations to bf16 while weights stay
    # fp32; without autocast the conv raises a dtype mismatch). On CPU we must NOT use bf16:
    # FlexiViT's patch-resize uses bicubic+antialias interpolation, which has no bf16 CPU
    # kernel ("compute_index_ranges_weights not implemented for BFloat16"). So bf16 only on cuda.
    use_amp = device.type == "cuda"
    for masked, label, idxs in tqdm(loader, desc=f"extract {split}"):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                              enabled=use_amp):
            tam = encoder(to_device(masked, device), patch_size=args.patch_size,
                          input_res=INPUT_RES, fast_pass=True)["tokens_and_masks"]
            T = next(getattr(tam, m).shape[3] for m in tam.modalities)
            feats = torch.stack([pool_per_timestep(tam, t, POOLING_TYPE) for t in range(T)],
                                dim=1)                    # (B, T, gH, gW, D)
        feats = feats.float()
        # Safety net: tiles are NaN-free after prep drops nodata tiles, so this should never
        # fire. If it does, a non-finite tile slipped through -- warn loudly rather than
        # silently caching NaN.
        if not torch.isfinite(feats).all():
            n_sanitized += int((~torch.isfinite(feats)).any(dim=(1, 2, 3, 4)).sum().item())
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        feats = feats.half().cpu()
        label = label.to(torch.int8).cpu()
        for b, idx in enumerate(idxs):
            torch.save({"feat": feats[b].clone(), "label": label[b].clone()},
                       split_out / f"{idx}.pt")
    if n_sanitized:
        print(f"[{split}] sanitized {n_sanitized} tiles that had non-finite encoder output.")
    return ds.n


def main() -> None:
    p = argparse.ArgumentParser(description="Cache OlmoEarth features for tiled UrbanSARFloods.")
    p.add_argument("--model_size", default="base", choices=list(MODEL_SIZE_TO_ID))
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--tile_size", type=int, default=64,
                   help="sub-tile size used at prep time; reads tiles from <tiles_root>_t<tile_size>")
    p.add_argument("--tiles_root", default="data/urbansarfloods_tiles")
    p.add_argument("--out_root", default="features")
    p.add_argument("--splits", default="train,valid")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    if args.tile_size % args.patch_size != 0:
        raise ValueError(f"tile_size {args.tile_size} must be divisible by patch_size {args.patch_size}")
    grid = args.tile_size // args.patch_size
    # Feature folder + tiles dir both encode tile_size so ps/tile variants never collide.
    name = f"usf_{args.model_size}_s1_ps{args.patch_size}_res{INPUT_RES}_t{args.tile_size}"
    out_dir = Path(args.out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = Path(f"{args.tiles_root}_t{args.tile_size}")
    print(f"Extraction config: {name} (patch_size={args.patch_size}, tile_size={args.tile_size}, "
          f"input_res={INPUT_RES}, grid={grid}) reading {tiles_dir}/")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Local weights avoid the HF download/rate-limit; falls back to ModelID if absent.
    local = Path("/scratch/timz/OlmoEarth-v1-Base")
    model_ref = str(local) if (args.model_size == "base" and local.exists()) \
        else getattr(ModelID, MODEL_SIZE_TO_ID[args.model_size])
    model = load_model_from_id(model_ref, load_weights=True)
    encoder = cast(nn.Module, model.encoder if hasattr(model, "encoder") else model)
    encoder = encoder.to(device).eval()
    for prm in encoder.parameters():
        prm.requires_grad = False
    normalizer = Normalizer(Strategy.COMPUTED)

    counts, T_dim, embed_dim = {}, None, None
    for split in [s for s in args.splits.split(",") if s]:
        counts[split] = extract_split(encoder, split, tiles_dir, out_dir, args, device, normalizer)
        if embed_dim is None and counts[split] > 0:
            sample = torch.load(out_dir / split / "0.pt")["feat"]
            T_dim, embed_dim = sample.shape[0], sample.shape[-1]

    meta = {
        "dataset": "urbansarfloods", "model_size": args.model_size,
        "patch_size": args.patch_size, "input_res": INPUT_RES, "modalities": ["sentinel1"],
        "pooling": str(POOLING_TYPE), "timesteps": T_dim, "embed_dim": embed_dim,
        "grid": grid, "tile_size": args.tile_size, "dtype": "float16", "counts": counts,
        "feature_shape": [T_dim, grid, grid, embed_dim], "num_classes": 3,
        "label_size": args.tile_size,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_dir}/meta.json: {meta}")


if __name__ == "__main__":
    main()
