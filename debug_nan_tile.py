"""Find the first NaN-producing tile and dissect it, to explain why ~1000 tiles NaN.

Walks the prepped 64x64 tiles, runs each through the SAME path as extraction, and stops at
the first non-finite encoder output. Then reports, for that tile:
  - raw 8-band SAR stats (per band: min/max/mean, constant?, non-finite?)
  - normalized S1 stats fed to the encoder
  - WHERE the NaN appears: encoder tokens (bf16 overflow) vs only after pooling (pooling bug,
    e.g. divide-by-zero unmasked-token count for S1-only input)
  - the token MASK values (to see if all tokens got masked out)
Run in salloc (GPU):
    source env_olmo.sh
    python -u debug_nan_tile.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "olmo_shims"))
import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()

import glob

import numpy as np
import torch

from urbansarfloods_dataset import sar8_to_olmoearth_sample
from olmoearth_pretrain.model_loader import load_model_from_id
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn
from olmoearth_pretrain.evals.finetune.model import to_device
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from finetune_olmoearth_pastis import pool_per_timestep

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, flush=True)
enc = load_model_from_id("/scratch/timz/OlmoEarth-v1-Base", load_weights=True).encoder.to(dev).eval()
norm = Normalizer(Strategy.COMPUTED)

fs = sorted(glob.glob("data/urbansarfloods_tiles/train/*.pt"))
print(f"walking {len(fs)} tiles until first NaN...", flush=True)

for f in fs:
    rec = torch.load(f)
    sar = rec["sar"].float()                              # (8,64,64)
    m = sar8_to_olmoearth_sample(sar.numpy(), rec.get("date", ""), norm)
    batch, _ = eval_collate_fn([(m, rec["label"].long())])
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        tam = enc(to_device(batch, dev), patch_size=4, input_res=20,
                  fast_pass=True)["tokens_and_masks"]
        mod = tam.modalities[0]
        raw_tokens = getattr(tam, mod).float()            # (B,gH,gW,T,BandSets,D)
        mask = getattr(tam, f"{mod}_mask")
        T = raw_tokens.shape[3]
        pooled = torch.stack([pool_per_timestep(tam, t, PoolingType.MEAN) for t in range(T)], 1).float()

    tokens_nan = not torch.isfinite(raw_tokens).all()
    pooled_nan = not torch.isfinite(pooled).all()
    if not pooled_nan:
        continue

    # ---- found it: dissect ----
    print("\n=== FIRST NaN TILE ===", flush=True)
    print("file:", f, "src:", rec.get("src"), "pos:", rec.get("pos"), "date:", rec.get("date"))
    print("\n-- raw 8-band SAR (0-3 coherence, 4-7 intensity dB) --")
    s = sar.numpy()
    for b in range(8):
        x = s[b]
        print(f"  band{b}: min={x.min():8.3f} max={x.max():8.3f} mean={x.mean():8.3f} "
              f"std={x.std():7.3f} constant={np.allclose(x, x.flat[0])} finite={np.isfinite(x).all()}")

    print("\n-- normalized S1 fed to encoder (H,W,T,[vv,vh]) --")
    s1 = m.sentinel1.float()
    print(f"  shape={tuple(s1.shape)} min={s1.min():.3f} max={s1.max():.3f} "
          f"finite={torch.isfinite(s1).all().item()}")
    for t in range(s1.shape[2]):
        for c, name in enumerate(("vv", "vh")):
            xx = s1[:, :, t, c]
            print(f"   t{t} {name}: min={xx.min():.3f} max={xx.max():.3f} "
                  f"constant={torch.allclose(xx, xx.flatten()[0])}")

    print("\n-- WHERE is the NaN --")
    print(f"  encoder raw tokens non-finite: {tokens_nan}")
    print(f"  pooled non-finite:             {pooled_nan}")
    if pooled_nan and not tokens_nan:
        print("  => tokens are FINITE but pooling produced NaN. Likely divide-by-zero over")
        print("     unmasked tokens (all tokens masked for this tile). Check mask below.")
    else:
        print("  => NaN already in encoder tokens (bf16 internal overflow, or bad token).")

    print("\n-- token mask --")
    mu, mc = torch.unique(mask.cpu(), return_counts=True)
    print("  mask unique:counts:", dict(zip(mu.tolist(), mc.tolist())))
    print("  mask shape:", tuple(mask.shape))
    # how many unmasked (ONLINE_ENCODER) tokens per (b,h,w) after slicing a timestep?
    from olmoearth_pretrain.datatypes import MaskValue
    online = (mask == MaskValue.ONLINE_ENCODER.value)
    print("  ONLINE_ENCODER token count:", int(online.sum().item()),
          "of", int(mask.numel()))
    break
else:
    print("no NaN tile found in the walk.", flush=True)
