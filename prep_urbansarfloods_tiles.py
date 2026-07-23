"""Tile UrbanSARFloods 512x512 samples into 64x64 sub-tiles for fast training.

Each source 512x512 tile is cut into an 8x8 grid of 64x64 sub-tiles. We keep the OlmoEarth
band mapping (S1 intensity VV/VH as a T=2 time series; coherence dropped) identical to
urbansarfloods_dataset.py, so downstream feature extraction sees the same inputs.

Output (per the user's choice): one .pt per KEPT sub-tile holding the 8-band SAR crop as
fp16 plus its int8 label crop and provenance:
    tiles/urbansarfloods/<split>/<idx>.pt = {
        "sar":   (8, 64, 64) float16,   # raw 8 bands (coherence 0-3 + intensity 4-7)
        "label": (64, 64) int8,         # {0,1,2}
        "src":   "<source_tif_name>",
        "pos":   (row, col),            # sub-tile position in the 8x8 grid
        "date":  "YYYYMMDD",
    }
We store the RAW 8 bands (not the OlmoEarth-mapped VV/VH) so the tiles stay a faithful crop
of the source; the OlmoEarth band selection/normalization happens at feature-extraction time
(reusing urbansarfloods_dataset's mapping). This keeps coherence available for later
experiments without re-tiling.

FILTERING (user choice): keep only sub-tiles that contain at least one flood pixel
(label == 1 or 2). Controlled by --flood_only (default True). NOTE: filtering val/test means
metrics are computed only over flood-containing regions -- not a standard val number (no
background false-positive coverage). Pass --flood_only false --split valid to regenerate a
complete val set later.

Runs in the OlmoEarth venv (needs rasterio). CPU-only, no model:
    source env_olmo.sh
    python -u prep_urbansarfloods_tiles.py --splits train,valid
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm

IMAGE_SIZE = 512
_DATE_RE = re.compile(r"(\d{8})")


def _sar_path(gt_path: Path) -> Path:
    return Path(str(gt_path).replace("/GT/", "/SAR/").replace("_GT.tif", "_SAR.tif"))


def _resolve(root: Path, rel: str) -> Path:
    p = rel.strip().lstrip("./")
    while p.startswith("../"):
        p = p[3:]
    return root / p


def _read_gt_paths(root: Path, split: str) -> list[Path]:
    fn = {"train": "Train_dataset.txt", "valid": "Valid_dataset.txt"}[split]
    with open(root / fn) as f:
        return [_resolve(root, line) for line in f if line.strip()]


def prep_split(root: Path, split: str, out_root: Path, tile_size: int,
               flood_only: bool, limit: int, overwrite: bool = False) -> None:
    if IMAGE_SIZE % tile_size != 0:
        raise ValueError(f"tile_size {tile_size} must divide {IMAGE_SIZE}")
    n_side = IMAGE_SIZE // tile_size
    out_dir = out_root / split
    # Skip if already tiled: prep drops tiles (no-flood / NaN), so the count isn't predictable
    # up front -- we treat "folder has >=1 .pt" as done. Re-tiling is idempotent but wasteful
    # (re-reads every source GeoTIFF). Pass --overwrite to force.
    existing = len(list(out_dir.glob("*.pt"))) if out_dir.exists() else 0
    if existing and not overwrite:
        print(f"[{split}] {existing} tiles already present in {out_dir}; skipping "
              f"(use --overwrite to re-tile).")
        return
    gt_paths = _read_gt_paths(root, split)
    if limit > 0:
        gt_paths = gt_paths[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    kept = 0
    seen = 0
    skipped_nan = 0
    skipped_nofloor = 0
    for gt_path in tqdm(gt_paths, desc=f"tile {split}"):
        sar_path = _sar_path(gt_path)
        with rasterio.open(sar_path) as s:
            sar = s.read().astype(np.float32)          # (8, 512, 512)
        with rasterio.open(gt_path) as g:
            label = g.read(1).astype(np.int64)         # (512, 512)
        m = _DATE_RE.search(sar_path.name)
        date = m.group(1) if m else "00000000"

        for r in range(n_side):
            for c in range(n_side):
                seen += 1
                y0, x0 = r * tile_size, c * tile_size
                lab = label[y0:y0 + tile_size, x0:x0 + tile_size]
                if flood_only and not np.any((lab == 1) | (lab == 2)):
                    skipped_nofloor += 1
                    continue
                sar_crop = sar[:, y0:y0 + tile_size, x0:x0 + tile_size]
                # DROP any sub-tile with a non-finite SAR pixel. Source scenes have NaN
                # nodata along the SAR swath edge (~diagonal); those pixels propagate NaN
                # through the encoder. We drop the whole tile (no imputation) so the cache
                # is clean and no downstream sanitizing is needed.
                if not np.isfinite(sar_crop).all():
                    skipped_nan += 1
                    continue
                torch.save(
                    {
                        "sar": torch.from_numpy(sar_crop).half(),      # (8,tile,tile) fp16
                        "label": torch.from_numpy(lab).to(torch.int8),  # (tile,tile)
                        "src": sar_path.name,
                        "pos": (r, c),
                        "date": date,
                    },
                    out_dir / f"{idx}.pt",
                )
                idx += 1
                kept += 1
    print(f"[{split}] source images: {len(gt_paths)} | sub-tiles seen: {seen} | "
          f"kept: {kept} | skipped (no flood): {skipped_nofloor} | "
          f"skipped (NaN nodata): {skipped_nan} ({'flood-only' if flood_only else 'all'}) -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Tile UrbanSARFloods 512 -> tile_size for fast training.")
    p.add_argument("--root", default="data/urban_sar_floods")
    p.add_argument("--out_root", default="data/urbansarfloods_tiles",
                   help="tiles go under <out_root>_t<tile_size>/<split>/ so tile sizes don't collide")
    p.add_argument("--tile_size", type=int, default=64,
                   help="sub-tile size; must divide 512 (e.g. 64, 32, 128)")
    p.add_argument("--splits", default="train,valid")
    p.add_argument("--flood_only", default="true",
                   help="keep only sub-tiles containing a flood pixel (class 1/2). true/false")
    p.add_argument("--limit", type=int, default=0, help="cap source images per split (debug)")
    p.add_argument("--overwrite", action="store_true",
                   help="re-tile even if the output split folder already has tiles")
    args = p.parse_args()

    flood_only = str(args.flood_only).strip().lower() in ("true", "1", "yes")
    root = Path(args.root)
    # Encode tile size in the output dir so t64 and t32 coexist.
    out_root = Path(f"{args.out_root}_t{args.tile_size}")
    print(f"Tiling 512 -> {args.tile_size} into {out_root}/ (flood_only={flood_only})")
    for split in [s for s in args.splits.split(",") if s]:
        prep_split(root, split, out_root, args.tile_size, flood_only, args.limit, args.overwrite)


if __name__ == "__main__":
    main()
