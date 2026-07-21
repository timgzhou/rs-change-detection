"""UrbanSARFloods -> OlmoEarth dataset.

Mirrors olmoearth_pretrain...PASTISRDataset's output contract so it drops into the same
pipeline (extract_olmoearth_features.py / finetune_olmoearth_pastis.py): __getitem__ returns
(MaskedOlmoEarthSample, label) and the batch collates with eval_collate_fn.

DATA (verified from the tiles + data_norm.txt):
  Each 512x512 GeoTIFF has 8 float32 bands:
    band 0-3 : interferometric COHERENCE (VV/VH x 2 date-pairs), range ~[0,1]
    band 4-7 : intensity in dB -> [date1_VH, date1_VV, date2_VH, date2_VV]
               (VV is ~5 dB brighter than VH; verified per-tile: b4<b5, b6<b7)
  GT: single band, pixel values 0=non-flood, 1=flooded-open, 2=flooded-urban.

MAPPING TO OLMOEARTH (decisions locked with the user):
  OlmoEarth S1 modality = VV,VH intensity (dB) as a TIME SERIES; it has NO coherence input.
  -> we feed intensity only (bands 4-7) as T=2 timesteps of [VV, VH]:
        t0 = [b5 (VV), b4 (VH)],  t1 = [b7 (VV), b6 (VH)]
     Coherence bands 0-3 are DROPPED (OlmoEarth can't consume them). This is in-distribution
     (dB intensity is exactly what OlmoEarth pretrained on); the tradeoff is that coherence --
     the signal that makes URBAN flood detectable -- is gone, so class 2 (FU) will be weak.
     That's acceptable: we evaluate NF vs FO and set FU aside for the pipeline check.
  Timestamps: the filename carries the EVENT date (YYYYMMDD...); we assign it to every
     timestep (OlmoEarth mainly uses month for seasonal encoding). Good enough for the check.
  Normalization: OlmoEarth's own COMPUTED stats via Normalizer (same as PASTISRDataset), so
     features live in the distribution the backbone expects. We do NOT use data_norm.txt
     (those are the dataset's own stats; OlmoEarth wants its pretraining stats).

Label: returned as int64 (H,W) with values {0,1,2}; caller decides ignore_index. For the
NF-vs-FO check, treat 2 as ignore (ignore_index=2) or merge to background per your metric.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample

# Intensity band indices (0-based) in the 8-band tile, as [VV, VH] per date.
# Verified: b4=VH,b5=VV (date1); b6=VH,b7=VV (date2). OlmoEarth wants [vv, vh] order.
_INTENSITY_VVVH_BY_DATE = [(5, 4), (7, 6)]   # -> T=2, each (vv_idx, vh_idx)
_DATE_RE = re.compile(r"(\d{8})")            # leading YYYYMMDD in the filename


def date_to_timestamps(date: str, T: int) -> torch.Tensor:
    """(T,3) [day, month(0-idx), year] from a YYYYMMDD string; same for all timesteps.
    OlmoEarth mainly uses month for seasonal encoding, so the event date is a fine proxy."""
    if date and len(date) == 8 and date.isdigit() and date != "00000000":
        year, month, day = int(date[:4]), int(date[4:6]), int(date[6:8])
    else:
        year, month, day = 2020, 6, 1
    ts = torch.tensor([day, month - 1, year], dtype=torch.long)
    return ts.unsqueeze(0).repeat(T, 1)


def sar8_to_olmoearth_sample(sar8, date: str, normalizer=None):
    """Map an 8-band UrbanSARFloods SAR crop -> a MaskedOlmoEarthSample (S1 intensity, T=2).

    sar8: (8, H, W) tensor/array (raw bands: 0-3 coherence, 4-7 intensity dB). We take the
    intensity bands as two timesteps of [VV, VH], optionally normalize with OlmoEarth's
    COMPUTED stats, and build the sample. Shared by UrbanSARFloodsDataset and the offline
    feature extractor so both feed the encoder identical inputs."""
    sar8 = np.asarray(sar8, dtype=np.float32)
    H, W = sar8.shape[1], sar8.shape[2]
    T = len(_INTENSITY_VVVH_BY_DATE)
    s1 = np.empty((H, W, T, 2), dtype=np.float32)          # (H,W,T,[vv,vh])
    for t, (vv_idx, vh_idx) in enumerate(_INTENSITY_VVVH_BY_DATE):
        s1[:, :, t, 0] = sar8[vv_idx]                       # vv
        s1[:, :, t, 1] = sar8[vh_idx]                       # vh
    if normalizer is not None:
        s1 = normalizer.normalize(Modality.SENTINEL1, s1)
    sample = OlmoEarthSample(
        timestamps=date_to_timestamps(date, T),
        sentinel1=torch.from_numpy(s1).float(),
    )
    return MaskedOlmoEarthSample.from_olmoearthsample(sample)


class UrbanSARFloodsDataset(torch.utils.data.Dataset):
    """OlmoEarth-format UrbanSARFloods (S1 intensity, T=2). See module docstring."""

    def __init__(self, root: str | Path, split: str = "train",
                 norm_stats_from_pretrained: bool = True):
        self.root = Path(root)
        split_file = {"train": "Train_dataset.txt", "valid": "Valid_dataset.txt"}[split]
        # split files list GT paths relative to root, e.g. "../03_FU/GT/<name>_GT.tif"
        self.gt_paths = []
        with open(self.root / split_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.gt_paths.append(self._resolve(line))
        self.split = split
        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        if norm_stats_from_pretrained:
            from olmoearth_pretrain.data.normalize import Normalizer, Strategy
            self.normalizer = Normalizer(Strategy.COMPUTED)

    def _resolve(self, rel: str) -> Path:
        # entries look like "../03_FU/GT/NAME_GT.tif"; drop the leading ".." and re-root.
        p = rel.lstrip("./")
        while p.startswith("../"):
            p = p[3:]
        return self.root / p

    @staticmethod
    def _sar_path(gt_path: Path) -> Path:
        # ".../03_FU/GT/NAME_GT.tif" -> ".../03_FU/SAR/NAME_SAR.tif"
        return Path(str(gt_path).replace("/GT/", "/SAR/").replace("_GT.tif", "_SAR.tif"))

    def __len__(self) -> int:
        return len(self.gt_paths)

    def __getitem__(self, idx: int):
        gt_path = self.gt_paths[idx]
        sar_path = self._sar_path(gt_path)

        with rasterio.open(sar_path) as s:
            sar = s.read().astype(np.float32)                 # (8, H, W)
        with rasterio.open(gt_path) as g:
            label = g.read(1).astype(np.int64)                # (H, W), {0,1,2}

        m = _DATE_RE.search(sar_path.name)
        date = m.group(1) if m else ""
        norm = self.normalizer if self.norm_stats_from_pretrained else None
        masked = sar8_to_olmoearth_sample(sar, date, norm)    # S1 intensity, T=2
        return masked, torch.from_numpy(label)
