"""UTAE model builders for uni- and multi-modal PASTIS segmentation.

The official utae-paps repo only ships single-stream UTAE; the multimodal fusion is
from the 2022 ISPRS paper (code not released), so early/late fusion are implemented
here, guided by that paper. All builders expose the same call interface:

    logits = model(data, dates)   # data/dates are dicts keyed by sat (e.g. {"S2": ...})
    # logits: (B, num_classes, H, W)

This matches what the vendored PASTIS_Dataset yields (per-sat dicts), so the training
loop stays modality-agnostic.

PASTIS channel counts: S2=10, S1A/S1D=3. num_classes=20 (class 19 = void, ignored).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .utae import UTAE

SAT_CHANNELS = {"S2": 10, "S1A": 3, "S1D": 3}

# UTAE config used for PASTIS in utae-paps/src/model_utils.py (the 63.1 mIoU setup).
_UTAE_KW = dict(
    encoder_widths=[64, 64, 64, 128],
    decoder_widths=[32, 32, 64, 128],
    str_conv_k=4, str_conv_s=2, str_conv_p=1,
    agg_mode="att_group", encoder_norm="group",
    n_head=16, d_model=256, d_k=4, pad_value=0, padding_mode="reflect",
)


def _resample_to(ref_dates: torch.Tensor, src_dates: torch.Tensor,
                 src: torch.Tensor) -> torch.Tensor:
    """Gather src timesteps nearest (in date) to each ref timestep. Per-sample.

    ref_dates: (T_ref,), src_dates: (T_src,), src: (T_src, C, H, W)
    returns (T_ref, C, H, W). Used for early fusion to put S1 on S2's time grid.
    """
    # nearest index in src for each ref date
    idx = (ref_dates[:, None] - src_dates[None, :]).abs().argmin(dim=1)  # (T_ref,)
    return src[idx]


class UniUTAE(nn.Module):
    """Single-satellite UTAE."""

    def __init__(self, sat: str, num_classes: int):
        super().__init__()
        self.sat = sat
        self.net = UTAE(input_dim=SAT_CHANNELS[sat], out_conv=[32, num_classes], **_UTAE_KW)

    def forward(self, data, dates):
        return self.net(data[self.sat], batch_positions=dates[self.sat])


class EarlyFusionUTAE(nn.Module):
    """Resample the secondary sat(s) onto the primary's date grid, concat channels,
    one UTAE. primary defaults to S2 (richer/denser); ~65.9 mIoU in the paper."""

    def __init__(self, sats: list[str], num_classes: int, primary: str = "S2"):
        super().__init__()
        assert primary in sats
        self.sats = sats
        self.primary = primary
        self.others = [s for s in sats if s != primary]
        in_dim = sum(SAT_CHANNELS[s] for s in sats)
        self.net = UTAE(input_dim=in_dim, out_conv=[32, num_classes], **_UTAE_KW)

    def forward(self, data, dates):
        ref = data[self.primary]                      # (B,T,C,H,W)
        ref_d = dates[self.primary]                   # (B,T)
        streams = [ref]
        for s in self.others:
            src, src_d = data[s], dates[s]
            # resample per-sample onto ref's time grid
            res = torch.stack([_resample_to(ref_d[b], src_d[b], src[b])
                               for b in range(ref.shape[0])])  # (B,T,C_s,H,W)
            streams.append(res)
        x = torch.cat(streams, dim=2)                 # concat on channel axis
        return self.net(x, batch_positions=ref_d)


class LateFusionUTAE(nn.Module):
    """One UTAE feature extractor per sat (return_maps -> last decoder map), concat the
    decoder features, then a 1x1 conv head. Closer to the paper's best (~66.3 mIoU)."""

    def __init__(self, sats: list[str], num_classes: int):
        super().__init__()
        self.sats = sats
        # encoder=False + return_maps=True makes UTAE return (out, maps); maps[-1] is the
        # full-res decoder feature map of width decoder_widths[0] (=32).
        self.nets = nn.ModuleDict({
            s: UTAE(input_dim=SAT_CHANNELS[s], out_conv=[32, num_classes],
                    return_maps=True, **_UTAE_KW)
            for s in sats
        })
        feat = 32 * len(sats)  # decoder_widths[0] per sat
        self.head = nn.Conv2d(feat, num_classes, kernel_size=1)

    def forward(self, data, dates):
        feats = []
        for s in self.sats:
            _, maps = self.nets[s](data[s], batch_positions=dates[s])
            feats.append(maps[-1])                    # (B,32,H,W)
        return self.head(torch.cat(feats, dim=1))


def build_model(modalities: list[str], fusion: str, num_classes: int = 20) -> nn.Module:
    """modalities: subset of ['S2','S1A','S1D']. fusion: 'early' | 'late' (ignored if
    uni-modal)."""
    if len(modalities) == 1:
        return UniUTAE(modalities[0], num_classes)
    if fusion == "early":
        return EarlyFusionUTAE(modalities, num_classes)
    if fusion == "late":
        return LateFusionUTAE(modalities, num_classes)
    raise ValueError(f"unknown fusion {fusion!r}; use 'early' or 'late'")
