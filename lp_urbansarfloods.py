"""Linear-probe flood segmentation on FROZEN OlmoEarth features (UrbanSARFloods).

Change-detection framing: each cached tile has per-timestep features (T=2, gH, gW, D) for
the pre-event (t0) and post-event (t1) S1 intensity. This probe CONCATENATES pre+post along
the feature dim -> (gH, gW, 2D) and applies a per-pixel head, so the linear layer can learn
from the pre->post difference (the flood signal), not just a single date. The OlmoEarth
backbone stays frozen; only the head trains.

Head (pa2px, per-pixel):
    concat pre,post -> (B, 2D, gH, gW) -> 1x1 conv (2D -> C*patch_size^2) -> unfold sub-pixels
    (b (c i j) gh gw -> b c (gh i) (gw j)) -> (B, C, 64, 64). Mirrors lp_pa2px from
    lp_on_cached_features.py but on concatenated pre/post features.

Metrics: 3 classes (0 non-flood, 1 flooded-open, 2 flooded-urban) but we HEADLINE mIoU over
{NF, FO} only -- flooded-urban (class 2) is dropped from the SAR intensity signal (coherence
was not fed), so its IoU is reported separately, not folded into the headline.

Runs in the OlmoEarth venv (for segmentation_metrics):
    source env_olmo.sh
    python -u lp_urbansarfloods.py --features usf_base_s1_ps4_res20
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "olmo_shims"))
import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.metrics import segmentation_metrics, _build_confusion_matrix

SCHEDULER_FACTOR, SCHEDULER_PATIENCE, SCHEDULER_MIN_LR, SCHEDULER_COOLDOWN = 0.2, 2, 1e-6, 0
NUM_CLASSES = 3            # 0 non-flood, 1 flooded-open, 2 flooded-urban
IGNORE_LABEL = -1
LABEL_SIZE = 64
HEADLINE_CLASSES = [0, 1]  # NF, FO -- class 2 (FU) reported separately, not in headline mIoU


class TiledFeatureDataset(torch.utils.data.Dataset):
    """Loads cached {"feat": (T,gH,gW,D) fp16, "label": (64,64) int8} tiles."""

    def __init__(self, feat_dir: Path, split: str):
        self.dir = feat_dir / split
        self.n = len(list(self.dir.glob("*.pt")))
        if self.n == 0:
            raise FileNotFoundError(f"no feature tiles in {self.dir}")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rec = torch.load(self.dir / f"{i}.pt")
        return rec["feat"].float(), rec["label"].long()     # (T,gH,gW,D), (64,64)


class ConcatPrePostPerPixelHead(nn.Module):
    """Concat pre+post features over the feature dim, then a per-pixel (pa2px) conv probe."""

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 label_size: int = LABEL_SIZE):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.label_size = label_size
        # in = 2*D (pre concat post); out = C * patch_size^2 (sub-pixel logits)
        self.probe = nn.Conv2d(2 * embed_dim, num_classes * patch_size * patch_size, kernel_size=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # (B, T=2, gH, gW, D)
        pre, post = feats[:, 0], feats[:, 1]                 # (B,gH,gW,D) each
        x = torch.cat([pre, post], dim=-1)                   # (B,gH,gW,2D)
        x = x.permute(0, 3, 1, 2).contiguous()               # (B,2D,gH,gW)
        logits = self.probe(x)                               # (B, C*p*p, gH, gW)
        p = self.patch_size
        logits = rearrange(logits, "b (c i j) gh gw -> b c (gh i) (gw j)",
                           c=self.num_classes, i=p, j=p)      # (B,C,gH*p,gW*p)
        if logits.shape[-2:] != (self.label_size, self.label_size):
            logits = F.interpolate(logits, size=(self.label_size, self.label_size),
                                   mode="bilinear", align_corners=True)
        return logits


def per_class_iou(preds, labels):
    """Per-class IoU from the confusion matrix. segmentation_metrics only returns aggregate
    miou (it hides per-class IoU), so we recompute here to headline NF/FO and split out FU."""
    conf = _build_confusion_matrix(preds, labels, NUM_CLASSES, IGNORE_LABEL)
    tp = conf.diagonal().float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    union = tp + fp + fn
    iou = tp / (union + 1e-8)
    return iou, union                                    # union==0 -> class absent


@torch.no_grad()
def evaluate(head, loader, device):
    head.eval()
    preds, labels = [], []
    for feats, label in loader:
        logits = head(feats.to(device))
        preds.append(logits.argmax(1).cpu())
        labels.append(label)
    preds, labels = torch.cat(preds), torch.cat(labels)
    res = segmentation_metrics(preds, labels, num_classes=NUM_CLASSES, ignore_label=IGNORE_LABEL)
    iou, union = per_class_iou(preds, labels)
    # Headline = mean IoU over NF/FO classes that are actually present.
    present = [c for c in HEADLINE_CLASSES if union[c] > 0]
    headline = float(sum(iou[c] for c in present) / len(present)) if present else float("nan")
    fu_iou = float(iou[2]) if union[2] > 0 else None
    return res, headline, fu_iou


def main() -> None:
    p = argparse.ArgumentParser(description="LP flood seg on frozen OlmoEarth features (concat pre/post).")
    p.add_argument("--features", required=True, help="folder under --out_root, e.g. usf_base_s1_ps4_res20")
    p.add_argument("--out_root", default="features")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = Path(args.out_root) / args.features
    meta = json.loads((feat_dir / "meta.json").read_text())
    embed_dim, patch_size = meta["embed_dim"], meta["patch_size"]
    assert meta["timesteps"] == 2, f"expected T=2 (pre/post), got {meta['timesteps']}"
    print(f"Features: {args.features} | shape {meta['feature_shape']} | concat pre+post -> 2D={2*embed_dim}")

    def loader(split, shuffle):
        return DataLoader(TiledFeatureDataset(feat_dir, split), batch_size=args.batch_size,
                          num_workers=args.num_workers, shuffle=shuffle,
                          pin_memory=device.type == "cuda")

    train_loader = loader("train", True)
    val_loader = loader("valid", False)

    head = ConcatPrePostPerPixelHead(embed_dim, NUM_CLASSES, patch_size).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    sched = ReduceLROnPlateau(opt, mode="max", factor=SCHEDULER_FACTOR,
                              patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR,
                              cooldown=SCHEDULER_COOLDOWN)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    # val IS the test set here (no separate val for a hyperparameter-free LP). To avoid
    # selecting on test, we do NOT keep the best epoch: we train a fixed number of epochs,
    # step the scheduler on TRAIN loss, and report the FINAL epoch's val=test mIoU. The
    # per-epoch val prints are just progress; the headline is the last line.
    final = None
    checked_finite = False
    for epoch in range(args.epochs):
        head.train()
        losses = []
        for feats, label in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
            feats, label = feats.to(device), label.to(device)
            if not checked_finite:      # features are NaN-free (prep drops nodata tiles); assert once
                assert torch.isfinite(feats).all(), "non-finite features -- re-run prep/extract"
                checked_finite = True
            logits = head(feats)
            loss = loss_fn(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = sum(losses) / max(len(losses), 1)
        sched.step(train_loss)                            # schedule on TRAIN loss, not test
        res, headline, fu = evaluate(head, val_loader, device)
        final = (headline, fu, res.metrics)
        print(f"epoch {epoch+1}/{args.epochs} | train_loss {train_loss:.4f} | "
              f"test mIoU(NF,FO) {headline:.4f} | FU IoU {fu if fu is None else round(fu,4)} | {res.metrics}")

    headline, fu, metrics = final
    print(f"\nFINAL (val-as-test) mIoU(NF,FO) {headline:.4f} | FU IoU "
          f"{fu if fu is None else round(fu,4)} | {metrics}")


if __name__ == "__main__":
    main()
