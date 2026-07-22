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
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.metrics import segmentation_metrics, _build_confusion_matrix

SCHEDULER_MIN_LR = 1e-6         # cosine annealing floor (eta_min)
NUM_CLASSES = 3            # 0 non-flood, 1 flooded-open, 2 flooded-urban
IGNORE_LABEL = -1
LABEL_SIZE = 64            # default; overridden per-run from meta.json (== tile_size)
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


class _SiamesePerPixelHead(nn.Module):
    """Base Siamese per-pixel (pa2px) head. Subclasses define how pre/post features are
    combined (_combine) and the resulting channel dim (in_dim). Shared: 1x1 conv to
    C*patch_size^2 sub-pixel logits, unfold to full res, interpolate to label size."""

    in_mult = 1                                              # in_dim = in_mult * embed_dim

    def __init__(self, embed_dim: int, num_classes: int, patch_size: int,
                 label_size: int = LABEL_SIZE):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.label_size = label_size
        self.probe = nn.Conv2d(self.in_mult * embed_dim,
                               num_classes * patch_size * patch_size, kernel_size=1)

    def _combine(self, pre, post):                           # (B,gH,gW,D) each -> (B,gH,gW,in)
        raise NotImplementedError

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # (B, T=2, gH, gW, D)
        pre, post = feats[:, 0], feats[:, 1]
        x = self._combine(pre, post)                         # (B,gH,gW,in)
        x = x.permute(0, 3, 1, 2).contiguous()               # (B,in,gH,gW)
        logits = self.probe(x)                               # (B, C*p*p, gH, gW)
        p = self.patch_size
        logits = rearrange(logits, "b (c i j) gh gw -> b c (gh i) (gw j)",
                           c=self.num_classes, i=p, j=p)      # (B,C,gH*p,gW*p)
        if logits.shape[-2:] != (self.label_size, self.label_size):
            logits = F.interpolate(logits, size=(self.label_size, self.label_size),
                                   mode="bilinear", align_corners=True)
        return logits


class ConcatPrePostPerPixelHead(_SiamesePerPixelHead):
    """concat: [pre, post] -> 2D. Learns from both dates jointly."""
    in_mult = 2

    def _combine(self, pre, post):
        return torch.cat([pre, post], dim=-1)                # (B,gH,gW,2D)


class DiffPrePostPerPixelHead(_SiamesePerPixelHead):
    """diff: post - pre -> D. Signed per-channel feature change (change-detection framing);
    keeps full magnitude+direction of change for the linear probe."""
    in_mult = 1

    def _combine(self, pre, post):
        return post - pre                                    # (B,gH,gW,D)


HEADS = {"concat": ConcatPrePostPerPixelHead, "diff": DiffPrePostPerPixelHead}


def build_head(name: str, embed_dim: int, num_classes: int, patch_size: int, label_size: int):
    if name not in HEADS:
        raise ValueError(f"head={name!r} not in {list(HEADS)}")
    return HEADS[name](embed_dim, num_classes, patch_size, label_size)


CLASS_NAMES = ["NF", "FO", "FU"]
# Class colormap for viz: NF=white background, FO=blue, FU=red.
CLASS_COLORS = ["#ffffff", "#1f77ff", "#e41a1c"]


@torch.no_grad()
def visualize_first_val(head, feat_dir: Path, tiles_root: Path, device, out_path: str):
    """Prediction vs GT on the first val tile, alongside pre/post SAR. NF=white, FO=blue,
    FU=red. Pre/post come from the RAW tiled SAR (data/urbansarfloods_tiles/valid/0.pt),
    which shares the same index as the cached feature tile."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    feat = torch.load(feat_dir / "valid" / "0.pt")
    logits = head(feat["feat"].float().unsqueeze(0).to(device))
    pred = logits.argmax(1)[0].cpu().numpy()
    gt = feat["label"].long().numpy()

    raw = torch.load(tiles_root / "valid" / "0.pt")["sar"].float().numpy()   # (8,64,64)
    # intensity bands: [b4=date1 VH, b5=date1 VV, b6=date2 VH, b7=date2 VV] -> pre/post VV,VH
    def fc(vv, vh):
        d = lambda x: np.clip((x - np.percentile(x, 2)) /
                              (np.percentile(x, 98) - np.percentile(x, 2) + 1e-6), 0, 1)
        return np.dstack([d(vv), d(vh), d(vv)])
    pre = fc(raw[5], raw[4])
    post = fc(raw[7], raw[6])

    cmap = ListedColormap(CLASS_COLORS)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
    ax[0].imshow(pre); ax[0].set_title("Pre-event (VV/VH)"); ax[0].axis("off")
    ax[1].imshow(post); ax[1].set_title("Post-event (VV/VH)"); ax[1].axis("off")
    ax[2].imshow(gt, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax[2].set_title("Ground truth"); ax[2].axis("off")
    ax[3].imshow(pred, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax[3].set_title("Prediction"); ax[3].axis("off")
    legend = [Patch(color=CLASS_COLORS[i], label=f"{i}: {CLASS_NAMES[i]}") for i in range(3)]
    ax[3].legend(handles=legend, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=115)
    plt.close(fig)
    print(f"Saved prediction visualization to {out_path}")


def per_class_stats(preds, labels):
    """Per-class IoU / precision / recall / F1 from the confusion matrix. segmentation_metrics
    only returns aggregate miou (it hides per-class values), so we recompute here."""
    conf = _build_confusion_matrix(preds, labels, NUM_CLASSES, IGNORE_LABEL)
    tp = conf.diagonal().float()
    fp = conf.sum(0).float() - tp                        # predicted c, true != c
    fn = conf.sum(1).float() - tp                        # true c, predicted != c
    union = tp + fp + fn
    iou = tp / (union + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1,
            "union": union, "support": tp + fn}          # support==0 -> class absent


def _fmt_per_class(stats) -> str:
    lines = []
    for c in range(NUM_CLASSES):
        if stats["support"][c] == 0:
            lines.append(f"    {CLASS_NAMES[c]}: (absent)")
            continue
        lines.append(f"    {CLASS_NAMES[c]}: IoU={stats['iou'][c]:.4f} "
                     f"P={stats['precision'][c]:.4f} R={stats['recall'][c]:.4f} "
                     f"F1={stats['f1'][c]:.4f}")
    return "\n".join(lines)


def append_results_csv(path: str, row: dict) -> None:
    """Append one result row to a CSV (write header if the file is new). One row per LP run,
    with per-class IoU/P/R/F1 flattened into columns so a whole sweep lands in one table."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def build_result_row(args, meta, headline, fu_iou, res_metrics, stats) -> dict:
    """Flatten a run's config + metrics into a flat dict for CSV."""
    row = {
        "features": args.features, "head": args.head, "weighted_ce": args.weighted_ce,
        "tile_size": meta.get("tile_size"), "patch_size": meta.get("patch_size"),
        "input_res": meta.get("input_res"), "epochs": args.epochs, "lr": args.lr,
        "seed": args.seed,
        "miou_NF_FO": round(headline, 4),
        "iou_FU": (round(fu_iou, 4) if fu_iou is not None else ""),
        "overall_acc": round(res_metrics.get("overall_acc", float("nan")), 4),
        "macro_f1": round(res_metrics.get("macro_f1", float("nan")), 4),
    }
    for c, name in enumerate(CLASS_NAMES):
        present = stats["support"][c] > 0
        for k in ("iou", "precision", "recall", "f1"):
            row[f"{name}_{k}"] = (round(float(stats[k][c]), 4) if present else "")
    return row


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
    stats = per_class_stats(preds, labels)
    # Headline = mean IoU over NF/FO classes that are actually present.
    present = [c for c in HEADLINE_CLASSES if stats["union"][c] > 0]
    headline = float(sum(stats["iou"][c] for c in present) / len(present)) if present else float("nan")
    fu_iou = float(stats["iou"][2]) if stats["union"][2] > 0 else None
    return res, headline, fu_iou, stats


def main() -> None:
    p = argparse.ArgumentParser(description="LP flood seg on frozen OlmoEarth features (concat pre/post).")
    p.add_argument("--features", required=True, help="folder under --out_root, e.g. usf_base_s1_ps4_res20")
    p.add_argument("--out_root", default="features")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--head", default="concat", choices=list(HEADS),
                   help="Siamese combine op: 'concat' ([pre,post]->2D) or 'diff' (post-pre->D).")
    p.add_argument("--weighted_ce", action="store_true",
                   help="use inverse-frequency class weights in CrossEntropy (default: plain CE). "
                        "Helps the rare flood classes against the NF-dominated pixel distribution.")
    p.add_argument("--tiles_root", default="data/urbansarfloods_tiles",
                   help="raw tiled SAR (for the post-training prediction visualization).")
    p.add_argument("--viz_out", default=None,
                   help="prediction PNG path; default auto-names per config: viz/<features>_<head>.png")
    p.add_argument("--results_csv", default="results/urbansarfloods_lp.csv",
                   help="append one result row (config + per-class P/R/F1) to this CSV.")
    args = p.parse_args()

    # Auto-name the viz per config so a sweep doesn't overwrite one file.
    if args.viz_out is None:
        wc = "_wce" if args.weighted_ce else ""
        args.viz_out = f"viz/{args.features}_{args.head}{wc}.png"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = Path(args.out_root) / args.features
    meta = json.loads((feat_dir / "meta.json").read_text())
    embed_dim, patch_size = meta["embed_dim"], meta["patch_size"]
    label_size = meta.get("label_size", LABEL_SIZE)     # == tile_size; head upsamples to this
    tile_size = meta.get("tile_size", label_size)
    assert meta["timesteps"] == 2, f"expected T=2 (pre/post), got {meta['timesteps']}"
    print(f"Features: {args.features} | shape {meta['feature_shape']} | label_size {label_size} | "
          f"head={args.head}")

    def loader(split, shuffle):
        return DataLoader(TiledFeatureDataset(feat_dir, split), batch_size=args.batch_size,
                          num_workers=args.num_workers, shuffle=shuffle,
                          pin_memory=device.type == "cuda")

    train_loader = loader("train", True)
    val_loader = loader("valid", False)

    head = build_head(args.head, embed_dim, NUM_CLASSES, patch_size, label_size).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    # Cosine annealing over the full run: lr goes args.lr -> SCHEDULER_MIN_LR by the last epoch.
    # Schedule-based (steps once per epoch on the epoch index), so no metric/plateau coupling.
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=SCHEDULER_MIN_LR)

    weight = None
    if args.weighted_ce:
        # Inverse-frequency weights from the train pixel counts (ignoring IGNORE_LABEL),
        # normalized to mean 1 so the overall loss scale is unchanged.
        counts = torch.zeros(NUM_CLASSES)
        for _, label in train_loader:
            valid = label[label != IGNORE_LABEL]
            counts += torch.bincount(valid.flatten(), minlength=NUM_CLASSES).float()
        freq = counts / counts.sum().clamp(min=1)
        inv = 1.0 / freq.clamp(min=1e-8)
        weight = (inv / inv.mean()).to(device)
        print(f"weighted CE: pixel counts {counts.tolist()} -> weights {[round(w,3) for w in weight.tolist()]}")
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, weight=weight)

    # val IS the test set here (no separate val for a hyperparameter-free LP). To avoid
    # selecting on test, we do NOT keep the best epoch: we train a fixed number of epochs
    # (lr cosine-annealed independently of any metric) and report the FINAL epoch's val=test
    # mIoU. The per-epoch val prints are just progress; the headline is the last line.
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
        sched.step()                                      # cosine anneal by epoch index
        res, headline, fu, stats = evaluate(head, val_loader, device)
        final = (headline, fu, res.metrics, stats)
        print(f"epoch {epoch+1}/{args.epochs} | train_loss {train_loss:.4f} | "
              f"test mIoU(NF,FO) {headline:.4f} | FU IoU {fu if fu is None else round(fu,4)}")
        print(_fmt_per_class(stats))

    headline, fu, metrics, stats = final
    print(f"\nFINAL (val-as-test) mIoU(NF,FO) {headline:.4f} | FU IoU "
          f"{fu if fu is None else round(fu,4)} | {metrics}")
    print("Per-class (NF/FO/FU):")
    print(_fmt_per_class(stats))

    row = build_result_row(args, meta, headline, fu, metrics, stats)
    append_results_csv(args.results_csv, row)
    print(f"Appended results to {args.results_csv}")

    visualize_first_val(head, feat_dir, Path(f"{args.tiles_root}_t{tile_size}"), device, args.viz_out)


if __name__ == "__main__":
    main()
