"""Visualize UTAE predictions on PASTIS + log test metrics to CSV.

Discovers every checkpoints/utae_*_best.pt, infers modalities + fusion from the
filename, rebuilds the matching model, then for each:
  - saves a 3-panel figure (RGB | prediction | ground truth, with metrics overlay)
    for the first train (folds 1-3) and first test (fold 5) sample,
  - evaluates on the full test set and appends a row to utae_pastis.csv.

Runs in the base env (torch 2.12), NOT env_olmo:
    source env_login.sh
    python -u visualize_utae_pastis.py

Note: UTAE works on full 128x128 patches, so these images are NOT the same tiles as
the OlmoEarth 64x64 viz -- not directly comparable image-to-image.
"""
import glob
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from utae.dataloader import PASTIS_Dataset
from utae.fusion import build_model
import utae_pastis as UP   # reuse evaluate / metrics / collate / constants

DATA_ROOT = "data/PASTIS-R"
CKPT_GLOB = "checkpoints/utae_*_best.pt"
OUT_DIR = "pastis_visualize"
NUM_CLASSES = UP.NUM_CLASSES
IGNORE_INDEX = UP.IGNORE_INDEX  # 19

# PASTIS class names + cmap (same scheme as the other viz scripts).
CLASSES = [
    'background', 'meadow', 'soft_winter_wheat', 'corn', 'winter_barley',
    'winter_rapeseed', 'spring_barley', 'sunflower', 'grapevine', 'beet',
    'winter_triticale', 'winter_durum_wheat', 'fruits_vegetables_flowers',
    'potatoes', 'leguminous_fodder', 'soybeans', 'orchard', 'mixed_cereal',
    'sorghum', 'void_label',
]
CMAP = plt.get_cmap('tab20', 20)


def _infer_run(ckpt_path):
    """Parse modalities + fusion from a name like utae_s2s1a_early_lr0.001_ep32_best.pt
    or utae_s2_lr0.001_ep2_best.pt. Returns (modalities, fusion)."""
    name = os.path.basename(ckpt_path)
    body = name[len("utae_"):].split("_lr")[0]   # e.g. "s2s1a_early" or "s2"
    fusion = "early"
    for fz in ("early", "late"):
        if body.endswith("_" + fz):
            fusion = fz
            body = body[: -(len(fz) + 1)]
            break
    # body is now the modality tag: s2 / s1a / s1d / s2s1a / ...
    sat_map = {"s2": "S2", "s1a": "S1A", "s1d": "S1D"}
    mods, i = [], 0
    for tok in ("s1a", "s1d", "s2"):      # match longer S1x before s2
        if tok in body:
            mods.append(sat_map[tok])
    # keep canonical order S2 first (matches build_model primary)
    mods = [m for m in ("S2", "S1A", "S1D") if m in mods]
    return mods, fusion


def visualize(rgb, pred, mask, filename):
    p = torch.from_numpy(pred).flatten()
    m = torch.from_numpy(mask).flatten()
    valid = m != IGNORE_INDEX
    cm = torch.bincount(m[valid] * NUM_CLASSES + p[valid],
                        minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    keep = [c for c in range(NUM_CLASSES) if c != IGNORE_INDEX]
    met = UP._metrics_from_confmat(cm[keep][:, keep])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb); axes[0].set_title('Input (RGB mean)'); axes[0].axis('off')
    for ax, data, title in zip(axes[1:], [pred, mask], ['Prediction', 'Ground truth']):
        ax.imshow(data, cmap=CMAP, vmin=0, vmax=19, interpolation='nearest')
        ax.set_title(title); ax.axis('off')
    fig.suptitle(f"miou={met['miou']:.3f}  acc={met['overall_acc']:.3f}  "
                 f"macro_acc={met['macro_acc']:.3f}", fontsize=14)
    present = np.unique(np.concatenate([pred.flatten(), mask.flatten()])).astype(int)
    present = [i for i in present if 0 <= i <= 19]
    patches = [mpatches.Patch(color=CMAP(i), label=f'{i}: {CLASSES[i]}') for i in present]
    fig.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close(fig)


def _rgb_of(split_folds, idx=0):
    """Raw (un-normalized) S2 -> time-averaged natural-color RGB. PASTIS S2 raw band
    order is [B2,B3,B4,...]; RGB = B4,B3,B2 = indices 2,1,0."""
    ds = PASTIS_Dataset(folder=DATA_ROOT, norm=False, folds=split_folds, sats=["S2"])
    (data, _), _ = ds[idx]
    s2 = data["S2"].float()                       # (T,10,128,128)
    rgb = s2.mean(0)[[2, 1, 0]].permute(1, 2, 0).numpy()
    return (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    available = sorted(glob.glob(CKPT_GLOB))
    if not available:
        print(f"No checkpoints matching {CKPT_GLOB}.")
        return
    print("Found:", [os.path.basename(c) for c in available])
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%m%d%H%M")

    SPLITS = {"train": [1, 2, 3], "test": [5]}
    for ckpt in available:
        mods, fusion = _infer_run(ckpt)
        tag = os.path.basename(ckpt).replace("_best.pt", "")
        print(f"\n{tag}: modalities={mods} fusion={fusion}")
        model = build_model(mods, fusion, NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        # visualize first sample of each split
        for split, folds in SPLITS.items():
            ds = PASTIS_Dataset(folder=DATA_ROOT, norm=True, target="semantic",
                                folds=folds, sats=mods)
            (data, dates), target = ds[0]
            batch = ((  {k: v.unsqueeze(0) for k, v in data.items()},
                        {k: v.unsqueeze(0) for k, v in dates.items()}), target.unsqueeze(0))
            with torch.no_grad():
                logits, _ = UP._forward(model, batch, device, mods)
            pred = logits.argmax(1)[0].cpu().numpy()
            mask = target.numpy()
            rgb = _rgb_of(folds, idx=0)
            visualize(rgb, pred, mask, os.path.join(OUT_DIR, f"{tag}_pred_{split}.png"))

        # full test-set eval + CSV (reuse utae_pastis' evaluate + CSV writer)
        test_loader = UP.make_loader(
            UP.UTAEConfig(modalities=mods, fusion=fusion), folds=[5], shuffle=False)
        res = UP.evaluate(model, test_loader, device, mods)
        print(f"  test miou={res['miou']:.4f} acc={res['overall_acc']:.4f}")
        UP._append_csv({
            "timestamp": ts,
            "checkpoint": os.path.basename(ckpt),
            "modalities": "+".join(mods),
            "fusion": fusion if len(mods) > 1 else "n/a",
            "test_miou": round(res["miou"], 6),
            "test_overall_acc": round(res["overall_acc"], 6),
            "test_macro_acc": round(res["macro_acc"], 6),
            "test_macro_f1": round(res["macro_f1"], 6),
        })


if __name__ == "__main__":
    main()
