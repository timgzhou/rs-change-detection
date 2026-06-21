"""Load the finetuned OlmoEarth checkpoint and visualize its prediction on the
first train sample and the first test sample.

NOTE: the visualized samples are NOT the same physical fields as pastis.py's. The
OlmoEarth pipeline (PASTISRProcessor) splits each 128x128 PASTIS patch into 4x 64x64
tiles and uses a fold-based train/val/test split, whereas pastis.py uses torchgeo's
random 60/20/20 split on full 128x128 patches. We use sample index 0 of each split as
requested, but they won't visually correspond across the two scripts.

Run in the OlmoEarth venv:
    source env_olmo.sh
    python -u visualize_olmoearth_pastis.py
"""
import os
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "olmo_shims"))
import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()  # MUST run before any olmoearth_pretrain import

from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import cast  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from einops import rearrange  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from torchmetrics.functional.classification import (  # noqa: E402
    multiclass_accuracy,
    multiclass_jaccard_index,
)

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id  # noqa: E402
from olmoearth_pretrain.evals.datasets.configs import DATASET_TO_CONFIG, TaskType  # noqa: E402
from olmoearth_pretrain.evals.datasets.pastis_dataset import PASTISRDataset  # noqa: E402
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn  # noqa: E402
from olmoearth_pretrain.evals.finetune.model import BackboneWithHead, to_device  # noqa: E402
from olmoearth_pretrain.nn.flexi_vit import PoolingType  # noqa: E402

# Reuse the finetune module so head construction / forward / guidance stay in sync.
import finetune_olmoearth_pastis as FT  # noqa: E402
from config import MODEL_SIZE_TO_ID  # noqa: E402

# ---- config (inherited from the finetune module) ----
DATA_SPLITS = FT.DATA_SPLITS
MODEL_ID = FT.MODEL_ID
DATASET = FT.DATASET
INPUT_MODALITIES = FT.INPUT_MODALITIES
POOLING_TYPE = FT.POOLING_TYPE
OUT_DIR = "pastis_visualize"
IGNORE_INDEX = -1  # OlmoEarth maps PASTIS void (19) -> -1

CKPT_GLOB = "checkpoints/*_best.pt"  # discovered at runtime; head inferred from name
RESULTS_CSV = "oe_pastis.csv"        # appended each run with test-set metrics per checkpoint
CSV_FIELDS = ["timestamp", "checkpoint", "head", "model_size",
              "test_miou", "test_overall_acc", "test_macro_acc", "test_macro_f1"]


def _append_csv(row: dict) -> None:
    """Append one result row to RESULTS_CSV, writing the header if the file is new."""
    import csv
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)

# PASTIS class names + colormap (same 20-class scheme as pastis.py).
CLASSES = [
    'background', 'meadow', 'soft_winter_wheat', 'corn', 'winter_barley',
    'winter_rapeseed', 'spring_barley', 'sunflower', 'grapevine', 'beet',
    'winter_triticale', 'winter_durum_wheat', 'fruits_vegetables_flowers',
    'potatoes', 'leguminous_fodder', 'soybeans', 'orchard', 'mixed_cereal',
    'sorghum', 'void_label',
]
CMAP = plt.get_cmap('tab20', 20)


def compute_metrics(pred, mask, num_classes=20, ignore_index=IGNORE_INDEX):
    """Per-image accuracy and IoU, each as (micro, macro). Mirrors pastis.py."""
    pred_t = torch.from_numpy(pred).long()
    mask_t = torch.from_numpy(mask).long()
    micro_acc = multiclass_accuracy(pred_t, mask_t, num_classes, average='micro', ignore_index=ignore_index)
    macro_acc = multiclass_accuracy(pred_t, mask_t, num_classes, average='macro', ignore_index=ignore_index)
    micro_iou = multiclass_jaccard_index(pred_t, mask_t, num_classes, average='micro', ignore_index=ignore_index)
    macro_iou = multiclass_jaccard_index(pred_t, mask_t, num_classes, average='macro', ignore_index=ignore_index)
    return micro_acc.item(), macro_acc.item(), micro_iou.item(), macro_iou.item()


def _draw_token_grid(ax, img_hw, grid_hw):
    """Overlay the OlmoEarth encoder's patch/token grid (grid_hw cells over img_hw px)."""
    H, W = img_hw
    gh, gw = grid_hw
    for k in range(1, gw):
        ax.axvline(k * W / gw - 0.5, color='white', lw=0.5, alpha=0.6)
    for k in range(1, gh):
        ax.axhline(k * H / gh - 0.5, color='white', lw=0.5, alpha=0.6)


def visualize(rgb, pred, mask, filename, grid_hw):
    """3-panel plot (RGB | prediction | ground truth) with metrics overlay and the
    OlmoEarth encoder token grid (grid_hw cells, e.g. 8x8 for 64px tiles @ patch_size 8)."""
    micro_acc, macro_acc, micro_iou, macro_iou = compute_metrics(pred, mask)
    img_hw = pred.shape  # (H, W) in pixels

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb); axes[0].set_title('Input (RGB mean)'); axes[0].axis('off')
    for ax, data, title in zip(axes[1:], [pred, mask], ['Prediction', 'Ground truth']):
        ax.imshow(data, cmap=CMAP, vmin=0, vmax=19, interpolation='nearest')
        ax.set_title(title); ax.axis('off')
    for ax in axes:
        _draw_token_grid(ax, img_hw, grid_hw)

    fig.suptitle(
        f'Acc micro={micro_acc:.3f} macro={macro_acc:.3f}  |  '
        f'IoU micro={micro_iou:.3f} macro={macro_iou:.3f}',
        fontsize=14,
    )
    present = np.unique(np.concatenate([pred.flatten(), mask.flatten()])).astype(int)
    present = [i for i in present if 0 <= i <= 19]
    patches = [mpatches.Patch(color=CMAP(i), label=f'{i}: {CLASSES[i]}') for i in present]
    fig.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close(fig)


def load_rgb(split, idx=0):
    """Raw (un-normalized) S2 -> natural-color RGB, time-averaged. 13-band L1C order:
    B04/B03/B02 = indices 3/2/1."""
    s2 = torch.load(Path(DATA_SPLITS) / f"pastis_r_{split}" / "s2_images" / f"{idx}.pt")
    rgb = s2.float().mean(0)[[3, 2, 1]].permute(1, 2, 0).numpy()  # (H, W, 3)
    return (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)


def _build_head(head, encoder, patch_size, task_config, device):
    return FT.build_head(head, encoder, patch_size, task_config).to(device)


def _first_sample_batch(split, head):
    """First sample (index 0) of `split`, in the batch form the head's forward expects."""
    ds = PASTISRDataset(path_to_splits=Path(DATA_SPLITS), split=split, partition="default",
                        norm_stats_from_pretrained=True, input_modalities=INPUT_MODALITIES)
    masked, label = eval_collate_fn([ds[0]])
    if not head.startswith("anyup"):
        return (masked, label)
    temporal = head in ("anyup_t1", "anyup_t2")
    rgb = FT._load_rgb_guidance(split, 0, temporal=temporal).unsqueeze(0)  # (1,3,64,64) or (1,T,3,64,64)
    return (masked, label, rgb)


def _infer_run(ckpt_path):
    """Infer (head, model_size) from a run-name checkpoint, e.g.
    oe_pastis_base_s2s1_anyup_t2_lr3e-4_ep32_best.pt -> ("anyup_t2","base")."""
    name = os.path.basename(ckpt_path)
    size = next((s for s in ("nano", "tiny", "base", "large") if f"_{s}_" in name), "base")
    if "_anyup_t1_" in name:
        head = "anyup_t1"
    elif "_anyup_t2_" in name:
        head = "anyup_t2"
    elif "_anyup_" in name:
        head = "anyup"
    else:
        head = "lp"
    # patch size encoded as _p<N>_ in run_name; default 4 for older names without it.
    m = re.search(r"_p(\d+)_", name)
    patch_size = int(m.group(1)) if m else 4
    return head, size, patch_size


def main():
    import glob
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_config = DATASET_TO_CONFIG[DATASET]
    assert task_config.task_type == TaskType.SEGMENTATION

    available = sorted(glob.glob(CKPT_GLOB))
    if not available:
        print(f"No checkpoints matching {CKPT_GLOB}. Train first (finetune_olmoearth_pastis.py).")
        return
    print("Found checkpoints:", [os.path.basename(c) for c in available])

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%m%d%H%M')

    # Cache encoders by model size (different checkpoints may use different sizes).
    encoders: dict[str, nn.Module] = {}
    for ckpt in available:
        head, size, patch_size = _infer_run(ckpt)
        tag = os.path.basename(ckpt).replace("_best.pt", "")
        if size not in encoders:
            m = load_model_from_id(getattr(ModelID, MODEL_SIZE_TO_ID[size]), load_weights=True)
            encoders[size] = cast(nn.Module, m.encoder if hasattr(m, "encoder") else m)
        encoder = encoders[size]

        FT.HEAD = head  # FT.make_loader / _forward_logits branch on this module global
        ft = _build_head(head, encoder, patch_size, task_config, device)
        # Lazy-init the head, then load finetuned weights.
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            FT._forward_logits(ft, _first_sample_batch("train", head), device, task_config, patch_size)
        ft.load_state_dict(torch.load(ckpt, map_location=device))
        ft.eval()

        # --- visualize first train + test sample ---
        for split in ["train", "test"]:
            batch = _first_sample_batch(split, head)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, label = FT._forward_logits(ft, batch, device, task_config, patch_size)
            pred = logits.argmax(dim=1)[0].cpu().numpy()
            mask = label[0].cpu().numpy()
            rgb = load_rgb(split, idx=0)
            grid_hw = (64 // patch_size, 64 // patch_size)  # encoder token grid (8x8)
            out = os.path.join(OUT_DIR, f"{tag}_pred_{split}.png")
            visualize(rgb, pred, mask, out, grid_hw=grid_hw)

        # --- evaluate on the full test set + log to CSV ---
        test_loader = FT.make_loader("test", shuffle=False)  # reads FT.HEAD set above
        res = FT._evaluate(ft, test_loader, device, task_config, patch_size)
        m = res.metrics
        print(f"  {tag}: test miou={m['miou']:.4f} acc={m['overall_acc']:.4f}")
        _append_csv({
            "timestamp": ts,
            "checkpoint": os.path.basename(ckpt),
            "head": head,
            "model_size": size,
            "test_miou": round(m["miou"], 6),
            "test_overall_acc": round(m["overall_acc"], 6),
            "test_macro_acc": round(m["macro_acc"], 6),
            "test_macro_f1": round(m["macro_f1"], 6),
        })


if __name__ == "__main__":
    main()

# python -u visualize_olmoearth_pastis.py