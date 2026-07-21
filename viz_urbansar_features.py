import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "olmo_shims"))
import olmo_bootstrap; olmo_bootstrap.apply()

import numpy as np, torch, rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from olmoearth_pretrain.model_loader import load_model_from_id
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from finetune_olmoearth_pastis import pool_per_timestep
from urbansarfloods_dataset import UrbanSARFloodsDataset

CROP, PS, INPUT_RES = 64, 4, 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

ds = UrbanSARFloodsDataset("data/urban_sar_floods", split="train")
masked, label = ds[0]
s1 = masked.sentinel1[:CROP, :CROP]
lab = label[:CROP, :CROP].numpy()

samp = OlmoEarthSample(timestamps=masked.timestamps, sentinel1=s1)
mc = MaskedOlmoEarthSample.from_olmoearthsample(samp)
batch, _ = eval_collate_fn([(mc, label[:CROP, :CROP])])

print("loading model...", flush=True)
model = load_model_from_id("/scratch/timz/OlmoEarth-v1-Base", load_weights=True)
enc = (model.encoder if hasattr(model, "encoder") else model).to(device).eval()
for p in enc.parameters():
    p.requires_grad = False

print("encoding...", flush=True)
from olmoearth_pretrain.evals.finetune.model import to_device
# Match the live pipeline: run the encoder under bf16 autocast (finetune_olmoearth_pastis
# does this). Without it, some inputs arrive as bf16 while weights stay float32 -> the
# "Input type BFloat16 and bias type float should be the same" conv error.
with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    tam = enc(to_device(batch, device), patch_size=PS, input_res=INPUT_RES,
              fast_pass=True)["tokens_and_masks"]
    T = next(getattr(tam, m).shape[3] for m in tam.modalities)
    feats = torch.stack([pool_per_timestep(tam, t, PoolingType.MEAN) for t in range(T)], 1)
    fmap = feats.mean(1)[0].float().cpu()          # (gH, gW, D)  (float() undoes bf16 for PCA)
print("feature map:", tuple(fmap.shape), flush=True)

X = fmap.reshape(-1, fmap.shape[-1]).float().numpy()
Xc = X - X.mean(0, keepdims=True)
_, _, Vt = np.linalg.svd(Xc, full_matrices=False)
rgb = (Xc @ Vt[:3].T).reshape(fmap.shape[0], fmap.shape[1], 3)
rgb = (rgb - rgb.min((0, 1))) / (np.ptp(rgb, axis=(0, 1)) + 1e-6)

with rasterio.open("data/urban_sar_floods/03_FU/SAR/20190329_Iran_ID_17_19_SAR.tif") as s:
    raw = s.read().astype(np.float32)[:, :CROP, :CROP]
db = lambda x: np.clip((x - np.percentile(x, 2)) / (np.percentile(x, 98) - np.percentile(x, 2) + 1e-6), 0, 1)
post = np.dstack([db(raw[7]), db(raw[6]), db(raw[7])])

CLS = ["non-flood", "flooded-open", "flooded-urban"]
cmap = ListedColormap(["#2c3e50", "#3498db", "#e74c3c"])
fig, ax = plt.subplots(1, 3, figsize=(16, 6))
ax[0].imshow(post); ax[0].set_title("Input 64x64 (post VV/VH)"); ax[0].axis("off")
ax[1].imshow(rgb, interpolation="nearest")
ax[1].set_title(f"OlmoEarth features PCA-RGB ({fmap.shape[0]}x{fmap.shape[1]}, input_res={INPUT_RES})"); ax[1].axis("off")
ax[2].imshow(lab, cmap=cmap, vmin=0, vmax=2, interpolation="nearest"); ax[2].set_title("Mask 64x64"); ax[2].axis("off")
present = np.unique(lab)
ax[2].legend(handles=[Patch(color=cmap(i), label=f"{i}:{CLS[i]}") for i in present],
             loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
out = "scratchpad_urbansar_features.png"
fig.savefig(out, bbox_inches="tight", dpi=115)
print("saved:", os.path.abspath(out), flush=True)
