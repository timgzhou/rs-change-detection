import lightning as L
from torchgeo.trainers import SemanticSegmentationTask
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback
from torchgeo.datamodules import PASTISDataModule
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from kornia.enhance import normalize


epochs=2
batch_size=64
num_workers=4
models_backbones={
    "unet": "resnet50",
    "deeplabv3+": "resnet50",
    "dpt": "tu-vit_base_patch16_224",
    "segformer": "tu-mix_transformer_b2",
}
model_index=0
from typing import Literal, cast
model_name = cast(Literal['unet', 'deeplabv3+', 'fcn', 'upernet', 'segformer', 'dpt'], list(models_backbones.keys())[model_index])
backbone=models_backbones[model_name]
print(f"{model_name=} || {backbone=} || {num_workers=} || {batch_size=}")

import os
_data_src = 'data'
if tmpdir := os.environ.get('SLURM_TMPDIR'):
    _data_dst = os.path.join(tmpdir, 'data')
    if not os.path.exists(os.path.join(_data_dst, 'PASTIS-R', 'metadata.geojson')):
        import time, zipfile, shutil
        os.makedirs(_data_dst, exist_ok=True)
        _zip_src = os.path.join(_data_src, 'PASTIS-R.zip')
        _zip_dst = os.path.join(tmpdir, 'PASTIS-R.zip')
        print(f"Copying zip to {tmpdir}...")
        _t0 = time.time()
        shutil.copy2(_zip_src, _zip_dst)
        print(f"Copy done in {time.time()-_t0:.1f}s. Extracting...")
        _t0 = time.time()
        with zipfile.ZipFile(_zip_dst) as zf:
            zf.extractall(_data_dst)
        os.remove(_zip_dst)
        print(f"Extract done in {time.time()-_t0:.1f}s.")
    else: print(f"data already exists at {os.path.join(_data_dst, 'PASTIS-R')}")
    _data_src = _data_dst
    
datamodule = PASTISDataModule(
    root=_data_src,
    batch_size=batch_size,
    num_workers=num_workers,
    bands='s2',
    mode='semantic',
)

########## DEBUG ###########
# import time
# datamodule.setup('fit')
# loader = datamodule.train_dataloader()
# t0 = time.time()
# for batch in loader:
#     pass
# print(f"Dataloader only: {time.time()-t0:.1f}s")
# raise SystemExit
############################

class UpsampleTo224(Callback):
    def _resize(self, batch):
        img = batch['image']
        if img.dim() == 5:
            # (B, C, T, H, W) -> flatten T into C, interpolate, unflatten
            B, C, T, H, W = img.shape
            img = img.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W)
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img = img.reshape(B, C, T, 224, 224).permute(0, 2, 1, 3, 4)
        else:
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        batch['image'] = img
        mask = batch['mask']
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        batch['mask'] = F.interpolate(mask.float(), size=(224, 224), mode='nearest').long().squeeze(1)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._resize(batch)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._resize(batch)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._resize(batch)


task = SemanticSegmentationTask(
    model=model_name,
    backbone=backbone,
    weights=None,
    in_channels=610,
    task='multiclass',
    num_classes=20,
    num_labels=None,
    labels=None,
    num_filters=3,
    pos_weight=None,
    loss='ce',
    class_weights=None,
    ignore_index=19,
    lr=0.001,
    patience=10,
    freeze_backbone=False,
    freeze_decoder=False,
)

wandb_logger = WandbLogger(project="pastis-test", log_model=False, name=f"{model_name}_{backbone}")
callbacks: list[Callback] = [UpsampleTo224()] if model_name in ('dpt', 'segformer') else []
trainer = L.Trainer(max_epochs=epochs, accelerator='gpu', devices=1, logger=wandb_logger, callbacks=callbacks)

trainer.fit(task, datamodule=datamodule)
trainer.test(task, datamodule=datamodule)

CLASSES = [
    'background', 'meadow', 'soft_winter_wheat', 'corn', 'winter_barley',
    'winter_rapeseed', 'spring_barley', 'sunflower', 'grapevine', 'beet',
    'winter_triticale', 'winter_durum_wheat', 'fruits_vegetables_flowers',
    'potatoes', 'leguminous_fodder', 'soybeans', 'orchard', 'mixed_cereal',
    'sorghum', 'void_label',
]
CMAP = plt.get_cmap('tab20', 20)

def visualize(batch, filename):
    x_raw = batch['image'][0:1]
    x = x_raw.to(task.device)
    if model_name in ('dpt', 'segformer'):
        if x.dim() == 5:
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W)
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    x = normalize(x, mean=datamodule.mean.to(task.device), std=datamodule.std.to(task.device))
    mask = batch['mask'][0].squeeze().numpy()

    with torch.no_grad():
        pred = task(x).argmax(dim=1).squeeze(0).cpu().numpy()

    img = x_raw[0].cpu()
    rgb = img.mean(0)[[3, 2, 1]].permute(1, 2, 0).numpy()
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb); axes[0].set_title('Input (RGB mean)'); axes[0].axis('off')
    for ax, data, title in zip(axes[1:], [pred, mask], ['Prediction', 'Ground truth']):
        ax.imshow(data, cmap=CMAP, vmin=0, vmax=19, interpolation='nearest')
        ax.set_title(title); ax.axis('off')

    present = np.unique(np.concatenate([pred.flatten(), mask.flatten()])).astype(int)
    patches = [mpatches.Patch(color=CMAP(i), label=f'{i}: {CLASSES[i]}') for i in present]
    fig.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    print(f"Saved prediction to {filename}")
    plt.close(fig)

datamodule.setup('fit')
visualize(next(iter(datamodule.train_dataloader())), 'prediction_train.png')

datamodule.setup('test')
visualize(next(iter(datamodule.test_dataloader())), 'prediction_test.png')

# python -u pastis.py