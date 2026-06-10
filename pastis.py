import lightning as L
from torchgeo.trainers import SemanticSegmentationTask
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback
from torchgeo.datamodules import PASTISDataModule
import torch
import torch.nn.functional as F
from kornia.enhance import normalize


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

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._resize(batch)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._resize(batch)

epochs=2

# PASTISDataModule pads all sequences to padding_length (default 61) timesteps.
# The model receives (B, T*C, H, W) = (B, 61*10, H, W) = (B, 610, H, W).

models_backbones={
    "unet": "resnet50",
    "deeplabv3+": "resnet50",
    "dpt": "tu-vit_base_patch16_224",
    "segformer": "tu-mix_transformer_b2",
}
model_index=2
model_name=list(models_backbones.keys())[model_index]
backbone=models_backbones[model_name]
print(f"{model_name=} || {backbone=}")
datamodule = PASTISDataModule(
    root='data',
    batch_size=64,
    num_workers=4,
    bands='s2',
    mode='semantic',
)

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
callbacks = [UpsampleTo224()] if model_name in ('dpt', 'segformer') else []
trainer = L.Trainer(max_epochs=epochs, accelerator='gpu', devices=1, logger=wandb_logger, callbacks=callbacks)
trainer.fit(task, datamodule=datamodule)
trainer.test(task, datamodule=datamodule)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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