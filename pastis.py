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
from typing import Literal, cast
from olmoearth_pretrain_minimal import ModelID, load_model_from_id
from olmoearth_utils import OlmoEarthSemanticSegmentationTask

epochs=64
batch_size=64
num_workers=4
save_checkpoint=True  # if True, save the final trained model checkpoint
models_backbones={
    "unet": "resnet50",
    "deeplabv3+": "resnet50",
    "dpt": "tu-vit_small_patch16_224",
    "segformer": "tu-mix_transformer_b2",
    "olmoearth": f"{ModelID.OLMOEARTH_V1_1_BASE}"
}
model_index=4
model_name = cast(Literal['unet', 'deeplabv3+', 'fcn', 'upernet', 'segformer', 'dpt'], list(models_backbones.keys())[model_index])
backbone=models_backbones[model_name]
print(f"{model_name=} || {backbone=} || {num_workers=} || {batch_size=}")

data_dir = 'data'

class UpsampleTo224(Callback):
    def _resize(self, batch):
        img = batch['image']
        if img.dim() == 5:
            # PASTIS layout is (B, T, C, H, W). Fold T into batch, interpolate, unfold.
            B, T, C, H, W = img.shape
            img = img.reshape(B * T, C, H, W)
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            img = img.reshape(B, T, C, 224, 224)
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

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._resize(batch)


datamodule = PASTISDataModule(
    root=data_dir,
    batch_size=batch_size,
    num_workers=num_workers,
    bands='s2',
    mode='semantic',
)
if model_name!="olmoearth":
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
else:
    task = OlmoEarthSemanticSegmentationTask(
        model=model_name,
        backbone=backbone,
        weights=None,
        in_channels=10,
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
if save_checkpoint:
    from lightning.pytorch.callbacks import ModelCheckpoint
    callbacks.append(ModelCheckpoint(
        dirpath='checkpoints',
        filename=f'{model_name}_{backbone}_{{epoch}}',
    ))
trainer = L.Trainer(max_epochs=epochs, accelerator='gpu', devices=1, logger=wandb_logger, callbacks=callbacks, enable_checkpointing=save_checkpoint)

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
IGNORE_INDEX = 19


from torchmetrics.functional.classification import multiclass_accuracy, multiclass_jaccard_index

def compute_metrics(pred, mask, num_classes=20, ignore_index=IGNORE_INDEX):
    """Per-image accuracy and IoU, each as (micro, macro)."""
    micro_acc = multiclass_accuracy(pred, mask, num_classes, average='micro', ignore_index=ignore_index)
    macro_acc = multiclass_accuracy(pred, mask, num_classes, average='macro', ignore_index=ignore_index)
    micro_iou = multiclass_jaccard_index(pred, mask, num_classes, average='micro', ignore_index=ignore_index)
    macro_iou = multiclass_jaccard_index(pred, mask, num_classes, average='macro', ignore_index=ignore_index)
    return micro_acc.item(), macro_acc.item(), micro_iou.item(), macro_iou.item()


def visualize(out, filename):
    """Plot the first sample of a predict_step output dict."""
    img = out['image'][0].cpu()           # (T, C, H, W) or (C, H, W) after aug
    pred_t = out['pred'][0].cpu()
    mask_t = out['mask'][0].squeeze().cpu()

    micro_acc, macro_acc, micro_iou, macro_iou = compute_metrics(pred_t, mask_t)

    pred = pred_t.numpy()
    mask = mask_t.numpy()

    # PASTIS S2 RGB bands are [2, 1, 0] (per torchgeo PASTIS.plot).
    if img.dim() == 4:                    # (T, C, H, W) -> mean over time for RGB
        rgb = img.mean(0)[[2, 1, 0]].permute(1, 2, 0).numpy()
    else:
        rgb = img[[2, 1, 0]].permute(1, 2, 0).numpy()
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb); axes[0].set_title('Input (RGB mean)'); axes[0].axis('off')
    for ax, data, title in zip(axes[1:], [pred, mask], ['Prediction', 'Ground truth']):
        ax.imshow(data, cmap=CMAP, vmin=0, vmax=19, interpolation='nearest')
        ax.set_title(title); ax.axis('off')

    fig.suptitle(
        f'Acc micro={micro_acc:.3f} macro={macro_acc:.3f}  |  '
        f'IoU micro={micro_iou:.3f} macro={macro_iou:.3f}',
        fontsize=14,
    )

    present = np.unique(np.concatenate([pred.flatten(), mask.flatten()])).astype(int)
    patches = [mpatches.Patch(color=CMAP(i), label=f'{i}: {CLASSES[i]}') for i in present]
    fig.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    print(f"Saved prediction to {filename}")
    plt.close(fig)

def predict_one_batch(dataloader):
    """Grab one batch, normalize via datamodule.aug (the same pipeline used in
    training), run the model, and return {image, mask, pred} for the first sample.

    Applies aug directly rather than relying on on_after_batch_transfer, which
    silently skips normalization unless the datamodule is attached to an active
    trainer in the right state.
    """
    task.eval()
    batch = next(iter(dataloader))
    batch = {k: (v.to(task.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    raw_image = batch['image'].clone()  # un-normalized, for the RGB display panel
    normed = datamodule.aug({'image': batch['image'], 'mask': batch['mask']})
    if model_name in ('dpt', 'segformer'):
        UpsampleTo224()._resize(normed)  # resize to 224x224, matching training
    with torch.no_grad():
        pred = task(normed['image']).argmax(dim=1)

    return {
        'image': raw_image[0:1].cpu(),
        'mask': normed['mask'][0:1].cpu(),  # use (possibly resized) mask to match pred
        'pred': pred[0:1].cpu(),
    }

from datetime import datetime
from zoneinfo import ZoneInfo
import os
ts = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%m%d%H%M')
out_dir = 'pastis_visualize'
os.makedirs(out_dir, exist_ok=True)

datamodule.setup('fit')
visualize(predict_one_batch(datamodule.train_dataloader()), os.path.join(out_dir, f'prediction_train_{model_name}_{ts}.png'))

datamodule.setup('test')
visualize(predict_one_batch(datamodule.test_dataloader()), os.path.join(out_dir, f'prediction_test_{model_name}_{ts}.png'))

# python -u pastis.py
