import lightning as L
from torchgeo.trainers import SemanticSegmentationTask
from lightning.pytorch.loggers import WandbLogger
from torchgeo.datamodules import PASTISDataModule
import torch

epochs=16

# PASTISDataModule pads all sequences to padding_length (default 61) timesteps.
# The model receives (B, T*C, H, W) = (B, 61*10, H, W) = (B, 610, H, W).

datamodule = PASTISDataModule(
    root='data',
    batch_size=64,
    num_workers=4,
    bands='s2',
    mode='semantic',
)

task = SemanticSegmentationTask(
    model='unet',
    backbone='resnet50',
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

wandb_logger = WandbLogger(project="pastis-test", log_model=False)
trainer = L.Trainer(max_epochs=epochs, accelerator='gpu', devices=1, logger=wandb_logger)
trainer.fit(task, datamodule=datamodule)


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

task.eval()

def visualize(batch, filename):
    x_raw = batch['image'][0:1]
    x = datamodule.aug(x_raw.to(task.device))
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
    plt.close(fig)

datamodule.setup('fit')
visualize(next(iter(datamodule.train_dataloader())), 'prediction_train.png')

datamodule.setup('test')
visualize(next(iter(datamodule.test_dataloader())), 'prediction_test.png')

# python -u pastis.py