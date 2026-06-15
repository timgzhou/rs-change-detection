"""Load a trained checkpoint and evaluate it on the PASTIS test set.

Usage:
    python -u test_checkpoint.py <path/to/checkpoint.ckpt>
    python -u test_checkpoint.py            # uses DEFAULT_CKPT below
"""
import os
import sys
import torch
import lightning as L
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.datamodules import PASTISDataModule
from torchgeo.datasets.utils import unbind_samples

DEFAULT_CKPT = 'pastis-test/wcs7wdbf/checkpoints/epoch=127-step=2816.ckpt'

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
print(f"Loading checkpoint: {ckpt_path}")

# Hyperparameters (model, backbone, in_channels, etc.) are restored from the
# checkpoint, so no need to re-specify the architecture here.
task = SemanticSegmentationTask.load_from_checkpoint(ckpt_path)

datamodule = PASTISDataModule(
    root='data',
    batch_size=64,
    num_workers=4,
    bands='s2',
    mode='semantic',
)

trainer = L.Trainer(accelerator='gpu', devices=1, logger=False)
trainer.test(task, datamodule=datamodule)


# ---- Visualize predictions on 4 test samples using torchgeo's plot() ----
N_SAMPLES = 4
out_dir = 'pastis_visualize'
os.makedirs(out_dir, exist_ok=True)

task.eval()
datamodule.setup('test')

batch = next(iter(datamodule.test_dataloader()))
batch = {k: (v.to(task.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

# Normalize with the datamodule's own aug pipeline (the same one used in training,
# applied directly to avoid on_after_batch_transfer's trainer-state dependency).
raw_image = batch['image'].clone()  # un-normalized, for plotting
raw_mask = batch['mask'].clone()    # original mask, for plotting/metrics

normed = datamodule.aug({'image': batch['image'], 'mask': batch['mask']})
with torch.no_grad():
    y_hat = task(normed['image'])

# Plot uses the raw (un-normalized) reflectance, which PASTIS.plot scales by /5000.
batch['image'] = raw_image
batch['mask'] = raw_mask
batch['prediction'] = y_hat.argmax(dim=1)

for key in ('image', 'mask', 'prediction'):
    batch[key] = batch[key].cpu()

samples = unbind_samples(batch)
for i in range(min(N_SAMPLES, len(samples))):
    fig = datamodule.plot(samples[i])
    path = os.path.join(out_dir, f'ckpt_pred_{i}.png')
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved {path}")


# python -u test_checkpoint.py