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
import os
from collections.abc import Sequence
from typing import Any, Literal
import kornia.augmentation as K
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from einops import rearrange
from matplotlib.figure import Figure
from torch import Tensor
from torchvision.models._api import WeightsEnum
from torchgeo.datamodules import BaseDataModule
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchgeo.models import FCN, get_weight

class OlmoEarthSemanticSegmentationTask(SemanticSegmentationTask):
    """Semantic Segmentation. for Olmoearth"""

    def __init__(
        self,
        model: str = 'olmoearth',
        backbone: ModelID = ModelID.OLMOEARTH_V1_1_BASE,
        weights: bool = True,
        in_channels: int = 3,
        task: Literal['binary', 'multiclass', 'multilabel'] = 'multiclass',
        num_classes: int | None = None,
        num_labels: int | None = None,
        labels: list[str] | None = None,
        num_filters: int = 3,
        pos_weight: Tensor | None = None,
        loss: Literal['ce', 'bce', 'jaccard', 'focal', 'dice'] = 'ce',
        class_weights: Tensor | Sequence[float] | None = None,
        ignore_index: int | None = None,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        """Initialize a new SemanticSegmentationTask instance.

        Args:
            in_channels: Number of input channels to model.
            task: One of 'binary', 'multiclass', or 'multilabel'.
            num_classes: Number of prediction classes (only for ``task='multiclass'``).
            num_labels: Number of prediction labels (only for ``task='multilabel'``).
            labels: List of class names.
            num_filters: Number of filters. Only applicable when model='fcn'.
            pos_weight: A weight of positive examples and used with 'bce' loss.
            loss: Name of the loss function, currently supports
                'ce', 'bce', 'jaccard', 'focal', and 'dice' loss.
            class_weights: Optional rescaling weight given to each
                class and used with 'ce' loss.
            ignore_index: Optional integer class index to ignore in the loss and
                metrics.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the
                decoder and segmentation head.
            freeze_decoder: Freeze the decoder network to linear probe
                the segmentation head.
        """
        self.weights = weights
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, T, C, H, W).

        Returns:
            Output tensor of shape (B, num_classes, H, W).
        """
        if x.ndim == 5:
            x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.model(x)
        return x

    def configure_models(self) -> None:
        """Initialize the model."""
        model: str = self.hparams['model']
        backbone: ModelID = self.hparams['backbone']
        weights = self.weights
        in_channels: int = self.hparams['in_channels']
        num_classes: int = (
            self.hparams['num_classes'] or self.hparams['num_labels'] or 1
        )

        self.model = load_model_from_id(backbone, load_weights=True)
            
        # Freeze backbone
        if self.hparams['freeze_backbone']:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams['freeze_decoder']:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch['image']
        y = batch['mask']
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.train_metrics(y_hat, y)

        if self.hparams['loss'] == 'bce':
            y = y.float()

        loss: Tensor = self.criterion(y_hat, y)
        self.log('train_loss', loss, batch_size=batch_size)

        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = batch['mask']
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.val_metrics(y_hat, y)

        if self.hparams['loss'] == 'bce':
            y = y.float()

        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, batch_size=batch_size)
        temp_datamodule = getattr(self.trainer, 'datamodule', None)
        temp_experiment = getattr(self.logger, 'experiment', None)
        if (
            batch_idx < 10
            and isinstance(temp_datamodule, BaseDataModule)
            and hasattr(temp_experiment, 'add_figure')
        ):
            datamodule = temp_datamodule
            if batch['image'].ndim == 5:
                _, T, C, _, _ = batch['image'].shape
                batch['image'] = rearrange(batch['image'], 'b t c h w -> b (t c) h w')

                aug = K.AugmentationSequential(
                    K.Denormalize(datamodule.mean, datamodule.std),
                    data_keys=None,
                    keepdim=True,
                )
                batch = aug(batch)
                batch['image'] = rearrange(
                    batch['image'], 'b (t c) h w -> b t c h w', t=T, c=C
                )
            else:
                aug = K.AugmentationSequential(
                    K.Denormalize(datamodule.mean, datamodule.std),
                    data_keys=None,
                    keepdim=True,
                )
                batch = aug(batch)
            match self.hparams['task']:
                case 'binary' | 'multilabel':
                    batch['prediction'] = (y_hat.sigmoid() >= 0.5).long()
                case 'multiclass':
                    batch['prediction'] = y_hat.argmax(dim=1)

            for key in ['image', 'mask', 'prediction']:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]

            fig: Figure | None = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = getattr(self.logger, 'experiment')
                summary_writer.add_figure(
                    f'image/{batch_idx}', fig, global_step=self.global_step
                )  # type: ignore[call-non-callable]
                plt.close()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = batch['mask']
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.test_metrics(y_hat, y)

        if self.hparams['loss'] == 'bce':
            y = y.float()

        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, batch_size=batch_size)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, Tensor]:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Dictionary with 'probabilities', 'bounds', and 'transform' keys.

        .. versionchanged:: 0.9
           Changed return type from Tensor to dict with probabilities, bounds,
           and transform keys.
        """
        x = batch['image']
        y_hat: Tensor = self(x)

        match self.hparams['task']:
            case 'binary' | 'multilabel':
                y_hat = y_hat.sigmoid()
            case 'multiclass':
                y_hat = y_hat.softmax(dim=1)

        return {
            'probabilities': y_hat,
            'bounds': batch.get('bounds'),
            'transform': batch.get('transform'),
        }
