from typing import Sequence, Tuple, Dict, Any
import os
import wandb
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinUNETR
from augmentations.mask import random_mask
import math
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.regression import PearsonCorrCoef

import matplotlib.pyplot as plt



class ContrastiveTransformer(pl.LightningModule):
    def __init__(
        self,
        patch_size: Sequence[int] = (4, 4, 4),
        learning_rate: float = 1e-4,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        mask_ratio: float = 0.6,
        temperature: float = 0.6,
        queue_size: int = 4096,
        momentum: float = 0.996,
        warmup_epochs: int = 1,
        max_epochs: int = 30,
        min_lr: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        # Main encoder
        self.encoder = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,
        )

        # Momentum encoder for MoCo
        self.encoder_m = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,
        )

        # Initialize momentum encoder
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_m.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Projection head
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *img_size)
            features = self.encoder.swinViT(dummy_input)[-1]
            encoder_dim = features.shape[1]

        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        # Momentum projection head
        self.projection_m = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        # Initialize momentum projection
        for param_q, param_k in zip(self.projection.parameters(), self.projection_m.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # MoCo queue
        self.register_buffer("queue", F.normalize(torch.randn(128, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.temperature = temperature
        self.momentum = momentum
        self.queue_size = queue_size
        self.mask_ratio = mask_ratio
        self.learning_rate = learning_rate

    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum encoder"""
        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_m.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        for param_q, param_k in zip(self.projection.parameters(), self.projection_m.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update MoCo queue"""
        if self.trainer is not None and self.trainer.world_size > 1:
            keys = self._concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.queue_size:
            self.queue[:, ptr:] = keys[:self.queue_size - ptr].T
            self.queue[:, :batch_size - (self.queue_size - ptr)] = keys[self.queue_size - ptr:].T
            ptr = batch_size - (self.queue_size - ptr)
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _concat_all_gather(self, tensor):
        """Gather tensors from all processes"""
        if not torch.distributed.is_initialized():
            return tensor

        tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)

    def forward_encoder(self, x):
        """Forward through encoder only"""
        features = self.encoder.swinViT(x)[-1]
        return features

    def forward_mae(self, x):
        """MAE forward pass"""
        masked_x, mask = random_mask(x, self.mask_ratio, 4)
        reconstruction = self.encoder(masked_x)
        return reconstruction, mask

    def forward_contrastive(self, x):
        """Contrastive forward pass"""
        features = self.forward_encoder(x)
        z = self.projection(features)
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def forward_momentum(self, x):
        """Forward through momentum encoder"""
        features = self.encoder_m.swinViT(x)[-1]
        z = self.projection_m(features)
        return F.normalize(z, dim=1)

    def contrastive_loss(self, q, k):
        """InfoNCE loss for MoCo with numerical stability"""
        # Add small epsilon for stability
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        l_pos = torch.clamp(l_pos, min=-1.0, max=1.0)
        l_neg = torch.clamp(l_neg, min=-1.0, max=1.0)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"NaN/Inf detected in logits: {logits}")
            print(f"l_pos stats: min={l_pos.min()}, max={l_pos.max()}, mean={l_pos.mean()}")
            print(f"l_neg stats: min={l_neg.min()}, max={l_neg.max()}, mean={l_neg.mean()}")

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


    def training_step(self, batch, batch_idx):
        view1 = batch["vol1"]
        view2 = batch["vol2"]

        # MAE loss
        recon1, mask1 = self.forward_mae(view1)
        recon2, mask2 = self.forward_mae(view2)
        loss_view1 = F.mse_loss(recon1[mask1], view1[mask1])
        loss_view2 = F.mse_loss(recon2[mask2], view2[mask2])
        mae_loss = 0.5 * (loss_view1 + loss_view2)

        # Update momentum encoder
        self._momentum_update()

        # Contrastive loss
        q1 = self.forward_contrastive(view1)
        q2 = self.forward_contrastive(view2)

        with torch.no_grad():
            k1 = self.forward_momentum(view1)
            k2 = self.forward_momentum(view2)

        # Cross-view contrastive loss
        loss_12 = self.contrastive_loss(q1, k2)
        loss_21 = self.contrastive_loss(q2, k1)
        contrastive_loss = 0.5 * (loss_12 + loss_21)

        # Update queue
        self._dequeue_and_enqueue(torch.cat([k1, k2]))

        # Total loss
        total_loss = mae_loss + contrastive_loss

        # Logging
        self.log_dict({
            "train/loss": total_loss,
            "train/mae_loss": mae_loss,
            "train/contrastive_loss": contrastive_loss,
        }, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        view1 = batch["vol1"]
        view2 = batch["vol2"]

        # MAE loss
        recon1, mask1 = self.forward_mae(view1)
        recon2, mask2 = self.forward_mae(view2)
        loss_view1 = F.mse_loss(recon1[mask1], view1[mask1])
        loss_view2 = F.mse_loss(recon2[mask2], view2[mask2])
        mae_loss = 0.5 * (loss_view1 + loss_view2)

        # Contrastive loss (no momentum update in validation)
        q1 = self.forward_contrastive(view1)
        q2 = self.forward_contrastive(view2)
        k1 = self.forward_momentum(view1)
        k2 = self.forward_momentum(view2)

        loss_12 = self.contrastive_loss(q1, k2)
        loss_21 = self.contrastive_loss(q2, k1)
        contrastive_loss = 0.5 * (loss_12 + loss_21)

        total_loss = mae_loss + 1.0 * contrastive_loss

        self.log_dict({
            "val/loss": total_loss,
            "val/mae_loss": mae_loss,
            "val/contrastive_loss": contrastive_loss,
        }, prog_bar=False, on_epoch=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01
        )

        # Calculate total steps
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(num_training_steps * self.hparams.warmup_epochs / self.hparams.max_epochs)

        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=num_warmup_steps
        )

        # Cosine decay scheduler
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(num_training_steps - num_warmup_steps), eta_min=self.hparams.min_lr
        )

        # Chain them together
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[num_warmup_steps]
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
            }
        }

class ClassificationFineTuner(pl.LightningModule):
    def __init__(
        self,
        patch_size: tuple[int],
        num_classes: int,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        in_channels: int = 1,
        learning_rate: float = 1e-5,
        freeze_encoder: bool = False,
        dropout_rate: float = 0.1,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        min_lr: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.freeze_encoder = freeze_encoder
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr

        # Initialize encoder with specified number of channels
        self.encoder = SwinUNETR(
            in_channels=in_channels,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,
        )

        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *img_size)
            features = self.encoder.swinViT(dummy_input)[-1]
            encoder_dim = features.shape[1]


        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder_dim, 64),  # Assuming encoder_dim=768, this is ~49K params
            nn.ReLU(),
            nn.Dropout(0.3),  # Higher dropout for few-shot
            nn.Linear(64, num_classes)
        )

        # Optionally freeze encoder
        if self.freeze_encoder:
            self._freeze_encoder()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.test_auroc = AUROC(task="binary")

    def _freeze_encoder(self):
        """Freeze encoder weights"""
        print("Freezing encoder weights")
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self):
        """Unfreeze encoder weights"""
        print("Unfreezing encoder weights")
        for param in self.encoder.parameters():
            param.requires_grad = True

    def adapt_to_multichannel(self, new_channels: int = 4, strategy: str = "average"):
        """
        Adapt model from 1 channel to multiple channels by modifying first conv layer

        Args:
            new_channels: Number of input channels (e.g., 4 for multi-modal MRI)
            strategy:
                - "average": Each channel gets 1/N of the original weights (preserves activation magnitude)
                - "copy": Each channel gets full copy of original weights
                - "first": Only first channel gets weights, others initialized randomly
        """
        # Get the first conv layer from SwinUNETR
        first_conv = self.encoder.swinViT.patch_embed.proj

        # Check if already has the right number of channels
        if first_conv.in_channels == new_channels:
            print(f"Model already has {new_channels} input channels")
            return

        # Save old weights
        old_weight = first_conv.weight.data.clone()  # [out_ch, 1, d, h, w]

        # Create new conv layer with more input channels
        new_conv = nn.Conv3d(
            in_channels=new_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        with torch.no_grad():
            if strategy == "average":
                # Each channel gets 1/N of the pretrained weights
                # This preserves the expected activation magnitude
                for i in range(new_channels):
                    new_conv.weight.data[:, i:i+1, :, :, :] = old_weight / new_channels

            elif strategy == "copy":
                # Each channel gets a full copy
                # Good if each modality is normalized similarly
                for i in range(new_channels):
                    new_conv.weight.data[:, i:i+1, :, :, :] = old_weight

            elif strategy == "first":
                # Only first channel gets pretrained weights
                nn.init.kaiming_normal_(new_conv.weight.data)
                new_conv.weight.data[:, 0:1, :, :, :] = old_weight

            # Copy bias if exists
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()

        # Replace the layer
        self.encoder.swinViT.patch_embed.proj = new_conv

        # Update hyperparameters
        self.hparams.in_channels = new_channels
        self.in_channels = new_channels

        print(f"✓ Adapted model from {first_conv.in_channels} to {new_channels} channels using '{strategy}' strategy")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        num_classes: int,
        in_channels: int = 1,
        multichannel_strategy: str = "average",
        **kwargs
    ):
        """
        Loads the encoder from a ContrastiveTransformer checkpoint and initializes
        a new ClassificationFineTuner model.
        """
        if not num_classes:
            raise ValueError("num_classes must be specified for fine-tuning.")

        # Load the pre-trained model to get its hparams and state_dict
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(checkpoint_path)

        # Create a new finetuner instance using pre-trained hparams
        # Override with any new kwargs provided by the user
        finetuner_hparams = pretrain_model.hparams
        finetuner_hparams.update(kwargs)
        finetuner_hparams['num_classes'] = num_classes

        # Initialize with 1 channel to match the pre-trained encoder
        finetuner_hparams['in_channels'] = 1

        # The `**finetuner_hparams` will pass img_size, feature_size, etc.
        model = cls(**finetuner_hparams)

        # Copy the encoder weights
        model.encoder.load_state_dict(pretrain_model.encoder.state_dict())

        print(f"✓ Loaded encoder weights from {checkpoint_path}")

        # Adapt to multi-channel input if necessary
        if in_channels > 1:
            model.adapt_to_multichannel(
                new_channels=in_channels,
                strategy=multichannel_strategy
            )

        return model


    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Called when loading a checkpoint during training continuation.
        This handles loading from both ContrastiveTransformer and ClassificationFineTuner checkpoints.
        """
        state_dict = checkpoint['state_dict']

        # Check if this is a ContrastiveTransformer checkpoint
        if 'encoder_m.swinViT.patch_embed.proj.weight' in state_dict:
            print("Detected ContrastiveTransformer checkpoint, filtering weights...")

            # Filter to only encoder weights
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.') and not k.startswith('encoder_m.'):
                    filtered_state_dict[k] = v

            checkpoint['state_dict'] = filtered_state_dict

    def forward(self, x):
        """Forward pass for inference"""
        features = self.encoder.swinViT(x)[-1]
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label'].long()

        logits = self(images).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        # Update metrics with raw logits or probs
        probs = torch.sigmoid(logits)
        self.train_acc.update(probs, labels)
        self.train_auroc.update(probs, labels)

        self.log_dict({
            'train/loss': loss,
            'train/acc': self.train_acc,
            'train/auroc': self.train_auroc,
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
            images, labels = batch['image'], batch['label'].long()

            logits = self(images).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

            # Update metrics
            probs = torch.sigmoid(logits)
            self.val_acc.update(probs, labels)
            self.val_auroc.update(probs, labels)

            self.log_dict({
                'val/loss': loss,
                'val/acc': self.val_acc,
                'val/auroc': self.val_auroc,
            }, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']

        # Forward pass
        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Logging
        self.log_dict({
            'test/loss': loss,
            'test/acc': acc,
        }, on_epoch=True)

        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        # Only optimize classifier if encoder is frozen
        if self.freeze_encoder:
            params = self.classifier.parameters()
        else:
            params = self.parameters()

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # Cosine annealing with warmup
        def lr_lambda(current_step: int):
            if self.trainer.estimated_stepping_batches is not None:
                num_training_steps = self.trainer.estimated_stepping_batches
                num_warmup_steps = (self.warmup_epochs * num_training_steps) // self.max_epochs
            else:
                num_training_steps = self.max_epochs
                num_warmup_steps = self.warmup_epochs

            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                decayed = (1 - self.min_lr / self.learning_rate) * cosine_decay + self.min_lr / self.learning_rate
                return decayed

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import math
import numpy as np
from scipy.stats import pearsonr
from monai.networks.nets import SwinUNETR


class RegressionFineTuner(pl.LightningModule):
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        in_channels: int = 1,
        learning_rate: float = 1e-4,
        freeze_encoder: bool = False,
        dropout_rate: float = 0.1,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        min_lr: float = 1e-6,
        loss_type: str = "mse",  # "mse", "mae", "huber", or "combined"
        loss_alpha: float = 0.5,  # For combined loss: alpha*MSE + (1-alpha)*MAE
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.freeze_encoder = freeze_encoder
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.loss_type = loss_type
        self.loss_alpha = loss_alpha

        # Initialize encoder with specified number of channels
        self.encoder = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,
        )

        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *img_size)
            features = self.encoder.swinViT(dummy_input)[-1]
            encoder_dim = features.shape[1]

        # Regression head - outputs single value for age
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder_dim, 64),  # Assuming encoder_dim=768, this is ~49K params
            nn.ReLU(),
            nn.Dropout(0.3),  # Higher dropout for few-shot
            nn.Linear(64, 1)
        )

        # For tracking validation predictions (correlation calculation)
        self.val_predictions = []
        self.val_targets = []

        # Optionally freeze encoder
        if self.freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        """Freeze encoder weights"""
        print("Freezing encoder weights")
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self):
        """Unfreeze encoder weights"""
        print("Unfreezing encoder weights")
        for param in self.encoder.parameters():
            param.requires_grad = True

    def adapt_to_multichannel(self, new_channels: int = 4, strategy: str = "average"):
        """
        Adapt model from 1 channel to multiple channels by modifying first conv layer

        Args:
            new_channels: Number of input channels (e.g., 2 for T1w + T2w)
            strategy:
                - "average": Each channel gets 1/N of the original weights
                - "copy": Each channel gets full copy of original weights
                - "first": Only first channel gets weights, others initialized randomly
        """
        # Get the first conv layer from SwinUNETR
        first_conv = self.encoder.swinViT.patch_embed.proj

        # Check if already has the right number of channels
        if first_conv.in_channels == new_channels:
            print(f"Model already has {new_channels} input channels")
            return

        # Save old weights
        old_weight = first_conv.weight.data.clone()  # [out_ch, 1, d, h, w]

        # Create new conv layer with more input channels
        new_conv = nn.Conv3d(
            in_channels=new_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        with torch.no_grad():
            if strategy == "average":
                # Each channel gets 1/N of the pretrained weights
                for i in range(new_channels):
                    new_conv.weight.data[:, i:i+1, :, :, :] = old_weight / new_channels

            elif strategy == "copy":
                # Each channel gets a full copy
                for i in range(new_channels):
                    new_conv.weight.data[:, i:i+1, :, :, :] = old_weight

            elif strategy == "first":
                # Only first channel gets pretrained weights
                nn.init.kaiming_normal_(new_conv.weight.data)
                new_conv.weight.data[:, 0:1, :, :, :] = old_weight

            # Copy bias if exists
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()

        # Replace the layer
        self.encoder.swinViT.patch_embed.proj = new_conv

        # Update hyperparameters
        self.hparams.in_channels = new_channels
        self.in_channels = new_channels

        print(f"✓ Adapted model from {first_conv.in_channels} to {new_channels} channels using '{strategy}' strategy")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        strict: bool = False,
        in_channels: int = None,
        multichannel_strategy: str = "average",
        **kwargs
    ):
        """
        Load from checkpoint with custom handling for architecture mismatch.
        """
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get hyperparameters from the original model
        loaded_hparams = checkpoint.get('hyper_parameters', {})

        # Extract relevant hyperparameters
        img_size = loaded_hparams.get('img_size', (96, 96, 96))
        feature_size = loaded_hparams.get('feature_size', 24)

        # Update with new parameters
        kwargs.update({
            'img_size': img_size,
            'feature_size': feature_size,
            'in_channels': 1,  # Always start with 1 channel to match checkpoint
        })

        # Create new model instance
        model = cls(**kwargs)

        # Extract and load only encoder weights
        state_dict = checkpoint['state_dict']
        encoder_state_dict = {}

        for k, v in state_dict.items():
            # Only load encoder weights
            if k.startswith('encoder.') and not k.startswith('encoder_m.'):
                encoder_state_dict[k] = v

        # Load encoder weights
        missing_keys, unexpected_keys = model.load_state_dict(
            encoder_state_dict,
            strict=False
        )

        print(f"Loaded encoder weights from {checkpoint_path}")
        print(f"Missing keys: {len(missing_keys)} (expected: regressor weights)")
        print(f"Unexpected keys: {len(unexpected_keys)}")

        # Adapt to multi-channel if needed
        if in_channels and in_channels > 1:
            model.adapt_to_multichannel(new_channels=in_channels, strategy=multichannel_strategy)

        return model

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handle loading from different checkpoint types"""
        state_dict = checkpoint['state_dict']

        # Check if this is a ContrastiveTransformer checkpoint
        if 'encoder_m.swinViT.patch_embed.proj.weight' in state_dict:
            print("Detected ContrastiveTransformer checkpoint, filtering weights...")

            # Filter to only encoder weights
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('encoder.') and not k.startswith('encoder_m.'):
                    filtered_state_dict[k] = v

            checkpoint['state_dict'] = filtered_state_dict

    def forward(self, x):
        """Forward pass for inference"""
        print(x.shape)
        features = self.encoder.swinViT(x)[-1]
        age_pred = self.regressor(features)
        return age_pred.squeeze(-1)  # Return shape: [batch_size]

    def compute_loss(self, pred, target):
        """Compute loss based on specified loss type"""
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        elif self.loss_type == "mae":
            return F.l1_loss(pred, target)
        elif self.loss_type == "huber":
            return F.smooth_l1_loss(pred, target)
        elif self.loss_type == "combined":
            mse = F.mse_loss(pred, target)
            mae = F.l1_loss(pred, target)
            return self.loss_alpha * mse + (1 - self.loss_alpha) * mae
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def training_step(self, batch, batch_idx):
        images = batch['image']
        ages = batch['label'].float()  # Ensure float type for regression

        # Forward pass
        age_pred = self(images)
        loss = self.compute_loss(age_pred, ages)

        # Calculate metrics
        mae = F.l1_loss(age_pred, ages)
        mse = F.mse_loss(age_pred, ages)
        rmse = torch.sqrt(mse)

        # Logging
        self.log_dict({
            'train/loss': loss,
            'train/mae': mae,
            'train/mse': mse,
            'train/rmse': rmse,
        }, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        ages = batch['label'].float()

        # Forward pass
        age_pred = self(images).squeeze(0)
        loss = self.compute_loss(age_pred, ages.float())

        # Calculate metrics
        mae = F.l1_loss(age_pred, ages)
        mse = F.mse_loss(age_pred, ages)
        rmse = torch.sqrt(mse)

        # Store predictions for epoch-level correlation
        self.val_predictions.extend(age_pred.detach().cpu().float().numpy())
        self.val_targets.extend(ages.detach().cpu().float().numpy())


        # Logging
        self.log_dict({
            'val/loss': loss,
            'val/mae': mae,
            'val/mse': mse,
            'val/rmse': rmse,
        }, prog_bar=True, on_epoch=True)

        return {'val_loss': loss, 'val_mae': mae}

    # def on_validation_epoch_end(self):
    #     """Calculate correlation coefficient at end of validation epoch"""
    #     if len(self.val_predictions) > 0:
    #         # Calculate Pearson correlation
    #         predictions = np.array(self.val_predictions)
    #         targets = np.array(self.val_targets)

    #         if len(predictions) > 1:  # Need at least 2 samples for correlation
    #             corr, _ = pearsonr(predictions, targets)
    #             self.log('val/correlation', corr, prog_bar=True)

    #         # Clear lists for next epoch
    #         self.val_predictions.clear()
    #         self.val_targets.clear()

    def test_step(self, batch, batch_idx):
        images = batch['image']
        ages = batch['age'].float()

        # Forward pass
        age_pred = self(images)
        loss = self.compute_loss(age_pred, ages)

        # Calculate metrics
        mae = F.l1_loss(age_pred, ages)
        mse = F.mse_loss(age_pred, ages)
        rmse = torch.sqrt(mse)

        # Store for correlation
        if not hasattr(self, 'test_predictions'):
            self.test_predictions = []
            self.test_targets = []


        self.test_predictions.extend(age_pred.detach().cpu().float().numpy())
        self.test_targets.extend(ages.detach().cpu().float().numpy())

        # Logging
        self.log_dict({
            'test/loss': loss,
            'test/mae': mae,
            'test/mse': mse,
            'test/rmse': rmse,
        }, on_epoch=True)

        return {'test_loss': loss, 'test_mae': mae}

    def on_test_epoch_end(self):
        """Calculate final test correlation"""
        if hasattr(self, 'test_predictions') and len(self.test_predictions) > 0:
            predictions = np.array(self.test_predictions)
            targets = np.array(self.test_targets)

            if len(predictions) > 1:
                corr, _ = pearsonr(predictions, targets)
                self.log('test/correlation', corr)

                # Print final results
                mae = np.mean(np.abs(predictions - targets))
                print(f"\nFinal Test Results:")
                print(f"MAE (Absolute Error): {mae:.2f} years")
                print(f"Correlation Coefficient: {corr:.4f}")

    def configure_optimizers(self):
        # Only optimize regressor if encoder is frozen
        if self.freeze_encoder:
            params = self.regressor.parameters()
        else:
            params = self.parameters()

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=0.01
        )

        return {
            'optimizer': optimizer,
        }

from monai.inferers import SlidingWindowInferer  # <-- Import added
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn as nn


import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
import wandb

from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from torch.optim.lr_scheduler import LambdaLR

# IMPORTANT: Make sure this import path is correct for your project structure
# from your_project.models.contrastive_transformer import ContrastiveTransformer






class SegmentationFineTuner(pl.LightningModule):
    """
    Fine-tunes a pre-trained SwinUNETR using a shared-weight fusion
    architecture, as seen in the user's RegressionFinetuner3.
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int, # Number of modalities
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        learning_rate: float = 1e-4,
        freeze_encoder: bool = True,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        min_lr: float = 1e-6,
        sw_batch_size: int = 4,
        sw_overlap: float = 0.5,
        log_image_frequency: int = 5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # We only need one SwinUNETR model, which will be shared across modalities
        self.encoder = SwinUNETR(
            in_channels=1, # The shared encoder sees one modality at a time
            out_channels=self.hparams.num_classes,
            feature_size=self.hparams.feature_size,
            use_checkpoint=False,
            use_v2=True,
        )

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.sliding_window_inferer = SlidingWindowInferer(
            roi_size=self.hparams.img_size, sw_batch_size=self.hparams.sw_batch_size,
            overlap=self.hparams.sw_overlap, mode="gaussian",
        )

        if self.hparams.freeze_encoder:
            self._freeze_backbone()

    @classmethod
    def load_from_pretrained(
        cls,
        pretrained_checkpoint_path: str,
        num_classes: int,
        in_channels: int,
        **kwargs
    ):
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(pretrained_checkpoint_path)
        finetuner_hparams = pretrain_model.hparams
        finetuner_hparams.update(kwargs)
        finetuner_hparams['num_classes'] = num_classes
        finetuner_hparams['in_channels'] = in_channels

        model = cls(**finetuner_hparams)
        print(f"\nShared-weight fusion model instantiated for {in_channels} modalities.")

        src_dict = pretrain_model.encoder.state_dict()
        src_dict.pop('out.conv.conv.weight', None)
        src_dict.pop('out.conv.conv.bias', None)

        msg = model.encoder.load_state_dict(src_dict, strict=False)
        print(f"✓ Pre-trained encoder loaded successfully.")
        print(f"  Missing keys: {len(msg.missing_keys)}")
        print(f"  Unexpected keys: {len(msg.unexpected_keys)}")

        return model

    def _freeze_backbone(self):
        """Freezes only the SwinViT part of the shared encoder."""
        print("Freezing SwinViT backbone weights...")
        for param in self.encoder.swinViT.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Implements the 'pack-process-revert-fuse' strategy.
        """
        # Input shape: [B, C, D, H, W]
        B, C, D, H, W = x.shape

        # 1. Pack modalities into the batch dimension
        # Reshaped to: [B * C, 1, D, H, W]
        x_reshaped = x.view(B * C, 1, D, H, W)

        # 2. Process with the shared encoder
        # This gets the raw logits from the full encoder-decoder path
        # Output shape: [B * C, Num_Classes, D, H, W]
        logits_reshaped = self.encoder(x_reshaped)

        # 3. Revert and Fuse
        # Reshape back to: [B, C, Num_Classes, D, H, W]
        _, Num_Classes, D_out, H_out, W_out = logits_reshaped.shape
        logits_per_modality = logits_reshaped.view(B, C, Num_Classes, D_out, H_out, W_out)

        # Fuse by averaging the logits from each modality's path
        # Output shape: [B, Num_Classes, D, H, W]
        fused_logits = logits_per_modality.mean(dim=1)

        return fused_logits

    def configure_optimizers(self):
        print("--- Configuring Optimizer ---")
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        print(f"Found {len(trainable_params)} trainable parameter tensors.")

        optimizer = torch.optim.AdamW(trainable_params, lr=self.hparams.learning_rate, weight_decay=0.01)

        def lr_lambda(current_step: int):
            num_training_steps = self.trainer.estimated_stepping_batches
            num_warmup_steps = int(num_training_steps * self.hparams.warmup_epochs / self.hparams.max_epochs)
            if current_step < num_warmup_steps: return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (1 - self.hparams.min_lr / self.hparams.learning_rate) * cosine_decay + self.hparams.min_lr / self.hparams.learning_rate
        scheduler = LambdaLR(optimizer, lr_lambda)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    # ... training_step, validation_step, etc. remain the same ...
    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.loss_function(outputs, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        # The inferer is passed the whole model ('self'), and its custom
        # forward pass will handle the fusion correctly.
        outputs = self.sliding_window_inferer(inputs=images, network=self)
        loss = self.loss_function(outputs, labels)
        post_pred = torch.argmax(outputs, dim=1, keepdim=True)
        self.dice_metric(y_pred=post_pred, y=labels)
        self.log('val/loss', loss, on_epoch=True)
        if self.current_epoch % self.hparams.log_image_frequency == 0:
            self._log_validation_images(batch, outputs, batch_idx)
        return loss

    def _log_validation_images(self, batch, outputs, batch_idx):
        if batch_idx > 0 or not hasattr(self, 'trainer') or self.trainer.global_rank != 0: return
        if not self.logger or not self.logger.experiment: return
        img, label = batch['image'][0].cpu().numpy(), batch['label'][0].squeeze().cpu().numpy()
        pred = torch.argmax(outputs[0], dim=0).cpu().numpy()
        vis_img = img[0] if img.ndim > 3 else img
        slice_idx_z, slice_idx_y, slice_idx_x = (
            np.argmax(np.sum(label, axis=(1, 2))),
            np.argmax(np.sum(label, axis=(0, 2))),
            np.argmax(np.sum(label, axis=(0, 1)))
        )
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Epoch {self.current_epoch} - Sample 0", fontsize=16)
        views = [
            ("Axial", vis_img[slice_idx_z, :, :], label[slice_idx_z, :, :], pred[slice_idx_z, :, :]),
            ("Coronal", vis_img[:, slice_idx_y, :], label[:, slice_idx_y, :], pred[:, slice_idx_y, :]),
            ("Sagittal", vis_img[:, :, slice_idx_x], label[:, :, slice_idx_x], pred[:, :, slice_idx_x]),
        ]
        for i, (title, img_slice, lbl_slice, pred_slice) in enumerate(views):
            axes[i].imshow(np.rot90(img_slice), cmap="gray")
            if np.any(lbl_slice): axes[i].contour(np.rot90(lbl_slice), colors='yellow', linewidths=0.8, alpha=0.9)
            if np.any(pred_slice): axes[i].contour(np.rot90(pred_slice), colors='red', linewidths=0.8, alpha=0.9)
            axes[i].set_title(title); axes[i].axis('off')
        self.logger.experiment.log({"Validation/Prediction vs Label": wandb.Image(fig)})
        plt.close(fig)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking: return
        val_dice = self.dice_metric.aggregate().item()
        self.log('val/dice', val_dice, prog_bar=True)
        self.dice_metric.reset()

    def test_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self.sliding_window_inferer(inputs=images, network=self)
        post_pred = torch.argmax(outputs, dim=1, keepdim=True)
        self.dice_metric_test(y_pred=post_pred, y=labels)

    def on_test_epoch_end(self):
        test_dice = self.dice_metric_test.aggregate().item()
        self.log('test/dice', test_dice)
        self.dice_metric_test.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch['image']
        return self.sliding_window_inferer(inputs=images, network=self)


import torch.nn.functional as F
import pytorch_lightning as pl
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple

# Assuming these imports from your project
from monai.networks.nets import SwinUNETR
from torchmetrics.classification import BinaryAccuracy, AUROC


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple

# --- Make sure your imports are correct ---
from monai.networks.nets import SwinUNETR
from torchmetrics.classification import BinaryAccuracy, AUROC
class ClassificationFinetuner2(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        feature_size: int = 24,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Frozen encoder
        self.encoder = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_v2=True,
        )

        # Determine dimensions for each layer output
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *img_size)
            all_features = self.encoder.swinViT(dummy)
            # Typically: [48, 96, 192, 384, 768] channels for different layers
            feature_dims = [f.shape[1] for f in all_features]

        # Separate pooling for each scale
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool3d(1) for _ in range(5)
        ])

        # Projection to common dimension for each scale
        common_dim = 64

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, common_dim),
                nn.LayerNorm(common_dim),  # Works with ANY batch size
                nn.ReLU()
            ) for dim in feature_dims
        ])

        self.classifier_head = nn.Sequential(
            nn.Linear(in_channels * 5 * common_dim, 128),
            nn.LayerNorm(128),  # Instead of BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self._freeze_encoder()

        # Metrics
        self.val_acc = BinaryAccuracy()
        self.val_auroc = AUROC(task="binary")

    def forward(self, x):
        B, C = x.shape[0], x.shape[1]
        x_reshaped = x.view(B * C, 1, *x.shape[2:])

        # Get ALL layer features
        all_features = self.encoder.swinViT(x_reshaped)  # List of 5 feature maps

        # Pool and project each scale
        pooled_features = []
        for i, features in enumerate(all_features):
            pooled = self.pools[i](features)  # [B*C, dim, 1, 1, 1]
            pooled = pooled.view(B * C, -1)   # [B*C, dim]
            projected = self.projections[i](pooled)  # [B*C, 64]
            pooled_features.append(projected)

        # Concatenate all scales
        multi_scale = torch.cat(pooled_features, dim=1)  # [B*C, 5*64]
        multi_scale = multi_scale.view(B, C * 5 * 64)    # [B, C*5*64]

        # Classify
        logits = self.classifier_head(multi_scale)
        return logits.squeeze(-1)

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
    def training_step(self, batch, batch_idx):
        # This part remains the same
        images, labels = batch['image'], batch['label'].float()
        logits = self(images)
        labels = labels.view(-1).float().to(logits.device)
        # print("logits:", logits.shape, "labels:", labels.shape)

        # print("logits:", logits)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_after_backward(self):
        # grads exist now
        g = [p.grad is not None and torch.isfinite(p.grad).all() for p in self.classifier_head.parameters()]
        self.log("dbg/cls_head_has_grads", float(all(g)), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label'].float()

        logits = self(images)
        labels = labels.view(-1).float().to(logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        probs = torch.sigmoid(logits)
        self.val_acc.update(probs, labels)
        self.val_auroc.update(probs, labels)
        self.log_dict({'val/loss': loss, 'val_acc': self.val_acc, 'val_auroc': self.val_auroc}, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        trainable_params = list(self.projections.parameters()) + \
                        list(self.classifier_head.parameters())

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        # Restarts every 50 epochs, with increasing periods
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,  # First restart after 50 epochs
            T_mult=2,  # Double the period after each restart (50, 100, 150)
            eta_min=1e-7
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }


    @classmethod
    def load_from_pretrained(
        cls,
        checkpoint_path: str,
        num_classes: int,
        in_channels: int,
        **kwargs
    ):
        """
        Loads only the SwinViT backbone weights from a ContrastiveTransformer checkpoint.
        """
        if not num_classes:
            raise ValueError("num_classes must be specified for fine-tuning.")

        # Load pretrained ContrastiveTransformer
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(checkpoint_path)

        finetuner_hparams = pretrain_model.hparams
        finetuner_hparams.update(kwargs)
        finetuner_hparams['num_classes'] = num_classes
        finetuner_hparams['in_channels'] = in_channels

        # Create finetuner with fresh weights
        model = cls(**finetuner_hparams)

        # --- Extract just swinViT weights ---
        src_dict = pretrain_model.encoder.swinViT.state_dict()
        dst_dict = model.encoder.swinViT.state_dict()

        # Keep only matching keys with identical shape
        filtered = {k: v for k, v in src_dict.items() if k in dst_dict and v.shape == dst_dict[k].shape}

        # Load into finetuner backbone
        msg = model.encoder.swinViT.load_state_dict(filtered, strict=False)

        print(f"\n✓ Loaded {len(filtered)} swinViT tensors from {checkpoint_path}")
        print(f"   Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}\n")

        return model




class RegressionFinetuner2(pl.LightningModule):
    """
    Implements multi-modal regression with multi-scale feature fusion.
    This version has INCREASED CAPACITY in the projection and regression heads
    to help learn more complex patterns.
    """
    def __init__(
        self,
        in_channels: int,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        learning_rate: float = 1e-4,
        freeze_encoder: bool = True,
        dropout_rate: float = 0.3,
        max_epochs: int = 100,
        min_lr: float = 1e-6,
        loss_type: str = "mae",
        loss_alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.register_buffer("target_min", torch.tensor(18.0))
        self.register_buffer("target_max", torch.tensor(110.0))

        self.encoder = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=self.hparams.feature_size,
            use_checkpoint=True,
            use_v2=True,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, *img_size)
            all_features = self.encoder.swinViT(dummy)
            self.feature_dims = [f.shape[1] for f in all_features]

        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool3d(1) for _ in range(5)
        ])

        # --- INCREASED CAPACITY 1 ---
        # Increased common_dim from 32 to 64
        common_dim = 64
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, common_dim),
                nn.LayerNorm(common_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.5)
            ) for dim in self.feature_dims
        ])

        # --- INCREASED CAPACITY 2 ---
        # Widened and deepened the regression head
        self.regression_head = nn.Sequential(
            nn.Linear(in_channels * 5 * common_dim, 128), # Widened from 64
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout_rate),
            nn.Linear(128, 128), # Added a second hidden layer
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        if self.hparams.freeze_encoder:
            self._freeze_encoder()

        self.val_corr = PearsonCorrCoef()

    def _freeze_encoder(self):
        print("Freezing encoder weights and setting to .eval().")
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.target_min) / (self.target_max - self.target_min)

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.target_max - self.target_min) + self.target_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[0], x.shape[1]
        x_reshaped = x.view(B * C, 1, *x.shape[2:])

        all_features = self.encoder.swinViT(x_reshaped)

        pooled_features = []
        for i, features in enumerate(all_features):
            pooled = self.pools[i](features)
            pooled = pooled.view(B * C, -1)
            projected = self.projections[i](pooled)
            pooled_features.append(projected)

        # Note: The input dimension to the head changes with common_dim
        multi_scale = torch.cat(pooled_features, dim=1)
        multi_scale = multi_scale.view(B, C * 5 * self.projections[0][0].out_features)

        output = self.regression_head(multi_scale)
        return output.squeeze(-1)

    def compute_loss(self, pred_normalized, target_normalized):
        loss_type = self.hparams.loss_type
        if loss_type == "mse":
            return F.mse_loss(pred_normalized, target_normalized)
        elif loss_type == "mae":
            return F.l1_loss(pred_normalized, target_normalized)
        elif loss_type == "huber":
            return F.smooth_l1_loss(pred_normalized, target_normalized)
        elif loss_type == "combined":
            mse = F.mse_loss(pred_normalized, target_normalized)
            mae = F.l1_loss(pred_normalized, target_normalized)
            return self.hparams.loss_alpha * mse + (1 - self.hparams.loss_alpha) * mae
        raise ValueError(f"Unknown loss type: {loss_type}")

    def training_step(self, batch, batch_idx):
        images, targets_original = batch['image'], batch['label'].float()
        targets_original = targets_original.view(-1)

        targets_normalized = self._normalize(targets_original)
        preds_normalized = self(images)
        loss = self.compute_loss(preds_normalized, targets_normalized)

        mae_original = F.l1_loss(self._unnormalize(preds_normalized.detach()), targets_original)

        self.log_dict({'train/loss': loss, 'train/mae_original': mae_original}, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets_original = batch['image'], batch['label'].float()
        targets_original = targets_original.view(-1)

        preds_normalized = self(images)
        targets_normalized = self._normalize(targets_original)
        loss = self.compute_loss(preds_normalized, targets_normalized)

        preds_original = self._unnormalize(preds_normalized.detach())

        mae_original = F.l1_loss(preds_original, targets_original)
        self.val_corr.update(preds_original, targets_original)

        self.log_dict({
            'val/loss': loss,
            'val/mae_original': mae_original,
            'val/correlation': self.val_corr,
        }, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        params = list(self.projections.parameters()) + list(self.regression_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate, weight_decay=0.01)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    @classmethod
    def load_from_pretrained(
        cls,
        checkpoint_path: str,
        in_channels: int,
        **kwargs
    ):
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(checkpoint_path)
        finetuner_hparams = pretrain_model.hparams
        finetuner_hparams.update(kwargs)
        finetuner_hparams['in_channels'] = in_channels

        model = cls(**finetuner_hparams)

        src_dict = pretrain_model.encoder.swinViT.state_dict()
        dst_dict = model.encoder.swinViT.state_dict()

        filtered_state_dict = {
            k: v for k, v in src_dict.items()
            if k in dst_dict and v.shape == dst_dict[k].shape
        }

        msg = model.encoder.swinViT.load_state_dict(filtered_state_dict, strict=False)

        print(f"\n✓ Loaded {len(filtered_state_dict)} swinViT tensors from {checkpoint_path}")
        print(f"  Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}\n")

        return model

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, Optional

from monai.networks.nets import SwinUNETR
from torchmetrics.regression import PearsonCorrCoef
from peft import get_peft_model, LoraConfig
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import PearsonCorrCoef
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from typing import Tuple
from peft import LoraConfig, get_peft_model



class RegressionFinetuner3(pl.LightningModule):
    """
    Implements a regression finetuner using a pre-trained SwinUNETR encoder.

    Key Features:
    - Loads weights from a self-supervised ContrastiveTransformer checkpoint.
    - Uses LoRA for parameter-efficient fine-tuning.
    - Normalizes regression targets using Z-score for robustness to outliers.
    - Uses MAE (L1Loss) as the objective function.
    - Logs a comparative distribution of train/validation labels at each epoch.
    """
    def __init__(
        self,
        in_channels: int,
        target_mean: float,
        target_std: float,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        lora_r: int = 128,
        lora_alpha: int = 16,
        learning_rate: float = 1e-4,
        dropout_rate: float = 0.1,
        max_epochs: int = 500,
        predict_uncertainty: bool = False,
        weight_decay: float = 1e-4,
        mixup_alpha: float = 0.4,          # Beta distribution α (0 disables MixUp)
        mixup_prob: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.all_train_targets_for_plot = None

        # Flags to ensure we only log one batch of images per run
        self.has_logged_train_batch = False
        self.has_logged_val_batch = False

        self.register_buffer("target_mean", torch.tensor(target_mean))
        self.register_buffer("target_std", torch.tensor(target_std))

        # 1. Base Encoder
        self.encoder = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=self.hparams.feature_size,
            use_checkpoint=True,
            use_v2=True,
        )

        # 2. Apply PEFT/LoRA Wrapper
        lora_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
        )
        self.encoder = get_peft_model(self.encoder, lora_config)
        # for name, param in self.encoder.base_model.model.named_parameters():
        #     param.requires_grad = True

        # 3. Multi-Scale Feature Extraction Setup
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.hparams.img_size)
            all_features = self.encoder.swinViT(dummy_input)
            self.feature_dims = [f.shape[1] for f in all_features]

        self.pools = nn.ModuleList([nn.AdaptiveAvgPool3d(1) for _ in range(5)])

        # 4. Projection & Regression Heads
        common_dim = 32
        self.projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, common_dim), nn.ReLU()) for dim in self.feature_dims]
        )

        output_dim = 2 if self.hparams.predict_uncertainty else 1
        # self.regression_head = nn.Sequential(
        #     nn.Linear(self.hparams.in_channels * 5 * common_dim, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Dropout(self.hparams.dropout_rate),
        #     nn.Linear(128, output_dim),
        #     # IMPORTANT: No Sigmoid, as Z-score targets are unbounded
        # )
        self.regression_head = nn.Sequential(
            nn.Linear(self.hparams.in_channels * 5 * common_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout_rate),
            nn.Linear(128, output_dim),
        )

        self.val_corr = PearsonCorrCoef()

        # Lists to store step outputs for logging
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.all_train_targets_for_plot = None

    def _log_batch_images(self, images, targets, preds, step_name: str):
        """Creates and logs a grid of image slices for a few samples in the batch."""
        if not self.logger:
            return

        # Move data to CPU and limit to a max of 4 samples
        images = images.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        num_samples = min(4, len(images))

        for i in range(num_samples):
            image = images[i]
            target = targets[i]
            pred = preds[i]

            # image shape is (C, D, H, W)
            num_modalities = image.shape[0]

            # Create a plot grid: one row per modality, 3 slices per row
            fig, axes = plt.subplots(num_modalities, 3, figsize=(12, 4 * num_modalities), squeeze=False)
            fig.suptitle(f"Sample {i} | True Age: {target:.1f} | Pred Age: {pred:.1f}", fontsize=16)

            for c in range(num_modalities):
                modality_vol = image[c]

                # Get central slices
                mid_d, mid_h, mid_w = [s // 2 for s in modality_vol.shape]
                axial_slice = modality_vol[mid_d, :, :]
                coronal_slice = modality_vol[:, mid_h, :]
                sagittal_slice = modality_vol[:, :, mid_w]

                # Plot axial slice
                axes[c, 0].imshow(axial_slice.T, cmap="bone", origin="lower")
                axes[c, 0].set_title(f"Modality {c} (Axial)")
                axes[c, 0].axis("off")

                # Plot coronal slice
                axes[c, 1].imshow(coronal_slice.T, cmap="bone", origin="lower")
                axes[c, 1].set_title(f"Modality {c} (Coronal)")
                axes[c, 1].axis("off")

                # Plot sagittal slice
                axes[c, 2].imshow(sagittal_slice.T, cmap="bone", origin="lower")
                axes[c, 2].set_title(f"Modality {c} (Sagittal)")
                axes[c, 2].axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

            self.logger.experiment.log({
                f"{step_name}/batch_visualization_{i}": wandb.Image(fig)
            })
            plt.close(fig)


    @classmethod
    def load_from_pretrained(
        cls,
        checkpoint_path: str,
        in_channels: int,
        target_mean: float,
        target_std: float,
        **kwargs
    ):
        """
        Loads a pretrained ContrastiveTransformer, creates an instance of this
        finetuner, and transfers the encoder weights.
        """
        print(f"Loading pretrained model from: {checkpoint_path}")
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(checkpoint_path)

        finetuner_hparams = pretrain_model.hparams
        finetuner_hparams.update(kwargs)
        finetuner_hparams['in_channels'] = in_channels
        finetuner_hparams['target_mean'] = target_mean
        finetuner_hparams['target_std'] = target_std

        model = cls(**finetuner_hparams)
        print("\nFinetuner instantiated. Now transferring weights...")

        src_dict = pretrain_model.encoder.swinViT.state_dict()
        base_encoder = model.encoder.base_model.model
        dst_dict = base_encoder.swinViT.state_dict()

        filtered_state_dict = {
            k: v for k, v in src_dict.items()
            if k in dst_dict and v.shape == dst_dict[k].shape
        }

        msg = base_encoder.swinViT.load_state_dict(filtered_state_dict, strict=False)

        print(f"\n✓ Loaded {len(filtered_state_dict)} swinViT tensors from {checkpoint_path}")
        print(f"  Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")

        print("\nEncoder wrapped with LoRA. Trainable parameters:")
        model.encoder.print_trainable_parameters()

        return model

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Z-score normalization."""
        eps = 1e-6
        return (x - self.target_mean) / (self.target_std + eps)

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverses Z-score normalization."""
        return x * self.target_std + self.target_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        x_reshaped = x.view(B * C, 1, D, H, W)
        all_features = self.encoder.swinViT(x_reshaped)

        pooled_features = [
            self.projections[i](self.pools[i](features).view(B * C, -1))
            for i, features in enumerate(all_features)
        ]

        multi_scale = torch.cat(pooled_features, dim=1).view(B, -1)
        output = self.regression_head(multi_scale)

        return output.squeeze(-1) if not self.hparams.predict_uncertainty else output

    def compute_loss(self, pred, target):
        """Computes the Mean Absolute Error (L1 Loss)."""
        if self.hparams.predict_uncertainty:
            pred = pred[:, 0]
        return F.l1_loss(pred, target)

    def training_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label'].float().view(-1)

        # Keep an unmixed copy just for the epoch-0 preview grid
        images_for_logging = images
        targets_for_logging = targets

        # Normalize targets *before* MixUp (affine -> mixing before/after is equivalent)
        targets_normalized = self._normalize(targets)

        # >>> MixUp here <<<
        images, targets_normalized, lam = self._maybe_mixup(images, targets_normalized)

        preds = self(images)
        loss = self.compute_loss(preds, targets_normalized)

        self.log('train/loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        # Log a clean, readable preview (non-mixed) only once at epoch 0
        if self.current_epoch == 0 and not self.has_logged_train_batch:
            preds_for_log = self(self._maybe_mixup(images_for_logging, targets_normalized)[0])  # run forward on clean batch
            preds_mean_norm = preds_for_log[:, 0] if self.hparams.predict_uncertainty else preds_for_log
            preds_original = self._unnormalize(preds_mean_norm.detach())
            self._log_batch_images(images_for_logging, targets_for_logging, preds_original, "train")
            self.has_logged_train_batch = True

        # Store labels on CPU for epoch-end aggregation (use original, non-mixed)
        self.training_step_outputs.append(targets_for_logging.detach().cpu())

        return loss

    def on_train_epoch_end(self):
        """Aggregates training labels at the end of the training epoch."""
        if self.training_step_outputs:
            # Concatenate all targets and store for the validation plot
            self.all_train_targets_for_plot = torch.cat(self.training_step_outputs).numpy()
            self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label'].float().view(-1)
        preds = self(images)

        targets_normalized = self._normalize(targets)
        loss = self.compute_loss(preds, targets_normalized)

        preds_mean_normalized = preds[:, 0] if self.hparams.predict_uncertainty else preds
        preds_original = self._unnormalize(preds_mean_normalized.detach())
        mae = F.l1_loss(preds_original, targets)
        self.val_corr.update(preds_original, targets)

        log_dict = {
            'val/loss': loss,
            'val/mae': mae,
            'val/correlation': self.val_corr,
        }

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)

        if self.current_epoch == 0 and not self.has_logged_val_batch:
            self._log_batch_images(images, targets, preds_original, "validation")
            self.has_logged_val_batch = True

        self.validation_step_outputs.append({'preds': preds_original, 'targets': targets})

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking and self.validation_step_outputs:
            if self.logger and self.trainer.global_rank == 0:
                preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu().numpy()
                targets = torch.cat([x['targets'] for x in self.validation_step_outputs]).cpu().numpy()

                fig, ax1 = plt.subplots(figsize=(12, 7))

                # Determine plot bounds from all available data
                min_val, max_val = targets.min(), targets.max()
                if self.all_train_targets_for_plot is not None:
                    min_val = min(min_val, self.all_train_targets_for_plot.min())
                    max_val = max(max_val, self.all_train_targets_for_plot.max())

                bins = np.linspace(min_val, max_val, num=50)

                # Plot validation ground truth
                ax1.hist(targets, bins=bins, alpha=0.6, color="blue", label="Ground Truth (Val)", density=True)

                # Overlay training ground truth as a step plot for clarity
                if self.all_train_targets_for_plot is not None:
                    ax1.hist(self.all_train_targets_for_plot, bins=bins, alpha=0.8, histtype='step',
                             linewidth=1.5, color="green", label="Ground Truth (Train)", density=True)

                # Plot predictions
                ax1.hist(preds, bins=bins, alpha=0.5, color="red", label="Predictions (Val)", density=True)

                ax1.set_title(f"Label Distributions & Predictions (Epoch {self.current_epoch})")
                ax1.set_xlabel("Value"); ax1.set_ylabel("Density")
                ax1.legend(); ax1.grid(True, alpha=0.3)

                self.logger.experiment.log({
                    "validation/prediction_distribution": wandb.Image(fig)
                })
                plt.close(fig)

            # Clear stored data for the next epoch
            self.validation_step_outputs.clear()
            self.all_train_targets_for_plot = None

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

    def _maybe_mixup(self, images: torch.Tensor, targets_norm: torch.Tensor):
        """
        Applies MixUp to 3D images and normalized regression targets with prob p.
        Returns (images, targets_norm, lam) where lam is the mixing coefficient used.
        """
        alpha = float(self.hparams.mixup_alpha)
        p = float(self.hparams.mixup_prob)
        if (not self.training) or alpha <= 0.0 or torch.rand(1, device=images.device).item() > p:
            return images, targets_norm, None

        # Sample lam ~ Beta(alpha, alpha); enforce lam >= 0.5 for symmetry (optional)
        lam = torch.distributions.Beta(alpha, alpha).sample().to(images.device)
        lam = torch.maximum(lam, 1.0 - lam)

        B = images.size(0)
        index = torch.randperm(B, device=images.device)

        mixed_images = lam * images + (1.0 - lam) * images[index]
        mixed_targets_norm = lam * targets_norm + (1.0 - lam) * targets_norm[index]

        return mixed_images, mixed_targets_norm, lam


import numpy as np
class RegressionHierarchicalFinetuner(pl.LightningModule):
    """
    Implements a hierarchical regression finetuner that combines:
    1. RegressionFinetuner3's SwinUNETR encoder for local (high-res) features
    2. BaseSupervisedModel's network encoder for global (low-res) features

    Key Features:
    - Dual-encoder architecture for multi-scale feature fusion
    - Uses LoRA for parameter-efficient fine-tuning on SwinUNETR
    - Compatible with HierarchicalDataset that provides local/global views
    - Normalizes regression targets using Z-score for robustness
    - Supports loading pretrained weights for both encoders separately

    Methods for loading pretrained weights:
    - load_from_pretrained(): Main class method that orchestrates loading both encoders
    - load_from_local_pretrained(): Load pretrained ContrastiveTransformer for local encoder
    - load_from_global_pretrained(): Load pretrained weights for global encoder
    """
    def __init__(
        self,
        in_channels: int,
        target_mean: float,
        target_std: float,
        global_config: dict,  # Config for global encoder (BaseSupervisedModel)
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        lora_r: int = 128,
        lora_alpha: int = 16,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.1,
        max_epochs: int = 500,
        predict_uncertainty: bool = False,
        weight_decay: float = 0.01,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.5,
        freeze_global_encoder: bool = True,
        global_checkpoint: str = None,  # Path to pretrained global encoder checkpoint
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.all_train_targets_for_plot = None

        # Flags to ensure we only log one batch of images per run
        self.has_logged_train_batch = False
        self.has_logged_val_batch = False

        self.register_buffer("target_mean", torch.tensor(target_mean))
        self.register_buffer("target_std", torch.tensor(target_std))

        # 1. Local Encoder (SwinUNETR with LoRA) - from RegressionFinetuner3
        self.local_encoder = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=self.hparams.feature_size,
            use_checkpoint=True,
            use_v2=True,
        )

        # Apply PEFT/LoRA Wrapper to local encoder
        lora_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
        )
        self.local_encoder = get_peft_model(self.local_encoder, lora_config)

        # 2. Global Encoder - from BaseSupervisedModel
        from models import networks
        model_factory = getattr(networks, global_config["model_name"])

        # Create the global encoder using the factory function
        self.global_encoder = model_factory(
            input_channels=global_config["num_modalities"],
            output_channels=global_config["num_classes"],
            mode="regression",  # Ensure regression mode
        )

        # Optionally freeze global encoder
        if self.hparams.freeze_global_encoder:
            for param in self.global_encoder.parameters():
                param.requires_grad = False

        # 3. Multi-Scale Feature Extraction Setup for local encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.hparams.img_size)
            local_features = self.local_encoder.swinViT(dummy_input)
            self.local_feature_dims = [f.shape[1] for f in local_features]

        self.local_pools = nn.ModuleList([nn.AdaptiveAvgPool3d(1) for _ in range(5)])

        # 4. Global feature extraction
        with torch.no_grad():
            dummy_global = torch.zeros(1, global_config["num_modalities"], *self.hparams.img_size)
            global_features = self.global_encoder(dummy_global)
            # Handle different output types from global encoder
            if isinstance(global_features, torch.Tensor):
                self.global_feature_dim = global_features.shape[1] if len(global_features.shape) > 1 else global_features.numel()
            else:
                # If it's a list or tuple, take the first element
                self.global_feature_dim = global_features[0].shape[1] if len(global_features[0].shape) > 1 else global_features[0].numel()

        # Global feature pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # 5. Projection & Fusion
        common_dim = 32

        # Local projections
        self.local_projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, common_dim), nn.ReLU()) for dim in self.local_feature_dims]
        )

        # Global projection
        self.global_projection = nn.Sequential(
            nn.Linear(self.global_feature_dim, common_dim),
            nn.ReLU()
        )

        # 6. Final regression head
        # Local: num_modalities * 5 scales * common_dim (each modality processed separately)
        # Global: common_dim
        local_channels = in_channels  # Account for all input modalities
        total_features = local_channels * 5 * common_dim + common_dim

        output_dim = 2 if self.hparams.predict_uncertainty else 1
        self.regression_head = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout_rate),
            nn.Linear(64, output_dim),
        )

        self.val_corr = PearsonCorrCoef()

        # Lists to store step outputs for logging
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def load_from_local_pretrained(self, local_checkpoint: str):
        """
        Load pretrained weights for the local encoder from a ContrastiveTransformer checkpoint.

        Args:
            local_checkpoint: Path to the ContrastiveTransformer checkpoint
        """
        print(f"Loading pretrained local encoder from: {local_checkpoint}")
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(local_checkpoint)

        # Transfer weights to local encoder
        src_dict = pretrain_model.encoder.swinViT.state_dict()
        base_encoder = self.local_encoder.base_model.model
        dst_dict = base_encoder.swinViT.state_dict()

        filtered_state_dict = {
            k: v for k, v in src_dict.items()
            if k in dst_dict and v.shape == dst_dict[k].shape
        }

        msg = base_encoder.swinViT.load_state_dict(filtered_state_dict, strict=False)

        print(f"✓ Loaded {len(filtered_state_dict)} swinViT tensors from {local_checkpoint}")
        print(f"  Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")

        print("Local encoder wrapped with LoRA. Trainable parameters:")
        self.local_encoder.print_trainable_parameters()

    def load_from_global_pretrained(self, global_checkpoint: str):
        """
        Load pretrained weights for the global encoder from a checkpoint.

        Args:
            global_checkpoint: Path to the global encoder checkpoint
        """
        try:
            print(f"Loading pretrained global encoder from: {global_checkpoint}")

            # Load checkpoint
            checkpoint = torch.load(global_checkpoint, map_location='cpu')

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Filter global encoder weights (typically starts with 'model.' or similar)
            global_encoder_weights = {}
            for key, value in state_dict.items():
                # Remove common prefixes to match the global encoder structure
                clean_key = key
                if key.startswith('model.'):
                    clean_key = key[6:]  # Remove 'model.'
                elif key.startswith('encoder.'):
                    clean_key = key[8:]  # Remove 'encoder.'

                global_encoder_weights[clean_key] = value

            # Load weights into global encoder (non-strict to handle architectural differences)
            missing_keys, unexpected_keys = self.global_encoder.load_state_dict(
                global_encoder_weights, strict=False
            )

            print(f"✓ Loaded global encoder weights from {global_checkpoint}")
            print(f"  Missing keys: {len(missing_keys)} | Unexpected keys: {len(unexpected_keys)}")

        except Exception as e:
            print(f"⚠️  Warning: Could not load global encoder weights from {global_checkpoint}")
            print(f"   Error: {e}")
            print("   Continuing with randomly initialized global encoder...")

    def _load_global_encoder_weights(self, checkpoint_path: str):
        """Load pretrained weights for the global encoder from a checkpoint."""
        # Delegate to the new method for backward compatibility
        self.load_from_global_pretrained(checkpoint_path)

    def _log_batch_images(self, local_images, global_images, targets, preds, step_name: str):
        """Creates and logs a grid of image slices for both local and global views."""
        if not self.logger:
            return

        # Move data to CPU and limit to a max of 2 samples to avoid cluttering
        local_images = local_images.detach().cpu().numpy()
        global_images = global_images.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        num_samples = min(2, len(local_images))

        for i in range(num_samples):
            local_img = local_images[i]
            global_img = global_images[i]
            target = targets[i]
            pred = preds[i]

            # Create a plot grid: rows for modalities, cols for [local_axial, local_coronal, global_axial, global_coronal]
            num_modalities = local_img.shape[0]
            fig, axes = plt.subplots(num_modalities, 4, figsize=(16, 4 * num_modalities), squeeze=False)
            fig.suptitle(f"Sample {i} | True: {target:.1f} | Pred: {pred:.1f} | Local vs Global", fontsize=16)

            for c in range(num_modalities):
                # Local views
                local_vol = local_img[c]
                mid_d, mid_h, mid_w = [s // 2 for s in local_vol.shape]

                axes[c, 0].imshow(local_vol[mid_d, :, :].T, cmap="bone", origin="lower")
                axes[c, 0].set_title(f"Local Mod {c} (Axial)")
                axes[c, 0].axis("off")

                axes[c, 1].imshow(local_vol[:, mid_h, :].T, cmap="bone", origin="lower")
                axes[c, 1].set_title(f"Local Mod {c} (Coronal)")
                axes[c, 1].axis("off")

                # Global views
                global_vol = global_img[c]
                mid_d_g, mid_h_g, mid_w_g = [s // 2 for s in global_vol.shape]

                axes[c, 2].imshow(global_vol[mid_d_g, :, :].T, cmap="bone", origin="lower")
                axes[c, 2].set_title(f"Global Mod {c} (Axial)")
                axes[c, 2].axis("off")

                axes[c, 3].imshow(global_vol[:, mid_h_g, :].T, cmap="bone", origin="lower")
                axes[c, 3].set_title(f"Global Mod {c} (Coronal)")
                axes[c, 3].axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            self.logger.experiment.log({
                f"{step_name}/hierarchical_batch_visualization_{i}": wandb.Image(fig)
            })
            plt.close(fig)

    @classmethod
    def load_from_pretrained(
        cls,
        local_checkpoint: str,
        in_channels: int,
        target_mean: float,
        target_std: float,
        global_config: dict,
        global_checkpoint: str = None,
        **kwargs
    ):
        """
        Creates an instance of the hierarchical finetuner and loads pretrained weights
        for both local and global encoders.

        Args:
            local_checkpoint: Path to pretrained ContrastiveTransformer for local encoder
            in_channels: Number of input channels
            target_mean: Target mean for normalization
            target_std: Target standard deviation for normalization
            global_config: Configuration dict for global encoder
            global_checkpoint: Path to pretrained global encoder checkpoint (optional)
            **kwargs: Additional arguments for model initialization

        Returns:
            RegressionHierarchicalFinetuner: Model with loaded pretrained weights
        """
        print("Creating hierarchical finetuner with pretrained encoders...")

        # Load ContrastiveTransformer to get hyperparameters
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(local_checkpoint)

        # Prepare model hyperparameters
        finetuner_hparams = pretrain_model.hparams.copy()
        finetuner_hparams.update(kwargs)
        finetuner_hparams['in_channels'] = in_channels
        finetuner_hparams['target_mean'] = target_mean
        finetuner_hparams['target_std'] = target_std
        finetuner_hparams['global_config'] = global_config
        # Note: global_checkpoint is not stored in hparams since it's handled explicitly

        # Create model instance
        model = cls(**finetuner_hparams)

        print("\nHierarchical finetuner instantiated. Loading pretrained weights...")

        # Load local encoder weights
        model.load_from_local_pretrained(local_checkpoint)

        # Load global encoder weights if provided
        if global_checkpoint and os.path.exists(global_checkpoint):
            model.load_from_global_pretrained(global_checkpoint)
        else:
            print("No global checkpoint provided or file not found. Using randomly initialized global encoder.")

        print("\n✅ Hierarchical model ready with pretrained encoders!")
        return model

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Z-score normalization."""
        eps = 1e-6
        return (x - self.target_mean) / (self.target_std + eps)

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverses Z-score normalization."""
        return x * self.target_std + self.target_mean

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass through both local and global encoders.

        Args:
            batch: Dict containing 'local' and 'global' image tensors

        Returns:
            torch.Tensor: Regression output
        """
        local_x = batch['local']  # High-res local patches
        global_x = batch['global']  # Low-res global view

        B, C, D, H, W = local_x.shape

        # 1. Local encoder (SwinUNETR with LoRA)
        local_x_reshaped = local_x.view(B * C, 1, D, H, W)
        local_features = self.local_encoder.swinViT(local_x_reshaped)

        # Pool and project local features
        local_pooled_features = [
            self.local_projections[i](self.local_pools[i](features).view(B * C, -1))
            for i, features in enumerate(local_features)
        ]
        local_multi_scale = torch.cat(local_pooled_features, dim=1).view(B, -1)

        # 2. Global encoder
        global_features = self.global_encoder(global_x)

        # Handle different output types from global encoder
        if isinstance(global_features, (list, tuple)):
            global_features = global_features[0]  # Take first output if multiple

        # Global adaptive pooling and projection
        if len(global_features.shape) > 2:
            global_pooled = self.global_pool(global_features).view(B, -1)
        else:
            global_pooled = global_features

        global_projected = self.global_projection(global_pooled)

        # 3. Concatenate local and global features
        combined_features = torch.cat([local_multi_scale, global_projected], dim=1)

        # 4. Final regression
        output = self.regression_head(combined_features)

        return output.squeeze(-1) if not self.hparams.predict_uncertainty else output

    def compute_loss(self, pred, target):
        """Computes the Mean Absolute Error (L1 Loss)."""
        if self.hparams.predict_uncertainty:
            pred = pred[:, 0]
        return F.l1_loss(pred, target)

    def training_step(self, batch, batch_idx):
        targets = batch['label'].float().view(-1)

        # Keep unmixed copies for logging
        batch_for_logging = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        targets_for_logging = targets.clone()

        # Normalize targets *before* MixUp
        targets_normalized = self._normalize(targets)

        # >>> MixUp here <<<
        batch, targets_normalized, lam = self._maybe_mixup(batch, targets_normalized)

        preds = self(batch)
        loss = self.compute_loss(preds, targets_normalized)

        self.log('train/loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        # Log visualization only once at epoch 0
        if self.current_epoch == 0 and not self.has_logged_train_batch:
            preds_for_log = self(batch_for_logging)
            preds_mean_norm = preds_for_log[:, 0] if self.hparams.predict_uncertainty else preds_for_log
            preds_original = self._unnormalize(preds_mean_norm.detach())
            self._log_batch_images(
                batch_for_logging['local'],
                batch_for_logging['global'],
                targets_for_logging,
                preds_original,
                "train"
            )
            self.has_logged_train_batch = True

        # Store labels for epoch-end aggregation
        self.training_step_outputs.append(targets_for_logging.detach().cpu())

        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch['label'].float().view(-1)
        preds = self(batch)

        targets_normalized = self._normalize(targets)
        loss = self.compute_loss(preds, targets_normalized)

        preds_mean_normalized = preds[:, 0] if self.hparams.predict_uncertainty else preds
        preds_original = self._unnormalize(preds_mean_normalized.detach())
        mae = F.l1_loss(preds_original, targets)
        self.val_corr.update(preds_original, targets)

        log_dict = {
            'val/loss': loss,
            'val/mae': mae,
            'val/correlation': self.val_corr,
        }

        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)

        if self.current_epoch == 0 and not self.has_logged_val_batch:
            self._log_batch_images(
                batch['local'],
                batch['global'],
                targets,
                preds_original,
                "validation"
            )
            self.has_logged_val_batch = True

        self.validation_step_outputs.append({'preds': preds_original, 'targets': targets})

    def on_train_epoch_end(self):
        """Aggregates training labels at the end of the training epoch."""
        if self.training_step_outputs:
            self.all_train_targets_for_plot = torch.cat(self.training_step_outputs).numpy()
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking and self.validation_step_outputs:
            if self.logger and self.trainer.global_rank == 0:
                preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu().numpy()
                targets = torch.cat([x['targets'] for x in self.validation_step_outputs]).cpu().numpy()

                fig, ax1 = plt.subplots(figsize=(12, 7))

                # Determine plot bounds
                min_val, max_val = targets.min(), targets.max()
                if self.all_train_targets_for_plot is not None:
                    min_val = min(min_val, self.all_train_targets_for_plot.min())
                    max_val = max(max_val, self.all_train_targets_for_plot.max())

                bins = np.linspace(min_val, max_val, num=50)

                # Plot distributions
                ax1.hist(targets, bins=bins, alpha=0.6, color="blue", label="Ground Truth (Val)", density=True)

                if self.all_train_targets_for_plot is not None:
                    ax1.hist(self.all_train_targets_for_plot, bins=bins, alpha=0.8, histtype='step',
                             linewidth=1.5, color="green", label="Ground Truth (Train)", density=True)

                ax1.hist(preds, bins=bins, alpha=0.5, color="red", label="Predictions (Val)", density=True)

                ax1.set_title(f"Hierarchical Model - Label Distributions & Predictions (Epoch {self.current_epoch})")
                ax1.set_xlabel("Value"); ax1.set_ylabel("Density")
                ax1.legend(); ax1.grid(True, alpha=0.3)

                self.logger.experiment.log({
                    "validation/hierarchical_prediction_distribution": wandb.Image(fig)
                })
                plt.close(fig)

            self.validation_step_outputs.clear()
            self.all_train_targets_for_plot = None

    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

    def _maybe_mixup(self, batch: dict, targets_norm: torch.Tensor):
        """
        Applies MixUp to both local and global images and normalized regression targets.
        """
        alpha = float(self.hparams.mixup_alpha)
        p = float(self.hparams.mixup_prob)
        if (not self.training) or alpha <= 0.0 or torch.rand(1, device=targets_norm.device).item() > p:
            return batch, targets_norm, None

        # Sample mixing coefficient
        lam = torch.distributions.Beta(alpha, alpha).sample().to(targets_norm.device)
        lam = torch.maximum(lam, 1.0 - lam)

        B = targets_norm.size(0)
        index = torch.randperm(B, device=targets_norm.device)

        # Mix both local and global images
        mixed_batch = {}
        for key in ['local', 'global']:
            if key in batch:
                mixed_batch[key] = lam * batch[key] + (1.0 - lam) * batch[key][index]
            else:
                mixed_batch[key] = batch[key]

        # Copy other keys as-is
        for key in batch:
            if key not in ['local', 'global']:
                mixed_batch[key] = batch[key]

        mixed_targets_norm = lam * targets_norm + (1.0 - lam) * targets_norm[index]

        return mixed_batch, mixed_targets_norm, lam

if __name__ == "__main__":
    model = ClassificationFineTuner.load_from_checkpoint(
        '/home/mg873uh/Projects_kb/checkpoints/contrastive_2gpu_1131/last.ckpt',
        num_classes=1,
        in_channels=4,
        multichannel_strategy='copy',
        freeze_encoder=True,
        learning_rate=1e-4,
        max_epochs=50
    )

