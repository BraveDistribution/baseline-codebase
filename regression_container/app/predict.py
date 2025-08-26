#!/usr/bin/env python
import argparse
import os
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.functional import sigmoid
from typing import List, Dict, Any, Tuple, Sequence
from monai.networks.nets.swin_unetr import SwinUNETR
import math
import nibabel as nib
import numpy as np

from yucca.functional.preprocessing import (
    preprocess_case_for_inference,
    reverse_preprocessing,
)

from torchmetrics.regression import PearsonCorrCoef

from yucca.modules.data.augmentation.transforms.cropping_and_padding import CropPad

import wandb
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, get_peft_model, LoraConfig

from pathlib import Path

def generate_random_mask(
    x: torch.Tensor,
    mask_ratio: float,
    patch_size: int,
    out_type: type = int,
):
    # assumes x is (B, C, H, W) or (B, C, H, W, Z)

    dim = len(x.shape) - 2
    assert dim in [2, 3]

    # check if all spatial dimensions are divisible by patch_size
    for i in range(2, len(x.shape)):
        assert x.shape[i] % patch_size == 0, f"Shape: {x.shape}, Patch size: {patch_size}, Dim {i} not divisible"

    mask = generate_1d_mask(x, mask_ratio, patch_size, out_type)
    mask = reshape_to_dim(mask, x.shape, patch_size)

    up_mask = upsample_mask(mask, patch_size)

    return up_mask


def generate_1d_mask(x: torch.Tensor, mask_ratio: float, patch_size: int, out_type: type):
    assert x.shape[1] in [1, 3], "Channel dim is not 1 or 3. Are you sure?"
    assert out_type in [int, bool]

    N = x.shape[0]
    # Calculate total number of patches by multiplying patches along each spatial dimension
    L = 1
    for i in range(2, len(x.shape)):
        L *= (x.shape[i] // patch_size)

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.randn(N, L, device=x.device)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 0 is keep 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    if out_type == bool:
        mask = mask.bool()  # (B, H * W)
    elif out_type == int:
        mask = mask.int()

    return mask  # (B, H * W) 0 or False is keep, 1 or True is remove


def reshape_to_dim(mask: torch.Tensor, original_shape: tuple, patch_size: int):
    dim = len(original_shape) - 2
    assert dim in [2, 3]
    assert len(mask.shape) == 2

    if dim == 2:
        h_patches = original_shape[2] // patch_size
        w_patches = original_shape[3] // patch_size
        return mask.reshape(-1, h_patches, w_patches)
    else:
        h_patches = original_shape[2] // patch_size
        w_patches = original_shape[3] // patch_size
        z_patches = original_shape[4] // patch_size
        return mask.reshape(-1, h_patches, w_patches, z_patches)


def upsample_mask(mask: torch.Tensor, scale: int):
    assert scale > 0
    assert len(mask.shape) in [3, 4]  # (B, H, W) or (B, H, W, Z)

    if len(mask.shape) == 3:
        mask = mask.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)  # (B, H * scale, W * scale)
    else:
        # (B, H * scale, W * scale, Z * scale)
        mask = mask.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2).repeat_interleave(scale, dim=3)

    return mask.unsqueeze(1)  # (B, C, H * scale, W * scale) or (B, C, H * scale, W * scale, Z * scale)


def random_mask(x, mask_ratio, mask_patch_size, mask_token=0):
    mask = generate_random_mask(x, mask_ratio, mask_patch_size, out_type=bool)
    assert isinstance(mask, torch.BoolTensor) or isinstance(
        mask, torch.cuda.BoolTensor
    ), mask.type()
    x[mask] = mask_token
    return x, mask

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
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.1,
        max_epochs: int = 500,
        predict_uncertainty: bool = False,
        weight_decay: float = 0.01,
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
        #     nn.Linear(self.hparams.in_channels * 5 * common_dim, 64),
        #     nn.LayerNorm(64),
        #     nn.ReLU(),
        #     nn.Dropout(self.hparams.dropout_rate),
        #     nn.Linear(64, output_dim),
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
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(checkpoint_path, strict=False)

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


def unnormalize(x: torch.Tensor, target_mean: float, target_std: float) -> torch.Tensor:
    """Reverses Z-score normalization."""
    return x * target_std + target_mean

def load_modalities(modality_paths: List[str]) -> List[nib.Nifti1Image]:
    """Load modality images from provided paths."""
    images = []
    for path in modality_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modality file not found: {path}")
        try:
            img = nib.load(path)
            images.append(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {path}: {str(e)}")

    return images

def save_output_txt(number: float | int, output_path: str):
    """Save a number (float or int) as plain text to a file."""

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if not output_path.endswith(".txt"):
        output_path = output_path + ".txt"

    with open(output_path, "w") as f:
        f.write(f"{number}")

def predict_from_config(
    modality_paths: List[str],
    predict_config: Dict[str, Any],
    reverse_preprocess: bool = False,
):
    """
    Run inference on input modality images using a task-specific configuration.

    Args:
        modality_paths: Paths to input modality images
        predict_config: Dictionary containing all the configuration parameters for prediction

    Returns:
        str: Path to saved prediction
    """
    # Load input images
    images = load_modalities(modality_paths)

    # Extract configuration parameters
    task_type = predict_config["task_type"]
    crop_to_nonzero = predict_config["crop_to_nonzero"]
    norm_op = predict_config["norm_op"]
    num_classes = predict_config["num_classes"]
    keep_aspect_ratio = predict_config.get("keep_aspect_ratio", True)
    patch_size = predict_config["patch_size"]
    model_path = predict_config["model_path"]

    # Define preprocessing parameters
    normalization_scheme = [norm_op] * len(modality_paths)
    target_spacing = [1.0, 1.0, 1.0]  # Isotropic 1mm spacing
    target_orientation = "RAS"

    test_image = images[0].get_fdata().astype(np.float32)
    test_path = Path("output/task3_test/test_image_clean_1.npy")
    np.save(test_path,test_image)


    # Apply preprocessing
    case_preprocessed, case_properties = preprocess_case_for_inference(
        crop_to_nonzero=crop_to_nonzero,
        images=images,
        intensities=None,  # Use default intensity normalization
        normalization_scheme=normalization_scheme,
        patch_size=patch_size,
        target_size=None,  # We use target_spacing instead
        target_spacing=target_spacing,
        target_orientation=target_orientation,
        allow_missing_modalities=False,
        keep_aspect_ratio=keep_aspect_ratio,
        transpose_forward=[0, 1, 2],  # Standard transpose order
    )

    x_np = case_preprocessed.squeeze(0).detach().numpy()

    test_path = Path("output/task3_test/test_image_1.npy")
    np.save(test_path,x_np)

    croppad = CropPad(patch_size=(96, 96, 96))
    out = croppad(
        packed_data_dict={"image": x_np},
        image_properties={"foreground_locations": []}
    )
    x_np = out["image"].astype(np.float32, copy=False)

    test_path = Path("output/task3_test/test_cropped_1.npy")
    np.save(test_path,x_np)

    case_preprocessed = torch.from_numpy(np.ascontiguousarray(x_np)).unsqueeze(0)

    # Load the model checkpoint directly with Lightning

    model = RegressionFinetuner3.load_from_pretrained(
            checkpoint_path=str(model_path),
            in_channels=2,
            target_min=18.0,
            target_max=120.0,
            feature_size=24,
            target_mean=61.87,
            target_std=15.089118845634706,
        )

    # Set model to evaluation mode
    model.eval()

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    case_preprocessed = case_preprocessed.to(device)

    # Run inference
    with torch.no_grad():
        # Set up sliding window parameters

        # Get prediction
        predictions = model(case_preprocessed)

    if reverse_preprocess:
        predictions_original, _ = reverse_preprocessing(
            crop_to_nonzero=crop_to_nonzero,
            images=predictions,
            image_properties=case_properties,
            n_classes=num_classes,
            transpose_forward=[0, 1, 2],
            transpose_backward=[0, 1, 2],
        )
        print(f"-- Prediction shape: {predictions_original.shape}")
        return predictions_original, images[0].affine
    else:
        print(f"-- Prediction shape: {predictions.shape}")
        return predictions, None

task3_config = {
    "task_name": "Task003_FOMO3",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("T1", "T2"),
    "norm_op": "volume_wise_znorm",
    "num_classes": 1,  # For regression, output dimension is 1
    "keep_aspect_ratio": True,
    "task_type": "regression",
    "label_extension": ".txt",
    "labels": {"regression": "Age"},  # Define as regression task
    "target_spacing": [1.0, 1.0, 1.0],
    "target_orientation": "RAS",
}

# Task-specific hardcoded configuration
predict_config = {
    # Import values from task_configs
    **task3_config,
    # Add inference-specific configs
    "model_path": "/app/weights/brano_full_regression.ckpt",  # Path to model (inside container!)
    "patch_size": (96, 96, 96),  # Patch size for inference
}


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on FOMO Task 1 (Infarct Detection)"
    )

    # Input and output paths using modality names from task config
    parser.add_argument(
        "--t1", type=str, required=True, help="Path to T1 image (NIfTI format)"
    )
    parser.add_argument(
        "--t2", type=str, required=True, help="Path to T2 image (NIfTI format)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for prediction"
    )

    # Parse arguments
    args = parser.parse_args()

    modality_paths = [args.t1, args.t2]
    output_path = args.output

    # Run prediction using the shared prediction logic
    predictions_original, _ = predict_from_config(
        modality_paths=modality_paths,
        predict_config=predict_config,
    )

    print(f"\n\n\n DATA PRED UNNORMALIZE: {predictions_original}, DATA PO: {int(unnormalize(predictions_original, target_mean=61.87, target_std=15.089118845634706))}")
    save_output_txt(int(unnormalize(predictions_original, target_mean=61.87, target_std=15.089118845634706)), output_path)


if __name__ == "__main__":
    main()