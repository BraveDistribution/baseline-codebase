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
from peft import LoraConfig, get_peft_model, get_peft_model

from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import SwinUNETR
from torch.optim.lr_scheduler import LambdaLR

from pathlib import Path

from monai.data import TestTimeAugmentation
from monai.transforms import Compose, RandFlipd, RandRotate90d, EnsureTyped

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

class SegmentationFineTuner(pl.LightningModule):
    """
    Fine-tunes a pre-trained SwinUNETR using a shared-weight fusion
    architecture, now upgraded with LoRA for parameter-efficient tuning.
    """
    def __init__(
        self,
        num_classes: int,
        in_channels: int, # Number of modalities
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        learning_rate: float = 1e-5,
        use_lora: bool = True, # Use LoRA instead of simple freezing
        lora_r: int = 8,
        lora_alpha: int = 16,
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
            use_checkpoint=False, # Checkpointing can interfere with PEFT
            use_v2=True,
        )

        # --- LoRA Integration ---
        if self.hparams.use_lora:
            print("--- Applying LoRA to SwinViT backbone ---")
            lora_config = LoraConfig(
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                target_modules=["qkv"], # Apply to query, key, value projections in attention
                lora_dropout=0.1,
                bias="none",
            )
            # Wrap the SwinViT part of the encoder. The rest of the model (CNN blocks,
            # decoder) is untouched and its trainability is determined by requires_grad.
            self.encoder.swinViT = get_peft_model(self.encoder.swinViT, lora_config)
            print("Trainable parameters with LoRA enabled:")
            self.encoder.swinViT.print_trainable_parameters()
            # By default, peft freezes the non-LoRA parts of the wrapped module.
            # We only need to ensure the decoder is trainable.
            for param in self.encoder.decoder1.parameters(): param.requires_grad = True
            for param in self.encoder.decoder2.parameters(): param.requires_grad = True
            for param in self.encoder.decoder3.parameters(): param.requires_grad = True
            for param in self.encoder.decoder4.parameters(): param.requires_grad = True
            for param in self.encoder.decoder5.parameters(): param.requires_grad = True
            for param in self.encoder.out.parameters(): param.requires_grad = True


        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.sliding_window_inferer = SlidingWindowInferer(
            roi_size=self.hparams.img_size, sw_batch_size=self.hparams.sw_batch_size,
            overlap=self.hparams.sw_overlap, mode="gaussian",
        )

    @classmethod
    def load_from_pretrained(
        cls,
        pretrained_checkpoint_path: str,
        num_classes: int,
        in_channels: int,
        **kwargs
    ):
        # NOTE: This assumes ContrastiveTransformer class is defined and accessible
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(pretrained_checkpoint_path)
        finetuner_hparams = {**pretrain_model.hparams, **kwargs}
        finetuner_hparams['num_classes'] = num_classes
        finetuner_hparams['in_channels'] = in_channels

        model = cls(**finetuner_hparams)
        print(f"\nShared-weight fusion model instantiated for {in_channels} modalities.")

        # Load weights, ignoring the final output layer of the pre-trained model
        src_dict = pretrain_model.encoder.state_dict()
        src_dict.pop('out.conv.conv.weight', None)
        src_dict.pop('out.conv.conv.bias', None)

        # Load into the main encoder. LoRA will be applied on top of these weights.
        msg = model.encoder.load_state_dict(src_dict, strict=False)
        print(f"✓ Pre-trained encoder loaded successfully.")
        print(f"  Missing keys: {len(msg.missing_keys)}")   # Should be the decoder keys
        print(f"  Unexpected keys: {len(msg.unexpected_keys)}") # Should be 0

        return model

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_reshaped = x.view(B * C, 1, D, H, W)
        logits_reshaped = self.encoder(x_reshaped)
        _, Num_Classes, D_out, H_out, W_out = logits_reshaped.shape
        logits_per_modality = logits_reshaped.view(B, C, Num_Classes, D_out, H_out, W_out)
        fused_logits = logits_per_modality.mean(dim=1)
        return fused_logits

    def configure_optimizers(self):
        # Pytorch Lightning will automatically find the trainable parameters.
        # With PEFT, this will be the LoRA weights and the decoder weights.
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01)

        def lr_lambda(current_step: int):
            num_training_steps = self.trainer.estimated_stepping_batches
            num_warmup_steps = int(num_training_steps * self.hparams.warmup_epochs / self.hparams.max_epochs)
            if current_step < num_warmup_steps: return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_lr_ratio = self.hparams.min_lr / self.hparams.learning_rate
            return (1 - min_lr_ratio) * cosine_decay + min_lr_ratio

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.loss_function(outputs, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self.sliding_window_inferer(inputs=images, network=self)
        loss = self.loss_function(outputs, labels)
        post_pred = torch.argmax(outputs, dim=1, keepdim=True)
        self.dice_metric(y_pred=post_pred, y=labels)
        self.log('val/loss', loss, on_epoch=True)
        if self.current_epoch % self.hparams.log_image_frequency == 0:
            self._log_validation_images(batch, outputs, batch_idx)
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking: return
        val_dice = self.dice_metric.aggregate().item()
        self.log('val/dice', val_dice, prog_bar=True)
        self.dice_metric.reset()
    def _log_validation_images(self, batch, outputs, batch_idx):
        if batch_idx > 0 or not hasattr(self, 'trainer') or self.trainer.global_rank != 0: return
        if not self.logger or not self.logger.experiment: return

        # --- Start of corrected block ---

        try:
            img, label = batch['image'][0].cpu().numpy(), batch['label'][0].squeeze().cpu().numpy()
            pred = torch.argmax(outputs[0], dim=0).cpu().numpy()
            vis_img = img[0] if img.ndim > 3 else img

            # Correctly find the slice with the largest area for the label
            # Sum over the other two axes to get a 1D array for each dimension
            slice_idx_z = np.argmax(np.sum(label, axis=(1, 2))) # Axial slice
            slice_idx_y = np.argmax(np.sum(label, axis=(0, 2))) # Coronal slice
            slice_idx_x = np.argmax(np.sum(label, axis=(0, 1))) # Sagittal slice

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

        except Exception as e:
            print(f"Error during validation image logging: {e}")

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

def save_segmentation(
    prediction: np.ndarray, affine: nib.Nifti1Image, output_path: str
):
    """Save prediction as a NIfTI file using affine from reference image."""
    # Create a new NIfTI image with the prediction data and reference affine
    pred_nifti = nib.Nifti1Image(prediction, affine)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Make sure output path has .nii.gz extension
    if not output_path.endswith((".nii", ".nii.gz")):
        output_path = output_path + ".nii.gz"

    # Save the prediction
    nib.save(pred_nifti, output_path)

def dbg(name, x):
    if isinstance(x, torch.Tensor):
        print(f"{name}: tensor shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}")
    elif isinstance(x, np.ndarray):
        print(f"{name}: ndarray shape={x.shape}, dtype={x.dtype}")
    else:
        print(f"{name}: {type(x)} -> {x}")


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

    # Load the model checkpoint directly with Lightning

    model = SegmentationFineTuner.load_from_checkpoint(str(model_path))

    # Set model to evaluation mode
    model.eval()

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    case_preprocessed = case_preprocessed.to(device)

    # Run inference
    with torch.no_grad():
        # Set up sliding window inferer
        inferer = SlidingWindowInferer(
            roi_size=patch_size,   # e.g. (96, 96, 96)
            sw_batch_size=1,
            overlap=0.5,
            mode="gaussian",
        )

        # Ensure model input is (1, C, D, H, W)
        if case_preprocessed.dim() == 4:
            case_preprocessed = case_preprocessed.unsqueeze(0)

        # --- MONAI TTA expects per-item (C, D, H, W), not batched ---
        tta_input = case_preprocessed.squeeze(0)  # -> (C, D, H, W)

        dbg("tta_input (C,D,H,W)", tta_input)

        tta_transform = Compose([
            RandFlipd(keys="image", prob=0.5, spatial_axis=0),
            RandFlipd(keys="image", prob=0.5, spatial_axis=1),
            RandFlipd(keys="image", prob=0.5, spatial_axis=2),
            RandRotate90d(keys="image", prob=0.5, max_k=3),
            EnsureTyped(keys="image"),
        ])
        # (optional) reproducibility
        tta_transform.set_random_state(seed=123)

        # Wrap the inferer+model; return softmax probabilities
        def _infer_fn(x: torch.Tensor) -> torch.Tensor:
            # x is (B, C, D, H, W)
            logits = inferer(inputs=x, network=model)
            return torch.softmax(logits, dim=1)  # (B, num_classes, D, H, W)

        # TTA config (ensure num_examples % batch_size == 0)
        tta_cfg = predict_config.get("tta", {"num_examples": 8, "batch_size": 2})
        n_examples = int(tta_cfg.get("num_examples", 8))
        tta_bs = int(tta_cfg.get("batch_size", 2))
        if n_examples % tta_bs != 0:
            for d in range(min(n_examples, tta_bs), 0, -1):
                if n_examples % d == 0:
                    tta_bs = d
                    break  # guarantees divisibility

        tta = TestTimeAugmentation(
            transform=tta_transform,
            batch_size=tta_bs,        # number of TTA realizations per loader batch
            num_workers=0,
            inferrer_fn=_infer_fn,
            device=device,
            image_key="image",
            orig_key="image",         # invert using the original image's meta
            output_device=device,
            return_full_data=False    # set True to get full stack
        )

        # Build the input dict MONAI expects (per-item tensor)
        data = {"image": tta_input}  # (C, D, H, W)

        # Run TTA: returns (mode, mean, std, vvc); use mean probs
        mode_pred, mean_pred, std_pred, vvc = tta(data, num_examples=n_examples)
        dbg("mean_pred (from TTA)", mean_pred)

        # Add batch dim back for downstream (B, C, D, H, W)
        predictions = mean_pred.unsqueeze(0)
        dbg("predictions (after unsqueeze)", predictions)

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

task2_config = {
    "task_name": "Task002_FOMO2",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("DWI", "T2FLAIR", "SWI_OR_T2STAR"),
    "norm_op": "volume_wise_znorm",
    "num_classes": 2,
    "keep_aspect_ratio": True,
    "task_type": "segmentation",
    "label_extension": ".txt",
    "labels": {0: "background", 1: "menigioma"},
    "target_spacing": [1.0, 1.0, 1.0],
    "target_orientation": "RAS",
}

# Task-specific hardcoded configuration
predict_config = {
    # Import values from task_configs
    **task2_config,
    # Add inference-specific configs
    "model_path": "/app/weights/brano__segmentacia.ckpt",  # Path to model (inside container!)
    "patch_size": (96, 96, 96),
}


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on FOMO Task 2 (Meningioma Segmentation)"
    )

    # Input and output paths using modality names from task config
    parser.add_argument(
        "--dwi_b1000", type=str, required=True, help="Path to DWI image (NIfTI format)"
    )
    parser.add_argument(
        "--flair",
        type=str,
        required=True,
        help="Path to T2FLAIR image (NIfTI format)",
    )
    parser.add_argument(
        "--swi", type=str, required=False, help="Path to SWI image (NIfTI format)"
    )
    parser.add_argument(
        "--t2s", type=str, required=False, help="Path to T2* image (NIfTI format)"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for prediction"
    )

    # Parse arguments
    args = parser.parse_args()

    assert (args.swi and not args.t2s) or (
        not args.swi and args.t2s
    ), "Either --swi or --t2s must be provided, but not both."

    # Map arguments to modality paths in expected order from task config
    modality_paths = [args.dwi_b1000, args.flair, args.swi or args.t2s]
    output_path = args.output

    # Run prediction using the shared prediction logic
    predictions, affine = predict_from_config(
        modality_paths=modality_paths,
        predict_config=predict_config,
        reverse_preprocess=True,
    )

    prediction_final = np.argmax(predictions[0], axis=0).astype(np.int32)

    save_segmentation(prediction_final, affine, output_path)


if __name__ == "__main__":
    main()