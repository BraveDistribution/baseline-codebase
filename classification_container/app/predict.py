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

from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy
from torch.optim.lr_scheduler import LambdaLR

from yucca.modules.data.augmentation.transforms.cropping_and_padding import CropPad

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
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(checkpoint_path, strict=False)

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
        f.write(f"{number:.3f}")

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

    x_np = case_preprocessed.squeeze(0).detach().numpy()


    croppad = CropPad(patch_size=(96, 96, 96))
    out = croppad(
        packed_data_dict={"image": x_np},
        image_properties={"foreground_locations": []}
    )
    x_np = out["image"].astype(np.float32, copy=False)
    case_preprocessed = torch.from_numpy(np.ascontiguousarray(x_np)).unsqueeze(0)

    # Load the model checkpoint directly with Lightning

    model = ClassificationFinetuner2.load_from_pretrained(
            checkpoint_path=str(model_path),
            num_classes=1,
            in_channels=4,
            freeze_encoder=True,
            learning_rate=1e-5, # We discussed using a lower LR for fine-tuning
            max_epochs=50
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

task1_config = {
    "task_name": "Task001_FOMO1",
    "crop_to_nonzero": True,
    "deep_supervision": False,
    "modalities": ("DWI", "T2FLAIR", "ADC", "SWI_OR_T2STAR"),
    "norm_op": "volume_wise_znorm",
    "num_classes": 1,
    "keep_aspect_ratio": True,
    "task_type": "classification",
    "label_extension": ".txt",
    "labels": {0: "Negative", 1: "Positive"},
    "target_spacing": [1.0, 1.0, 1.0],
    "target_orientation": "RAS",
}

# Task-specific hardcoded configuration
predict_config = {
    # Import values from task_configs
    **task1_config,
    # Add inference-specific configs
    "model_path": "/app/weights/brano_checkpoint_classification_258.ckpt",  # Path to model (inside container!)
    "patch_size": (96, 96, 96),  # Patch size for inference
}


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on FOMO Task 1 (Infarct Detection)"
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
        "--adc", type=str, required=True, help="Path to ADC image (NIfTI format)"
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
    modality_paths = [args.dwi_b1000, args.flair, args.adc, args.swi or args.t2s]
    output_path = args.output

    # Run prediction using the shared prediction logic
    predictions_original, _ = predict_from_config(
        modality_paths=modality_paths,
        predict_config=predict_config,
    )

    # softmax output to get probability
    probabilities = sigmoid(predictions_original)

    save_output_txt(float(probabilities), output_path)


if __name__ == "__main__":
    main()