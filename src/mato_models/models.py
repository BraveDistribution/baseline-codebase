from typing import Sequence, Tuple, Dict, Any, List
import os
import wandb
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.losses.hausdorff_loss import HausdorffDTLoss

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
from torch.optim.lr_scheduler import CosineAnnealingLR
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
        print("--- Applying LoRA to SwinViT backbone ---")
        lora_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            target_modules=["qkv"], # Apply to query, key, value projections in attention
            lora_dropout=0.1,
            bias="none",
        )
        self.encoder.swinViT = get_peft_model(self.encoder.swinViT, lora_config)
        print("Trainable parameters with LoRA enabled:")
        self.encoder.swinViT.print_trainable_parameters()
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

        try:
            img, label = batch['image'][0].cpu().numpy(), batch['label'][0].squeeze().cpu().numpy()
            pred = torch.argmax(outputs[0], dim=0).cpu().numpy()
            vis_img = img[0] if img.ndim > 3 else img

            # Correctly find the slice with the largest area for the label
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






import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import wandb
import math

from typing import Tuple
from torch.optim.lr_scheduler import LambdaLR

from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import SwinUNETR

# Make sure you have peft installed: pip install peft
from peft import get_peft_model, LoraConfig

# Assuming the ContrastiveTransformer class from your pre-training is available
# from pretrain_module import ContrastiveTransformer

# ==============================================================================
# Final, Corrected PyTorch Lightning Module
# Author: MICCAI-Prodigy
# FIX: Programmatically handles input/output channel mismatch during weight loading.
# ==============================================================================

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple, List
import numpy as np

# --- MONAI Imports ---
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss, HausdorffDTLoss
from monai.metrics import DiceMetric
from monai.inferers import SlidingWindowInferer
from monai.transforms import AsDiscrete, KeepLargestConnectedComponent

# --- PEFT/LoRA Imports ---
from peft import get_peft_model, LoraConfig


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
        images, labels, label_seg = batch['image'], batch['reg_label'].float(), batch["label"]
        logits = self(images)
        labels = labels.view(-1).float().to(logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_after_backward(self):
        # grads exist now
        g = [p.grad is not None and torch.isfinite(p.grad).all() for p in self.classifier_head.parameters()]
        self.log("dbg/cls_head_has_grads", float(all(g)), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        images, labels, label_seg = batch['image'], batch['reg_label'].float(), batch["label"]
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

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            T_mult=2,
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
    CORRECTED "BEST SHOT" CONFIGURATION:
    - Encoder: Hardcoded to in_channels=1, as required.
    - Forward Pass: Correctly reshapes multi-channel input to be processed by the 1-channel encoder.
    - Architecture: Uses ONLY the final, most abstract feature map for regression.
    - Normalization: Uses a robust log-transform (log1p) + Z-score.
    - Loss: Uses a simple and stable L1 (MAE) loss, with an added loss for Brain Age Gap (BAG) correlation.
    """
    def __init__(
        self,
        in_channels: int,
        target_mean: float,     # IMPORTANT: Mean of the LOG-TRANSFORMED training ages
        target_std: float,      # IMPORTANT: Std dev of the LOG-TRANSFORMED training ages
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        lora_r: int = 128,
        lora_alpha: int = 128,
        learning_rate: float = 1e-4,
        dropout_rate: float = 0.1,
        predict_uncertainty: bool = False,
        weight_decay: float = 1e-5,
        bag_loss_weight: float = 0.1,  # NEW HYPERPARAMETER for BAG loss
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.register_buffer("bias_a", torch.tensor(1.0))  # y* = a * yhat + b
        self.register_buffer("bias_b", torch.tensor(0.0))
        self.best_val_mae = float("inf")
        self.best_bias_epoch = -1
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.all_train_targets_for_plot = None
        self.has_logged_train_batch = False
        self.has_logged_val_batch = False
        
        self.register_buffer("target_mean", torch.tensor(target_mean))
        self.register_buffer("target_std", torch.tensor(target_std))
        
        # NOTE: Metrics are now initialized in the setup() method for correct device placement.
        self.train_bag_corr_metric = None
        self.val_corr = None

        # --- MODEL ARCHITECTURE ---

        # 1. Base Encoder - Hardcoded to 1 input channel, as you correctly stated.
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

        # 3. Define Regression Head
        with torch.no_grad():
            # The dummy input for calculation must also be 1-channel.
            dummy_input = torch.zeros(1, 1, *self.hparams.img_size)
            all_features = self.encoder.swinViT(dummy_input)
            final_feature_dim = all_features[-1].shape[1]

        self.pool = nn.AdaptiveAvgPool3d(1)
        
        # The input to the head will be features from ALL input channels concatenated side-by-side.
        regressor_input_dim = self.hparams.in_channels * final_feature_dim
        
        output_dim = 2 if self.hparams.predict_uncertainty else 1
        self.regression_head = nn.Sequential(
            nn.Linear(regressor_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout_rate),
            nn.Linear(256, output_dim),
        )

    def setup(self, stage: str):
        """Initializes metrics after the model has been moved to the correct device."""
        if self.train_bag_corr_metric is None:
            self.train_bag_corr_metric = PearsonCorrCoef().to(self.device)
        if self.val_corr is None:
            self.val_corr = PearsonCorrCoef().to(self.device)
            # This is also a good place to initialize the validation BAG correlation metric
            self.val_bag_corr_metric = PearsonCorrCoef().to(self.device)
            
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
        original_target_mean: float,
        **kwargs
    ):
        """
        Loads a pretrained ContrastiveTransformer, creates an instance of this
        finetuner, and transfers the encoder weights.
        """
        print(f"Loading pretrained model from: {checkpoint_path}")
        # Assuming ContrastiveTransformer is defined elsewhere and handles its own state
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(checkpoint_path)

        finetuner_hparams = pretrain_model.hparams
        finetuner_hparams.update(kwargs)
        finetuner_hparams['in_channels'] = in_channels
        finetuner_hparams['target_mean'] = target_mean
        finetuner_hparams['target_std'] = target_std
        finetuner_hparams['original_target_mean'] = original_target_mean

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
        """Applies Log Transform, then Z-score normalization."""
        eps = 1e-6
        x_transformed = torch.log1p(x)
        return (x_transformed - self.target_mean) / (self.target_std + eps)

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverses Z-score normalization, then inverse Log Transform."""
        x_transformed = x * self.target_std + self.target_mean
        return torch.expm1(x_transformed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        # --- CORRECTED LOGIC ---
        # Reshape the input to treat the channels as items in the batch dimension.
        # Input shape: [B, C, D, H, W] -> Reshaped: [B*C, 1, D, H, W]
        x_reshaped = x.view(B * C, 1, D, H, W)
        
        all_features = self.encoder.swinViT(x_reshaped)

        # Use ONLY the final, most abstract feature map
        final_features = all_features[-1]

        # Pool features. Shape is now [B*C, feature_dim, 1, 1, 1]
        pooled_features = self.pool(final_features)
        
        # Reshape back to the original batch size B, which concatenates the channel features.
        # Shape: [B*C, feature_dim] -> [B, C * feature_dim]
        flattened_features = pooled_features.view(B, -1)
        
        output = self.regression_head(flattened_features)
        # --- END CORRECTION ---

        return output.squeeze(-1) if not self.hparams.predict_uncertainty else output

    def compute_loss(self, preds_original, targets_original):
        mae_loss = F.l1_loss(preds_original, targets_original)
        
        # Compute BAG
        bag = preds_original - targets_original
        
        # Compute correlation manually for gradient flow
        bag_mean = bag.mean()
        targets_mean = targets_original.mean()
        
        bag_centered = bag - bag_mean
        targets_centered = targets_original - targets_mean
        
        # Pearson correlation
        numerator = (bag_centered * targets_centered).sum()
        denominator = torch.sqrt((bag_centered ** 2).sum() * (targets_centered ** 2).sum())
        
        # Add small epsilon to avoid division by zero
        correlation = numerator / (denominator + 1e-8)
        
        # We want to minimize the absolute correlation
        bag_correlation_loss = torch.abs(correlation)
        
        total_loss = mae_loss + self.hparams.bag_loss_weight * bag_correlation_loss
        
        return total_loss, mae_loss, bag_correlation_loss

    def training_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label'].float().view(-1)
        
        # NOTE: We need unnormalized predictions for the BAG loss calculation.
        preds_normalized = self(images)
        preds_mean_normalized = preds_normalized[:, 0] if self.hparams.predict_uncertainty else preds_normalized
        preds_original = self._unnormalize(preds_mean_normalized)
        
        targets_normalized = self._normalize(targets)

        total_loss, mae_loss, bag_corr_loss = self.compute_loss(preds_original, targets)
        
        self.log('train/loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/mae_loss', mae_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/bag_corr_loss', bag_corr_loss, on_step=False, on_epoch=True, sync_dist=True)

        self.training_step_outputs.append(targets.detach().cpu())
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label'].float().view(-1)
        preds_normalized = self(images)
        targets_normalized = self._normalize(targets)
        
        preds_mean_normalized = preds_normalized[:, 0] if self.hparams.predict_uncertainty else preds_normalized
        preds_original = self._unnormalize(preds_mean_normalized.detach())
        mae = F.l1_loss(preds_original, targets)
        
        self.val_corr.update(preds_original, targets)
        
        # We also want to log the BAG correlation on the validation set for monitoring
        bag = preds_original - targets
        self.val_bag_corr_metric.update(bag, targets)
        
        log_dict = {
            'val/mae': mae, 'val/correlation': self.val_corr, 'val/bag_corr': self.val_bag_corr_metric, 'val/loss': mae,
        }
        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.append({'preds': preds_original, 'targets': targets})

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = self.trainer.estimated_stepping_batches,
            eta_min = self.hparams.learning_rate / 50
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    def on_train_epoch_end(self):
        """Aggregates training labels at the end of the training epoch."""
        if self.training_step_outputs:
            self.all_train_targets_for_plot = torch.cat(self.training_step_outputs).numpy()
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self.validation_step_outputs:
            return

        # ----- gather epoch preds/targets -----
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs], dim=0)
        targets = torch.cat([x['targets'] for x in self.validation_step_outputs], dim=0)

        # DDP: gather across ranks
        if self.trainer.world_size > 1:
            preds = self.all_gather(preds).reshape(-1)
            targets = self.all_gather(targets).reshape(-1)

        # ----- fit linear correction y = a * yhat + b -----
        x = preds.detach()
        y = targets.detach()

        x_mean = x.mean()
        y_mean = y.mean()
        x_var  = x.var(unbiased=False)

        # guard: if predictions are (nearly) constant this epoch
        if x_var < 1e-8:
            a = torch.tensor(1.0, device=x.device)
            b = torch.tensor(0.0, device=x.device)
        else:
            cov = ((x - x_mean) * (y - y_mean)).mean()
            a = cov / (x_var + 1e-8)
            b = y_mean - a * x_mean

        # evaluate raw vs corrected
        mae_raw = F.l1_loss(x, y)
        x_corr = a * x + b
        mae_corr = F.l1_loss(x_corr, y)

        # BAG corr after correction (diagnostic only)
        bag = x_corr - y
        bag_centered = bag - bag.mean()
        y_centered = y - y.mean()
        denom = torch.sqrt((bag_centered.pow(2).sum()) * (y_centered.pow(2).sum())) + 1e-8
        bag_r = (bag_centered * y_centered).sum() / denom

        # ----- persist best (a,b) on rank 0 -----
        if self.trainer.is_global_zero:
            # keep the (a,b) that yields the best val MAE so far
            if mae_corr.item() < self.best_val_mae:
                self.best_val_mae = mae_corr.item()
                self.bias_a.copy_(a.detach())
                self.bias_b.copy_(b.detach())
                self.best_bias_epoch = int(self.current_epoch)

            # log both current-epoch and running-best
            self.log_dict({
                "val/mae_raw_epoch": mae_raw,
                "val/mae_corr_epoch": mae_corr,
                "val/bag_corr_after": bag_r,
                "val/bias_a_epoch": a,
                "val/bias_b_epoch": b,
                "val/best_mae_corr": torch.tensor(self.best_val_mae, device=a.device),
            }, prog_bar=False, on_epoch=True, sync_dist=True)

            if self.logger:
                self.logger.log_metrics({
                    "val/best_bias_epoch": self.best_bias_epoch
                }, step=self.global_step)

        # ----- your existing histogram plot -----
        if self.logger and self.trainer.is_global_zero:
            preds_np = x.detach().cpu().numpy()
            targets_np = y.detach().cpu().numpy()
            import matplotlib.pyplot as plt
            import numpy as np
            fig, ax1 = plt.subplots(figsize=(12, 7))
            min_val, max_val = targets_np.min(), targets_np.max()
            if self.all_train_targets_for_plot is not None:
                min_val = min(min_val, self.all_train_targets_for_plot.min())
                max_val = max(max_val, self.all_train_targets_for_plot.max())
            bins = np.linspace(min_val, max_val, num=50)
            ax1.hist(targets_np, bins=bins, alpha=0.6, color="blue", label="Ground Truth (Val)", density=True)
            if self.all_train_targets_for_plot is not None:
                ax1.hist(self.all_train_targets_for_plot, bins=bins, alpha=0.8, histtype='step',
                        linewidth=1.5, color="green", label="Ground Truth (Train)", density=True)
            ax1.hist(preds_np, bins=bins, alpha=0.5, color="red", label="Predictions (Val)", density=True)
            ax1.set_title(f"Label Distributions & Predictions (Epoch {self.current_epoch})")
            ax1.set_xlabel("Value"); ax1.set_ylabel("Density")
            ax1.legend(); ax1.grid(True, alpha=0.3)
            self.logger.experiment.log({
                "validation/prediction_distribution": wandb.Image(fig)
            })
            plt.close(fig)

        # clear epoch caches
        self.validation_step_outputs.clear()
        self.all_train_targets_for_plot = None

        
    def _maybe_mixup(self, images: torch.Tensor, targets_norm: torch.Tensor):
        """
        Applies MixUp to 3D images and normalized regression targets with prob p.
        Returns (images, targets_norm, lam) where lam is the mixing coefficient used.
        """
        alpha = float(self.hparams.mixup_alpha)
        p = float(self.hparams.mixup_prob)
        if (not self.training) or alpha <= 0.0 or torch.rand(1, device=images.device).item() > p:
            return images, targets_norm, None

        lam = torch.distributions.Beta(alpha, alpha).sample().to(images.device)
        lam = torch.maximum(lam, 1.0 - lam)

        B = images.size(0)
        index = torch.randperm(B, device=images.device)

        mixed_images = lam * images + (1.0 - lam) * images[index]
        mixed_targets_norm = lam * targets_norm + (1.0 - lam) * targets_norm[index]

        return mixed_images, mixed_targets_norm, lam

    def apply_bias_correction(self, yhat: torch.Tensor) -> torch.Tensor:
        # y* = a * yhat + b (buffers live on the correct device)
        return self.bias_a * yhat + self.bias_b
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.networks.nets import SwinUNETR
# Ensure 'peft' library is installed (pip install peft)
from peft import LoraConfig, get_peft_model
from torchmetrics.regression import PearsonCorrCoef
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

# Assuming wandb is installed if used in the logger
try:
    import wandb
except ImportError:
    wandb = None



class SegmentationProtoNet(pl.LightningModule):
    """
    Implements Few-Shot Segmentation using Prototypical Networks on a frozen
    pre-trained Swin Transformer encoder using multi-scale features (Hypercolumns).
    
    The "training" phase consists of a single epoch to calculate the class prototypes.
    """
    def __init__(
        self,
        in_channels: int, # Number of modalities
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        sw_batch_size: int = 4,
        sw_overlap: float = 0.5,
        log_image_frequency: int = 1,
        **kwargs,
    ):
        super().__init__()
        # Clean up hyperparameters irrelevant for a frozen model
        kwargs.pop('learning_rate', None)
        kwargs.pop('max_epochs', None)
        kwargs.pop('freeze_encoder', None)
        kwargs.pop('num_classes', None) # Classes are implicitly 2 (BG/FG) for ProtoNets

        self.save_hyperparameters()

        # 1. Base Encoder (Using SwinUNETR structure just for the SwinViT component)
        # We instantiate the full model to easily access the SwinViT component.
        self.encoder_model = SwinUNETR(
            in_channels=1, # Shared encoder processes 1 channel at a time
            out_channels=2, # Dummy value, decoder is not used
            feature_size=self.hparams.feature_size,
            use_checkpoint=False,
            # use_v2=True, # Uncomment if your pretraining used V2 and MONAI supports it
        )
        self.encoder = self.encoder_model.swinViT

        # 2. Freeze the encoder completely
        self._freeze_encoder()

        # 3. Determine Feature Dimensions and Strategy
        self.feature_dims = self._get_feature_dims()
        # Define which stages of the SwinViT to use (Hypercolumns).
        # Stages 1, 2, 3 provide a balance between resolution and semantic depth.
        self.used_stages = [1, 2, 3]
        self.total_feature_dim = sum(self.feature_dims[i] for i in self.used_stages)

        # 4. Prototypes (Registered as buffers so they are saved with the model)
        # Prototypes for Background (0) and Foreground (1)
        self.register_buffer("prototype_bg", torch.zeros(self.total_feature_dim))
        self.register_buffer("prototype_fg", torch.zeros(self.total_feature_dim))
        self.register_buffer("prototypes_calculated", torch.tensor(False))

        # 5. Metrics and Inference
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.sliding_window_inferer = SlidingWindowInferer(
            roi_size=self.hparams.img_size, sw_batch_size=self.hparams.sw_batch_size,
            overlap=self.hparams.sw_overlap, mode="gaussian",
        )

        # Accumulators for prototype calculation (initialized during training)
        self.fg_sum = None
        self.bg_sum = None
        self.fg_count = 0
        self.bg_count = 0

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval() # Set to eval mode (important for BN/Dropout if present)
        print("--- Encoder (SwinViT) is completely frozen ---")

    def _get_feature_dims(self) -> List[int]:
        # Helper to determine the output dimensions of the SwinViT stages
        try:
            with torch.no_grad():
                # Use CPU for dummy input if GPU memory is tight during init
                dummy_input = torch.zeros(1, 1, *self.hparams.img_size)
                # Temporarily move encoder to CPU if needed for initialization check
                original_device = next(self.encoder.parameters()).device
                all_features = self.encoder.to('cpu')(dummy_input)
                self.encoder.to(original_device) # Move back
                return [f.shape[1] for f in all_features]
        except Exception as e:
            print(f"Warning: Automatic feature detection failed ({e}). Using fallback.")
            # Fallback based on common configurations
            if self.hparams.feature_size == 24:
                return [24, 48, 96, 192, 384]
            elif self.hparams.feature_size == 48:
                return [48, 96, 192, 384, 768]
            raise ValueError("Could not determine feature dimensions.")

    @classmethod
    def load_from_pretrained(
        cls,
        pretrained_checkpoint_path: str,
        in_channels: int,
        **kwargs
    ):
        # Requires ContrastiveTransformer to be defined in the environment
        # This assumes ContrastiveTransformer is imported or defined in the scope.
        try:
            # Replace 'ContrastiveTransformer' with the actual class name if different
            pretrain_model = ContrastiveTransformer.load_from_checkpoint(pretrained_checkpoint_path)
        except NameError:
            print("Error: ContrastiveTransformer class definition not found.")
            raise
        except Exception as e:
            print(f"Error loading checkpoint {pretrained_checkpoint_path}: {e}")
            raise

        # Combine hyperparameters
        finetuner_hparams = pretrain_model.hparams.copy()
        finetuner_hparams.update(kwargs)
        finetuner_hparams['in_channels'] = in_channels

        model = cls(**finetuner_hparams)

        # Load weights into the SwinViT component
        # Assumes the pre-trained model stores the SwinViT under encoder.swinViT
        src_dict = pretrain_model.encoder.swinViT.state_dict()
        msg = model.encoder.load_state_dict(src_dict, strict=False)
        
        print(f"\n✓ Pre-trained encoder loaded from {pretrained_checkpoint_path}")
        print(f"  Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")
        
        # Ensure encoder remains frozen after loading
        model._freeze_encoder()

        return model

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts multi-scale features, fuses modalities, and creates hypercolumns.
        Input: [B, C, D, H, W]
        Output: [B, F_total, D', H', W'] (where D', H', W' are downsampled)
        """
        B, C, D, H, W = x.shape
        # Reshape to process modalities independently: [B*C, 1, D, H, W]
        x_reshaped = x.view(B * C, 1, D, H, W)

        # Ensure encoder is in eval mode
        self.encoder.eval() 
        with torch.no_grad():
            all_features = self.encoder(x_reshaped)

        # Target shape for upsampling (Shape of the earliest used stage)
        target_shape = all_features[self.used_stages[0]].shape[2:]

        fused_features = []
        for i in self.used_stages:
            features = all_features[i]
            # Upsample if necessary (Trilinear interpolation)
            if features.shape[2:] != target_shape:
                features = F.interpolate(features, size=target_shape, mode='trilinear', align_corners=False)
            
            # Reshape back to separate modalities: [B, C, F_i, D', H', W']
            _, F_i, D_out, H_out, W_out = features.shape
            features_per_modality = features.view(B, C, F_i, D_out, H_out, W_out)
            
            # Fuse modalities by averaging features (Robust fusion)
            # [B, F_i, D', H', W']
            fused_modality_features = features_per_modality.mean(dim=1)
            fused_features.append(fused_modality_features)

        # Concatenate across feature dimension (Hypercolumns)
        # [B, F_total, D', H', W']
        hypercolumns = torch.cat(fused_features, dim=1)
        return hypercolumns

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs segmentation using the calculated prototypes.
        Input: [B, C, D, H, W]
        Output: [B, 2, D, H, W] (Logits for BG/FG)
        """
        if not self.prototypes_calculated:
            # Return zeros if prototypes are not ready (e.g., during sanity check)
            return torch.zeros(x.shape[0], 2, *x.shape[2:], device=self.device)

        # 1. Extract Features
        # [B, F_total, D', H', W']
        features = self.extract_features(x)
        B, prototype, D_p, H_p, W_p = features.shape

        # 2. Calculate Distances
        # Reshape features for distance calculation: [B, F, N_voxels]
        features_flat = features.view(B, prototype, -1)

        # Calculate squared Euclidean distance to prototypes: ||x - p||^2
        # Prototypes: [F]
        
        # Calculate distance for BG
        # Broadcasting: (B, F, N_voxels) - (F, 1) -> (B, F, N_voxels)
        dist_bg = torch.sum((features_flat - self.prototype_bg.view(prototype, 1))**2, dim=1)
        # Calculate distance for FG
        dist_fg = torch.sum((features_flat - self.prototype_fg.view(prototype, 1))**2, dim=1)
        # Result: [B, N_voxels]
        
        # 3. Convert distances to logits
        # ProtoNets use negative distance as logits (closer means higher logit)
        logits_flat = torch.stack([-dist_bg, -dist_fg], dim=1) # [B, 2, N_voxels]

        # 4. Reshape and Upsample
        logits = logits_flat.view(B, 2, D_p, H_p, W_p)
        
        # Upsample logits back to the original image resolution
        logits_upsampled = F.interpolate(logits, size=x.shape[2:], mode='trilinear', align_corners=False)

        return logits_upsampled

    # --- Training Loop (Prototype Calculation) ---
    # We use the training loop (1 epoch) to iterate over the dataset and calculate prototypes.

    def on_train_start(self):
        # Initialize accumulators at the start of the fitting process
        if self.prototypes_calculated:
            print("Prototypes already calculated. Skipping training phase.")
            return

        print("\nStarting Prototype Calculation Phase (Training Epoch)...")
        # Initialize accumulators on the correct device and dtype
        self.fg_sum = torch.zeros(self.total_feature_dim, device=self.device, dtype=torch.float32)
        self.bg_sum = torch.zeros(self.total_feature_dim, device=self.device, dtype=torch.float32)
        self.fg_count = 0
        self.bg_count = 0

    def training_step(self, batch, batch_idx):
        if self.prototypes_calculated:
            return None

        images, labels = batch['image'], batch['label']
        # Ensure labels are binary (0 or 1)
        labels = (labels > 0).long()

        # 1. Extract Features
        features = self.extract_features(images)

        # 2. Align Labels with Feature Resolution (using nearest neighbor)
        labels_downsampled = F.interpolate(labels.float(), size=features.shape[2:], mode='nearest').long()

        # 3. Accumulate Features
        B = images.shape[0]
        for i in range(B):
            f = features[i] # [F, D', H', W']
            l = labels_downsampled[i].squeeze(0) # [D', H', W']

            fg_mask = (l == 1)
            bg_mask = (l == 0)

            # Efficiently gather and sum features using masking
            if fg_mask.any():
                # Permute to [D', H', W', F] and mask to get [N_fg, F]
                fg_features = f.permute(1, 2, 3, 0)[fg_mask] 
                self.fg_sum += fg_features.sum(dim=0)
                self.fg_count += fg_features.shape[0]

            if bg_mask.any():
                # Permute and mask to get [N_bg, F]
                bg_features = f.permute(1, 2, 3, 0)[bg_mask]
                self.bg_sum += bg_features.sum(dim=0)
                self.bg_count += bg_features.shape[0]
        
        # Log progress
        self.log('proto/fg_voxels_accumulated', float(self.fg_count), on_step=True, prog_bar=True)

        # No loss is returned as there is no optimization
        return None

    def on_train_epoch_end(self):
        if self.prototypes_calculated:
            return

        # Calculate the final prototypes by averaging the accumulated features
        print("\nFinalizing Prototype Calculation...")
        
        # Use distributed reduction if using multiple GPUs (DDP)
        if self.trainer.world_size > 1:
             # Sum counts across GPUs
             fg_count_total = torch.tensor(self.fg_count, device=self.device)
             torch.distributed.all_reduce(fg_count_total, op=torch.distributed.ReduceOp.SUM)
             self.fg_count = fg_count_total.item()

             bg_count_total = torch.tensor(self.bg_count, device=self.device)
             torch.distributed.all_reduce(bg_count_total, op=torch.distributed.ReduceOp.SUM)
             self.bg_count = bg_count_total.item()

             # Sum features across GPUs
             torch.distributed.all_reduce(self.fg_sum, op=torch.distributed.ReduceOp.SUM)
             torch.distributed.all_reduce(self.bg_sum, op=torch.distributed.ReduceOp.SUM)

        if self.fg_count > 0:
            # Update the buffer with the calculated prototype
            self.prototype_fg.copy_(self.fg_sum / self.fg_count)
        else:
            print("Warning: No foreground voxels found. FG prototype remains zero.")
        
        if self.bg_count > 0:
            self.prototype_bg.copy_(self.bg_sum / self.bg_count)
        else:
             print("Warning: No background voxels found. BG prototype remains zero.")

        self.prototypes_calculated = torch.tensor(True)
        print(f"✓ Prototypes calculated successfully.")

        # Clear accumulators
        self.fg_sum = None
        self.bg_sum = None

    def configure_optimizers(self):
        # No optimizer needed as the model has no trainable parameters
        return None

    # --- Validation Loop ---

    def validation_step(self, batch, batch_idx):
        if not self.prototypes_calculated:
            return

        images, labels = batch['image'], batch['label']
        labels = (labels > 0).long()

        # Use sliding window inference, which calls the forward pass
        outputs = self.sliding_window_inferer(inputs=images, network=self)

        # Convert logits to predictions (argmax)
        post_pred = torch.argmax(outputs, dim=1, keepdim=True)
        
        self.dice_metric(y_pred=post_pred, y=labels)

        if (self.current_epoch % self.hparams.log_image_frequency == 0) or self.hparams.log_image_frequency == 1:
            self._log_validation_images(batch, outputs, batch_idx)

    def on_validation_epoch_end(self):
        if not self.prototypes_calculated or self.trainer.sanity_checking: 
            return
            
        try:
            val_dice = self.dice_metric.aggregate().item()
            self.log('val/dice', val_dice, prog_bar=True)
            self.log('val/loss', val_dice, prog_bar=True)
        except RuntimeError:
            # Handle case where aggregation fails (e.g., no foreground present in batch)
            self.log('val/dice', 0.0, prog_bar=True)
            

        self.dice_metric.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if not self.prototypes_calculated:
             raise RuntimeError("Prototypes must be calculated (via trainer.fit()) before prediction.")
        images = batch['image']
        return self.sliding_window_inferer(inputs=images, network=self)

    def _log_validation_images(self, batch, outputs, batch_idx):
        # Visualization logic (adapted from the original SegmentationFineTuner)
        if batch_idx > 0 or not hasattr(self, 'trainer') or self.trainer.global_rank != 0: return
        if not self.logger or not self.logger.experiment or wandb is None: return
        
        img, label = batch['image'][0].cpu().numpy(), batch['label'][0].squeeze().cpu().numpy()
        pred = torch.argmax(outputs[0], dim=0).cpu().numpy()
        
        # Use the first modality for visualization background
        vis_img = img[0] 

        # Find slices with the most foreground voxels
        slice_idx_z = np.argmax(np.sum(label, axis=(1, 2)))
        slice_idx_y = np.argmax(np.sum(label, axis=(0, 2)))
        slice_idx_x = np.argmax(np.sum(label, axis=(0, 1)))

        # If no foreground exists, use the center slice
        if np.sum(label) == 0:
            slice_idx_z, slice_idx_y, slice_idx_x = [s // 2 for s in vis_img.shape]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Note: Epoch number might be 0 or 1 depending on how the trainer is configured.
        fig.suptitle(f"Validation - Sample 0 (Prototypes Ready: {self.prototypes_calculated.item()})", fontsize=16)
        
        views = [
            ("Axial", vis_img[slice_idx_z, :, :], label[slice_idx_z, :, :], pred[slice_idx_z, :, :]),
            ("Coronal", vis_img[:, slice_idx_y, :], label[:, slice_idx_y, :], pred[:, slice_idx_y, :]),
            ("Sagittal", vis_img[:, :, slice_idx_x], label[:, :, slice_idx_x], pred[:, :, slice_idx_x]),
        ]
        
        for i, (title, img_slice, lbl_slice, pred_slice) in enumerate(views):
            axes[i].imshow(np.rot90(img_slice), cmap="gray")
            # Plot ground truth contour (Yellow)
            if np.any(lbl_slice): axes[i].contour(np.rot90(lbl_slice), colors='yellow', linewidths=0.8, alpha=0.9)
            # Plot prediction contour (Red)
            if np.any(pred_slice): axes[i].contour(np.rot90(pred_slice), colors='red', linewidths=0.8, alpha=0.9)
            axes[i].set_title(title); axes[i].axis('off')
            
        self.logger.experiment.log({"Validation/Prediction vs Label": wandb.Image(fig)})
        plt.close(fig)
        
        
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import PearsonCorrCoef
from typing import Tuple
import matplotlib.pyplot as plt
import wandb
import numpy as np

# Assuming SwinUNETR and other dependencies are imported correctly
# from monai.networks.nets import SwinUNETR
# from peft import get_peft_model, LoraConfig # No longer needed

class RegressionFinetuner4(pl.LightningModule):
    """
    EARLY FUSION CONFIGURATION:
    - Encoder: Accepts in_channels directly (e.g., 2 for two modalities).
    - Forward Pass: Processes multi-channel input in a single pass.
    - Architecture: Fully trainable encoder (no LoRA/PEFT). Uses the final feature map for regression.
    - Normalization: Uses a robust log-transform (log1p) + Z-score.
    - Loss: Uses a simple and stable L1 (MAE) loss, with an added loss for Brain Age Gap (BAG) correlation.
    """
    def __init__(
        self,
        in_channels: int,        # Set to 2 for two modalities
        target_mean: float,      # IMPORTANT: Mean of the LOG-TRANSFORMED training ages
        target_std: float,       # IMPORTANT: Std dev of the LOG-TRANSFORMED training ages
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        learning_rate: float = 1e-4,
        dropout_rate: float = 0.1,
        predict_uncertainty: bool = False,
        weight_decay: float = 1e-5,
        bag_loss_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        # Note: lora_r and lora_alpha are ignored but kept in signature for compatibility
        self.save_hyperparameters()
        self.register_buffer("bias_a", torch.tensor(1.0))
        self.register_buffer("bias_b", torch.tensor(0.0))
        self.best_val_mae = float("inf")
        self.best_bias_epoch = -1
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.all_train_targets_for_plot = None
        self.has_logged_train_batch = False
        self.has_logged_val_batch = False
        
        self.register_buffer("target_mean", torch.tensor(target_mean))
        self.register_buffer("target_std", torch.tensor(target_std))
        
        self.train_bag_corr_metric = None
        self.val_corr = None

        # --- MODEL ARCHITECTURE ---

        # 1. Base Encoder - Accepts multiple input channels for early fusion.
        self.encoder = SwinUNETR(
            in_channels=self.hparams.in_channels, # Changed from 1
            out_channels=1,
            feature_size=self.hparams.feature_size,
            use_checkpoint=True,
            use_v2=True,
        )
        # NOTE: LoRA wrapper has been removed. The entire encoder is now trainable.

        # 2. Define Regression Head
        with torch.no_grad():
            # Dummy input now reflects the multi-channel input
            dummy_input = torch.zeros(1, self.hparams.in_channels, *self.hparams.img_size)
            all_features = self.encoder.swinViT(dummy_input)
            final_feature_dim = all_features[-1].shape[1]

        self.pool = nn.AdaptiveAvgPool3d(1)
        
        # Input to the head is the feature dimension from the single forward pass.
        regressor_input_dim = final_feature_dim # Changed from self.hparams.in_channels * final_feature_dim
        
        output_dim = 2 if self.hparams.predict_uncertainty else 1
        self.regression_head = nn.Sequential(
            nn.Linear(regressor_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout_rate),
            nn.Linear(256, output_dim),
        )

    def setup(self, stage: str):
        """Initializes metrics after the model has been moved to the correct device."""
        if self.train_bag_corr_metric is None:
            self.train_bag_corr_metric = PearsonCorrCoef().to(self.device)
        if self.val_corr is None:
            self.val_corr = PearsonCorrCoef().to(self.device)
            self.val_bag_corr_metric = PearsonCorrCoef().to(self.device)
            
    def _log_batch_images(self, images, targets, preds, step_name: str):
        """Creates and logs a grid of image slices for a few samples in the batch."""
        if not self.logger:
            return

        images = images.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        num_samples = min(4, len(images))

        for i in range(num_samples):
            image = images[i]
            target = targets[i]
            pred = preds[i]
            num_modalities = image.shape[0]
            fig, axes = plt.subplots(num_modalities, 3, figsize=(12, 4 * num_modalities), squeeze=False)
            fig.suptitle(f"Sample {i} | True Age: {target:.1f} | Pred Age: {pred:.1f}", fontsize=16)

            for c in range(num_modalities):
                modality_vol = image[c]
                mid_d, mid_h, mid_w = [s // 2 for s in modality_vol.shape]
                axial_slice = modality_vol[mid_d, :, :]
                coronal_slice = modality_vol[:, mid_h, :]
                sagittal_slice = modality_vol[:, :, mid_w]

                axes[c, 0].imshow(axial_slice.T, cmap="bone", origin="lower")
                axes[c, 0].set_title(f"Modality {c} (Axial)")
                axes[c, 0].axis("off")
                axes[c, 1].imshow(coronal_slice.T, cmap="bone", origin="lower")
                axes[c, 1].set_title(f"Modality {c} (Coronal)")
                axes[c, 1].axis("off")
                axes[c, 2].imshow(sagittal_slice.T, cmap="bone", origin="lower")
                axes[c, 2].set_title(f"Modality {c} (Sagittal)")
                axes[c, 2].axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
        original_target_mean: float,
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
        finetuner_hparams['original_target_mean'] = original_target_mean

        model = cls(**finetuner_hparams)
        print("\nFinetuner instantiated. Now transferring weights...")

        src_dict = pretrain_model.encoder.swinViT.state_dict()
        dst_dict = model.encoder.swinViT.state_dict() # Adjusted for direct encoder access

        filtered_state_dict = {
            k: v for k, v in src_dict.items()
            if k in dst_dict and v.shape == dst_dict[k].shape
        }

        msg = model.encoder.swinViT.load_state_dict(filtered_state_dict, strict=False)

        print(f"\n✓ Loaded {len(filtered_state_dict)} swinViT tensors from {checkpoint_path}")
        print(f"  Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")
        
        # LoRA-specific prints removed
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal trainable parameters: {num_trainable:,}")

        return model
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Log Transform, then Z-score normalization."""
        eps = 1e-6
        x_transformed = torch.log1p(x)
        return (x_transformed - self.target_mean) / (self.target_std + eps)

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverses Z-score normalization, then inverse Log Transform."""
        x_transformed = x * self.target_std + self.target_mean
        return torch.expm1(x_transformed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        
        # --- EARLY FUSION LOGIC ---
        # The encoder now handles the multi-channel input directly.
        # Input shape: [B, C, D, H, W]
        all_features = self.encoder.swinViT(x)

        # Use ONLY the final, most abstract feature map
        final_features = all_features[-1]

        # Pool features. Shape is now [B, feature_dim, 1, 1, 1]
        pooled_features = self.pool(final_features)
        
        # Flatten for the regression head.
        # Shape: [B, feature_dim]
        flattened_features = pooled_features.view(B, -1)
        
        output = self.regression_head(flattened_features)
        # --- END FUSION LOGIC ---

        return output.squeeze(-1) if not self.hparams.predict_uncertainty else output

    def compute_loss(self, preds_original, targets_original):
        mae_loss = F.l1_loss(preds_original, targets_original)
        bag = preds_original - targets_original
        
        # Manual correlation for gradient flow
        bag_mean, targets_mean = bag.mean(), targets_original.mean()
        bag_centered, targets_centered = bag - bag_mean, targets_original - targets_mean
        
        numerator = (bag_centered * targets_centered).sum()
        denominator = torch.sqrt((bag_centered ** 2).sum() * (targets_centered ** 2).sum())
        correlation = numerator / (denominator + 1e-8)
        
        bag_correlation_loss = torch.abs(correlation)
        total_loss = mae_loss + self.hparams.bag_loss_weight * bag_correlation_loss
        
        return total_loss, mae_loss, bag_correlation_loss

    def training_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label'].float().view(-1)
        
        preds_normalized = self(images)
        preds_mean_normalized = preds_normalized[:, 0] if self.hparams.predict_uncertainty else preds_normalized
        preds_original = self._unnormalize(preds_mean_normalized)
        
        targets_normalized = self._normalize(targets)

        total_loss, mae_loss, bag_corr_loss = self.compute_loss(preds_original, targets)
        
        self.log('train/loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/mae_loss', mae_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/bag_corr_loss', bag_corr_loss, on_step=False, on_epoch=True, sync_dist=True)

        self.training_step_outputs.append(targets.detach().cpu())
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['label'].float().view(-1)
        preds_normalized = self(images)
        
        preds_mean_normalized = preds_normalized[:, 0] if self.hparams.predict_uncertainty else preds_normalized
        preds_original = self._unnormalize(preds_mean_normalized.detach())
        mae = F.l1_loss(preds_original, targets)
        
        self.val_corr.update(preds_original, targets)
        
        bag = preds_original - targets
        self.val_bag_corr_metric.update(bag, targets)
        
        log_dict = {
            'val/mae': mae, 'val/correlation': self.val_corr, 'val/bag_corr': self.val_bag_corr_metric, 'val/loss': mae,
        }
        self.log_dict(log_dict, prog_bar=True, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.append({'preds': preds_original, 'targets': targets})

    def configure_optimizers(self):
        # All parameters are now trainable since LoRA is removed
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = self.trainer.estimated_stepping_batches,
            eta_min = self.hparams.learning_rate / 50
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    # on_train_epoch_end, on_validation_epoch_end, _maybe_mixup, and apply_bias_correction
    # methods are unchanged and can be copied directly from your original code.
    # For brevity, I am omitting them here, but they should be included in your final class definition.

    def on_train_epoch_end(self):
        """Aggregates training labels at the end of the training epoch."""
        if self.training_step_outputs:
            self.all_train_targets_for_plot = torch.cat(self.training_step_outputs).numpy()
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self.validation_step_outputs:
            return

        preds = torch.cat([x['preds'] for x in self.validation_step_outputs], dim=0)
        targets = torch.cat([x['targets'] for x in self.validation_step_outputs], dim=0)

        if self.trainer.world_size > 1:
            preds = self.all_gather(preds).reshape(-1)
            targets = self.all_gather(targets).reshape(-1)

        x = preds.detach()
        y = targets.detach()
        x_mean, y_mean = x.mean(), y.mean()
        x_var  = x.var(unbiased=False)

        if x_var < 1e-8:
            a = torch.tensor(1.0, device=x.device)
            b = torch.tensor(0.0, device=x.device)
        else:
            cov = ((x - x_mean) * (y - y_mean)).mean()
            a = cov / (x_var + 1e-8)
            b = y_mean - a * x_mean

        mae_raw = F.l1_loss(x, y)
        x_corr = a * x + b
        mae_corr = F.l1_loss(x_corr, y)
        bag = x_corr - y
        bag_centered = bag - bag.mean()
        y_centered = y - y.mean()
        denom = torch.sqrt((bag_centered.pow(2).sum()) * (y_centered.pow(2).sum())) + 1e-8
        bag_r = (bag_centered * y_centered).sum() / denom

        if self.trainer.is_global_zero:
            if mae_corr.item() < self.best_val_mae:
                self.best_val_mae = mae_corr.item()
                self.bias_a.copy_(a.detach())
                self.bias_b.copy_(b.detach())
                self.best_bias_epoch = int(self.current_epoch)

            self.log_dict({
                "val/mae_raw_epoch": mae_raw,
                "val/mae_corr_epoch": mae_corr,
                "val/bag_corr_after": bag_r,
                "val/bias_a_epoch": a,
                "val/bias_b_epoch": b,
                "val/best_mae_corr": torch.tensor(self.best_val_mae, device=a.device),
            }, prog_bar=False, on_epoch=True, sync_dist=True)

            if self.logger:
                self.logger.log_metrics({
                    "val/best_bias_epoch": self.best_bias_epoch
                }, step=self.global_step)

        if self.logger and self.trainer.is_global_zero:
            preds_np, targets_np = x.detach().cpu().numpy(), y.detach().cpu().numpy()
            fig, ax1 = plt.subplots(figsize=(12, 7))
            min_val, max_val = targets_np.min(), targets_np.max()
            if self.all_train_targets_for_plot is not None:
                min_val = min(min_val, self.all_train_targets_for_plot.min())
                max_val = max(max_val, self.all_train_targets_for_plot.max())
            bins = np.linspace(min_val, max_val, num=50)
            ax1.hist(targets_np, bins=bins, alpha=0.6, color="blue", label="Ground Truth (Val)", density=True)
            if self.all_train_targets_for_plot is not None:
                ax1.hist(self.all_train_targets_for_plot, bins=bins, alpha=0.8, histtype='step',
                         linewidth=1.5, color="green", label="Ground Truth (Train)", density=True)
            ax1.hist(preds_np, bins=bins, alpha=0.5, color="red", label="Predictions (Val)", density=True)
            ax1.set_title(f"Label Distributions & Predictions (Epoch {self.current_epoch})")
            ax1.set_xlabel("Value"); ax1.set_ylabel("Density")
            ax1.legend(); ax1.grid(True, alpha=0.3)
            self.logger.experiment.log({
                "validation/prediction_distribution": wandb.Image(fig)
            })
            plt.close(fig)

        self.validation_step_outputs.clear()
        self.all_train_targets_for_plot = None

    def _maybe_mixup(self, images: torch.Tensor, targets_norm: torch.Tensor):
        alpha = float(self.hparams.mixup_alpha)
        p = float(self.hparams.mixup_prob)
        if (not self.training) or alpha <= 0.0 or torch.rand(1, device=images.device).item() > p:
            return images, targets_norm, None
        lam = torch.distributions.Beta(alpha, alpha).sample().to(images.device)
        lam = torch.maximum(lam, 1.0 - lam)
        B = images.size(0)
        index = torch.randperm(B, device=images.device)
        mixed_images = lam * images + (1.0 - lam) * images[index]
        mixed_targets_norm = lam * targets_norm + (1.0 - lam) * targets_norm[index]
        return mixed_images, mixed_targets_norm, lam

    def apply_bias_correction(self, yhat: torch.Tensor) -> torch.Tensor:
        return self.bias_a * yhat + self.bias_b
    
    
class ClassificationFinetuner3(pl.LightningModule):
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
            out_channels=2,
            feature_size=feature_size,
            use_v2=True,
        )
        self.dice_ce_loss = DiceCELoss(to_onehot_y=True, softmax=True)

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

        all_features = self.encoder.swinViT(x_reshaped)

        enc0 = self.encoder1(x[:, 0, :, :, :])
        enc1 = self.encoder2(all_features[0])
        enc2 = self.encoder3(all_features[1])
        enc3 = self.encoder4(all_features[2])
        dec4 = self.encoder10(all_features[4])
        dec3 = self.decoder5(dec4, all_features[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        seg = self.decoder1(dec0, enc0)

        pooled_features = []
        for i, features in enumerate(all_features):
            pooled = self.pools[i](features)  # [B*C, dim, 1, 1, 1]
            pooled = pooled.view(B * C, -1)   # [B*C, dim]
            projected = self.projections[i](pooled)  # [B*C, 64]
            pooled_features.append(projected)
        multi_scale = torch.cat(pooled_features, dim=1)  # [B*C, 5*64]
        multi_scale = multi_scale.view(B, C * 5 * 64)    # [B, C*5*64]
        logits = self.classifier_head(multi_scale)
        return logits.squeeze(-1), seg

    def _freeze_encoder(self):
        for p in self.encoder.swinViT.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def training_step(self, batch, batch_idx):
        images, labels, label_seg = batch['image'], batch['reg_label'].float(), batch["label"]
        logits, segs = self(images)
        labels = labels.view(-1).float().to(logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_after_backward(self):
        # grads exist now
        g = [p.grad is not None and torch.isfinite(p.grad).all() for p in self.classifier_head.parameters()]
        self.log("dbg/cls_head_has_grads", float(all(g)), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        images, labels, label_seg = batch['image'], batch['reg_label'].float(), batch["label"]
        logits, segs = self(images)
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

class SimpleFPN_Decoder(nn.Module):
    """
    A lightweight Feature Pyramid Network (FPN) style decoder.
    It upsamples and fuses features from different scales of the encoder.
    """
    def __init__(self, feature_dims: List[int], img_size: Tuple[int, int, int], common_dim: int = 64):
        super().__init__()
        self.img_size = img_size
        
        # 1. Project encoder features to a common dimension
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(dim, common_dim, kernel_size=1),
                # Use GroupNorm as batch size is very small in fine-tuning
                nn.GroupNorm(8, common_dim), 
                nn.ReLU(inplace=True)
            )
             for dim in feature_dims
        ])

        # 2. Fusion and final output layers
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(common_dim * len(feature_dims), common_dim*2, kernel_size=3, padding=1),
            nn.GroupNorm(8, common_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(common_dim*2, common_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, common_dim),
            nn.ReLU(inplace=True),
        )
        # Output channel is 1 for binary segmentation logits
        self.output_conv = nn.Conv3d(common_dim, 1, kernel_size=1)

    def forward(self, features: List[torch.Tensor]):
        upsampled_features = []

        for i, feature in enumerate(features):
            # Project
            projected = self.projections[i](feature)
            # Upsample to the target image resolution
            upsampled = F.interpolate(
                projected,
                size=self.img_size,
                mode='trilinear',
                align_corners=False
            )
            upsampled_features.append(upsampled)

        # Fuse (Concatenate)
        concatenated = torch.cat(upsampled_features, dim=1)

        # Process fused features and output logits
        fused = self.fusion_conv(concatenated)
        logits = self.output_conv(fused)
        return logits

# -----------------------------------------------------------------
# Main Module: Classification Finetuner with Multi-Task Learning
# -----------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, List
from torchmetrics.classification import BinaryAccuracy, AUROC

# Assuming these imports are available
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
# Note: ContrastiveTransformer must be defined in the scope for load_from_pretrained to work
# from your_project import ContrastiveTransformer 

# -----------------------------------------------------------------
# 1. Updated FPN Decoder
# -----------------------------------------------------------------

class SimpleFPN_Decoder(nn.Module):
    """
    A lightweight FPN decoder that returns both segmentation logits and high-level features.
    """
    def __init__(self, feature_dims: List[int], img_size: Tuple[int, int, int], common_dim: int = 64):
        super().__init__()
        self.img_size = img_size
        
        # Projections
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(dim, common_dim, kernel_size=1),
                # Use GroupNorm as batch size is very small
                nn.GroupNorm(8, common_dim), 
                nn.ReLU(inplace=True)
            )
             for dim in feature_dims
        ])

        # Fusion layers
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(common_dim * len(feature_dims), common_dim*2, kernel_size=3, padding=1),
            nn.GroupNorm(8, common_dim*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(common_dim*2, common_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, common_dim),
            nn.ReLU(inplace=True),
        )
        # Output channel is 1 for binary segmentation logits
        self.output_conv = nn.Conv3d(common_dim, 1, kernel_size=1)

    def forward(self, features: List[torch.Tensor]):
        upsampled_features = []

        for i, feature in enumerate(features):
            projected = self.projections[i](feature)
            upsampled = F.interpolate(
                projected,
                size=self.img_size,
                mode='trilinear',
                align_corners=False
            )
            upsampled_features.append(upsampled)

        concatenated = torch.cat(upsampled_features, dim=1)

        # [CHANGE] Capture features before the final convolution
        fused_features = self.fusion_conv(concatenated)
        logits = self.output_conv(fused_features)
        
        # [CHANGE] Return both logits and the features for late fusion
        return logits, fused_features


class ClassificationFinetunerMTL(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        feature_size: int = 24,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        lambda_seg: float = 0.5, # Weight for the segmentation loss (Hyperparameter)
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.img_size = self.hparams.img_size if 'img_size' in self.hparams else img_size

        # --- Frozen Encoder (Same as original) ---
        self.encoder = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_v2=True,
        )

        # Determine feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *self.img_size)
            self.encoder.eval()
            all_features = self.encoder.swinViT(dummy)
            feature_dims = [f.shape[1] for f in all_features]

        # --- Classification Path Projections (Same as original) ---
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool3d(1) for _ in range(5)
        ])

        self.common_dim_cls = 64 # Dimension for classification features
        self.cls_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.common_dim_cls),
                nn.LayerNorm(self.common_dim_cls),
                nn.ReLU()
            ) for dim in feature_dims
        ])
        
        # --- Segmentation Decoder ---
        # NOTE: Assumes SimpleFPN_Decoder is defined and imported/available, returning (logits, features)
        self.common_dim_seg = 32 # Dimension for the segmentation features (auxiliary task)
        self.segmentation_decoder = SimpleFPN_Decoder(
            feature_dims=feature_dims, 
            img_size=self.img_size, 
            common_dim=self.common_dim_seg
        )
        
        # Pooling layer for the segmentation features (for fusion)
        self.seg_pool = nn.AdaptiveAvgPool3d(1)

        # --- Classifier Head (UPDATED for Fusion) ---
        # Input = (C * 5 scales * ClsDim) + SegDim
        cls_input_dim = (in_channels * 5 * self.common_dim_cls) + self.common_dim_seg
        
        self.classifier_head = nn.Sequential(
            nn.Linear(cls_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self._freeze_encoder()

        # --- Losses and Metrics ---
        # Segmentation loss: Dice + CrossEntropy
        self.seg_loss_fn = DiceCELoss(sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)
        self.val_acc = BinaryAccuracy()
        self.val_auroc = AUROC(task="binary")

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def _prepare_segmentation_labels(self, seg_labels):
        """Robustly prepares segmentation labels to ensure shape (B, 1, H, W, D)."""
        # If (B, C, H, W, D), select the first channel (as per user requirement)
        if seg_labels.ndim == 5:
             # Keep the channel dimension
             seg_labels = seg_labels[:, 0:1, ...]

        # Ensure channel dimension if input is (B, H, W, D)
        if seg_labels.ndim == 4:
             seg_labels = seg_labels.unsqueeze(1)
        
        return seg_labels

    def forward(self, x):
        # x shape: (B, C, H, W, D)
        B, C = x.shape[0], x.shape[1]
        # Reshape (B, C, H, W, D) -> (B*C, 1, H, W, D)
        x_reshaped = x.view(B * C, 1, *x.shape[2:])

        # ------------------------------------------------------
        # A. Shared Feature Extraction (Frozen)
        # ------------------------------------------------------
        self.encoder.eval() 
        with torch.no_grad():
            # Each map has shape [B*C, Dim, H', W', D']
            all_features = self.encoder.swinViT(x_reshaped) 

        # ------------------------------------------------------
        # B. Segmentation Path (Trainable, DWI only - Channel 0)
        # ------------------------------------------------------
        # Isolate DWI features
        dwi_features = []
        for features in all_features:
            f_shape = features.shape
            # 1. Unstack: [B*C, Dim, ...] -> [B, C, Dim, ...]
            f_reshaped = features.view(B, C, f_shape[1], *f_shape[2:])
            # 2. Select DWI (Channel 0): [B, Dim, ...]
            f_dwi = f_reshaped[:, 0, ...] 
            dwi_features.append(f_dwi)

        # Run the segmentation decoder
        segmentation_logits, fused_seg_features = self.segmentation_decoder(dwi_features)

        # Pool the segmentation features for fusion
        pooled_seg_features = self.seg_pool(fused_seg_features) # [B, SegDim, 1, 1, 1]
        pooled_seg_features = pooled_seg_features.view(B, -1)   # [B, SegDim]

        # ------------------------------------------------------
        # C. Classification Path (Trainable, ALL channels)
        # ------------------------------------------------------
        pooled_cls_features = []
        for i, features in enumerate(all_features):
            # Process the B*C features
            pooled = self.pools[i](features)           # [B*C, dim, 1, 1, 1]
            pooled = pooled.view(B * C, -1)            # [B*C, dim]
            projected = self.cls_projections[i](pooled) # [B*C, ClsDim] 
            pooled_cls_features.append(projected)

        # Concatenate scales and reshape
        multi_scale = torch.cat(pooled_cls_features, dim=1)      # [B*C, 5*ClsDim]
        # Reshape back to batch dimension B
        multi_scale = multi_scale.view(B, C * 5 * self.common_dim_cls) # [B, C*5*ClsDim]

        # ------------------------------------------------------
        # D. FUSION
        # ------------------------------------------------------
        # Combine the multi-modal features and the segmentation features
        combined_features = torch.cat([multi_scale, pooled_seg_features], dim=1)

        # Classify
        classification_logits = self.classifier_head(combined_features)
        
        return classification_logits.squeeze(-1), segmentation_logits


    def training_step(self, batch, batch_idx):
        images, cls_labels_raw, seg_labels_raw = batch['image'], batch['reg_label'].float(), batch["label"].float()
        
        cls_logits, seg_logits = self(images)

        # 1. Classification Loss (BCE)
        cls_labels = cls_labels_raw.view(-1).to(cls_logits.device)
        loss_cls = F.binary_cross_entropy_with_logits(cls_logits, cls_labels)

        # 2. Segmentation Loss (DiceCE)
        seg_labels = self._prepare_segmentation_labels(seg_labels_raw).to(seg_logits.device)
             
        # Handle potential size mismatches if dataloader output != img_size
        if seg_logits.shape[2:] != seg_labels.shape[2:]:
            seg_logits = F.interpolate(seg_logits, size=seg_labels.shape[2:], mode='trilinear', align_corners=False)

        loss_seg = self.seg_loss_fn(seg_logits, seg_labels)

        # 3. Total Loss (MTL): L_total = L_cls + lambda * L_seg
        loss = loss_cls + self.hparams.lambda_seg * loss_seg

        self.log('train/loss_total', loss, prog_bar=True)
        self.log('train/loss_cls', loss_cls)
        self.log('train/loss_seg', loss_seg, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, cls_labels_raw = batch['image'], batch['reg_label'].float()
        
        cls_logits, seg_logits = self(images)
        cls_labels = cls_labels_raw.view(-1).to(cls_logits.device)

        loss_cls = F.binary_cross_entropy_with_logits(cls_logits, cls_labels)
        loss = loss_cls
        
        # Check if masks are available for auxiliary loss and visualization
        has_masks = "label" in batch and batch["label"] is not None
        
        if has_masks:
            seg_labels_raw = batch["label"].float().to(seg_logits.device)
            seg_labels = self._prepare_segmentation_labels(seg_labels_raw)
                
            # Handle potential size mismatches
            if seg_logits.shape[2:] != seg_labels.shape[2:]:
                seg_logits_resized = F.interpolate(seg_logits, size=seg_labels.shape[2:], mode='trilinear', align_corners=False)
            else:
                seg_logits_resized = seg_logits

            loss_seg = self.seg_loss_fn(seg_logits_resized, seg_labels)
            # Include auxiliary loss in total validation loss
            loss += self.hparams.lambda_seg * loss_seg
            self.log('val/loss_seg', loss_seg, on_epoch=True)

        # Primary Assessment Metric: AUROC
        probs = torch.sigmoid(cls_logits)
        self.val_acc.update(probs, cls_labels)
        self.val_auroc.update(probs, cls_labels)
        self.log_dict({'val/loss': loss, 'val_acc': self.val_acc, 'val_auroc': self.val_auroc}, on_epoch=True, prog_bar=True)

        # --- [NEW] 2. Visualization (Only on the first batch if masks are available) ---
        # We visualize during the step to avoid storing outputs across the epoch.
        if batch_idx == 0 and has_masks:
            # We use the resized logits for visualization as they match the GT mask shape
            self._log_validation_images(batch, seg_logits, batch_idx)

    # [NEW] 1. Predict step implementation
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Performs prediction on a batch.
        Returns classification probabilities (primary task) and segmentation logits.
        """
        # Check if 'image' key exists, otherwise assume the batch itself is the image tensor
        if isinstance(batch, dict) and 'image' in batch:
            images = batch['image']
        else:
            images = batch

        # Run the forward pass
        cls_logits, seg_logits = self(images)
        # Calculate probabilities for the classification task
        cls_probs = torch.sigmoid(cls_logits)
        
        # Return both outputs
        return {
            "classification_probs": cls_probs,
            "segmentation_logits": seg_logits
        }


    def configure_optimizers(self):
        # Ensure all trainable parts are included (Encoder is frozen).
        # We simply filter parameters that require gradients.
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50, T_mult=2, eta_min=1e-7
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    # Keep the loading mechanism exactly the same (as requested)
    @classmethod
    def load_from_pretrained(
        cls,
        checkpoint_path: str,
        num_classes: int, # Kept for compatibility with original request
        in_channels: int,
        **kwargs
    ):
        """
        Loads only the SwinViT backbone weights from a ContrastiveTransformer checkpoint.
        """
        # Load pretrained ContrastiveTransformer (Requires definition of ContrastiveTransformer)
        try:
            # Assuming ContrastiveTransformer is defined elsewhere
            pretrain_model = ContrastiveTransformer.load_from_checkpoint(checkpoint_path)
        except NameError:
            print("Error: ContrastiveTransformer class not defined. Cannot load pretrained model.")
            raise
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
            raise

        # Handle potential mismatch in hparams structure
        if hasattr(pretrain_model, 'hparams') and pretrain_model.hparams:
             # Convert to dict if it's a Namespace or similar object
             finetuner_hparams = dict(pretrain_model.hparams)
        else:
             print("Warning: Pretrained model does not have 'hparams'. Using kwargs only.")
             finetuner_hparams = {}

        finetuner_hparams.update(kwargs)
        finetuner_hparams['in_channels'] = in_channels

        # Create finetuner with fresh weights (decoder/heads will be randomly initialized)
        model = cls(**finetuner_hparams)

        # --- Extract just swinViT weights ---
        # Assuming the backbone is located at encoder.swinViT
        try:
            src_dict = pretrain_model.encoder.swinViT.state_dict()
        except AttributeError:
            print("Error: Could not find 'encoder.swinViT' in the pretrained model structure.")
            raise
            
        dst_dict = model.encoder.swinViT.state_dict()

        # Keep only matching keys with identical shape
        filtered = {k: v for k, v in src_dict.items() if k in dst_dict and v.shape == dst_dict[k].shape}

        # Load into finetuner backbone
        msg = model.encoder.swinViT.load_state_dict(filtered, strict=False)

        print(f"\n✓ Loaded {len(filtered)} swinViT tensors from {checkpoint_path}")
        print(f"    Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}\n")
        
        # Ensure the encoder remains frozen after loading
        model._freeze_encoder()

        return model

    
    def _log_validation_images(self, batch, outputs, batch_idx):
        # Visualization logic (adapted from the original SegmentationFineTuner)
        if batch_idx > 0 or not hasattr(self, 'trainer') or self.trainer.global_rank != 0: return
        if not self.logger or not self.logger.experiment or wandb is None: return
        
        label = batch['label'][0, 0, :, :, :].cpu().numpy()
        img = batch['image'][0].cpu().numpy()
        pred = (torch.sigmoid(outputs[0]) > 0.5).squeeze(0).cpu().numpy()
        
        # Use the first modality for visualization background
        vis_img = img[0] 

        # Find slices with the most foreground voxels
        slice_idx_z = np.argmax(np.sum(label, axis=(1, 2)))
        slice_idx_y = np.argmax(np.sum(label, axis=(0, 2)))
        slice_idx_x = np.argmax(np.sum(label, axis=(0, 1)))

        # If no foreground exists, use the center slice
        if np.sum(label) == 0:
            slice_idx_z, slice_idx_y, slice_idx_x = [s // 2 for s in vis_img.shape]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        views = [
            ("Axial", vis_img[slice_idx_z, :, :], label[slice_idx_z, :, :], pred[slice_idx_z, :, :]),
            ("Coronal", vis_img[:, slice_idx_y, :], label[:, slice_idx_y, :], pred[:, slice_idx_y, :]),
            ("Sagittal", vis_img[:, :, slice_idx_x], label[:, :, slice_idx_x], pred[:, :, slice_idx_x]),
        ]
        
        for i, (title, img_slice, lbl_slice, pred_slice) in enumerate(views):
            axes[i].imshow(np.rot90(img_slice), cmap="gray")
            # Plot ground truth contour (Yellow)
            if np.any(lbl_slice): axes[i].contour(np.rot90(lbl_slice), colors='yellow', linewidths=0.8, alpha=0.9)
            # Plot prediction contour (Red)
            if np.any(pred_slice): axes[i].contour(np.rot90(pred_slice), colors='red', linewidths=0.8, alpha=0.9)
            axes[i].set_title(title); axes[i].axis('off')
            
        self.logger.experiment.log({"Validation/Prediction vs Label": wandb.Image(fig)})
        plt.close(fig)


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


