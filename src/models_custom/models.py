from typing import Tuple, Optional, Union, Dict, Any

import torch
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
import wandb

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.swin_unetr import SwinUNETR, PatchMerging, PatchMergingV2
from augmentations.mask import random_mask
from pytorch_lightning.utilities import rank_zero_only


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}

class SwinUNETRPretraining(pl.LightningModule):
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 24,
        spatial_dims: int = 3,
        depths: Tuple[int, ...] = (2, 2, 2, 2),
        num_heads: Tuple[int, ...] = (2, 4, 8, 16),
        window_size: Tuple[int, int, int] = (7, 7, 7),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = True,
        mask_ratio: float = 0.6,
        mask_patch_size: int = 4,
        rec_loss_masked_only: bool = True,
        projection_dim: int = 128,
        projection_hidden_dim: int = 2048,
        temperature: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 10,
        epochs: int = 300,
        steps_per_epoch: int = 1000,
        cosine_period_ratio: float = 1.0,
        mae_weight: float = 1.0,
        contrastive_weight: float = 5.0,
        normalize: bool = True,
        use_v2: bool = False,
        disable_image_logging: bool = False,
        debug_losses: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        

        self.temperature = temperature
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            depths=depths,
            num_heads=num_heads,
            use_v2=use_v2,
        )
        self.max_image_logs_per_epoch = 10
        self.image_log_interval = 5
        self.batch_counter = 0  # Counter for tracking batch numbers
        encoder_out_dim = feature_size * 8
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *img_size)
            encoder_output = self.forward_encoder(dummy_input)
            encoder_out_dim = encoder_output.shape[1]  # Get the actual channel dimension
        print(f"Encoder output dimension: {encoder_out_dim}")
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder_out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )
        self.moco_m = 0.996                 # momentum for key encoder
        self.K = 4096                       # queue size
        self.proj_dim = projection_dim
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / self.temperature)))
        self.encoder_m = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            depths=depths,
            num_heads=num_heads,
            use_v2=use_v2,
        )
        for p_q, p_k in zip(self.swin_unetr.parameters(), self.encoder_m.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False
        self.projection_head_m = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder_out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim),
        )
        for p_q, p_k in zip(self.projection_head.parameters(), self.projection_head_m.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False
        self.register_buffer("queue", F.normalize(torch.randn(self.proj_dim, self.K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        mse_reduction = "none" if debug_losses else "mean"
        self._rec_loss_fn = nn.MSELoss(reduction=mse_reduction)
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        self.rec_loss_masked_only = rec_loss_masked_only
        self.mae_weight = mae_weight
        self.contrastive_weight = contrastive_weight
        self.disable_image_logging = disable_image_logging
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        m = self.moco_m
        for p_q, p_k in zip(self.swin_unetr.parameters(), self.encoder_m.parameters()):
            p_k.data = p_k.data * m + p_q.data * (1. - m)
        for p_q, p_k in zip(self.projection_head.parameters(), self.projection_head_m.parameters()):
            p_k.data = p_k.data * m + p_q.data * (1. - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        if self.trainer.world_size and self.trainer.world_size > 1 and dist.is_initialized():
            keys = self._concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        assert self.K % batch_size == 0, "K must be divisible by total batch size for simple pointer"
        self.queue[:, ptr:ptr+batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _concat_all_gather(self, tensor):
        """Gather tensors from all processes; no gradients."""
        tensors_gather = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(tensors_gather, tensor, async_op=False)
        return torch.cat(tensors_gather, dim=0)

    def _encode_online(self, x):
        h = self.swin_unetr.swinViT(x)[-1]
        z = self.projection_head(h)
        return F.normalize(z, dim=1)

    @torch.no_grad()
    def _encode_momentum(self, x):
        h = self.encoder_m.swinViT(x)[-1]
        z = self.projection_head_m(h)
        return F.normalize(z, dim=1)

    def forward_mae(self, x):
        """Forward pass for masked autoencoder - just use full SwinUNETR"""
        masked_x, mask = random_mask(x, self.mask_ratio, self.mask_patch_size)
        reconstruction = self.swin_unetr(masked_x)
        return reconstruction, mask
    
    def forward_encoder(self, x):
        """Forward pass through encoder only for contrastive learning"""
        hidden_states = self.swin_unetr.swinViT(x)
        return hidden_states[-1]  # Return last hidden state
        
    def contrastive_loss(self, q, k):
        """
        MoCo InfoNCE:
        q: queries from online encoder   [B, C] (normalized)
        k: keys from momentum encoder    [B, C] (normalized, no grad)
        queue:                           [C, K]
        """
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) * self.logit_scale.exp()
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits.float(), labels)

    def rec_loss(self, y_hat, y, mask=None):
        """Reconstruction MSE loss"""
        if mask is not None and self.rec_loss_masked_only:
            y_masked = y.clone()
            y_hat_masked = y_hat.clone()
            y_masked[mask] = 0
            y_hat_masked[mask] = 0
            return self._rec_loss_fn(y_hat_masked, y_masked)
        return self._rec_loss_fn(y_hat, y)
        

    def compute_pos_neg_sims(self, z1, z2):
        """
        Computes average positive similarity and average negative similarity.
        z1, z2 should be L2-normalized feature vectors.
        """
        pos_sims = torch.sum(z1 * z2, dim=1)
        avg_pos_sim = pos_sims.mean().item()
        sim_matrix = torch.matmul(z1, z2.T) 
        batch_size = z1.shape[0]
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=z1.device)
        neg_sims = sim_matrix[neg_mask]
        avg_neg_sim = neg_sims.mean().item()
        return avg_pos_sim, avg_neg_sim

    
    def training_step(self, batch, batch_idx):
        view1 = batch["view1"]
        view2 = batch["view2"]
        
        # Log view slices every 6th batch
        # self.log_view_slices(view1, view2, batch_idx)
        
        recon1, mask1 = self.forward_mae(view1)
        recon2, mask2 = self.forward_mae(view2)
        mae_loss1 = self.rec_loss(recon1, view1, mask=mask1)
        mae_loss2 = self.rec_loss(recon2, view2, mask=mask2)
        mae_loss = 0.5 * (mae_loss1 + mae_loss2)
        with torch.no_grad():
            self._momentum_update_key_encoder()
        q12 = self._encode_online(view1)          # [B, C], normalized
        with torch.no_grad():
            k12 = self._encode_momentum(view2)    # [B, C], normalized
        loss_12 = self.contrastive_loss(q12, k12)  # (view1_i, view2_i) as positives
        q21 = self._encode_online(view2)
        with torch.no_grad():
            k21 = self._encode_momentum(view1)
        loss_21 = self.contrastive_loss(q21, k21)
        contrastive_loss = 0.5 * (loss_12 + loss_21)
        with torch.no_grad():
            self._dequeue_and_enqueue(k12)
            self._dequeue_and_enqueue(k21)
        with torch.no_grad():
            pos_sim = (q12 * k12).sum(dim=1).mean()
            neg_sim = (q12 @ self.queue.clone().detach()).mean()
        self.log("train/pos_sim", pos_sim, prog_bar=True, on_step=True, batch_size=view1.size(0))
        self.log("train/neg_sim", neg_sim, prog_bar=True, on_step=True, batch_size=view1.size(0))
        total_loss = self.mae_weight * mae_loss + self.contrastive_weight * contrastive_loss
        self.log_dict({
            "train/loss": total_loss,
            "train/mae_loss": mae_loss,
            "train/contrastive_loss": contrastive_loss,
        }, prog_bar=True, on_step=True, on_epoch=True, batch_size=view1.size(0))
        return total_loss


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        view1 = batch["view1"]
        view2 = batch["view2"]
        recon1, mask1 = self.forward_mae(view1)
        recon2, mask2 = self.forward_mae(view2)
        mae_loss1 = self.rec_loss(recon1, view1, mask=mask1)
        mae_loss2 = self.rec_loss(recon2, view2, mask=mask2)
        mae_loss = 0.5 * (mae_loss1 + mae_loss2)
        q12 = self._encode_online(view1)
        k12 = self._encode_momentum(view2)
        loss_12 = self.contrastive_loss(q12, k12)
        q21 = self._encode_online(view2)
        k21 = self._encode_momentum(view1)
        loss_21 = self.contrastive_loss(q21, k21)
        contrastive_loss = 0.5 * (loss_12 + loss_21)
        total_loss = self.mae_weight * mae_loss + self.contrastive_weight * contrastive_loss
        S = (q12 @ k12.T)
        pos_sim = S.diag().mean()
        neg_sim = (q12 @ self.queue.clone().detach()).mean()
        self.log_dict({
            "val/loss": total_loss,
            "val/mae_loss": mae_loss,
            "val/contrastive_loss": contrastive_loss,
            "val/logit_scale": self.logit_scale.exp(),
            "val/dbg_pos_sim": pos_sim,
            "val/dbg_neg_sim": neg_sim,
        }, prog_bar=False, on_step=False, on_epoch=True, batch_size=view1.size(0))

        return total_loss

    def log_view_slices(self, view1, view2, batch_idx):
        """Log middle slices of view1 and view2 to wandb every 6th batch."""
        if self.hparams.disable_image_logging:
            return
            
        # Only log every 6th batch
        if batch_idx % 6 != 0:
            return
            
        try:
            # Get the middle slice indices for each dimension (assuming 3D: D, H, W)
            # view1 and view2 should have shape [B, C, D, H, W]
            batch_size = view1.shape[0]
            depth_idx = view1.shape[2] // 2
            height_idx = view1.shape[3] // 2
            width_idx = view1.shape[4] // 2
            
            # Take the first sample from the batch for visualization
            view1_sample = view1[0, 0].cpu().numpy()  # [D, H, W]
            view2_sample = view2[0, 0].cpu().numpy()  # [D, H, W]
            
            # Create figure with subplots for different slice orientations
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'View1 and View2 Middle Slices - Batch {batch_idx}', fontsize=16)
            
            # View1 slices
            axes[0, 0].imshow(view1_sample[depth_idx, :, :], cmap='gray')
            axes[0, 0].set_title(f'View1 - Axial (depth={depth_idx})')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(view1_sample[:, height_idx, :], cmap='gray')
            axes[0, 1].set_title(f'View1 - Coronal (height={height_idx})')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(view1_sample[:, :, width_idx], cmap='gray')
            axes[0, 2].set_title(f'View1 - Sagittal (width={width_idx})')
            axes[0, 2].axis('off')
            
            # View2 slices
            axes[1, 0].imshow(view2_sample[depth_idx, :, :], cmap='gray')
            axes[1, 0].set_title(f'View2 - Axial (depth={depth_idx})')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(view2_sample[:, height_idx, :], cmap='gray')
            axes[1, 1].set_title(f'View2 - Coronal (height={height_idx})')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(view2_sample[:, :, width_idx], cmap='gray')
            axes[1, 2].set_title(f'View2 - Sagittal (width={width_idx})')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Log to wandb
            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({
                    f"train/view_slices_batch_{batch_idx}": wandb.Image(fig),
                    "train/step": self.global_step
                })
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not log view slices for batch {batch_idx}: {str(e)}")

    def configure_optimizers(self):
        # Separate params: encoder gets WD, projector gets no WD
        encoder_params = []
        projector_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "projection_head" in name:
                projector_params.append(param)
            else:
                encoder_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "weight_decay": self.hparams.weight_decay},
                {"params": projector_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999)
        )

        return optimizer


class SwinUNETRClassification(SwinUNETRPretraining):
    def __init__(
        self, 
        *args,
        num_classes=1,
        freeze_encoder=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if freeze_encoder:
            for param in self.swin_unetr.parameters():
                param.requires_grad = False
        self.save_hyperparameters()
        self.num_classes = num_classes

        with torch.no_grad():
            dummy_input = torch.zeros(1, self.hparams.in_channels, *self.hparams.img_size)
            encoder_output = self.forward_encoder(dummy_input)
            encoder_out_dim = encoder_output.shape[1]

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder_out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(512, num_classes)  # 1 output for binary classification
        )
        
        # Initialize metrics for binary classification
        from torchmetrics import MetricCollection
        from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC
        
        # Train metrics
        self.train_metrics = MetricCollection({
            "train/accuracy": Accuracy(task="binary"),
            "train/precision": Precision(task="binary"),
            "train/recall": Recall(task="binary"),
            "train/f1": F1Score(task="binary"),
            "train/auroc": AUROC(task="binary"),
        })
        
        # Validation metrics
        self.val_metrics = MetricCollection({
            "val/accuracy": Accuracy(task="binary"),
            "val/precision": Precision(task="binary"),
            "val/recall": Recall(task="binary"),
            "val/f1": F1Score(task="binary"),
            "val/auroc": AUROC(task="binary"),
        })

    def forward(self, x):
        """
        Forward pass for classification.
        x: [B, C, D, H, W]
        Returns: logits for classification
        """
        encoder_output = self.forward_encoder(x)
        logits = self.classification_head(encoder_output) 
        return logits

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        logits = self.forward(x)
        
        # Squeeze logits from [B, 1] to [B]
        logits = logits.squeeze(-1)  # [B, 1] -> [B]
        
        # Make sure y is float for BCE loss and long for metrics
        y_float = y.float()
        y_long = y.long()
        
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(logits, y_float)
        
        # Calculate probabilities for metrics
        probs = torch.sigmoid(logits)
        
        # Update metrics
        self.train_metrics.update(probs, y_long)
        
        # Log loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        logits = self.forward(x)
        
        # Squeeze logits from [B, 1] to [B]
        logits = logits.squeeze(-1)  # [B, 1] -> [B]
        
        # Make sure y is float for BCE loss and long for metrics
        y_float = y.float()
        y_long = y.long()
        
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(logits, y_float)
        
        # Calculate probabilities for metrics
        probs = torch.sigmoid(logits)
        
        # Update metrics
        self.val_metrics.update(probs, y_long)
        
        # Log loss
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        
        return loss

    def on_train_epoch_start(self):
        # Reset metrics at the start of epoch
        self.train_metrics.reset()
        
    def on_validation_epoch_start(self):
        # Reset metrics at the start of epoch
        self.val_metrics.reset()
        
    def on_train_epoch_end(self):
        # Compute and log metrics at the end of epoch
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        
    def on_validation_epoch_end(self):
        # Compute and log metrics at the end of epoch
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Much lower learning rate for few-shot learning
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-5,  # Start with 1e-5 instead of default (likely 1e-3 or 1e-4)
            betas=(0.9, 0.999),
            weight_decay=0.01  # Slightly lower weight decay
        )
        
        # Add cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-7
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }

        
class SwinUNETRRegression(SwinUNETRPretraining):
    def __init__(
        self, 
        *args,
        num_outputs=1,  # Number of regression targets
        freeze_encoder=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if freeze_encoder:
            for param in self.swin_unetr.parameters():
                param.requires_grad = False
        self.save_hyperparameters()
        self.num_outputs = num_outputs

        with torch.no_grad():
            dummy_input = torch.zeros(1, self.hparams.in_channels, *self.hparams.img_size)
            encoder_output = self.forward_encoder(dummy_input)
            encoder_out_dim = encoder_output.shape[1]

        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(encoder_out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Slightly less dropout for regression
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs)  # Direct output without activation
        )
        
        # Initialize metrics for regression
        from torchmetrics import MetricCollection
        from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
        
        # Train metrics
        self.train_metrics = MetricCollection({
            "train/mae": MeanAbsoluteError(),
            "train/mse": MeanSquaredError(),
            "train/rmse": MeanSquaredError(squared=False),
            "train/r2": R2Score(),
            "train/pearson": PearsonCorrCoef(),
        })
        
        # Validation metrics
        self.val_metrics = MetricCollection({
            "val/mae": MeanAbsoluteError(),
            "val/mse": MeanSquaredError(),
            "val/rmse": MeanSquaredError(squared=False),
            "val/r2": R2Score(),
            "val/pearson": PearsonCorrCoef(),
        })

    def forward(self, x):
        """
        Forward pass for regression.
        x: [B, C, D, H, W]
        Returns: continuous values for regression
        """
        encoder_output = self.forward_encoder(x)
        predictions = self.regression_head(encoder_output) 
        return predictions

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]  # Continuous target values
        
        predictions = self.forward(x)
        
        if self.num_outputs == 1:
            predictions = predictions.squeeze(-1)  # [B, 1] -> [B]
            y = y.squeeze(-1) if y.dim() > 1 else y
        
        loss = F.mse_loss(predictions, y)
        self.train_metrics.update(predictions, y)
        
        # Log loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]  # Continuous target values
        
        predictions = self.forward(x)
        
        # Handle single vs multi-output regression
        if self.num_outputs == 1:
            predictions = predictions.squeeze(-1)  # [B, 1] -> [B]
            y = y.squeeze(-1) if y.dim() > 1 else y
        
        # Calculate loss
        loss = F.mse_loss(predictions, y)
        
        # Update metrics
        self.val_metrics.update(predictions, y)
        
        # Log loss
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        
        return loss

    def on_train_epoch_start(self):
        # Reset metrics at the start of epoch
        self.train_metrics.reset()
        
    def on_validation_epoch_start(self):
        # Reset metrics at the start of epoch
        self.val_metrics.reset()
        
    def on_train_epoch_end(self):
        # Compute and log metrics at the end of epoch
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)
        
    def on_validation_epoch_end(self):
        # Compute and log metrics at the end of epoch
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # Optimizer configuration for regression
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,  # Slightly higher than classification since regression often needs more updates
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # You might want to use ReduceLROnPlateau for regression
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # Minimize the loss
            factor=0.5,  # Reduce LR by half
            patience=10,  # Wait 10 epochs before reducing
            min_lr=1e-7,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",  # Monitor validation loss
                "interval": "epoch",
                "frequency": 1,
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """Convenience method for inference"""
        x = batch["image"] if isinstance(batch, dict) else batch
        predictions = self.forward(x)
        if self.num_outputs == 1:
            predictions = predictions.squeeze(-1)
        return predictions


class SwinUNETRPretrainingDistributed(SwinUNETRPretraining):
    """
    Enhanced version with better distributed training support
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # DON'T access self.trainer here - it doesn't exist yet!
        # Just set up the model normally
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Fixed version with better error handling"""
        # Gather keys from all GPUs
        if self.trainer.world_size and self.trainer.world_size > 1 and dist.is_initialized():
            keys = self._concat_all_gather(keys)
        
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        
        # Ensure K is divisible by the total batch size
        if self.K % batch_size != 0:
            # Adjust batch size to fit evenly
            usable_batch_size = batch_size - (batch_size % (self.K // 100))  # Use up to 1% of K
            keys = keys[:usable_batch_size]
            batch_size = usable_batch_size
        
        # Update queue
        if ptr + batch_size > self.K:
            # Handle wrap-around
            self.queue[:, ptr:self.K] = keys[:self.K-ptr].T
            self.queue[:, :ptr+batch_size-self.K] = keys[self.K-ptr:].T
        else:
            self.queue[:, ptr:ptr+batch_size] = keys.T
        
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    @rank_zero_only
    def log_view_slices(self, view1, view2, batch_idx):
        """Only log from rank 0 to avoid duplicate logging"""
        super().log_view_slices(view1, view2, batch_idx)
    
    def on_train_epoch_start(self):
        """Ensure all processes are synchronized at epoch start"""
        if dist.is_initialized():
            dist.barrier()
    
    def configure_optimizers(self):
        """Enhanced optimizer config with better LR scaling for multi-GPU"""
        # Scale learning rate by number of GPUs (linear scaling rule)
        base_lr = self.hparams.learning_rate
        if self.trainer and self.trainer.world_size:
            scaled_lr = base_lr * self.trainer.world_size
            print(f"Scaling learning rate from {base_lr} to {scaled_lr} for {self.trainer.world_size} GPUs")
        else:
            scaled_lr = base_lr
        
        # Separate params: encoder gets WD, projector gets no WD
        encoder_params = []
        projector_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "projection_head" in name:
                projector_params.append(param)
            else:
                encoder_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "weight_decay": self.hparams.weight_decay},
                {"params": projector_params, "weight_decay": 0.0},
            ],
            lr=scaled_lr,
            betas=(0.9, 0.999)
        )

        # Add warmup for distributed training
        if self.trainer and self.trainer.world_size and self.trainer.world_size > 1:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
            
            # Warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer, 
                start_factor=0.1, 
                total_iters=self.hparams.warmup_epochs
            )
            
            # Main scheduler
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.epochs - self.hparams.warmup_epochs,
                eta_min=1e-6
            )
            
            # Combined scheduler
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.hparams.warmup_epochs]
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        
        return optimizer