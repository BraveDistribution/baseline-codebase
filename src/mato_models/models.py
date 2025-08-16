from typing import Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinUNETR
from augmentations.mask import random_mask


class ContrastiveTransformer(pl.LightningModule):
    def __init__(
        self, 
        patch_size: Sequence[int] = (4, 4, 4),
        learning_rate: float = 1e-4,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        mask_ratio: float = 0.6,
        temperature: float = 0.1,
        queue_size: int = 4096,
        momentum: float = 0.996,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Main encoder
        self.encoder = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,
        )
        
        # Momentum encoder for MoCo
        self.encoder_m = SwinUNETR(
            img_size=img_size,
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
        """InfoNCE loss for MoCo"""
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
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
        total_loss = mae_loss + 5.0 * contrastive_loss
        
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
        
        total_loss = mae_loss + 5.0 * contrastive_loss
        
        self.log_dict({
            "val/loss": total_loss,
            "val/mae_loss": mae_loss,
            "val/contrastive_loss": contrastive_loss,
        }, prog_bar=False, on_epoch=True, sync_dist=True)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        return optimizer

    

    