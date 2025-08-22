from typing import Sequence, Tuple, Dict, Any
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
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
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
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update learning rate at each step
                'frequency': 1,
            }
        }


class ClassificationFineTuner(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        in_channels: int = 1,
        learning_rate: float = 1e-4,
        freeze_encoder: bool = False,
        dropout_rate: float = 0.1,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        min_lr: float = 1e-6,
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
        num_classes: int = None,
        strict: bool = False,
        in_channels: int = None,
        multichannel_strategy: str = "average",
        **kwargs
    ):
        """
        Load from checkpoint with custom handling for architecture mismatch.
        
        This overrides the default load_from_checkpoint to handle loading
        from the ContrastiveTransformer checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            num_classes: Number of classes for classification
            in_channels: Number of input channels (if different from checkpoint)
            multichannel_strategy: Strategy for adapting weights if in_channels > 1
                - "average": Each channel gets 1/N of original weights
                - "copy": Each channel gets full copy
                - "first": Only first channel gets weights
        """
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get hyperparameters from the original model
        loaded_hparams = checkpoint.get('hyper_parameters', {})
        
        # Extract relevant hyperparameters that we want to keep
        img_size = loaded_hparams.get('img_size', (96, 96, 96))
        feature_size = loaded_hparams.get('feature_size', 24)
        
        # Update with new parameters for classification
        kwargs.update({
            'img_size': img_size,
            'feature_size': feature_size,
            'in_channels': 1,  # Always start with 1 channel to match checkpoint
        })
        
        # num_classes is required for classification
        if num_classes is None:
            raise ValueError("num_classes must be specified for classification fine-tuning")
        kwargs['num_classes'] = num_classes
        
        # Create new model instance with 1 channel first
        model = cls(**kwargs)
        
        # Extract and load only encoder weights
        state_dict = checkpoint['state_dict']
        encoder_state_dict = {}
        
        for k, v in state_dict.items():
            # Only load encoder weights (not encoder_m, projection, or projection_m)
            if k.startswith('encoder.') and not k.startswith('encoder_m.'):
                encoder_state_dict[k] = v
        
        # Load encoder weights into the model
        missing_keys, unexpected_keys = model.load_state_dict(
            encoder_state_dict, 
            strict=False
        )
        
        print(f"Loaded encoder weights from {checkpoint_path}")
        print(f"Missing keys: {len(missing_keys)} (expected: classifier weights)")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        # NOW adapt to multi-channel if needed (after weights are loaded)
        if in_channels and in_channels > 1:
            model.adapt_to_multichannel(new_channels=in_channels, strategy=multichannel_strategy)
        
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
        images = batch['image']
        labels = batch['label'].long()
        
        # Forward pass
        logits = self(images).squeeze()  # Remove ALL singleton dimensions
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())  # BCE expects float labels
        
        # Calculate accuracy
        preds = torch.sigmoid(logits).round()
        acc = (preds == labels).float().mean()

        # Logging
        self.log_dict({
            'train/loss': loss,
            'train/acc': acc,
        }, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label'].long()
        
        logits = self(images).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        preds = torch.sigmoid(logits)
        acc = (preds == labels).float().mean()
        self.log_dict({
            'val/loss': loss,
            'val/acc': acc,
        }, prog_bar=True, on_epoch=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
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

class SegmentationFineTuner(pl.LightningModule):
    """
    Fine-tunes a pre-trained SwinUNETR model for a semantic segmentation task.
    Includes sliding window inference for validation, testing, and prediction.
    """
    def __init__(
        self,
        num_classes: int,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        in_channels: int = 1,
        learning_rate: float = 1e-4,
        freeze_encoder: bool = False,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        min_lr: float = 1e-6,
        sw_batch_size: int = 4,        # <-- Sliding window batch size
        sw_overlap: float = 0.5,       # <-- Sliding window overlap
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

        self.encoder = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=True,
            use_v2=True,
        )

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric_test = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

        self.sliding_window_inferer = SlidingWindowInferer(
            roi_size=img_size,
            sw_batch_size=sw_batch_size,
            overlap=sw_overlap,
            mode="gaussian",     
        )

        if self.freeze_encoder:
            self._freeze_encoder()
            
    def _freeze_encoder(self):
        """Freeze encoder (SwinViT) weights."""
        print("Freezing encoder weights.")
        for param in self.encoder.swinViT.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self):
        """Unfreeze encoder (SwinViT) weights."""
        print("Unfreezing encoder weights.")
        for param in self.encoder.swinViT.parameters():
            param.requires_grad = True
            
    def adapt_to_multichannel(self, new_channels: int = 4, strategy: str = "average"):
        """Adapt model from 1 channel to multiple channels by modifying the first conv layer."""
        first_conv = self.encoder.swinViT.patch_embed.proj
        if first_conv.in_channels == new_channels:
            print(f"Model already has {new_channels} input channels.")
            return

        old_weight = first_conv.weight.data.clone()
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
                new_conv.weight.data = old_weight.repeat(1, new_channels, 1, 1, 1) / new_channels
            elif strategy == "copy":
                 new_conv.weight.data = old_weight.repeat(1, new_channels, 1, 1, 1)
            else: # "first" or default
                nn.init.kaiming_normal_(new_conv.weight.data)
                new_conv.weight.data[:, 0:1, :, :, :] = old_weight

            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data.clone()

        self.encoder.swinViT.patch_embed.proj = new_conv
        self.hparams.in_channels = new_channels
        self.in_channels = new_channels
        print(f"✓ Adapted model from {first_conv.in_channels} to {new_channels} channels using '{strategy}' strategy.")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        num_classes: int,
        strict: bool = False,
        in_channels: int = None,
        multichannel_strategy: str = "average",
        **kwargs
    ):
        """Loads a pre-trained encoder from a ContrastiveTransformer checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        loaded_hparams = checkpoint.get('hyper_parameters', {})

        img_size = loaded_hparams.get('img_size', (96, 96, 96))
        feature_size = loaded_hparams.get('feature_size', 24)

        kwargs.update({
            'img_size': img_size,
            'feature_size': feature_size,
            'num_classes': num_classes,
            'in_channels': 1,
        })

        model = cls(**kwargs)
        state_dict = checkpoint['state_dict']
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.') and not k.startswith('encoder_m.'):
                encoder_state_dict[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded encoder weights from {checkpoint_path}")
        print(f"Missing keys: {len(missing_keys)} (expected: decoder weights)")
        print(f"Unexpected keys: {len(unexpected_keys)}")

        if in_channels and in_channels > 1:
            model.adapt_to_multichannel(new_channels=in_channels, strategy=multichannel_strategy)

        return model


    def forward(self, x):
        """Forward pass for training on a single patch."""
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self(images)
        loss = self.loss_function(outputs, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']
        outputs = self.sliding_window_inferer(inputs=images, network=self.encoder)
        loss = self.loss_function(outputs, labels)

        post_pred = torch.argmax(outputs, dim=1, keepdim=True)
        self.dice_metric(y_pred=post_pred, y=labels)

        self.log('val/loss', loss, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        val_dice = self.dice_metric.aggregate().item()
        self.log('val/dice', val_dice, prog_bar=True)
        self.dice_metric.reset()

    def on_test_epoch_end(self):
        test_dice = self.dice_metric_test.aggregate().item()
        self.log('test/dice', test_dice)
        self.dice_metric_test.reset()

    def test_step(self, batch, batch_idx):
        full_data_tensor = batch['data']
        images = full_data_tensor[:, :-1]  # All channels except the last
        labels = full_data_tensor[:, -1:]  # The last channel is the label
        outputs = self.sliding_window_inferer(inputs=images, network=self.encoder)
        # Calculate metrics as before
        post_pred = torch.argmax(outputs, dim=1, keepdim=True)
        self.dice_metric_test(y_pred=post_pred, y=labels)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch['data']
        prediction = self.sliding_window_inferer(inputs=images, network=self.encoder)
        return {"prediction": prediction, "case_id": batch["case_id"]}

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.freeze_encoder:
            decoder_params = [p for name, p in self.encoder.named_parameters() if not name.startswith('swinViT.') and p.requires_grad]
            params = decoder_params
            print(f"Optimizing {len(params)} decoder parameters.")
        else:
            params = self.parameters()

        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=0.01)

        def lr_lambda(current_step: int):
            num_training_steps = self.trainer.estimated_stepping_batches
            if num_training_steps == float('inf'): # Handle edge case
                num_training_steps = self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())

            num_warmup_steps = int(num_training_steps * self.warmup_epochs / self.max_epochs)
            
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed = (1 - self.min_lr / self.learning_rate) * cosine_decay + self.min_lr / self.learning_rate
            return decayed

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


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