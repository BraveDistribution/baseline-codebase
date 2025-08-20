import pytorch_lightning as pl
import torch
import torch.nn as nn
from monai.networks.nets import MaskedAutoEncoderViT


class SelfSupervisedMAE(pl.LightningModule):
    """
    Independent PyTorch Lightning implementation of MONAI's MaskedAutoencoderViT.
    Includes utility methods for encoder extraction and transfer learning.
    """

    def __init__(
        self,
        input_channels: int = 1,
        img_size: int = 96,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        num_heads: int = 12,
        learning_rate: float = 1e-4,
    ):
        """
        Initialize the self-supervised MAE model.

        Args:
            input_channels (int): Number of input channels (default: 1)
            img_size (int): Input image size (default: 96)
            patch_size (int): Patch size for vision transformer (default: 16)
            mask_ratio (float): Ratio of patches to mask (default: 0.75)
            num_heads (int): Number of attention heads (default: 12)
            learning_rate (float): Learning rate (default: 1e-4)
        """
        super().__init__()
        self.save_hyperparameters()

        # Initialize MONAI's MaskedAutoencoderViT
        self.model = MaskedAutoEncoderViT(
            in_channels=input_channels,
            img_size=img_size,
            patch_size=patch_size,
            masking_ratio=mask_ratio,
            num_heads=num_heads,
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        """Forward pass through the MAE model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning."""
        x = batch["image"]
        y = batch.get("label", x)  # Use input as target if label not present
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning."""
        x = batch["image"]
        y = batch.get("label", x)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        """Configure optimizer for PyTorch Lightning."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def get_encoder(self):
        """
        Extract the encoder part of the MAE model.

        Returns:
            torch.nn.Module: The encoder part of the masked autoencoder
        """
        return self.model.encoder

    def get_encoder_features(self, x, mask_ratio=None):
        """
        Get encoded features from the encoder without reconstruction.

        Args:
            x (torch.Tensor): Input tensor
            mask_ratio (float, optional): Mask ratio for encoding. If None, uses model's default.

        Returns:
            tuple: (encoded_features, mask, ids_restore)
        """
        if mask_ratio is not None:
            original_mask_ratio = self.model.mask_ratio
            self.model.mask_ratio = mask_ratio

        # Forward through encoder only
        with torch.no_grad():
            features, mask, ids_restore = self.model.forward_encoder(x, mask_ratio)

        if mask_ratio is not None:
            self.model.mask_ratio = original_mask_ratio

        return features, mask, ids_restore

    def freeze_encoder(self):
        """
        Freeze the encoder parameters for transfer learning.
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """
        Unfreeze the encoder parameters.
        """
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def get_encoder_state_dict(self):
        """
        Get the state dictionary of the encoder only.

        Returns:
            dict: State dictionary of the encoder
        """
        encoder_state_dict = {}
        model_state_dict = self.model.state_dict()

        for key, value in model_state_dict.items():
            if key.startswith('encoder'):
                encoder_state_dict[key] = value

        return encoder_state_dict

    def load_encoder_weights(self, checkpoint_path):
        """
        Load encoder weights from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Filter encoder weights
        encoder_weights = {}
        for key, value in state_dict.items():
            if 'model.encoder' in key:
                # Remove 'model.' prefix if present
                new_key = key.replace('model.', '') if key.startswith('model.') else key
                encoder_weights[new_key] = value

        # Load weights into encoder
        self.model.encoder.load_state_dict(encoder_weights, strict=False)
        print(f"Loaded encoder weights from {checkpoint_path}")


# Convenience function for quick instantiation
def create_mae_model(
    input_channels: int = 1,
    img_size: int = 96,
    patch_size: int = 16,
    mask_ratio: float = 0.75,
    num_heads: int = 12,
    learning_rate: float = 1e-4,
):
    """
    Create a self-supervised MAE model with specified parameters.

    Returns:
        SelfSupervisedMAE: Configured MAE model
    """
    return SelfSupervisedMAE(
        input_channels=input_channels,
        img_size=img_size,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        num_heads=num_heads,
        learning_rate=learning_rate,
    )
