import pytorch_lightning as pl

from monai.networks.nets import ViT

class VITSelfSupervisedPretraining(pl.LightningModule):
    def __init__(self, in_channels=1, patch_size=(16, 16, 16), img_size=(96, 96, 96)):
        self.vit = ViT(in_channels=in_channels, patch_size=patch_size, img_size=img_size)

    def forward(self, x):
        pass
    
    def training_step(self, batch, batch_idx):
        x = batch['image']
    
    def validation_step(self, batch, batch_idx):
        x = batch['image']
        pass
    
    def configure_optimizers(self):
        pass