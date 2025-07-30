import torch
import pytorch_lightning as pl

from torch.utils.networks import Conv3d

class FeatureExtractor(torch.nn.Module):
    def __init__(self, block_counts, input_channels, output_channels):
        super(FeatureExtractor, self).__init__()
        self.block_counts = block_counts
        self.input_channels = input_channels
        self.output_channels = output_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
        

class PretrainingModel(pl.LightningModule):
    def __init__(self):
        self.feature_extractor = FeatureExtractor(
            block_counts=[2, 2, 2, 2],
            input_channels=1,
            output_channels=64
        )