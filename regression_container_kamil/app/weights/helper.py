# import torch

# Load checkpoint
# ckpt1 = torch.load('best-checkpoint-v4.ckpt', map_location='cpu')
# ckpt2 = torch.load('weights.ckpt', map_location='cpu')
# print(ckpt2.keys())
# state_dict = ckpt1['state_dict']

# for name, param in state_dict.items():
#     print(f"{name}: {tuple(param.shape)}")

# print(ckpt2["encoder.swinViT.patch_embed.proj.weight"])
# print(ckpt1["state_dict"]["encoder.swinViT.patch_embed.proj.weight"])
# print(ckpt1["state_dict"].keys() == ckpt2.keys())
# print(torch.equal(ckpt1["state_dict"], ckpt2.keys()))

import argparse
import os
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.functional import sigmoid
from typing import List, Dict, Any, Tuple, Sequence, Literal
from monai.networks.nets.swin_unetr import SwinUNETR
import math
import nibabel as nib
import numpy as np

import sys

from yucca.functional.preprocessing import (
    preprocess_case_for_inference,
    reverse_preprocessing,
)

from torchmetrics.regression import PearsonCorrCoef

from yucca.modules.data.augmentation.transforms.cropping_and_padding import CropPad

import wandb
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, get_peft_model

from pytorch_lightning.loggers import WandbLogger

from pathlib import Path

from yucca.modules.networks.networks.YuccaNet import YuccaNet

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

class PearsonCorrLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = (y_pred - y_pred.mean()) / (y_pred.std() + self.eps)
        y_true = (y_true - y_true.mean()) / (y_true.std() + self.eps)
        return 1.0 - (y_pred * y_true).mean()  # minimize → maximize corr


class CCCLoss(nn.Module):
    # Concordance correlation: better when scale & bias drift exist
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_hat, y):
        y_hat_mu, y_mu = y_hat.mean(), y.mean()
        y_hat_var = y_hat.var(unbiased=False) + self.eps
        y_var = y.var(unbiased=False) + self.eps
        cov = ((y_hat - y_hat_mu) * (y - y_mu)).mean()
        ccc = (2*cov) / (y_hat_var + y_var + (y_hat_mu - y_mu).pow(2) + self.eps)
        return 1 - ccc


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, delta=1.0):
        super().__init__()
        self.robust = nn.SmoothL1Loss(beta=delta)  # Huber
        self.corr = PearsonCorrLoss()
        self.ccc = CCCLoss()
        self.alpha, self.beta = alpha, beta

    def forward(self, y_hat, y):
        return self.alpha*self.robust(y_hat, y) + self.beta*self.corr(y_hat, y) + (1-self.alpha-self.beta)*self.ccc(y_hat, y)

class ConvDropoutNormNonlin(nn.Module):
    """
    2D Convolutional layers
    Arguments:
    num_in_filters {int} -- number of input filters
    num_out_filters {int} -- number of output filters
    kernel_size {tuple} -- size of the convolving kernel
    stride {tuple} -- stride of the convolution (default: {(1, 1)})
    activation {str} -- activation function (default: {'relu'})
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.5, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)

        if self.dropout_op is not None and self.dropout_op_kwargs["p"] is not None and self.dropout_op_kwargs["p"] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.norm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.activation = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.activation(self.norm(x))


class ConvDropoutNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.norm(x)


class DoubleConvDropoutNormNonlin(nn.Module):
    """
    2D Convolutional layers
    Arguments:
    num_in_filters {int} -- number of input filters
    num_out_filters {int} -- number of output filters
    kernel_size {tuple} -- size of the convolving kernel
    stride {tuple} -- stride of the convolution (default: {(1, 1)})
    activation {str} -- activation function (default: {'relu'})
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.5, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv1 = ConvDropoutNormNonlin(
            input_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )
        self.conv2 = ConvDropoutNormNonlin(
            output_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MultiLayerConvDropoutNormNonlin(nn.Module):
    """
    2D Convolutional layers
    Arguments:
    num_in_filters {int} -- number of input filters
    num_out_filters {int} -- number of output filters
    num_layers {int} -- number of conv layers, must be at least 1
    kernel_size {tuple} -- size of the convolving kernel
    stride {tuple} -- stride of the convolution (default: {(1, 1)})
    activation {str} -- activation function (default: {'relu'})
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        num_layers=2,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.5, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        assert num_layers >= 1, "Number of layers must be at least 1, got {}".format(num_layers)
        self.num_layers = num_layers

        self.conv1 = ConvDropoutNormNonlin(
            input_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        for layer in range(2, num_layers + 1):
            setattr(
                self,
                f"conv{layer}",
                ConvDropoutNormNonlin(
                    output_channels,
                    output_channels,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                ),
            )

    def forward(self, x):
        x = self.conv1(x)
        for layer in range(2, self.num_layers + 1):
            x = getattr(self, f"conv{layer}")(x)

        return x

    @staticmethod
    def get_block_constructor(n_layers):
        def _block(input_channels, output_channels, **kwargs):
            return MultiLayerConvDropoutNormNonlin(input_channels, output_channels, num_layers=n_layers, **kwargs)

        return _block


class DoubleLayerResBlock(nn.Module):
    """
    2D Convolutional layers
    Arguments:
    num_in_filters {int} -- number of input filters
    num_out_filters {int} -- number of output filters
    num_layers {int} -- number of conv layers, must be at least 1
    kernel_size {tuple} -- size of the convolving kernel
    stride {tuple} -- stride of the convolution (default: {(1, 1)})
    activation {str} -- activation function (default: {'relu'})
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        assert conv_kwargs["dilation"] == 1, "Dilation must be 1 for residual blocks"

        self.conv1 = ConvDropoutNormNonlin(
            input_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        if (conv_kwargs["stride"] != 1) or (input_channels != output_channels):
            self.downsample_skip = nn.Sequential(
                conv_op(input_channels, output_channels, kernel_size=1, padding=0, stride=conv_kwargs["stride"], bias=False),
                norm_op(output_channels, **norm_op_kwargs),
            )
        else:
            self.downsample_skip = lambda x: x

        self.conv2 = ConvDropoutNorm(
            output_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        self.final_nonlin = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.conv2(x)

        x += self.downsample_skip(residual)
        x = self.final_nonlin(x)

        return x


class MultiLayerResBlock(nn.Module):
    """
    2D Convolutional layers
    Arguments:
    num_in_filters {int} -- number of input filters
    num_out_filters {int} -- number of output filters
    num_layers {int} -- number of conv layers, must be at least 1
    kernel_size {tuple} -- size of the convolving kernel
    stride {tuple} -- stride of the convolution (default: {(1, 1)})
    activation {str} -- activation function (default: {'relu'})
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        num_layers=2,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.5, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        assert num_layers >= 1, "Number of layers must be at least 1, got {}".format(num_layers)

        assert conv_kwargs["stride"] == 1, "Stride must be 1 for residual blocks"
        assert conv_kwargs["conv_dilation"] == 1, "Dilation must be 1 for residual blocks"

        self.num_layers = num_layers

        self.conv1 = ConvDropoutNormNonlin(
            input_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        if (conv_kwargs["stride"] != 1) or (input_channels != output_channels):
            self.downsample_skip = nn.Sequential(
                conv_op(input_channels, output_channels, kernel_size=1, padding=0, stride=conv_kwargs["stride"], bias=False),
                norm_op(output_channels, **norm_op_kwargs),
            )
        else:
            self.downsample_skip = lambda x: x

        for layer in range(2, num_layers + 1):
            if layer < num_layers:
                setattr(
                    self,
                    f"conv{layer}",
                    ConvDropoutNormNonlin(
                        output_channels,
                        output_channels,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    ),
                )
            else:
                # Last layer does not have activation, is added after residual
                setattr(
                    self,
                    f"conv{layer}",
                    ConvDropoutNorm(
                        output_channels,
                        output_channels,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    ),
                )

        self.final_nonlin = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        for layer in range(2, self.num_layers + 1):
            x = getattr(self, f"conv{layer}")(x)

        x += self.downsample_skip(residual)
        x = self.final_nonlin(x)

        return x

    @staticmethod
    def get_block_constructor(n_layers):
        def _block(input_channels, output_channels, **kwargs):
            return MultiLayerResBlock(input_channels, output_channels, num_layers=n_layers, **kwargs)

        return _block

class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.depthconv = conv_op(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.pointwiseconv = conv_op(input_channels, output_channels, kernel_size=1)

class ClsRegHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = x[-1]  # only use bottleneck repr
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class UNet(YuccaNet):
    def __init__(
        self,
        mode: Literal[
            "mae", "classification", "segmentation", "regression", "enc"
        ] = "segmentation",  # prediction mode
        input_channels: int = 1,
        output_channels: int = 1,
        starting_filters: int = 64,
        encoder_block: nn.Module = MultiLayerConvDropoutNormNonlin.get_block_constructor(
            2
        ),
        decoder_block: nn.Module = MultiLayerConvDropoutNormNonlin.get_block_constructor(
            2
        ),
        use_skip_connections: bool = False,
        deep_supervision: bool = False,
    ):
        super().__init__()

        self.encoder_block = encoder_block
        self.decoder_block = decoder_block

        self.encoder = UNetEncoder(
            input_channels=input_channels,
            starting_filters=starting_filters,
            basic_block=encoder_block,
        )
        self.num_classes = output_channels
        self.mode = mode

        # Set up the decoder based on the mode
        if mode == "mae":
            self.decoder = UNetDecoder(
                output_channels=output_channels,
                use_skip_connections=use_skip_connections,
                basic_block=decoder_block,
                starting_filters=starting_filters,
            )
        elif mode == "segmentation":
            self.decoder = UNetDecoder(
                output_channels=output_channels,
                use_skip_connections=True,
                deep_supervision=deep_supervision,
                basic_block=decoder_block,
                starting_filters=starting_filters,
            )
        elif mode in ["classification", "regression"]:
            self.decoder = ClsRegHead(
                in_channels=starting_filters * 16, num_classes=output_channels
            )
        elif mode == "enc":
            self.decoder = nn.Identity()
        else:
            raise ValueError(
                "Invalid mode. Choose from 'mae', 'segmentation', 'classification', 'regression', or 'enc'"
            )

    def forward(self, x):
        enc = self.encoder(x)
        return self.decoder(enc)


class UNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        starting_filters: int = 64,
        conv_op=nn.Conv3d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        weightInitializer=None,
        basic_block=DoubleConvDropoutNormNonlin,
    ) -> None:
        super().__init__()

        # Task specific
        self.filters = starting_filters

        # Model parameters
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.weightInitializer = weightInitializer
        self.basic_block = basic_block

        self.pool_op = nn.MaxPool3d

        self.in_conv = self.basic_block(
            input_channels=input_channels,
            output_channels=self.filters,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool1 = self.pool_op(2)
        self.encoder_conv1 = self.basic_block(
            input_channels=self.filters,
            output_channels=self.filters * 2,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool2 = self.pool_op(2)
        self.encoder_conv2 = self.basic_block(
            input_channels=self.filters * 2,
            output_channels=self.filters * 4,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool3 = self.pool_op(2)
        self.encoder_conv3 = self.basic_block(
            input_channels=self.filters * 4,
            output_channels=self.filters * 8,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool4 = self.pool_op(2)
        self.encoder_conv4 = self.basic_block(
            input_channels=self.filters * 8,
            output_channels=self.filters * 16,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, x):
        x0 = self.in_conv(x)

        x1 = self.pool1(x0)
        x1 = self.encoder_conv1(x1)

        x2 = self.pool2(x1)
        x2 = self.encoder_conv2(x2)

        x3 = self.pool3(x2)
        x3 = self.encoder_conv3(x3)

        x4 = self.pool4(x3)
        x4 = self.encoder_conv4(x4)

        return [x0, x1, x2, x3, x4]


class UNetDecoder(nn.Module):
    def __init__(
        self,
        output_channels: int = 1,
        starting_filters: int = 64,
        conv_op=nn.Conv3d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        },
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        dropout_in_decoder=False,
        weightInitializer=None,
        basic_block=DoubleConvDropoutNormNonlin,
        deep_supervision=False,
        use_skip_connections=True,
    ) -> None:
        super().__init__()

        # Task specific
        self.num_classes = output_channels
        self.filters = starting_filters

        # Model parameters
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.weightInitializer = weightInitializer
        self.basic_block = basic_block
        self.deep_supervision = deep_supervision
        self.use_skip_connections = use_skip_connections

        self.upsample = torch.nn.ConvTranspose3d

        # Decoder
        if not dropout_in_decoder:
            old_dropout_p = self.dropout_op_kwargs["p"]
            self.dropout_op_kwargs["p"] = 0.0

        self.upsample1 = self.upsample(
            self.filters * 16, self.filters * 8, kernel_size=2, stride=2
        )
        self.decoder_conv1 = self.basic_block(
            input_channels=self.filters * (16 if self.use_skip_connections else 8),
            output_channels=self.filters * 8,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample2 = self.upsample(
            self.filters * 8, self.filters * 4, kernel_size=2, stride=2
        )
        self.decoder_conv2 = self.basic_block(
            input_channels=self.filters * (8 if self.use_skip_connections else 4),
            output_channels=self.filters * 4,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample3 = self.upsample(
            self.filters * 4, self.filters * 2, kernel_size=2, stride=2
        )
        self.decoder_conv3 = self.basic_block(
            input_channels=self.filters * (4 if self.use_skip_connections else 2),
            output_channels=self.filters * 2,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample4 = self.upsample(
            self.filters * 2, self.filters, kernel_size=2, stride=2
        )
        self.decoder_conv4 = self.basic_block(
            input_channels=self.filters * (2 if self.use_skip_connections else 1),
            output_channels=self.filters,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.out_conv = self.conv_op(self.filters, self.num_classes, kernel_size=1)

        if self.deep_supervision:
            self.ds_out_conv0 = self.conv_op(
                self.filters * 16, self.num_classes, kernel_size=1
            )
            self.ds_out_conv1 = self.conv_op(
                self.filters * 8, self.num_classes, kernel_size=1
            )
            self.ds_out_conv2 = self.conv_op(
                self.filters * 4, self.num_classes, kernel_size=1
            )
            self.ds_out_conv3 = self.conv_op(
                self.filters * 2, self.num_classes, kernel_size=1
            )

        if not dropout_in_decoder:
            self.dropout_op_kwargs["p"] = old_dropout_p

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, xs):
        # We assume xs contains 5 elements. One for each of the skip connections and the bottleneck representation
        # The contents of xs is: [first skip connection, ..., last skip connection, bottleneck]
        assert isinstance(xs, list), type(xs)
        assert len(xs) == 5

        x_enc = xs[4]

        if self.use_skip_connections:
            x5 = torch.cat([self.upsample1(x_enc), xs[3]], dim=1)
            x5 = self.decoder_conv1(x5)

            x6 = torch.cat([self.upsample2(x5), xs[2]], dim=1)
            x6 = self.decoder_conv2(x6)

            x7 = torch.cat([self.upsample3(x6), xs[1]], dim=1)
            x7 = self.decoder_conv3(x7)

            x8 = torch.cat([self.upsample4(x7), xs[0]], dim=1)
            x8 = self.decoder_conv4(x8)
        else:
            x5 = self.decoder_conv1(self.upsample1(x_enc))
            x6 = self.decoder_conv2(self.upsample2(x5))
            x7 = self.decoder_conv3(self.upsample3(x6))
            x8 = self.decoder_conv4(self.upsample4(x7))

        # We only want to do multiple outputs during training, therefore it is only enabled
        # when grad is also enabled because that means we're training. And if for some reason
        # grad is enabled and you're not training, then there's other, bigger problems.
        if self.deep_supervision and torch.is_grad_enabled():
            ds0 = self.ds_out_conv0(xs[4])
            ds1 = self.ds_out_conv1(x5)
            ds2 = self.ds_out_conv2(x6)
            ds3 = self.ds_out_conv3(x7)
            ds4 = self.out_conv(x8)
            return [ds4, ds3, ds2, ds1, ds0]

        logits = self.out_conv(x8)

        return logits


def unet_b(
    mode: str = "segmentation",
    input_channels: int = 1,
    output_channels: int = 1,
):
    return UNet(
        mode=mode,
        input_channels=input_channels,
        output_channels=output_channels,
        use_skip_connections=True,
        starting_filters=32,
    )


def unet_b_lw_dec(
    input_channels: int = 1,
    output_channels: int = 1,
):
    unet_model = UNet(
        input_channels=input_channels,
        output_channels=output_channels,
        decoder_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        use_skip_connections=False,
        starting_filters=32,
    )

    return unet_model


def unet_xl(
    mode: str = "segmentation",
    input_channels: int = 1,
    output_channels: int = 1,
):
    return UNet(
        input_channels=input_channels,
        output_channels=output_channels,
        mode=mode,
        use_skip_connections=True,
    )


def light_weight_decoder(
    output_channels: int = 1,
    use_skip_connections: bool = False,
    starting_filters: int = 64,
):
    decoder_block = MultiLayerConvDropoutNormNonlin.get_block_constructor(1)
    return UNetDecoder(
        output_channels=output_channels,
        starting_filters=starting_filters,
        use_skip_connections=use_skip_connections,
        basic_block=decoder_block,
    )


def standard_decoder(
    output_channels: int = 1,
    use_skip_connections: bool = False,
    starting_filters: int = 64,
    deep_supervision: bool = False,
):
    decoder_block = MultiLayerConvDropoutNormNonlin.get_block_constructor(2)
    return UNetDecoder(
        output_channels=output_channels,
        starting_filters=starting_filters,
        use_skip_connections=use_skip_connections,
        basic_block=decoder_block,
        deep_supervision=deep_supervision,
    )


def unet_xl_lw_dec(
    input_channels: int = 1,
    output_channels: int = 1,
):
    unet_model = UNet(
        input_channels=input_channels,
        output_channels=output_channels,
        decoder_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(1),
        use_skip_connections=False,
    )

    return unet_model

def load_pretrained_weights(weights_path, compile_flag):
    """Load pretrained weights with handling for compiled models and PyTorch Lightning checkpoints."""
    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))

    # Extract the state_dict from PyTorch Lightning checkpoint if needed
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        print("Loading from PyTorch Lightning checkpoint")
        state_dict = checkpoint["state_dict"]
    else:
        print("Loading from standard model checkpoint")
        state_dict = checkpoint

    # Handle compiled checkpoints when loading to uncompiled model
    if isinstance(state_dict, dict) and len(state_dict) > 0:
        first_key = next(iter(state_dict))
        if "_orig_mod" in first_key and not compile_flag:
            print("Converting compiled model weights to uncompiled format")
            uncompiled_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace("_orig_mod.", "")
                uncompiled_state_dict[new_key] = state_dict[key]
            state_dict = uncompiled_state_dict

    return state_dict

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

class RegressionHierarchicalFinetuner(pl.LightningModule):
    """
    Implements a hierarchical regression finetuner that combines:
    1. RegressionFinetuner3's SwinUNETR encoder for local (high-res) features
    2. BaseSupervisedModel's network encoder for global (low-res) features

    Key Features:
    - Dual-encoder architecture for multi-scale feature fusion
    - Uses LoRA for parameter-efficient fine-tuning on SwinUNETR
    - Compatible with HierarchicalDataset that provides local/global views
    - Normalizes regression targets using Z-score for robustness
    - Supports loading pretrained weights for both encoders separately

    Methods for loading pretrained weights:
    - load_from_pretrained(): Main class method that orchestrates loading both encoders
    - load_from_local_pretrained(): Load pretrained ContrastiveTransformer for local encoder
    - load_from_global_pretrained(): Load pretrained weights for global encoder
    """
    def __init__(
        self,
        in_channels: int,
        target_mean: float,
        target_std: float,
        global_config: dict,  # Config for global encoder (BaseSupervisedModel)
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 24,
        lora_r: int = 128,
        lora_alpha: int = 16,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.1,
        max_epochs: int = 500,
        predict_uncertainty: bool = False,
        weight_decay: float = 0.01,
        mixup_alpha: float = 0.4,
        mixup_prob: float = 0.5,
        freeze_global_encoder: bool = True,
        global_checkpoint: str = None,  # Path to pretrained global encoder checkpoint
        # Combo loss parameters
        combo_alpha: float = 0.5,  # Weight for Huber loss
        combo_beta: float = 0.3,   # Weight for Pearson correlation loss
        combo_delta: float = 1.0,  # Huber loss delta parameter
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

        # 1. Local Encoder (SwinUNETR with LoRA) - from RegressionFinetuner3
        self.local_encoder = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=self.hparams.feature_size,
            use_checkpoint=True,
            use_v2=True,
        )

        # Apply PEFT/LoRA Wrapper to local encoder
        lora_config = LoraConfig(
            r=self.hparams.lora_r,
            lora_alpha=self.hparams.lora_alpha,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
        )
        self.local_encoder = get_peft_model(self.local_encoder, lora_config)

        # 2. Global Encoder - from BaseSupervisedMode

        model_factory = getattr(sys.modules[__name__], global_config["model_name"])

        # Create only the encoder part by directly using UNetEncoder
        # First get the encoder block from the model factory to match the configuration
        temp_model = model_factory(
            input_channels=global_config["num_modalities"],
            output_channels=global_config["num_classes"],
        )
        # Extract the encoder configuration and create only the encoder
        self.global_encoder = UNetEncoder(
            input_channels=global_config["num_modalities"],
            starting_filters=temp_model.encoder.filters,
            basic_block=temp_model.encoder_block,
        )
        # Delete temporary model to free memory
        del temp_model

        # Optionally freeze global encoder
        if self.hparams.freeze_global_encoder:
            for param in self.global_encoder.parameters():
                param.requires_grad = False

        # 3. Multi-Scale Feature Extraction Setup for local encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.hparams.img_size)
            local_features = self.local_encoder.swinViT(dummy_input)
            self.local_features = local_features #INFO: Stored to object for debugging (test_hierarchical_feat_dims.py)
            self.local_feature_dims = [f.shape[1] for f in local_features]

        self.local_pools = nn.ModuleList([nn.AdaptiveAvgPool3d(1) for _ in range(5)])

        # 4. Global feature extraction
        with torch.no_grad():
            dummy_global = torch.zeros(1, global_config["num_modalities"], *self.hparams.img_size)
            global_features = self.global_encoder(dummy_global)
            self.global_features = global_features #INFO: Stored to object for debugging (test_hierarchical_feat_dims.py)
            # Handle different output types from global encoder
            if isinstance(global_features, torch.Tensor):
                self.global_feature_dims = [global_features.shape[1]] if len(global_features.shape) > 1 else [global_features.numel()]
            else:
                # If it's a list or tuple, extract all feature dimensions (same as local)
                self.global_feature_dims = [f.shape[1] for f in global_features]

        # Global feature pooling - now we need multiple pools for multi-scale features
        self.global_pools = nn.ModuleList([nn.AdaptiveAvgPool3d(1) for _ in range(len(self.global_feature_dims))])

        # 5. Projection & Fusion
        common_dim = 32

        # Local projections
        self.local_projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, common_dim), nn.ReLU()) for dim in self.local_feature_dims]
        )

        # Global projections - now handles multiple features like local
        self.global_projections = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, common_dim), nn.ReLU()) for dim in self.global_feature_dims]
        )

        # 6. Balanced feature fusion for 50%-50% split
        # Calculate dimensions for balanced representation
        local_total_dim = in_channels * len(self.local_feature_dims) * common_dim  # Fixed: use local_feature_dims
        global_total_dim = len(self.global_feature_dims) * common_dim  # Now handles multiple global features

        # Create balanced feature dimensions (50%-50% split)
        balanced_dim = 128  # Total balanced feature dimension
        local_balanced_dim = balanced_dim // 2  # 50% for local features
        global_balanced_dim = balanced_dim // 2  # 50% for global features

        # Projection layers to balance the features
        self.local_balance_projection = nn.Sequential(
            nn.Linear(local_total_dim, local_balanced_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.global_balance_projection = nn.Sequential(
            nn.Linear(global_total_dim, global_balanced_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 7. Final regression head with balanced features
        total_features = balanced_dim

        output_dim = 2 if self.hparams.predict_uncertainty else 1
        self.regression_head = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout_rate),
            nn.Linear(64, output_dim),
        )

        self.val_corr = PearsonCorrCoef()

        # Initialize combo loss
        self.combo_loss = ComboLoss(
            alpha=self.hparams.combo_alpha,
            beta=self.hparams.combo_beta,
            delta=self.hparams.combo_delta
        )

        # Lists to store step outputs for logging
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def load_from_local_pretrained(self, local_checkpoint: str):
        """
        Load pretrained weights for the local encoder from a ContrastiveTransformer checkpoint.

        Args:
            local_checkpoint: Path to the ContrastiveTransformer checkpoint
        """
        print(f"Loading pretrained local encoder from: {local_checkpoint}")
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(local_checkpoint)

        # Transfer weights to local encoder
        src_dict = pretrain_model.encoder.swinViT.state_dict()
        base_encoder = self.local_encoder.base_model.model
        dst_dict = base_encoder.swinViT.state_dict()

        filtered_state_dict = {
            k: v for k, v in src_dict.items()
            if k in dst_dict and v.shape == dst_dict[k].shape
        }

        msg = base_encoder.swinViT.load_state_dict(filtered_state_dict, strict=False)

        print(f"✓ Loaded {len(filtered_state_dict)} swinViT tensors from {local_checkpoint}")
        print(f"  Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")

        print("Local encoder wrapped with LoRA. Trainable parameters:")
        self.local_encoder.print_trainable_parameters()

    def load_from_global_pretrained(self, global_checkpoint: str):
        """
        Load pretrained weights for the global encoder from a checkpoint.

        Args:
            global_checkpoint: Path to the global encoder checkpoint
        """
        try:
            print(f"Loading pretrained global encoder from: {global_checkpoint}")

            # Load checkpoint using the utility function
            # Note: We assume the global encoder is not compiled for now
            state_dict = load_pretrained_weights(global_checkpoint, compile_flag=False)

            # Filter global encoder weights (typically starts with 'model.' or similar)
            global_encoder_weights = {}
            for key, value in state_dict.items():
                # Remove common prefixes to match the global encoder structure
                clean_key = key
                if key.startswith('model.encoder.'):
                    clean_key = key[14:]  # Remove 'model.encoder.'
                elif key.startswith('model.'):
                    clean_key = key[6:]  # Remove 'model.'
                elif key.startswith('encoder.'):
                    clean_key = key[8:]  # Remove 'encoder.'

                global_encoder_weights[clean_key] = value

            # Load weights into global encoder (non-strict to handle architectural differences)
            load_result = self.global_encoder.load_state_dict(
                global_encoder_weights, strict=False
            )

            # Handle different return types from load_state_dict
            if load_result is not None:
                if isinstance(load_result, tuple) and len(load_result) == 2:
                    missing_keys, unexpected_keys = load_result
                else:
                    # In some PyTorch versions, it might return just missing_keys or a different format
                    missing_keys = getattr(load_result, 'missing_keys', [])
                    unexpected_keys = getattr(load_result, 'unexpected_keys', [])
            else:
                # load_state_dict returned None - this is also valid in some cases
                missing_keys, unexpected_keys = [], []

            print(f"✓ Loaded global encoder weights from {global_checkpoint}")
            print(f"  Missing keys: {len(missing_keys)} | Unexpected keys: {len(unexpected_keys)}")

        except Exception as e:
            print(f"⚠️  Warning: Could not load global encoder weights from {global_checkpoint}")
            print(f"   Error: {e}")
            print("   Continuing with randomly initialized global encoder...")


    def _log_batch_images(self, local_images, global_images, targets, preds, step_name: str):
        """Creates and logs a grid of image slices for both local and global views."""
        if not self.logger:
            return

        # Use simple approach - just check if logger has experiment attribute
        if not isinstance(self.logger, WandbLogger):
            return

        # Move data to CPU and limit to a max of 2 samples to avoid cluttering
        local_images = local_images.detach().cpu().float().numpy()
        global_images = global_images.detach().cpu().float().numpy()
        targets = targets.detach().cpu().float().numpy()
        preds = preds.detach().cpu().float().numpy()
        num_samples = min(2, len(local_images))

        for i in range(num_samples):
            local_img = local_images[i]
            global_img = global_images[i]
            target = targets[i]
            pred = preds[i]

            # Create a plot grid: rows for modalities, cols for [local_axial, local_coronal, global_axial, global_coronal]
            num_modalities = local_img.shape[0]
            fig, axes = plt.subplots(num_modalities, 4, figsize=(16, 4 * num_modalities), squeeze=False)
            fig.suptitle(f"Sample {i} | True: {target:.1f} | Pred: {pred:.1f} | Local vs Global", fontsize=16)

            for c in range(num_modalities):
                # Local views
                local_vol = local_img[c]
                mid_d, mid_h, mid_w = [s // 2 for s in local_vol.shape]

                axes[c, 0].imshow(local_vol[mid_d, :, :].T, cmap="bone", origin="lower")
                axes[c, 0].set_title(f"Local Mod {c} (Axial)")
                axes[c, 0].axis("off")

                axes[c, 1].imshow(local_vol[:, mid_h, :].T, cmap="bone", origin="lower")
                axes[c, 1].set_title(f"Local Mod {c} (Coronal)")
                axes[c, 1].axis("off")

                # Global views
                global_vol = global_img[c]
                mid_d_g, mid_h_g, mid_w_g = [s // 2 for s in global_vol.shape]

                axes[c, 2].imshow(global_vol[mid_d_g, :, :].T, cmap="bone", origin="lower")
                axes[c, 2].set_title(f"Global Mod {c} (Axial)")
                axes[c, 2].axis("off")

                axes[c, 3].imshow(global_vol[:, mid_h_g, :].T, cmap="bone", origin="lower")
                axes[c, 3].set_title(f"Global Mod {c} (Coronal)")
                axes[c, 3].axis("off")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            self.logger.experiment.log({
                f"{step_name}/hierarchical_batch_visualization_{i}": wandb.Image(fig)
            })
            plt.close(fig)

    @classmethod
    def load_from_pretrained(
        cls,
        local_checkpoint: str,
        in_channels: int,
        target_mean: float,
        target_std: float,
        global_config: dict,
        global_checkpoint: str = None,
        **kwargs
    ):
        """
        Creates an instance of the hierarchical finetuner and loads pretrained weights
        for both local and global encoders.

        Args:
            local_checkpoint: Path to pretrained ContrastiveTransformer for local encoder
            in_channels: Number of input channels
            target_mean: Target mean for normalization
            target_std: Target standard deviation for normalization
            global_config: Configuration dict for global encoder
            global_checkpoint: Path to pretrained global encoder checkpoint (optional)
            **kwargs: Additional arguments for model initialization

        Returns:
            RegressionHierarchicalFinetuner: Model with loaded pretrained weights
        """
        print("Creating hierarchical finetuner with pretrained encoders...")

        # Load ContrastiveTransformer to get hyperparameters
        pretrain_model = ContrastiveTransformer.load_from_checkpoint(local_checkpoint)

        # Prepare model hyperparameters
        finetuner_hparams = pretrain_model.hparams.copy()
        finetuner_hparams.update(kwargs)
        finetuner_hparams['in_channels'] = in_channels
        finetuner_hparams['target_mean'] = target_mean
        finetuner_hparams['target_std'] = target_std
        finetuner_hparams['global_config'] = global_config
        # Note: global_checkpoint is not stored in hparams since it's handled explicitly

        # Create model instance
        model = cls(**finetuner_hparams)

        print("\nHierarchical finetuner instantiated. Loading pretrained weights...")

        # Load local encoder weights
        model.load_from_local_pretrained(local_checkpoint)

        # Load global encoder weights if provided
        if global_checkpoint and os.path.exists(global_checkpoint):
            model.load_from_global_pretrained(global_checkpoint)
        else:
            print("No global checkpoint provided or file not found. Using randomly initialized global encoder.")

        print("\n✅ Hierarchical model ready with pretrained encoders!")
        return model

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Z-score normalization."""
        eps = 1e-6
        return (x - self.target_mean) / (self.target_std + eps)

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverses Z-score normalization."""
        return x * self.target_std + self.target_mean

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass through both local and global encoders.

        Args:
            batch: Dict containing 'local' and 'global' image tensors

        Returns:
            torch.Tensor: Regression output
        """
        local_x = batch['local']  # High-res local patches
        global_x = batch['global']  # Low-res global view

        B, C, D, H, W = local_x.shape

        # 1. Local encoder (SwinUNETR with LoRA)
        local_x_reshaped = local_x.view(B * C, 1, D, H, W)
        local_features = self.local_encoder.swinViT(local_x_reshaped)

        # Pool and project local features
        local_pooled_features = [
            self.local_projections[i](self.local_pools[i](features).view(B * C, -1))
            for i, features in enumerate(local_features)
        ]
        local_multi_scale = torch.cat(local_pooled_features, dim=1).view(B, C, -1).view(B, -1)

        # 2. Global encoder
        global_features = self.global_encoder(global_x)

        # Handle different output types from global encoder and process all features
        if isinstance(global_features, (list, tuple)):
            # Pool and project all global features (same as local processing)
            global_pooled_features = [
                self.global_projections[i](self.global_pools[i](features).view(B, -1))
                for i, features in enumerate(global_features)
            ]
            global_multi_scale = torch.cat(global_pooled_features, dim=1)
        else:
            # Handle single tensor case
            if len(global_features.shape) > 2:
                global_pooled = self.global_pools[0](global_features).view(B, -1)
            else:
                global_pooled = global_features
            global_multi_scale = self.global_projections[0](global_pooled)

        # 3. Balance local and global features for 50%-50% representation
        local_balanced = self.local_balance_projection(local_multi_scale)
        global_balanced = self.global_balance_projection(global_multi_scale)

        # Debug: Check feature norms to understand contribution balance
        if self.training and torch.rand(1).item() < 0.01:  # 1% chance to log during training
            local_norm = torch.norm(local_balanced, dim=1).mean().item()
            global_norm = torch.norm(global_balanced, dim=1).mean().item()
            local_var = torch.var(local_balanced, dim=1).mean().item()
            global_var = torch.var(global_balanced, dim=1).mean().item()
            print(f"Feature balance - Local norm: {local_norm:.3f}, Global norm: {global_norm:.3f}")
            print(f"Feature variance - Local var: {local_var:.3f}, Global var: {global_var:.3f}")

        # 4. Concatenate balanced features (now 50%-50% contribution)
        combined_features = torch.cat([local_balanced, global_balanced], dim=1)

        # 5. Final regression with balanced feature representation
        output = self.regression_head(combined_features)

        return output.squeeze(-1) if not self.hparams.predict_uncertainty else output

    def compute_loss(self, pred, target):
        """Computes the combo loss (Huber + Pearson + CCC)."""
        if self.hparams.predict_uncertainty:
            pred = pred[:, 0]
        return self.combo_loss(pred, target)

    def training_step(self, batch, batch_idx):
        targets = batch['label'].float().view(-1)

        # Keep unmixed copies for logging
        batch_for_logging = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        targets_for_logging = targets.clone()

        # Normalize targets *before* MixUp
        targets_normalized = self._normalize(targets)

        # >>> MixUp here <<<
        batch, targets_normalized, lam = self._maybe_mixup(batch, targets_normalized)

        preds = self(batch)
        loss = self.compute_loss(preds, targets_normalized)

        self.log('train/loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        # Log visualization only once at epoch 0
        if self.current_epoch == 0 and not self.has_logged_train_batch:
            preds_for_log = self(batch_for_logging)
            preds_mean_norm = preds_for_log[:, 0] if self.hparams.predict_uncertainty else preds_for_log
            preds_original = self._unnormalize(preds_mean_norm.detach())
            self._log_batch_images(
                batch_for_logging['local'],
                batch_for_logging['global'],
                targets_for_logging,
                preds_original,
                "train"
            )
            self.has_logged_train_batch = True

        # Store labels for epoch-end aggregation
        self.training_step_outputs.append(targets_for_logging.detach().cpu())

        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch['label'].float().view(-1)
        preds = self(batch)

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
            self._log_batch_images(
                batch['local'],
                batch['global'],
                targets,
                preds_original,
                "validation"
            )
            self.has_logged_val_batch = True

        self.validation_step_outputs.append({'preds': preds_original, 'targets': targets})

    def on_train_epoch_end(self):
        """Aggregates training labels at the end of the training epoch."""
        if self.training_step_outputs:
            self.all_train_targets_for_plot = torch.cat(self.training_step_outputs).numpy()
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking and self.validation_step_outputs:
            if self.logger and self.trainer.global_rank == 0:
                preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu().float().numpy()
                targets = torch.cat([x['targets'] for x in self.validation_step_outputs]).cpu().float().numpy()

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

                ax1.set_title(f"Hierarchical Model - Label Distributions & Predictions (Epoch {self.current_epoch})")
                ax1.set_xlabel("Value"); ax1.set_ylabel("Density")
                ax1.legend(); ax1.grid(True, alpha=0.3)

                if hasattr(self.logger, 'experiment'):
                    self.logger.experiment.log({
                        "validation/prediction_distribution": wandb.Image(fig)
                    })
                else:
                    print("Invalid logger: Skipping validation/prediction_distribution logging.")

                plt.close(fig)

            # Clear stored data for the next epoch
            self.validation_step_outputs.clear()
            self.all_train_targets_for_plot = None

    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        total_steps = self.trainer.estimated_stepping_batches

        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.hparams.learning_rate,
        #     total_steps=total_steps,
        #     pct_start=0.1,
        #     anneal_strategy='cos',
        # )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = self.trainer.estimated_stepping_batches, # Total number of training steps
            eta_min = self.hparams.learning_rate / 50 # Go down to 2% of the max LR
        )


        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }

    def _maybe_mixup(self, batch: dict, targets_norm: torch.Tensor):
        """
        Applies MixUp to both local and global images and normalized regression targets.
        """
        alpha = float(self.hparams.mixup_alpha)
        p = float(self.hparams.mixup_prob)
        if (not self.training) or alpha <= 0.0 or torch.rand(1, device=targets_norm.device).item() > p:
            return batch, targets_norm, None

        # Sample mixing coefficient
        lam = torch.distributions.Beta(alpha, alpha).sample().to(targets_norm.device)
        lam = torch.maximum(lam, 1.0 - lam)

        B = targets_norm.size(0)
        index = torch.randperm(B, device=targets_norm.device)

        # Mix both local and global images
        mixed_batch = {}
        for key in ['local', 'global']:
            if key in batch:
                mixed_batch[key] = lam * batch[key] + (1.0 - lam) * batch[key][index]
            else:
                mixed_batch[key] = batch[key]

        # Copy other keys as-is
        for key in batch:
            if key not in ['local', 'global']:
                mixed_batch[key] = batch[key]

        mixed_targets_norm = lam * targets_norm + (1.0 - lam) * targets_norm[index]

        return mixed_batch, mixed_targets_norm, lam


model = RegressionHierarchicalFinetuner.load_from_checkpoint(str("/home/branislav/fomo25/baseline-codebase/regression_container_kamil/app/weights/hierarchical_v10.ckpt"))

state_dict = model.state_dict()
# print(state_dict.keys())  # list of all parameter & buffer names

ckpt = torch.load("/home/branislav/fomo25/baseline-codebase/regression_container_kamil/app/weights/hierarchical_v10.ckpt")
# print(ckpt.keys())
# print(ckpt["state_dict"].keys())

# 1) Key differences

state_model = model.state_dict()
state_ckpt = ckpt["state_dict"]

keys_model = set(state_model.keys())
keys_ckpt = set(state_ckpt.keys())

print("Missing in model:", keys_ckpt - keys_model)
print("Unexpected in model:", keys_model - keys_ckpt)

# 2) Compare shapes and values for overlapping keys
for k in sorted(keys_ckpt & keys_model):
    w_model = state_model[k]
    w_ckpt = state_ckpt[k]

    if w_model.shape != w_ckpt.shape:
        print(f"❌ Shape mismatch: {k}: model {tuple(w_model.shape)} vs ckpt {tuple(w_ckpt.shape)}")
    else:
        if not torch.allclose(w_model, w_ckpt, rtol=1e-5, atol=1e-7):
            diff = (w_model - w_ckpt).abs()
            print(f"⚠️ Value mismatch: {k}: max diff={diff.max().item():.3e}, mean diff={diff.mean().item():.3e}")

all_match = True

for k in state_model.keys():
    w_model = state_model[k]
    w_ckpt = state_ckpt[k]

    if w_model.shape != w_ckpt.shape:
        print(f"❌ Shape mismatch: {k}: model {tuple(w_model.shape)} vs ckpt {tuple(w_ckpt.shape)}")
        all_match = False
    elif not torch.allclose(w_model, w_ckpt, rtol=1e-5, atol=1e-7):
        diff = (w_model - w_ckpt).abs()
        print(f"⚠️ Value mismatch: {k}: max diff={diff.max().item():.3e}, mean diff={diff.mean().item():.3e}")
        all_match = False

if all_match:
    print("\n✅ All weights match exactly within tolerance!")
else:
    print("\n⚠️ Some weights differ (see above).")