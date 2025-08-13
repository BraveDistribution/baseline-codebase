"""
Example usage of the SwinUNETRClassification model (finetune)
"""

from augmentations.finetune_augmentation_presets import get_finetune_augmentation_params
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from yucca.modules.data.augmentation.YuccaAugmentationComposer import (
    YuccaAugmentationComposer,
)
from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
from yucca.modules.callbacks.loggers import YuccaLogger
from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset
from yucca.pipeline.configuration.split_data import get_split_config
from data.dataset import FOMODataset
import os
import fire
import logging
import wandb
from utils.utils import (
    SimplePathConfig)

from models_custom.models import SwinUNETRClassification
from data.task_configs import task1_config, task2_config, task3_config

import torch
import torch.nn as nn


def adapt_to_multichannel(model, new_channels=4, strategy="average"):
    """
    Adapt model from 1 channel to multiple channels by modifying first conv layer
    
    Args:
        model: Your SwinUNETR model (already created)
        new_channels: Number of input channels you want (e.g., 4)
        strategy: 
            - "average": Each channel gets 1/4 of the original weights
            - "copy": Each channel gets full copy of original weights
            - "first": Only first channel gets weights, others initialized randomly
    
    Example:
        model = SwinUNETRClassification.load_from_checkpoint("pretrained_1ch.ckpt")
        adapt_to_multichannel(model, new_channels=4, strategy="average")
    """
    # Get the first conv layer
    first_conv = model.swin_unetr.swinViT.patch_embed.proj
    
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
            # Good if each modality was normalized similarly during training
            for i in range(new_channels):
                new_conv.weight.data[:, i:i+1, :, :, :] = old_weight
                
        elif strategy == "first":
            # Only first channel gets pretrained weights
            nn.init.kaiming_normal_(new_conv.weight.data)
            new_conv.weight.data[:, 0:1, :, :, :] = old_weight
        
        # Copy bias
        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data.clone()
    
    # Replace the layer
    model.swin_unetr.swinViT.patch_embed.proj = new_conv
    
    # Update model's input channels hyperparameter
    model.hparams.in_channels = new_channels
    
    print(f"✓ Adapted model from 1 to {new_channels} channels using '{strategy}' strategy")
    
    return model

def finetune(checkpoint_path: str | None, 
             model_type: str, 
             data_path: str, 
             model_dir: str,
             patch_size=96,
             augmentation_preset: str = 'basic',
             epochs: int = 100,
             batch_size: int = 8,
             split_method: str = 'simple_train_val_split',
             split_idx: int = 0,
             experiment: str = 'default_experiment', 
             num_workers: int = 12,
             split_param: float = 0.2,
            ):
    """Finetune a model based on the provided checkpoint path and model type."""
    logging.getLogger().setLevel(logging.INFO)
    model = None
    data_module=None

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename=f"{model_type}-{{epoch:02d}}-{{val_loss:.2f}}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )

    wandb_logger = WandbLogger(
        project="fomo-finetuning",
        name=f"{model_type}",
        log_model=True,
    )

    # Create dataset splits
    if split_method == "kfold":
        split_param = int(split_param)
    elif split_method == "simple_train_val_split":
        split_param = float(split_param)
    else:
        split_param = split_param


    if model_type == 'classification': 
        task_cfg = task1_config
        task_name = task_cfg["task_name"]
        data_dir = data_path
        train_data_dir = os.path.join(data_dir, task_name)
        path_config = SimplePathConfig(train_data_dir=train_data_dir)
        splits_config = get_split_config(
            method=split_method,
            param=split_param,
            path_config=path_config,
        )

        labels = task_cfg['labels']
        aug_params = get_finetune_augmentation_params(augmentation_preset)
        tt_preset = "classification" if model_type == "regression" else model_type 
        model = SwinUNETRClassification.load_from_checkpoint(checkpoint_path, strict=False)
        adapt_to_multichannel(model, new_channels=4, strategy="average")
        augmenter = YuccaAugmentationComposer(
            patch_size=[patch_size, patch_size, patch_size],
            task_type_preset=tt_preset,
            parameter_dict=aug_params,
            deep_supervision=False,
        )
        data_module = YuccaDataModule(
            train_dataset_class=(
                YuccaTrainDataset if model_type == "segmentation" else FOMODataset
            ),
            composed_train_transforms=augmenter.train_transforms,
            composed_val_transforms=augmenter.val_transforms,
            patch_size=[patch_size, patch_size, patch_size],
            batch_size=batch_size,
            train_data_dir=train_data_dir,
            image_extension=".npy",
            task_type=model_type,
            splits_config=splits_config,
            split_idx=split_idx,
            num_workers=num_workers,
            val_sampler=None,
        )
            # Print dataset information
        
        print("Train dataset: ", data_module.splits_config.train(split_idx))
        print("Val dataset: ", data_module.splits_config.val(split_idx))
        print("Run type: ", model_type)
    
    if model_type == 'segmentation':
        task_cfg = task2_config
        pass

    if model_type == 'regression': 
        task_cfg = task3_config
        pass
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=[0],
        callbacks=[model_checkpoint_callback],
        logger=[wandb_logger],
        precision='16-mixed',
        limit_train_batches=100,
    )
    trainer.fit(model, datamodule=data_module)
    wandb.finish()
if __name__ == "__main__":
    fire.Fire(finetune)

