#!/usr/bin/env python
"""
Simple script to visualize volumes from the finetune data pipeline.
Shows orthogonal slices of multimodal volumes and their labels.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from augmentations.finetune_augmentation_presets import get_finetune_augmentation_params
from yucca.modules.data.augmentation.YuccaAugmentationComposer import YuccaAugmentationComposer
from yucca.modules.data.data_modules.YuccaDataModule import YuccaDataModule
from yucca.modules.data.datasets.YuccaDataset import YuccaTrainDataset
from yucca.pipeline.configuration.split_data import get_split_config
from data.dataset import FOMODataset
from data.task_configs import task1_config, task2_config, task3_config
from utils.utils import SimplePathConfig


def get_task_config(taskid):
    if taskid == 1:
        return task1_config
    elif taskid == 2:
        return task2_config
    elif taskid == 3:
        return task3_config
    else:
        raise ValueError(f"Unknown taskid: {taskid}. Supported IDs are 1, 2, and 3")


def plot_orthogonal_slices_multimodal(volumes, title="Volume", modality_names=None):
    """Plot orthogonal slices of multimodal 3D volumes."""
    if isinstance(volumes, torch.Tensor):
        volumes = volumes.numpy()

    # volumes should be (C, D, H, W) where C is number of modalities
    if volumes.ndim != 4:
        raise ValueError(f"Expected 4D volume (C,D,H,W), got shape {volumes.shape}")

    num_modalities, d, h, w = volumes.shape

    if modality_names is None:
        modality_names = [f"Modality {i+1}" for i in range(num_modalities)]

    # Create subplots: one row per modality, 3 columns per row (axial, sagittal, coronal)
    fig, axes = plt.subplots(num_modalities, 3, figsize=(12, 4 * num_modalities))
    fig.suptitle(title, fontsize=14)

    # If only one modality, axes might not be 2D
    if num_modalities == 1:
        axes = axes.reshape(1, -1)

    for mod_idx in range(num_modalities):
        volume = volumes[mod_idx]
        mod_name = modality_names[mod_idx]

        # Axial slice (middle of depth)
        axes[mod_idx, 0].imshow(volume[d//2, :, :], cmap='gray', origin='lower')
        axes[mod_idx, 0].set_title(f'{mod_name} - Axial (slice {d//2})')
        axes[mod_idx, 0].axis('off')

        # Sagittal slice (middle of width)
        axes[mod_idx, 1].imshow(volume[:, :, w//2], cmap='gray', origin='lower')
        axes[mod_idx, 1].set_title(f'{mod_name} - Sagittal (slice {w//2})')
        axes[mod_idx, 1].axis('off')

        # Coronal slice (middle of height)
        axes[mod_idx, 2].imshow(volume[:, h//2, :], cmap='gray', origin='lower')
        axes[mod_idx, 2].set_title(f'{mod_name} - Coronal (slice {h//2})')
        axes[mod_idx, 2].axis('off')

    plt.tight_layout()
    return fig


def plot_3d_label_volume(label_volume, title="3D Label Volume"):
    """Plot orthogonal slices of a 3D label volume (for segmentation tasks)."""
    if isinstance(label_volume, torch.Tensor):
        label_volume = label_volume.numpy()

    # Remove channel dimension if present
    if label_volume.ndim == 4 and label_volume.shape[0] == 1:
        label_volume = label_volume[0]
    elif label_volume.ndim != 3:
        raise ValueError(f"Expected 3D label volume, got shape {label_volume.shape}")

    d, h, w = label_volume.shape

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=12)

    # Use a colormap that highlights different label values
    cmap = 'viridis' if label_volume.max() > 1 else 'gray'

    axes[0].imshow(label_volume[d//2, :, :], cmap=cmap, origin='lower')
    axes[0].set_title(f'Axial (slice {d//2})')
    axes[0].axis('off')

    axes[1].imshow(label_volume[:, :, w//2], cmap=cmap, origin='lower')
    axes[1].set_title(f'Sagittal (slice {w//2})')
    axes[1].axis('off')

    axes[2].imshow(label_volume[:, h//2, :], cmap=cmap, origin='lower')
    axes[2].set_title(f'Coronal (slice {h//2})')
    axes[2].axis('off')

    plt.tight_layout()
    return fig


def main():
    # Configuration - SET TASK ID HERE (1, 2, or 3)
    taskid = 1  # Change this to 1, 2, or 3 to visualize different tasks

    # Get task configuration
    task_cfg = get_task_config(taskid)
    task_type = task_cfg["task_type"]
    task_name = task_cfg["task_name"]
    modalities = task_cfg["modalities"]
    labels_dict = task_cfg["labels"]

    # Configuration
    data_dir = "/home/mg873uh/Projects_kb/data/finetuning_preproc/Unified_2.6667mm_float16"
    train_data_dir = os.path.join(data_dir, task_name)
    patch_size = (96, 96, 96)
    num_samples = 3
    augmentation_preset = "basic"

    print(f"Visualizing Task {taskid}: {task_name}")
    print(f"Task type: {task_type}")
    print(f"Modalities: {modalities}")
    print(f"Labels: {labels_dict}")
    print(f"Loading {num_samples} samples from: {train_data_dir}")
    print(f"Augmentation preset: {augmentation_preset}")
    print("=" * 50)

    # Setup data pipeline exactly as in finetune.py
    path_config = SimplePathConfig(train_data_dir=train_data_dir)
    splits_config = get_split_config(
        method="simple_train_val_split",
        param=0.2,
        path_config=path_config,
    )

    # Configure augmentations based on preset
    aug_params = get_finetune_augmentation_params(augmentation_preset)
    # Use the classification augmentation preset for regression
    tt_preset = "classification" if task_type == "regression" else task_type
    augmenter = YuccaAugmentationComposer(
        patch_size=patch_size,
        task_type_preset=tt_preset,
        parameter_dict=aug_params,
        deep_supervision=False,
    )

    # Create the data module that handles loading and batching
    data_module = YuccaDataModule(
        train_dataset_class=(
            YuccaTrainDataset if task_type == "segmentation" else FOMODataset
        ),
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
        patch_size=patch_size,
        batch_size=1,
        train_data_dir=train_data_dir,
        image_extension=".npy",
        task_type=task_type,
        splits_config=splits_config,
        split_idx=0,
        num_workers=0,
        val_sampler=None,
    )

    data_module.setup("fit")

    # Load and visualize samples
    for i in range(min(num_samples, len(data_module.train_dataset))):
        print(f"\nSample {i+1}:")

        # Get training data (with augmentations and labels)
        train_sample = data_module.train_dataset[i]
        training = train_sample['image']
        training_label = train_sample['label']

        filename = os.path.basename(train_sample.get('file_path', f'sample_{i}'))

        print(f"  File: {filename}")
        print(f"  Training shape: {training.shape}, range: [{training.min():.3f}, {training.max():.3f}]")

        # Print label information
        if task_type in ["classification", "regression"]:
            if isinstance(training_label, torch.Tensor):
                label_value = training_label.item() if training_label.numel() == 1 else training_label.numpy()
            else:
                label_value = training_label

            if task_type == "classification":
                label_name = labels_dict.get(int(label_value), f"Unknown({label_value})")
                print(f"  Label: {label_value} ({label_name})")
            else:  # regression
                print(f"  Label: {label_value}")

        elif task_type == "segmentation":
            if isinstance(training_label, torch.Tensor):
                label_shape = training_label.shape
                unique_labels = torch.unique(training_label).numpy()
            else:
                label_shape = training_label.shape
                unique_labels = np.unique(training_label)
            print(f"  Label shape: {label_shape}")
            print(f"  Unique label values: {unique_labels}")

        # Convert to numpy if needed
        if isinstance(training, torch.Tensor):
            training = training.numpy()

        # Ensure we have 4D arrays (C, D, H, W)
        if training.ndim == 3:
            training = training[np.newaxis, ...]

        num_modalities = training.shape[0]
        max_mods_to_show = num_modalities

        # Create plot for training data only
        fig, axes = plt.subplots(1, max_mods_to_show, figsize=(15, 5))
        fig.suptitle(f"Sample {i+1}: {filename}", fontsize=16)

        # Handle case where we have only one modality (axes won't be a list)
        if max_mods_to_show == 1:
            axes = [axes]

        for mod_idx in range(max_mods_to_show):
            # Training volume for this modality
            train_vol = training[mod_idx]
            d, h, w = train_vol.shape

            axes[mod_idx].imshow(train_vol[d//2, :, :], cmap='gray', origin='lower')
            mod_name = modalities[mod_idx] if mod_idx < len(modalities) else f"Mod_{mod_idx}"
            axes[mod_idx].set_title(f'Training - {mod_name}')
            axes[mod_idx].axis('off')

        plt.tight_layout()
        plt.show()

        # For segmentation tasks, show the 3D label volume separately
        if task_type == "segmentation":
            if isinstance(training_label, torch.Tensor):
                label_vol = training_label.numpy()
            else:
                label_vol = training_label

            # Remove channel dimension if present
            if label_vol.ndim == 4 and label_vol.shape[0] == 1:
                label_vol = label_vol[0]

            if label_vol.ndim == 3:
                label_fig = plot_3d_label_volume(label_vol, f"Sample {i+1} - Label Volume")
                plt.show()


if __name__ == "__main__":
    main()
