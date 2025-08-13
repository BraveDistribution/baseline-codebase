#!/usr/bin/env python
"""
Simple script to visualize volumes from the SelfSupervisedModel data pipeline.
Shows orthogonal slices of original and augmented volumes.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from augmentations.augmentation_composer import get_pretrain_augmentations, get_val_augmentations
from data.datamodule import PretrainDataModule
from data.pretrain_split import get_pretrain_split_config
from utils.utils import SimplePathConfig


def plot_orthogonal_slices(volume, title="Volume"):
    """Plot orthogonal slices of a 3D volume."""
    if isinstance(volume, torch.Tensor):
        volume = volume.numpy()

    # Remove channel dimension if present
    if volume.ndim == 4:
        volume = volume[0]

    d, h, w = volume.shape

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=12)

    axes[0].imshow(volume[d//2, :, :], cmap='gray', origin='lower')
    axes[0].set_title(f'Axial (slice {d//2})')
    axes[0].axis('off')

    axes[1].imshow(volume[:, :, w//2], cmap='gray', origin='lower')
    axes[1].set_title(f'Sagittal (slice {w//2})')
    axes[1].axis('off')

    axes[2].imshow(volume[:, h//2, :], cmap='gray', origin='lower')
    axes[2].set_title(f'Coronal (slice {h//2})')
    axes[2].axis('off')

    plt.tight_layout()
    return fig


def main():
    # Configuration
    data_dir = "/home/mg873uh/Projects_kb/data/pretrain_preproc/FOMO60k_1.0mm_float16"
    patch_size = (240, 256, 256)
    num_samples = 3
    augmentation_preset = "all"

    print(f"Loading {num_samples} samples from: {data_dir}")
    print(f"Augmentation preset: {augmentation_preset}")
    print("=" * 50)

    # Setup data pipeline exactly as in SelfSupervisedModel
    path_config = SimplePathConfig(train_data_dir=data_dir)
    splits_config = get_pretrain_split_config(
        method="simple_train_val_split",
        idx=0,
        split_ratio=0.01,
        path_config=path_config,
    )

    # Get augmentations
    train_transforms = get_pretrain_augmentations(patch_size, augmentation_preset)
    val_transforms = get_val_augmentations()

    # Create data module
    data_module = PretrainDataModule(
        patch_size=patch_size,
        batch_size=1,
        num_workers=0,
        splits_config=splits_config,
        split_idx=0,
        train_data_dir=data_dir,
        composed_train_transforms=train_transforms,
        composed_val_transforms=val_transforms,
        dataset='contrastive',  # Use 'contrastive' to match the original context
        crop=False,  # Use the same crop setting as in the original context
    )

    data_module.setup("fit")

    # Load and visualize samples
    for i in range(min(num_samples, len(data_module.train_dataset))):
        print(f"\nSample {i+1}:")

        # Get original (validation transforms only)
        val_sample = data_module.val_dataset[i]
        original = val_sample['image']

        # Get augmented (training transforms)
        train_sample = data_module.train_dataset[i]
        augmented = train_sample['image']

        filename = os.path.basename(val_sample.get('file_path', f'sample_{i}'))

        print(f"  File: {filename}")
        print(f"  Original shape: {original.shape}, range: [{original.min():.3f}, {original.max():.3f}]")
        print(f"  Augmented shape: {augmented.shape}, range: [{augmented.min():.3f}, {augmented.max():.3f}]")

        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f"Sample {i+1}: {filename}", fontsize=14)

        # Original volume
        orig_vol = original[0] if original.ndim == 4 else original
        d1, h1, w1 = orig_vol.shape

        axes[0, 0].imshow(orig_vol[d1//2, :, :], cmap='gray', origin='lower')
        axes[0, 0].set_title('Original - Axial')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(orig_vol[:, :, w1//2], cmap='gray', origin='lower')
        axes[0, 1].set_title('Original - Sagittal')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(orig_vol[:, h1//2, :], cmap='gray', origin='lower')
        axes[0, 2].set_title('Original - Coronal')
        axes[0, 2].axis('off')

        # Augmented volume
        aug_vol = augmented[0] if augmented.ndim == 4 else augmented
        d2, h2, w2 = aug_vol.shape

        axes[1, 0].imshow(aug_vol[d2//2, :, :], cmap='gray', origin='lower')
        axes[1, 0].set_title(f'Augmented - Axial')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(aug_vol[:, :, w2//2], cmap='gray', origin='lower')
        axes[1, 1].set_title('Augmented - Sagittal')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(aug_vol[:, h2//2, :], cmap='gray', origin='lower')
        axes[1, 2].set_title('Augmented - Coronal')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
