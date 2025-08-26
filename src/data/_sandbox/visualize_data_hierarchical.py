#!/usr/bin/env python
"""
Script to visualize volumes from the hierarchical finetune data pipeline.
Shows orthogonal slices of multimodal volumes from both local (high-res) and global (low-res) views.
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
from data.dataset import FOMODataset, HierarchicalDataset
from data.task_configs import task1_config, task2_config, task3_config
from utils.utils import SimplePathConfig


# ============================================================================
# CONFIGURATION - MODIFY THESE VARIABLES
# ============================================================================
TASKID = 1                          # Task ID (1, 2, or 3)
USE_VALIDATION_DATA = False         # True for validation data, False for training data
NUM_SAMPLES = 3                     # Number of samples to visualize
AUGMENTATION_PRESET = "basic"       # "all", "basic", or "none"
PATCH_SIZE = (96, 96, 96)           # 3D patch size
SHOW_MODALITIES_SEPARATELY = False   # Show each modality in separate subplots
LOCAL_DATA_DIR = "/home/mg873uh/Projects_kb/data/finetuning_preproc"
GLOBAL_DATA_DIR = "/home/mg873uh/Projects_kb/data/finetuning_preproc/Unified_2.6667mm_float16"
# ============================================================================


def get_task_config(taskid):
    if taskid == 1:
        return task1_config
    elif taskid == 2:
        return task2_config
    elif taskid == 3:
        return task3_config
    else:
        raise ValueError(f"Unknown taskid: {taskid}. Supported IDs are 1, 2, and 3")


def plot_hierarchical_sample(local_volume, global_volume, title="Hierarchical Sample", modality_names=None):
    """Plot both local and global volumes side by side for comparison."""
    if isinstance(local_volume, torch.Tensor):
        local_volume = local_volume.numpy()
    if isinstance(global_volume, torch.Tensor):
        global_volume = global_volume.numpy()

    # Ensure both volumes are 4D (C, D, H, W)
    if local_volume.ndim == 3:
        local_volume = local_volume[np.newaxis, ...]
    if global_volume.ndim == 3:
        global_volume = global_volume[np.newaxis, ...]

    num_modalities = local_volume.shape[0]

    if modality_names is None:
        modality_names = [f"Modality {i+1}" for i in range(num_modalities)]

    if SHOW_MODALITIES_SEPARATELY:
        # Create subplots: rows for modalities, columns for [local_axial, global_axial, local_sagittal, global_sagittal, local_coronal, global_coronal]
        fig, axes = plt.subplots(num_modalities, 6, figsize=(18, 3 * num_modalities))
        fig.suptitle(title, fontsize=16)

        # If only one modality, axes might not be 2D
        if num_modalities == 1:
            axes = axes.reshape(1, -1)

        for mod_idx in range(num_modalities):
            local_vol = local_volume[mod_idx]
            global_vol = global_volume[mod_idx]
            mod_name = modality_names[mod_idx]

            # Get middle slices
            ld, lh, lw = local_vol.shape
            gd, gh, gw = global_vol.shape

            # Row: Local Axial, Global Axial, Local Sagittal, Global Sagittal, Local Coronal, Global Coronal

            # Axial slices
            axes[mod_idx, 0].imshow(local_vol[ld//2, :, :], cmap='gray', origin='lower')
            axes[mod_idx, 0].set_title(f'{mod_name} - Local Axial')
            axes[mod_idx, 0].axis('off')

            axes[mod_idx, 1].imshow(global_vol[gd//2, :, :], cmap='gray', origin='lower')
            axes[mod_idx, 1].set_title(f'{mod_name} - Global Axial')
            axes[mod_idx, 1].axis('off')

            # Sagittal slices
            axes[mod_idx, 2].imshow(local_vol[:, :, lw//2], cmap='gray', origin='lower')
            axes[mod_idx, 2].set_title(f'{mod_name} - Local Sagittal')
            axes[mod_idx, 2].axis('off')

            axes[mod_idx, 3].imshow(global_vol[:, :, gw//2], cmap='gray', origin='lower')
            axes[mod_idx, 3].set_title(f'{mod_name} - Global Sagittal')
            axes[mod_idx, 3].axis('off')

            # Coronal slices
            axes[mod_idx, 4].imshow(local_vol[:, lh//2, :], cmap='gray', origin='lower')
            axes[mod_idx, 4].set_title(f'{mod_name} - Local Coronal')
            axes[mod_idx, 4].axis('off')

            axes[mod_idx, 5].imshow(global_vol[:, gh//2, :], cmap='gray', origin='lower')
            axes[mod_idx, 5].set_title(f'{mod_name} - Global Coronal')
            axes[mod_idx, 5].axis('off')

    else:
        # Simplified view: just show axial slices of all modalities
        fig, axes = plt.subplots(2, num_modalities, figsize=(4 * num_modalities, 8))
        fig.suptitle(title, fontsize=16)

        if num_modalities == 1:
            axes = axes.reshape(-1, 1)

        for mod_idx in range(num_modalities):
            local_vol = local_volume[mod_idx]
            global_vol = global_volume[mod_idx]
            mod_name = modality_names[mod_idx]

            ld, lh, lw = local_vol.shape
            gd, gh, gw = global_vol.shape

            # Top row: Local views
            axes[0, mod_idx].imshow(local_vol[ld//2, :, :], cmap='gray', origin='lower')
            axes[0, mod_idx].set_title(f'{mod_name} - Local ({local_vol.shape})')
            axes[0, mod_idx].axis('off')

            # Bottom row: Global views
            axes[1, mod_idx].imshow(global_vol[gd//2, :, :], cmap='gray', origin='lower')
            axes[1, mod_idx].set_title(f'{mod_name} - Global ({global_vol.shape})')
            axes[1, mod_idx].axis('off')

    plt.tight_layout()
    return fig


def print_sample_info(sample, sample_idx, task_type, labels_dict):
    """Print detailed information about a sample."""
    print(f"\n{'='*60}")
    print(f"SAMPLE {sample_idx + 1}")
    print(f"{'='*60}")

    # File path
    if 'file_path' in sample:
        filename = os.path.basename(sample['file_path'])
        print(f"File: {filename}")

    # Local volume info
    if 'local' in sample:
        local_vol = sample['local']
        if isinstance(local_vol, torch.Tensor):
            local_shape = local_vol.shape
            local_range = f"[{local_vol.min():.3f}, {local_vol.max():.3f}]"
        else:
            local_shape = local_vol.shape
            local_range = f"[{local_vol.min():.3f}, {local_vol.max():.3f}]"
        print(f"Local volume shape: {local_shape}, range: {local_range}")

    # Global volume info
    if 'global' in sample:
        global_vol = sample['global']
        if isinstance(global_vol, torch.Tensor):
            global_shape = global_vol.shape
            global_range = f"[{global_vol.min():.3f}, {global_vol.max():.3f}]"
        else:
            global_shape = global_vol.shape
            global_range = f"[{global_vol.min():.3f}, {global_vol.max():.3f}]"
        print(f"Global volume shape: {global_shape}, range: {global_range}")

    # Label info
    if 'label' in sample:
        label = sample['label']
        if isinstance(label, torch.Tensor):
            label_value = label.item() if label.numel() == 1 else label.numpy()
        else:
            label_value = label

        if task_type == "classification":
            label_name = labels_dict.get(int(label_value), f"Unknown({label_value})")
            print(f"Label: {label_value} ({label_name})")
        elif task_type == "regression":
            print(f"Label: {label_value}")
        elif task_type == "segmentation":
            if isinstance(label, torch.Tensor):
                unique_labels = torch.unique(label).numpy()
            else:
                unique_labels = np.unique(label)
            print(f"Label shape: {label.shape if hasattr(label, 'shape') else 'scalar'}")
            print(f"Unique label values: {unique_labels}")


def main():
    # Get task configuration
    task_cfg = get_task_config(TASKID)
    task_type = task_cfg["task_type"]
    task_name = task_cfg["task_name"]
    modalities = task_cfg["modalities"]
    labels_dict = task_cfg["labels"]
    train_data_dir = os.path.join(LOCAL_DATA_DIR, task_name)

    print(f"{'='*60}")
    print(f"HIERARCHICAL DATA VISUALIZATION")
    print(f"{'='*60}")
    print(f"Task {TASKID}: {task_name}")
    print(f"Task type: {task_type}")
    print(f"Modalities: {modalities}")
    print(f"Labels: {labels_dict}")
    print(f"Data source: {'Validation' if USE_VALIDATION_DATA else 'Training'}")
    print(f"Samples to show: {NUM_SAMPLES}")
    print(f"Augmentation preset: {AUGMENTATION_PRESET}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Data dir: {train_data_dir}")

    # Setup data pipeline exactly as in finetune.py
    path_config = SimplePathConfig(train_data_dir=train_data_dir)
    splits_config = get_split_config(
        method="simple_train_val_split",
        param=0.2,
        path_config=path_config,
    )

    # Configure augmentations based on preset
    aug_params = get_finetune_augmentation_params(AUGMENTATION_PRESET)
    # Use the classification augmentation preset for regression
    tt_preset = "classification" if task_type == "regression" else task_type
    augmenter = YuccaAugmentationComposer(
        patch_size=PATCH_SIZE,
        task_type_preset=tt_preset,
        parameter_dict=aug_params,
        deep_supervision=False,
    )

    # Select hierarchical dataset
    dataset_class = HierarchicalDataset

    # Create the data module exactly as in finetune.py with hierarchical flag
    data_module = YuccaDataModule(
        train_dataset_class=dataset_class,
        composed_train_transforms=augmenter.train_transforms,
        composed_val_transforms=augmenter.val_transforms,
        patch_size=PATCH_SIZE,
        batch_size=1,  # Use batch size 1 for visualization
        train_data_dir=train_data_dir,
        image_extension=".npy",
        task_type=task_type,
        splits_config=splits_config,
        split_idx=0,
        num_workers=0,  # Use 0 workers for visualization
        val_sampler=None,
    )

    data_module.setup("fit")

    # Choose dataset based on configuration
    if USE_VALIDATION_DATA:
        dataset = data_module.val_dataset
        print(f"Using validation dataset with {len(dataset)} samples")
    else:
        dataset = data_module.train_dataset
        print(f"Using training dataset with {len(dataset)} samples")

    # Load and visualize samples
    num_to_show = min(NUM_SAMPLES, len(dataset))
    print(f"Showing {num_to_show} samples...")

    for i in range(num_to_show):
        # Get sample from hierarchical dataset
        sample = dataset[i]

        # Print detailed sample information
        print_sample_info(sample, i, task_type, labels_dict)

        # Extract local and global volumes
        local_volume = sample['local']
        global_volume = sample['global']

        # Convert to numpy if needed
        if isinstance(local_volume, torch.Tensor):
            local_volume = local_volume.numpy()
        if isinstance(global_volume, torch.Tensor):
            global_volume = global_volume.numpy()

        # Ensure we have 4D arrays (C, D, H, W)
        if local_volume.ndim == 3:
            local_volume = local_volume[np.newaxis, ...]
        if global_volume.ndim == 3:
            global_volume = global_volume[np.newaxis, ...]

        # Get filename for title
        filename = "Unknown"
        if 'file_path' in sample:
            filename = os.path.basename(sample['file_path'])

        # Create the hierarchical visualization
        title = f"Sample {i+1}: {filename}"
        if 'label' in sample:
            label = sample['label']
            if isinstance(label, torch.Tensor):
                label_value = label.item() if label.numel() == 1 else label.numpy()
            else:
                label_value = label

            if task_type == "classification":
                label_name = labels_dict.get(int(label_value), f"Unknown({label_value})")
                title += f" | Label: {label_value} ({label_name})"
            elif task_type == "regression":
                title += f" | Label: {label_value}"

        fig = plot_hierarchical_sample(
            local_volume,
            global_volume,
            title=title,
            modality_names=modalities[:local_volume.shape[0]]  # Use modality names from config
        )

        plt.show()

        # Print comparison info
        print(f"\nResolution comparison:")
        print(f"  Local volume:  {local_volume.shape} (high resolution)")
        print(f"  Global volume: {global_volume.shape} (low resolution)")

    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
