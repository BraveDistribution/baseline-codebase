#!/usr/bin/env python3
"""
Simple visualization script for CombinedPretrainingV2 dataset.
Shows orthogonal cuts (axial, sagittal, coronal) for view1 and view2 from a few samples.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch

# Add the parent directory to the path to import dataset
import sys
sys.path.append('/home/mg873uh/Projects_kb/baseline-codebase/src')

from data.dataset import CombinedPretrainingV2


def create_orthogonal_cuts(volume, title=""):
    """
    Create orthogonal cuts (axial, sagittal, coronal) for a 3D volume.

    Args:
        volume: numpy array of shape (C, D, H, W) or (D, H, W)
        title: string title for the plot

    Returns:
        fig, axes for the plot
    """
    # Handle channel dimension
    if volume.ndim == 4:
        volume = volume[0]  # Take first channel
    elif volume.ndim == 3:
        pass  # Already correct shape
    else:
        raise ValueError(f"Expected 3D or 4D volume, got {volume.ndim}D")

    D, H, W = volume.shape

    # Get middle slices
    mid_axial = D // 2
    mid_sagittal = W // 2
    mid_coronal = H // 2

    # Create the plots
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Axial slice (XY plane)
    axes[0].imshow(volume[mid_axial, :, :], cmap='gray', origin='lower')
    axes[0].set_title(f'{title} - Axial (slice {mid_axial}/{D})')
    axes[0].axis('off')

    # Sagittal slice (YZ plane)
    axes[1].imshow(volume[:, :, mid_sagittal], cmap='gray', origin='lower')
    axes[1].set_title(f'{title} - Sagittal (slice {mid_sagittal}/{W})')
    axes[1].axis('off')

    # Coronal slice (XZ plane)
    axes[2].imshow(volume[:, mid_coronal, :], cmap='gray', origin='lower')
    axes[2].set_title(f'{title} - Coronal (slice {mid_coronal}/{H})')
    axes[2].axis('off')

    plt.tight_layout()
    return fig, axes


def visualize_contrastive_samples(num_samples=3):
    """
    Visualize samples from CombinedPretrainingV2 dataset.
    Shows view1 and view2 for each sample with orthogonal cuts.
    """

    # Dataset configuration
    data_dir = "/home/mg873uh/Projects_kb/data/pretrain_preproc/FOMO60k_2.667mm_float16"
    patch_size = (240, 256, 256)

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please update the data_dir path in the script")
        return

    # Get list of available files to determine patients
    available_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    if not available_files:
        print(f"No .npy files found in {data_dir}")
        return

    print(f"Found {len(available_files)} .npy files")

    # Extract unique patients (assuming filename pattern: sub_XXX_ses_YYY_modality.npy)
    patients = set()
    for filename in available_files:
        if filename.startswith('sub_'):
            parts = filename.split('_')
            if len(parts) >= 2:
                patient_id = parts[1]
                patients.add(patient_id)

    patients = list(patients)[:10]  # Take first 10 patients
    print(f"Using {len(patients)} patients: {patients}")

    # Create dataset
    dataset = CombinedPretrainingV2(
        patients=set(patients),
        patch_size=patch_size,
        data_dir=data_dir,
        mode='multimodal',  # or 'augmentation'
        crop=False,  # Use the same crop setting as in the original context
    )

    print(f"Dataset created with {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0  # Use 0 for debugging
    )

    # Visualize samples
    sample_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= num_samples:
            break

        view1 = batch['view1'][0]  # Remove batch dimension
        view2 = batch['view2'][0]  # Remove batch dimension
        patient = batch['patient'][0]
        path1 = batch['path1'][0]
        path2 = batch['path2'][0]

        # Convert to numpy if tensor
        if torch.is_tensor(view1):
            view1 = view1.numpy()
        if torch.is_tensor(view2):
            view2 = view2.numpy()

        print(f"\nSample {sample_count + 1}:")
        print(f"  Patient: {patient}")
        print(f"  View1 shape: {view1.shape}")
        print(f"  View2 shape: {view2.shape}")
        print(f"  Path1: {os.path.basename(path1)}")
        print(f"  Path2: {os.path.basename(path2)}")
        print(f"  View1 range: [{view1.min():.3f}, {view1.max():.3f}]")
        print(f"  View2 range: [{view2.min():.3f}, {view2.max():.3f}]")

        # Create visualization
        fig = plt.figure(figsize=(15, 8))

        # View1 cuts
        plt.subplot(2, 3, 1)
        mid_slice = view1.shape[-3] // 2 if view1.ndim == 4 else view1.shape[0] // 2
        img1 = view1[0, mid_slice, :, :] if view1.ndim == 4 else view1[mid_slice, :, :]
        plt.imshow(img1, cmap='gray', origin='lower')
        plt.title(f'View1 - Axial\n{os.path.basename(path1)}')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        mid_slice = view1.shape[-1] // 2
        img1 = view1[0, :, :, mid_slice] if view1.ndim == 4 else view1[:, :, mid_slice]
        plt.imshow(img1, cmap='gray', origin='lower')
        plt.title('View1 - Sagittal')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        mid_slice = view1.shape[-2] // 2
        img1 = view1[0, :, mid_slice, :] if view1.ndim == 4 else view1[:, mid_slice, :]
        plt.imshow(img1, cmap='gray', origin='lower')
        plt.title('View1 - Coronal')
        plt.axis('off')

        # View2 cuts
        plt.subplot(2, 3, 4)
        mid_slice = view2.shape[-3] // 2 if view2.ndim == 4 else view2.shape[0] // 2
        img2 = view2[0, mid_slice, :, :] if view2.ndim == 4 else view2[mid_slice, :, :]
        plt.imshow(img2, cmap='gray', origin='lower')
        plt.title(f'View2 - Axial\n{os.path.basename(path2)}')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        mid_slice = view2.shape[-1] // 2
        img2 = view2[0, :, :, mid_slice] if view2.ndim == 4 else view2[:, :, mid_slice]
        plt.imshow(img2, cmap='gray', origin='lower')
        plt.title('View2 - Sagittal')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        mid_slice = view2.shape[-2] // 2
        img2 = view2[0, :, mid_slice, :] if view2.ndim == 4 else view2[:, mid_slice, :]
        plt.imshow(img2, cmap='gray', origin='lower')
        plt.title('View2 - Coronal')
        plt.axis('off')

        plt.suptitle(f'Sample {sample_count + 1} - Patient {patient}', fontsize=14)
        plt.tight_layout()
        plt.show()

        sample_count += 1

    print(f"\nVisualized {sample_count} samples from the CombinedPretrainingV2 dataset")


if __name__ == "__main__":
    print("Visualizing CombinedPretrainingV2 dataset...")
    try:
        visualize_contrastive_samples(num_samples=3)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
