#!/usr/bin/env python3
"""
Simple NPY Image Comparator - Compare .npy files across folders with orthogonal cuts
Set the folders and case_name variables below and run the script.
Run in interactive Python environment. (Outputs plots)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===== SET THESE VARIABLES =====
folders = [
    #"/home/mg873uh/Projects_kb/data/pretrain_preproc/FOMO60k",
    "/home/mg873uh/Projects_kb/data/finetuning_preproc/Task001_FOMO1",
    "/home/mg873uh/Projects_kb/data/finetuning_preproc/Task001_FOMO1_2.667mm_float16",
]
case_name = "FOMO1_sub_1" # Case name without file extension

# For multi-volume data, specify which volume to show
# Example: [2] for shape [4,100,100,100] to show volume at index 2
# Example: [1,2] for shape [2,4,100,100,100] to show volume at indices [1,2]
show_volume = None  # Set to None to use default behavior, or specify indices like [2] or [1,2]
# ===============================

def find_npy_files(folders, case_name):
    """Find .npy files matching case_name in each folder."""
    files = {}
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: {folder} does not exist")
            continue

        npy_files = list(folder_path.glob(f"*{case_name}*.npy"))
        if npy_files:
            files[folder_path.name] = npy_files[0]
        else:
            print(f"Warning: {case_name} not found in {folder}")
    return files

def load_images(file_dict):
    """Load all .npy files."""
    images = {}
    for folder_name, file_path in file_dict.items():
        try:
            data = np.load(file_path)
            images[folder_name] = data
            print(f"Loaded {file_path}: shape={data.shape}, dtype={data.dtype}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return images

def get_orthogonal_slices(data, show_volume=None):
    """Get orthogonal slices through center of 3D volume.

    Args:
        data: Input array with shape >= 3D
        show_volume: List of indices to select specific volume from multi-dimensional data.
                    Example: [2] for shape [4,100,100,100] selects volume at index 2
                    Example: [1,2] for shape [2,4,100,100,100] selects volume at indices [1,2]
    """
    original_shape = data.shape

    # Handle multi-dimensional data (more than 3 dimensions)
    if len(data.shape) > 3:
        if show_volume is not None:
            # Validate show_volume indices
            expected_dims = len(data.shape) - 3
            if len(show_volume) != expected_dims:
                raise ValueError(f"show_volume must have {expected_dims} indices for data shape {data.shape}, got {len(show_volume)}")

            # Check if indices are valid
            for i, idx in enumerate(show_volume):
                if idx >= data.shape[i]:
                    raise ValueError(f"Index {idx} is out of bounds for dimension {i} with size {data.shape[i]}")

            # Select the specific volume using the provided indices
            data = data[tuple(show_volume)]
            print(f"Selected volume {show_volume} from shape {original_shape}, resulting shape: {data.shape}")
        else:
            # Default behavior: take the last 3 dimensions and select first volume for each extra dimension
            # This preserves the original behavior
            default_indices = tuple(0 for _ in range(len(data.shape) - 3))
            data = data[default_indices]
            print(f"Using default volume selection {default_indices} from shape {original_shape}, resulting shape: {data.shape}")

    # Ensure we have exactly 3D data at this point
    if len(data.shape) != 3:
        raise ValueError(f"Expected 3D data after volume selection, got shape {data.shape}")

    cx, cy, cz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
    return data[:, :, cz], data[cx, :, :], data[:, cy, :]  # axial, sagittal, coronal

def plot_comparison(images, case_name, show_volume=None):
    """Create comparison plot of orthogonal slices."""
    if not images:
        print("No images to plot")
        return

    n_folders = len(images)
    fig, axes = plt.subplots(3, n_folders, figsize=(4*n_folders, 10))

    if n_folders == 1:
        axes = axes.reshape(-1, 1)

    # Get global min/max for consistent scaling
    all_values = []
    for data in images.values():
        axial, sagittal, coronal = get_orthogonal_slices(data, show_volume)
        for slice_data in [axial, sagittal, coronal]:
            all_values.extend(slice_data.flatten())
    vmin, vmax = np.min(all_values), np.max(all_values)

    view_names = ['Axial', 'Sagittal', 'Coronal']
    folder_names = list(images.keys())

    for col, folder_name in enumerate(folder_names):
        data = images[folder_name]
        slices = get_orthogonal_slices(data, show_volume)

        for row, (slice_data, view_name) in enumerate(zip(slices, view_names)):
            ax = axes[row, col]
            im = ax.imshow(slice_data.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)

            if row == 0:  # Title on top row
                volume_info = f"\nVolume: {show_volume}" if show_volume is not None else ""
                ax.set_title(f"{folder_name}\nShape: {data.shape}\nType: {data.dtype}{volume_info}", fontsize=10)
            if col == 0:  # View labels on left
                ax.set_ylabel(view_name, fontsize=12, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])

    plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.suptitle(f'NPY Comparison: {case_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the comparator."""
    # Find and load files
    files = find_npy_files(folders, case_name)
    if not files:
        print("No files found!")
        return

    images = load_images(files)
    if not images:
        print("No images loaded!")
        return

    # Create plot
    plot_comparison(images, case_name, show_volume)

if __name__ == "__main__":
    main()
