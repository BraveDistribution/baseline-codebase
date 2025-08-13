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
    "/home/mg873uh/Projects_kb/data/pretrain_preproc/FOMO60k_1.0mm_float16",
    "/home/mg873uh/Projects_kb/data/pretrain_preproc/FOMO60k_2.667mm_float16"
]
case_name = "sub_10000_ses_1_flair"
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

def get_orthogonal_slices(data):
    """Get orthogonal slices through center of 3D volume."""
    if len(data.shape) == 4:
        data = data[:, :, :, 0]  # Take first channel/timepoint

    cx, cy, cz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2
    return data[:, :, cz], data[cx, :, :], data[:, cy, :]  # axial, sagittal, coronal

def plot_comparison(images, case_name):
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
        axial, sagittal, coronal = get_orthogonal_slices(data)
        for slice_data in [axial, sagittal, coronal]:
            all_values.extend(slice_data.flatten())
    vmin, vmax = np.min(all_values), np.max(all_values)

    view_names = ['Axial', 'Sagittal', 'Coronal']
    folder_names = list(images.keys())

    for col, folder_name in enumerate(folder_names):
        data = images[folder_name]
        slices = get_orthogonal_slices(data)

        for row, (slice_data, view_name) in enumerate(zip(slices, view_names)):
            ax = axes[row, col]
            im = ax.imshow(slice_data.T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)

            if row == 0:  # Title on top row
                ax.set_title(f"{folder_name}\nShape: {data.shape}\nType: {data.dtype}", fontsize=10)
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
    plot_comparison(images, case_name)

if __name__ == "__main__":
    main()
