#!/usr/bin/env python
"""
Debug script to visualize the data that goes into the hierarchical regression model during inference.
This script shows 3 orthogonal cuts (axial, coronal, sagittal) for both local and global data.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import nibabel as nib
from typing import List, Dict, Any

# Add the regression container app to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(current_dir, 'regression_container_kamil', 'app')
sys.path.insert(0, app_path)

try:
    from predict import (
        load_modalities, 
        img_preprocess_for_global_branch,
        predict_config,
        RegressionHierarchicalFinetuner
    )
except ImportError as e:
    print(f"Error importing from predict.py: {e}")
    print(f"Make sure the script is run from the baseline-codebase directory")
    print(f"Looking for predict.py in: {app_path}")
    sys.exit(1)

try:
    from yucca.functional.preprocessing import preprocess_case_for_inference
except ImportError as e:
    print(f"Error importing yucca: {e}")
    print("Make sure yucca is installed or available in the environment")
    sys.exit(1)


def visualize_3_orthogonal_cuts(volume, title, fig, row_idx, num_cols=3, cmap='gray'):
    """
    Visualize 3 orthogonal cuts (axial, coronal, sagittal) of a 3D volume.
    
    Args:
        volume: 3D numpy array [D, H, W]
        title: Title for the visualization
        fig: matplotlib figure
        row_idx: Row index in the subplot grid
        num_cols: Number of columns per row (3 for axial, coronal, sagittal)
        cmap: Colormap for visualization
    """
    D, H, W = volume.shape
    
    # Calculate middle slices
    mid_d = D // 2  # Axial
    mid_h = H // 2  # Coronal  
    mid_w = W // 2  # Sagittal
    
    # Axial cut (xy plane at mid z)
    ax1 = fig.add_subplot(4, num_cols, row_idx * num_cols + 1)
    axial_slice = volume[mid_d, :, :]
    ax1.imshow(axial_slice.T, cmap=cmap, origin='lower', aspect='equal')
    ax1.set_title(f'{title} - Axial (z={mid_d})')
    ax1.axis('off')
    
    # Coronal cut (xz plane at mid y)
    ax2 = fig.add_subplot(4, num_cols, row_idx * num_cols + 2)
    coronal_slice = volume[:, mid_h, :]
    ax2.imshow(coronal_slice.T, cmap=cmap, origin='lower', aspect='equal')
    ax2.set_title(f'{title} - Coronal (y={mid_h})')
    ax2.axis('off')
    
    # Sagittal cut (yz plane at mid x)
    ax3 = fig.add_subplot(4, num_cols, row_idx * num_cols + 3)
    sagittal_slice = volume[:, :, mid_w]
    ax3.imshow(sagittal_slice.T, cmap=cmap, origin='lower', aspect='equal')
    ax3.set_title(f'{title} - Sagittal (x={mid_w})')
    ax3.axis('off')
    
    return [ax1, ax2, ax3]


def visualize_hierarchical_data(local_data, global_data, modality_names, output_path):
    """
    Create a comprehensive visualization of local and global data for each modality.
    
    Args:
        local_data: Local preprocessed data [C, D, H, W]
        global_data: Global preprocessed data [C, D, H, W] 
        modality_names: List of modality names
        output_path: Path to save the visualization
    """
    num_modalities = local_data.shape[0]
    
    # Create figure with enough space for all modalities and both local/global views
    fig = plt.figure(figsize=(18, 6 * num_modalities))
    fig.suptitle('Hierarchical Model Input Data Visualization\n(Local vs Global Processing)', 
                 fontsize=16, fontweight='bold')
    
    for mod_idx, modality_name in enumerate(modality_names):
        # Local data for this modality
        local_vol = local_data[mod_idx]  # [D, H, W]
        global_vol = global_data[mod_idx]  # [D, H, W]
        
        print(f"Modality {modality_name}:")
        print(f"  Local shape: {local_vol.shape}, range: [{local_vol.min():.3f}, {local_vol.max():.3f}]")
        print(f"  Global shape: {global_vol.shape}, range: [{global_vol.min():.3f}, {global_vol.max():.3f}]")
        
        # Row for local data
        row_idx = mod_idx * 2
        visualize_3_orthogonal_cuts(
            local_vol, 
            f'{modality_name} Local (High-res)', 
            fig, 
            row_idx
        )
        
        # Row for global data  
        row_idx = mod_idx * 2 + 1
        visualize_3_orthogonal_cuts(
            global_vol, 
            f'{modality_name} Global (Low-res)', 
            fig, 
            row_idx
        )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.show()


def add_statistics_text(fig, local_data, global_data, modality_names):
    """Add statistical information as text to the figure."""
    stats_text = "Data Statistics:\n"
    
    for mod_idx, modality_name in enumerate(modality_names):
        local_vol = local_data[mod_idx]
        global_vol = global_data[mod_idx]
        
        stats_text += f"\n{modality_name}:\n"
        stats_text += f"  Local:  shape={local_vol.shape}, mean={local_vol.mean():.3f}, std={local_vol.std():.3f}\n"
        stats_text += f"  Global: shape={global_vol.shape}, mean={global_vol.mean():.3f}, std={global_vol.std():.3f}\n"
    
    # Add text box to the figure
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))


def debug_inference_data(modality_paths: List[str], output_dir: str = "debug_output"):
    """
    Debug the data preprocessing pipeline by visualizing local and global data.
    
    Args:
        modality_paths: List of paths to input modality files
        output_dir: Directory to save debug outputs
    """
    print("=== Debugging Hierarchical Model Input Data ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input images
    print(f"Loading modalities from: {modality_paths}")
    images = load_modalities(modality_paths)
    
    # Extract configuration parameters
    crop_to_nonzero = predict_config["crop_to_nonzero"]
    norm_op = predict_config["norm_op"]
    keep_aspect_ratio = predict_config.get("keep_aspect_ratio", True)
    patch_size = predict_config["patch_size"]
    
    # Define preprocessing parameters
    normalization_scheme = [norm_op] * len(modality_paths)
    target_spacing = [1.0, 1.0, 1.0]  # Isotropic 1mm spacing
    target_orientation = "RAS"
    
    print(f"Preprocessing configuration:")
    print(f"  - Crop to nonzero: {crop_to_nonzero}")
    print(f"  - Normalization: {normalization_scheme}")
    print(f"  - Target spacing: {target_spacing}")
    print(f"  - Patch size: {patch_size}")
    
    # Apply local preprocessing (high-resolution)
    print("\n=== LOCAL PREPROCESSING (High-resolution) ===")
    local_preprocessed, case_properties = preprocess_case_for_inference(
        crop_to_nonzero=crop_to_nonzero,
        images=images,
        intensities=None,
        normalization_scheme=normalization_scheme,
        patch_size=patch_size,
        target_size=None,
        target_spacing=target_spacing,
        target_orientation=target_orientation,
        allow_missing_modalities=False,
        keep_aspect_ratio=keep_aspect_ratio,
        transpose_forward=[0, 1, 2],
    )
    
    # Convert to numpy and remove batch dimension if present
    if len(local_preprocessed.shape) == 5:
        local_data = local_preprocessed.squeeze(0).numpy()  # [C, D, H, W]
    else:
        local_data = local_preprocessed.numpy()  # [C, D, H, W]
    
    print(f"Local data shape: {local_data.shape}")
    
    # Apply global preprocessing (low-resolution)
    print("\n=== GLOBAL PREPROCESSING (Low-resolution) ===")
    global_enc_input = []
    
    for modality_idx in range(local_data.shape[0]):
        modality_data = local_data[modality_idx]  # [D, H, W]
        
        print(f"Processing modality {modality_idx} for global branch...")
        processed_img = img_preprocess_for_global_branch(
            modality_data,
            case_properties,
            target_spacing=2.6667,  # Lower resolution
            target_shape=(96, 96, 96),
            target_element_type='float32'
        )
        global_enc_input.append(processed_img)
    
    global_data = np.stack(global_enc_input, axis=0)  # [C, D, H, W]
    print(f"Global data shape: {global_data.shape}")
    
    # Modality names
    modality_names = predict_config.get("modalities", [f"Modality_{i}" for i in range(local_data.shape[0])])
    
    # Save data statistics
    stats_file = os.path.join(output_dir, "data_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("=== Hierarchical Model Input Data Statistics ===\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Input files: {modality_paths}\n")
        f.write(f"  - Modalities: {modality_names}\n")
        f.write(f"  - Crop to nonzero: {crop_to_nonzero}\n")
        f.write(f"  - Normalization: {normalization_scheme}\n")
        f.write(f"  - Target spacing: {target_spacing}\n")
        f.write(f"  - Patch size: {patch_size}\n\n")
        
        for mod_idx, modality_name in enumerate(modality_names):
            local_vol = local_data[mod_idx]
            global_vol = global_data[mod_idx]
            
            f.write(f"{modality_name} Statistics:\n")
            f.write(f"  Local Data:\n")
            f.write(f"    Shape: {local_vol.shape}\n")
            f.write(f"    Mean: {local_vol.mean():.6f}\n")
            f.write(f"    Std: {local_vol.std():.6f}\n")
            f.write(f"    Min: {local_vol.min():.6f}\n")
            f.write(f"    Max: {local_vol.max():.6f}\n")
            f.write(f"    Non-zero voxels: {np.count_nonzero(local_vol)}/{local_vol.size}\n")
            
            f.write(f"  Global Data:\n")
            f.write(f"    Shape: {global_vol.shape}\n")
            f.write(f"    Mean: {global_vol.mean():.6f}\n")
            f.write(f"    Std: {global_vol.std():.6f}\n")
            f.write(f"    Min: {global_vol.min():.6f}\n")
            f.write(f"    Max: {global_vol.max():.6f}\n")
            f.write(f"    Non-zero voxels: {np.count_nonzero(global_vol)}/{global_vol.size}\n\n")
    
    print(f"Data statistics saved to: {stats_file}")
    
    # Create visualization
    viz_file = os.path.join(output_dir, "hierarchical_data_visualization.png")
    visualize_hierarchical_data(local_data, global_data, modality_names, viz_file)
    
    # Save the processed data as numpy arrays for further analysis if needed
    np.save(os.path.join(output_dir, "local_data.npy"), local_data)
    np.save(os.path.join(output_dir, "global_data.npy"), global_data)
    print(f"Processed data saved to: {output_dir}/local_data.npy and {output_dir}/global_data.npy")
    
    # Test model loading (optional verification)
    try:
        print("\n=== MODEL LOADING TEST ===")
        model_path = predict_config["model_path"]
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            model = RegressionHierarchicalFinetuner.load_from_checkpoint(str(model_path))
            model.eval()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # Convert to tensors and test forward pass
            local_tensor = torch.from_numpy(local_data).unsqueeze(0).to(device)  # Add batch dim
            global_tensor = torch.from_numpy(global_data).unsqueeze(0).to(device)  # Add batch dim
            
            batch = {
                'local': local_tensor,
                'global': global_tensor
            }
            
            with torch.no_grad():
                predictions = model(batch)
                predictions = predictions.squeeze(0)
            
            print(f"Model inference successful!")
            print(f"  Prediction shape: {predictions.shape}")
            print(f"  Prediction value: {predictions.item():.3f}")
            
            # Save prediction info
            with open(os.path.join(output_dir, "model_test_result.txt"), 'w') as f:
                f.write("=== Model Loading and Inference Test ===\n\n")
                f.write(f"Model path: {model_path}\n")
                f.write(f"Device: {device}\n")
                f.write(f"Input local shape: {local_tensor.shape}\n")
                f.write(f"Input global shape: {global_tensor.shape}\n")
                f.write(f"Prediction shape: {predictions.shape}\n")
                f.write(f"Prediction value: {predictions.item():.6f}\n")
            
        else:
            print(f"Model file not found: {model_path}")
    except Exception as e:
        print(f"Model loading/inference failed: {e}")
    
    print(f"\n=== Debug completed. All outputs saved to: {output_dir} ===")


def main():
    parser = argparse.ArgumentParser(
        description="Debug hierarchical regression model input data processing"
    )
    
    parser.add_argument(
        "--t1", type=str, required=True, 
        help="Path to T1 image (NIfTI format)"
    )
    parser.add_argument(
        "--t2", type=str, required=True, 
        help="Path to T2 image (NIfTI format)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="debug_output",
        help="Output directory for debug files (default: debug_output)"
    )
    
    args = parser.parse_args()
    
    # Verify input files exist
    for path in [args.t1, args.t2]:
        if not os.path.exists(path):
            print(f"Error: Input file not found: {path}")
            return
    
    modality_paths = [args.t1, args.t2]
    
    # Run debug analysis
    debug_inference_data(modality_paths, args.output_dir)


if __name__ == "__main__":
    main()
