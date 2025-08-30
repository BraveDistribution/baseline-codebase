# Debug Inference Data Visualization

This script helps debug the hierarchical regression model by visualizing the local and global data that goes into the model during inference. It shows 3 orthogonal cuts (axial, coronal, sagittal) for both high-resolution local data and low-resolution global data.

## Files

- `debug_inference_data.py` - Main debug script
- `create_dummy_data.py` - Creates dummy NIfTI files for testing

## Setup

1. Make sure you're in the baseline-codebase directory
2. Ensure all dependencies are installed (matplotlib, nibabel, numpy, torch)
3. Ensure the regression model and yucca are available

## Usage

### Option 1: Test with dummy data

```bash
# Create dummy T1 and T2 files
python create_dummy_data.py

# Run debug visualization
python debug_inference_data.py --t1 dummy_data/dummy_T1.nii.gz --t2 dummy_data/dummy_T2.nii.gz --output-dir debug_results
```

### Option 2: Test with real data

```bash
# Use your actual T1 and T2 files
python debug_inference_data.py --t1 /path/to/your/T1.nii.gz --t2 /path/to/your/T2.nii.gz --output-dir debug_results
```

## Output

The script generates several files in the output directory:

1. **`hierarchical_data_visualization.png`** - Main visualization showing 3 orthogonal cuts for each modality (local vs global)
2. **`data_statistics.txt`** - Detailed statistics about the processed data
3. **`local_data.npy`** - Preprocessed local (high-res) data as numpy array
4. **`global_data.npy`** - Preprocessed global (low-res) data as numpy array  
5. **`model_test_result.txt`** - Results from testing model loading and inference

## What to Look For

When debugging your inference performance, check:

1. **Data Range and Normalization**: Are the intensity ranges reasonable after preprocessing?
2. **Spatial Resolution**: Do the local and global data show the expected resolution differences?
3. **Anatomical Structure**: Can you see brain structures clearly in both local and global views?
4. **Preprocessing Artifacts**: Are there any unexpected cropping, padding, or resampling artifacts?
5. **Model Input Consistency**: Do the tensor shapes and value ranges match what the model expects?

## Visualization Layout

The visualization shows:
- **Row 1**: T1 Local (high-resolution) - Axial, Coronal, Sagittal
- **Row 2**: T1 Global (low-resolution) - Axial, Coronal, Sagittal  
- **Row 3**: T2 Local (high-resolution) - Axial, Coronal, Sagittal
- **Row 4**: T2 Global (low-resolution) - Axial, Coronal, Sagittal

## Troubleshooting

If you get import errors:
1. Make sure you're running from the baseline-codebase directory
2. Check that the regression_container_kamil/app/predict.py file exists
3. Ensure yucca and other dependencies are installed

If visualization looks wrong:
1. Check the data statistics file for unusual value ranges
2. Verify that your input NIfTI files are valid
3. Check if the model weights file exists at the expected path

## Integration with Training/Validation

This debug script uses the same preprocessing pipeline as your actual inference, so any issues you see here should correspond to issues in your training/validation performance. Use this to verify that:

1. The data preprocessing is working correctly
2. The model receives the expected input format
3. Local and global branches get appropriately different resolutions
4. Intensity normalization is consistent

## Sample Output Interpretation

- **Good data**: Clear anatomical structures, reasonable intensity ranges, no obvious artifacts
- **Problem indicators**: 
  - All zeros or constant values
  - Extreme intensity ranges (e.g., -1000 to +1000 after normalization)
  - Missing or distorted anatomy
  - Unexpected shapes or dimensions
  - Large differences between modalities that should be similar
