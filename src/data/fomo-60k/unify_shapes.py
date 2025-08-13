import numpy as np
import argparse
import os
import SimpleITK as sitk
import pickle

def _resample_image(img_data, target_spacing, current_spacing=None):
    """
    Resample a NIfTI image to a target spacing using SimpleITK.

    Args:
        img_data (np.ndarray): Image data array
        target_spacing (float): Target pixel/voxel spacing
        current_spacing (tuple, optional): Current spacing. If None, assumes isotropic spacing of 1.0

    Returns:
        np.ndarray: Resampled image data
    """

    # Set current spacing if not provided, otherwise assume isotropic spacing of 1.0
    if current_spacing is None:
        current_spacing = [1.0] * len(img_data.shape)

    # Check if resampling is needed - avoid unnecessary computation
    if all(abs(cs - target_spacing) < 1e-6 for cs in current_spacing):
        return img_data

    # Convert numpy array to SimpleITK image
    sitk_image = sitk.GetImageFromArray(img_data)
    sitk_image.SetSpacing(current_spacing)

    # Calculate new size based on target spacing
    original_size = sitk_image.GetSize()
    original_spacing = sitk_image.GetSpacing()

    new_size = []
    for i in range(len(original_size)):
        new_size.append(int(round(original_size[i] * (original_spacing[i] / target_spacing))))

    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([target_spacing] * sitk_image.GetDimension())
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    # Resample the image
    resampled_image = resampler.Execute(sitk_image)

    # Convert back to numpy array
    resampled_array = sitk.GetArrayFromImage(resampled_image)

    return resampled_array


def unify_dataset_shape(
    src_dataset_path: str,
    dst_dataset_path: str,
    target_spacing: float = 1.0,
    target_shape: tuple = None,
    target_element_type: str = None,
    do_conversion: bool = False
)   -> None:
    """
    Find the maximum size in each dimension so that all the images fits in and
    unify the shape of all images in the dataset to that common size.
    Place the sample in the center of the image and fill the empty space with zeros by padding.
    Save the unified images to the destination directory while preserving the original subfolders structure.
    Resample the images to the specified target spacing using metadata from pickle files.
    This function assumes that the images are in npy format with corresponding pkl metadata files.
    It will also create the destination directory if it does not exist.

    Args:
        src_dataset_path (str): Path to the source dataset directory.
        dst_dataset_path (str): Path to the destination dataset directory.
        target_spacing (float): The target spacing to resample the images to.
    """

    # ensure dir exists
    if not os.path.exists(dst_dataset_path):
        os.makedirs(dst_dataset_path)

    exclusion_list = [
        "sub_9147_ses_2_dwi",   # 0: 363
        "sub_10162_ses_2_dwi",  # 0: 256
        "sub_10162_ses_2_dwi_2",# 0: 256
        "sub_8084_ses_1_t1",    # 1: 269
        "sub_9508_ses_1_t1",    # 1: 263
        "sub_1953_ses_1_t1",    # 1: 263
        "sub_2019_ses_1_flair", # 1: 263
        "sub_2019_ses_1_t1",    # 1: 262
        "sub_8486_ses_1_t1",    # 1: 262
        "sub_8486_ses_1_dwi",   # 1: 261
        "sub_9349_ses_1_t1",    # 1: 261
        "sub_3213_ses_1_t1",    # 1: 261
        "sub_3614_ses_1_t1",    # 1: 260
        "sub_7642_ses_1_dwi",   # 1: 260
        "sub_7642_ses_1_t1",    # 1: 259
        "sub_4961_ses_1_t1",    # 1: 258
        "sub_3028_ses_1_dwi_2", # 1: 258
        "sub_7642_ses_1_flair", # 1: 257
        "sub_4961_ses_1_flair"  # 1: 257
]

    # Convert exclusion list to set for faster lookup
    exclusion_set = set(exclusion_list)

    # Find the maximum size in each dimension after resampling
    max_shape = None
    resampled_shapes = {}

    for root, _, files in os.walk(src_dataset_path):
        for file in files:
            if file.endswith('.npy'):
                # Get the corresponding pickle file
                base_name = os.path.splitext(file)[0]

                # Check if this file should be excluded
                if base_name in exclusion_set:
                    print(f"Skipping excluded file: {file}")
                    continue

                pkl_file = base_name + '.pkl'
                pkl_path = os.path.join(root, pkl_file)

                if not os.path.exists(pkl_path):
                    print(f"Warning: No pickle file found for {file}, using default spacing and loading size from npy file...")
                    # Use default spacing and get size from npy file
                    current_spacing = [1.0, 1.0, 1.0]
                    npy_path = os.path.join(root, file)
                    img_data = np.load(npy_path)
                    current_size = list(img_data.shape)
                else:
                    # Load metadata from pickle file
                    with open(pkl_path, 'rb') as f:
                        metadata = pickle.load(f)

                    # Get current spacing and size from metadata
                    current_spacing = metadata.get('new_spacing', [1.0, 1.0, 1.0])
                    current_size = metadata.get('new_size', metadata.get('size_after_transpose', None))

                    if current_size is None:
                        print(f"Warning: No size information found in {pkl_file}, loading npy file...")
                        npy_path = os.path.join(root, file)
                        img_data = np.load(npy_path)
                        current_size = list(img_data.shape)

                # Calculate new size after resampling to target spacing
                resampled_size = []
                for i, (curr_sp, size) in enumerate(zip(current_spacing, current_size)):
                    new_size = int(round(size * (curr_sp / target_spacing)))
                    resampled_size.append(new_size)

                resampled_size = tuple(resampled_size)
                resampled_shapes[os.path.join(root, file)] = resampled_size

                # Update maximum shape
                if max_shape is None:
                    max_shape = resampled_size
                else:
                    max_shape = tuple(max(s, m) for s, m in zip(resampled_size, max_shape))

    print(f"Maximum resampled shape: {max_shape}")
    print(f"Requested target shape: {target_shape}")
    if target_shape is not None:
        #TODO[Khamyl]: To get rid of this assert implement cropping and padding
        assert len(target_shape) == len(max_shape), f"Target shape {target_shape} and max shape {max_shape} must have the same number of dimensions."
        assert all(ts >= ms for ts, ms in zip(target_shape, max_shape)), \
            f"Each element of target_shape {target_shape} must be >= corresponding element in max_shape {max_shape}"
    max_shape = tuple(max_shape) if target_shape is None else tuple(target_shape)
    print(f"The shape will be: {max_shape}")

    # Calculate total dataset size if all files are converted to max_shape
    total_files = len(resampled_shapes)
    elements_per_file = np.prod(max_shape)
    total_elements = total_files * elements_per_file

    # Get bytes per element from the first .npy file
    bytes_per_element = 4  # default fallback
    data_type = "unknown"

    if resampled_shapes and target_element_type is None:
        # Get the first file path from resampled_shapes
        first_file_path = next(iter(resampled_shapes.keys()))
        try:
            # Load the first .npy file to determine data type
            sample_data = np.load(first_file_path)
            bytes_per_element = sample_data.dtype.itemsize
            data_type = str(sample_data.dtype)
            print(f"Detected data type: {data_type} ({bytes_per_element} bytes per element)")
        except Exception as e:
            print(f"Warning: Could not load {first_file_path} to determine data type: {e}")
            print(f"Using default assumption: float32 (4 bytes per element)")
            data_type = "float32 (assumed)"
    else:
        # Use the specified target element type if provided
        if target_element_type is not None:
            data_type = target_element_type
            if   data_type == "float16": bytes_per_element = 2
            elif data_type == "float32": bytes_per_element = 4
            elif data_type == "float64": bytes_per_element = 8
            else:
                raise ValueError(f"Unsupported target element type: {target_element_type}")

            print(f"Requested data type: {data_type} ({bytes_per_element} bytes per element)")

    total_size_bytes = total_elements * bytes_per_element
    total_size_gb = total_size_bytes / (1024**3)

    print(f"Total files to be processed: {total_files}")
    print(f"One file size after resampling: ({elements_per_file * bytes_per_element / (1024**2)} MB each)")
    print(f"Estimated total dataset size: {total_size_gb:.2f} GB (data type: {data_type})")

    if not do_conversion:
        print("Analysis complete. Set `do_conversion=True` to perform the conversion.")
        return

    # Resample and save images with the unified shape
    for root, _, files in os.walk(src_dataset_path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)

                # Skip if we don't have resampling info for this file
                if file_path not in resampled_shapes:
                    continue

                # Get the corresponding pickle file for spacing info
                base_name = os.path.splitext(file)[0]

                # Check if this file should be excluded (double-check in case of inconsistency)
                if base_name in exclusion_set:
                    print(f"Skipping excluded file during processing: {file}")
                    continue

                pkl_file = base_name + '.pkl'
                pkl_path = os.path.join(root, pkl_file)

                if not os.path.exists(pkl_path):
                    # Use default spacing if no pickle file exists
                    current_spacing = [1.0, 1.0, 1.0]
                else:
                    with open(pkl_path, 'rb') as f:
                        metadata = pickle.load(f)
                    current_spacing = metadata.get('new_spacing', [1.0, 1.0, 1.0])

                # Load and resample the image
                img_data = np.load(file_path)

                # Resample to target spacing
                resampled_img_data = _resample_image(img_data, target_spacing, current_spacing)

                # Pad to the maximum shape and center the image
                padded_img_data = np.zeros(max_shape, dtype=data_type)
                offsets = [(m - s) // 2 for s, m in zip(resampled_img_data.shape, max_shape)]
                slices = tuple(slice(offset, offset + s) for offset, s in zip(offsets, resampled_img_data.shape))
                padded_img_data[slices] = resampled_img_data

                # Save the unified image
                relative_path = os.path.relpath(root, src_dataset_path)
                dst_dir = os.path.join(dst_dataset_path, relative_path)

                # Ensure dir exists
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)

                dst_file_path = os.path.join(dst_dir, file)
                np.save(dst_file_path, padded_img_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path", type=str, required=True, help="Path to pretrain data"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to put preprocessed pretrain data",
    )
    parser.add_argument(
        "--target_spacing",
        type=float,
        default=1.0,
        help="Target spacing to resample images to (default: 1.0)",
    )
    parser.add_argument(
        "--target_shape",
        type=int,
        nargs=3,
        default=None,
        help="Target shape for the output images (e.g., 96 96 96). If not specified, will use the shape from analysis.",
    )
    parser.add_argument(
        "--target_element_type",
        type=str,
        help="Target element type for the output images (e.g., 'float16', 'float32', 'float64'). If not specified, will infer from the first file.",
    )
    parser.add_argument(
        "--do_conversion",
        action="store_true",
        help="Do conversion after analysis"
    )
    args = parser.parse_args()

    unify_dataset_shape(
        src_dataset_path=args.in_path,
        dst_dataset_path=args.out_path,
        target_spacing=args.target_spacing,
        target_shape=args.target_shape,
        target_element_type=args.target_element_type,
        do_conversion=args.do_conversion
    )
