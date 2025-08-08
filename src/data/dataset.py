from curses import meta
import torchvision
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from typing import Tuple, Optional, Literal
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from yucca.modules.data.augmentation.transforms.cropping_and_padding import CropPad
from yucca.modules.data.augmentation.transforms.formatting import NumpyToTorch

from batchgenerators.utilities.file_and_folder_operations import join


class FOMODataset(Dataset):
    """
    Dataset class for FOMO downstream tasks. Supports classification and regression tasks.
    For segmentation tasks, use YuccaTrainDataset from the Yucca library instead.
    """

    def __init__(
        self,
        samples: list,
        patch_size: Tuple[int, int, int],
        composed_transforms: Optional[torchvision.transforms.Compose] = None,
        task_type: Literal["classification", "regression"] = "classification",
        allow_missing_modalities: Optional[bool] = False,  # For compatibility
        p_oversample_foreground: Optional[float] = None,  # For compatibility
    ):
        super().__init__()
        # Support only non-segmentation tasks
        assert task_type in [
            "classification",
            "regression",
        ], f"Unsupported task type: {task_type}. For segmentation use YuccaTrainDataset instead."

        self.task_type = task_type
        self.all_files = samples
        self.composed_transforms = composed_transforms
        self.patch_size = patch_size

        self.croppad = CropPad(patch_size=self.patch_size)
        self.to_torch = NumpyToTorch()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        case = self.all_files[idx]

        # single modality
        assert isinstance(case, str)

        data = self._load_volume(case)
        label = self._load_label(case)
        data_dict = {
            "file_path": case,
            "image": data,
            "label": label,
        }

        metadata = {"foreground_locations": []}
        return self._transform(data_dict, metadata)

    def _transform(self, data_dict, metadata=None):
        # Pad the image and label to ensure the entire volume is included
        image = data_dict["image"]
        label = data_dict.get("label")

        # Calculate padding sizes to make the image and label cubic
        max_dim = max(image.shape[-3:])
        pad_sizes = [(max_dim - dim) // 2 for dim in image.shape[-3:]]
        pad_sizes = [(pad, max_dim - dim - pad) for pad, dim in zip(pad_sizes, image.shape[-3:])]

        # Apply padding
        image = np.pad(image, [(0, 0)] + pad_sizes, mode="constant", constant_values=0)
        if label is not None:
            label = np.pad(label, pad_sizes, mode="constant", constant_values=0)

        # Resize to 96x96x96
        resize_shape = (96, 96, 96)
        image = torch.nn.functional.interpolate(
            torch.tensor(image).unsqueeze(0), size=resize_shape, mode="trilinear", align_corners=False
        ).squeeze(0).numpy()
        if label is not None:
            label = torch.nn.functional.interpolate(
                torch.tensor(label).unsqueeze(0).unsqueeze(0), size=resize_shape, mode="trilinear", align_corners=False
            ).squeeze(0).squeeze(0).numpy()

        data_dict["image"] = image
        data_dict["label"] = label

        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)

        return self.to_torch(data_dict)

    def _load_volume_and_header(self, file):
        vol = self._load_volume(file)
        header = load_pickle(file[: -len(".npy")] + ".pkl")
        return vol, header

    def _load_label(self, file):
        # For classification and regression, labels are in .txt files
        txt_file = file + ".txt"
        if self.task_type == "classification":
            return np.loadtxt(txt_file, dtype=int)
        else:  # regression
            reg_label = np.loadtxt(txt_file, dtype=float)
            reg_label = np.atleast_1d(reg_label)
            return reg_label

    def _load_volume(self, file):
        file = file + ".npy"

        try:
            vol = np.load(file, "r")
        except ValueError:
            vol = np.load(file, allow_pickle=True)

        return vol


class PretrainDataset(Dataset):
    def __init__(
        self,
        samples: list,
        patch_size: Tuple[int, int, int],
        data_dir: str,
        pre_aug_patch_size: Optional[Tuple[int, int, int]] = None,
        composed_transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        self.all_files = samples
        self.data_dir = data_dir
        self.composed_transforms = composed_transforms
        self.patch_size = patch_size
        self.pre_aug_patch_size = pre_aug_patch_size

        self.croppad = CropPad(patch_size=self.pre_aug_patch_size or self.patch_size)
        self.to_torch = NumpyToTorch()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        case = self.all_files[idx]

        # single modality
        assert isinstance(case, str)
        data = self._load_volume(case)

        # Ensure volume does not contain NaNs or Infs, which can sometimes
        # occur in large pretraining datasets.
        if np.isnan(data).any() or np.isinf(data).any():
            if "DISABLE_NAN_WARNING" not in os.environ:
                print("A case contains NaNs or infs. We have corrected this, but consider handling this with different preprocessing or skipping affected cases.")
                print(f"Affected Case: {case}")
                print("Set DISABLE_NAN_WARNING=1 to disable this warning.")
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0, copy=True)

        data_dict = {
            "file_path": case
        }  # metadata that can be very useful for debugging.
        metadata = {"foreground_locations": []}
        data_dict["image"] = data

        return self._transform(data_dict, metadata)

    def _transform(self, data_dict, metadata=None):
                # Pad the image and label to ensure the entire volume is included
        image = data_dict["image"]
        label = data_dict.get("label")

        # Calculate padding sizes to make the image and label cubic
        max_dim = max(image.shape[-3:])
        pad_sizes = [(max_dim - dim) // 2 for dim in image.shape[-3:]]
        pad_sizes = [(pad, max_dim - dim - pad) for pad, dim in zip(pad_sizes, image.shape[-3:])]

        # Apply padding
        image = np.pad(image, [(0, 0)] + pad_sizes, mode="constant", constant_values=0)
        if label is not None:
            label = np.pad(label, pad_sizes, mode="constant", constant_values=0)

        # Resize to 96x96x96
        resize_shape = (96, 96, 96)
        image = torch.nn.functional.interpolate(
            torch.tensor(image).unsqueeze(0), size=resize_shape, mode="trilinear", align_corners=False
        ).squeeze(0).numpy()
        if label is not None:
            label = torch.nn.functional.interpolate(
                torch.tensor(label).unsqueeze(0).unsqueeze(0), size=resize_shape, mode="trilinear", align_corners=False
            ).squeeze(0).squeeze(0).numpy()

        data_dict["image"] = image
        data_dict["label"] = label

        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)
        return self.to_torch(data_dict)

    def _load_volume_and_header(self, file):
        vol = self._load_volume(file)
        header = load_pickle(file[: -len(".npy")] + ".pkl")
        return vol, header

    def _load_volume(self, file):
        file = file + ".npy"
        path = join(self.data_dir, file)

        try:
            vol = np.load(path, "r")
        except ValueError:
            vol = np.load(path, allow_pickle=True)

        # Add channel dimension if it doesn't exist
        if len(vol.shape) == 3:
            vol = vol[np.newaxis, ...]

        return vol


from enum import Enum

class ModalityIndex(Enum):
    T1 = 0
    T2 = 1
    FLAIR = 2
    T1CE = 3
    DWI = 4
    PD = 5

    @classmethod
    def from_string(cls, modality: str):
        return cls[modality.upper()]

class PretrainDatasetCombinedPatient(Dataset):
    def __init__(
        self,
        patients: list,
        patch_size: Tuple[int, int, int],
        data_dir: str,
        pre_aug_patch_size: Optional[Tuple[int, int, int]] = None,
        composed_transforms: Optional[torchvision.transforms.Compose] = None,
    ):
        self.patients = patients
        self.all_files = {}
        self.data_dir = data_dir
        self.composed_transforms = composed_transforms
        self.patch_size = patch_size
        self.pre_aug_patch_size = pre_aug_patch_size
        self.croppad = CropPad(patch_size=self.pre_aug_patch_size or self.patch_size)
        self.to_torch = NumpyToTorch()
        self.get_all_patient_file_paths()  # Load all patient file paths on initialization
    
    def get_all_patient_file_paths(self):
        self.all_files = {}  # Reset all_files dictionary
        
        # Get all .npy files in the data directory
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.npy'):
                continue
                
            # Parse filename to extract patient, session, and modality
            parts = filename.split('_')
            if len(parts) < 5 or not parts[0] == 'sub':
                continue

            patient_id = "sub_" + parts[1]
            
            if patient_id not in self.patients:
                continue
                
            print(filename)
            # Extract session and modality
            if parts[2] == 'ses' and len(parts) > 4:
                session_num = parts[3]
                # The modality is the last part before the extension
                modality_name = parts[4].split('.')[0]
                
                try:
                    print(f'Processing file: {filename} for patient {patient_id}, session {session_num}, modality {modality_name}')
                    modality = ModalityIndex.from_string(modality_name.upper())
                    # Create the session key
                    session_key = f"{patient_id}/ses_{session_num}"
                    
                    # Initialize dictionary for this session if not exists
                    if session_key not in self.all_files:
                        self.all_files[session_key] = {}
                        
                    # Store the full path to the file
                    self.all_files[session_key][modality] = os.path.join(self.data_dir, filename)
                except (KeyError, IndexError):
                    print(f'Skipping file: {filename} - unknown modality')
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        pass

    def _transform(self, data_dict, metadata=None):
        pass


    def _load_volume_and_header(self, file):
        pass

    def _load_volume(self, file):
        pass
