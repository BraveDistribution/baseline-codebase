from curses import meta
import torchvision
import numpy as np
import torch
import logging
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

        # logging.info('LOADING CASE: %s', case)
        # logging.info(data.shape)

        metadata = {"foreground_locations": []}
        return self._transform(data_dict, metadata)

    def _transform(self, data_dict, metadata=None):
        # Pad the image and label to ensure the entire volume is included
        label = data_dict["label"]
        data_dict["label"] = None
        data_dict = self.croppad(data_dict, metadata)

        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)

        data_dict["label"] = label

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
        data_dict = self.croppad(data_dict, metadata)

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



import random
import numpy as np
import torch
import pytorch_lightning as pl
from typing import Tuple, Optional, Dict, List
import torchvision
import os
from os.path import join



import re
SESSION_RE = re.compile(
    r'^sub_(?P<patient>[^_]+)_ses_(?P<session>[^_]+)_(?P<modality>[^_.]+?)(?:_(?P<variant>\d+))?\.npy$'
)

class CombinedPretraining(Dataset):
    def __init__(
        self,
        patients: set[str],
        patch_size: Tuple[int, int, int],
        data_dir: str,
        pre_aug_patch_size: Optional[Tuple[int, int, int]] = None,
        composed_transforms: Optional[torchvision.transforms.Compose] = None,
        mode: str = 'multimodal',  # 'multimodal' or 'augmentation'
    ):
        super().__init__()
        self.data_dir = data_dir
        self.patients_included = patients
        self.composed_transforms = None
        self.patch_size = patch_size
        self.pre_aug_patch_size = pre_aug_patch_size
        self.mode = mode

        self.croppad = CropPad(patch_size=self.pre_aug_patch_size or self.patch_size)
        self.to_torch = NumpyToTorch()
        self.patients = {}  # Changed to dict for easier access

        # Populate patient data on initialization
        self.populate_paths()

        # Create index mapping for efficient sampling
        self._create_index_mapping()

    def populate_paths(self):
        """patients_sessions: {patient: {session: [full_paths...]}}"""
        self.patients_sessions = {}
        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".npy"):
                continue
            if filename.split('_')[1] not in self.patients_included:
                continue
            m = SESSION_RE.match(filename)
            if not m:
                print(f"Warning: Unexpected filename: {filename}")
                continue
            patient = m.group("patient")
            session = m.group("session")
            full = os.path.join(self.data_dir, filename)

            self.patients_sessions.setdefault(patient, {}).setdefault(session, []).append(full)

        # (optional) still keep a flat per-patient list if other code expects it
        self.patients = {p: sum(sdict.values(), []) for p, sdict in self.patients_sessions.items()}


    def _shared_random_crop(self, v1: np.ndarray, v2: np.ndarray, patch_size=(96, 96, 96)):
        """
        Apply the same random crop to both volumes so they are spatially aligned.
        v1, v2: (C, D, H, W) arrays on the same grid.
        patch_size: tuple of ints (pd, ph, pw) for depth, height, width.
        """
        pd, ph, pw = patch_size
        D, H, W = v1.shape[-3:]

        # Random crop coordinates
        sd = 0 if D <= pd else random.randint(0, D - pd)
        sh = 0 if H <= ph else random.randint(0, H - ph)
        sw = 0 if W <= pw else random.randint(0, W - pw)

        # Apply crop to both
        v1c = v1[..., sd:sd+pd, sh:sh+ph, sw:sw+pw]
        v2c = v2[..., sd:sd+pd, sh:sh+ph, sw:sw+pw]

        # If any dim is smaller than patch, pad both identically
        def _pad_to_patch(x):
            Cd, Ch, Cw = x.shape[-3:]
            pad_d = max(0, pd - Cd)
            pad_h = max(0, ph - Ch)
            pad_w = max(0, pw - Cw)
            if pad_d or pad_h or pad_w:
                x = np.pad(
                    x,
                    ((0, 0),
                    (0, pad_d),
                    (0, pad_h),
                    (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )
            return x

        v1c = _pad_to_patch(v1c)
        v2c = _pad_to_patch(v2c)

        return v1c, v2c


    def _create_index_mapping(self):
        """Create mapping from index to (patient, modality_indices) for efficient sampling"""
        self.index_to_patient = []

        if self.mode == 'multimodal':
            # Each index corresponds to a patient (sample pairs from their modalities)
            for patient_name, modalities in self.patients.items():
                if len(modalities) >= 2:  # Only include patients with at least 2 modalities
                    self.index_to_patient.append((patient_name, modalities))
        else:  # 'augmentation' mode
            # Each index corresponds to a single modality (augment same image twice)
            for patient_name, modalities in self.patients.items():
                for modality_path in modalities:
                    self.index_to_patient.append((patient_name, [modality_path]))

    def __len__(self):
        """Return number of valid samples based on mode"""
        return len(self.index_to_patient)

    def __getitem__(self, idx):
        patient_name, available_modalities = self.index_to_patient[idx]
        if self.mode == 'multimodal':
            i, j = random.sample(range(len(available_modalities)), 2)
            path1, path2 = available_modalities[i], available_modalities[j]
        else:
            path1 = path2 = available_modalities[0]

        def _load_raw(p):
            if p.endswith(".npy"): p = p[:-4]
            x = self._load_volume(p)
            if np.isnan(x).any() or np.isinf(x).any():
                x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0, copy=True)
            return x

        v1 = _load_raw(path1)
        v2 = _load_raw(path2)

        v1c, v2c = self._shared_random_crop(v1, v2)
        d1 = {"image": v1c}
        d2 = {"image": v2c}

        if self.composed_transforms is not None:
            # IMPORTANT: this should contain ONLY intensity / non-spatial augs
            d1 = self.composed_transforms(d1)
            d2 = self.composed_transforms(d2)

        d1 = self.to_torch(d1)
        d2 = self.to_torch(d2)

        return {
            "view1": d1["image"],
            "view2": d2["image"],
            "patient": patient_name,
            "path1": path1,
            "path2": path2,
        }


    def _load_and_transform(self, file_path: str) -> Dict:
        """Load a single volume and apply transforms"""
        # Remove .npy extension if it's already there
        if file_path.endswith('.npy'):
            file_path = file_path[:-4]

        data = self._load_volume(file_path)

        # Handle NaN/Inf values
        if np.isnan(data).any() or np.isinf(data).any():
            if "DISABLE_NAN_WARNING" not in os.environ:
                print(f"Warning: NaNs or infs found in {file_path}")
                print("Set DISABLE_NAN_WARNING=1 to disable this warning.")
            data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0, copy=True)

        data_dict = {
            "file_path": file_path,
            "image": data
        }

        metadata = {"foreground_locations": []}
        return self._transform(data_dict, metadata)

    def _transform(self, data_dict, metadata=None):
        data_dict = self.croppad(data_dict, metadata)

        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)

        return self.to_torch(data_dict)

    def _load_volume(self, file):
        file = file + ".npy"
        path = join(self.data_dir, file) if not file.startswith(self.data_dir) else file

        try:
            vol = np.load(path, 'r')
        except ValueError:
            vol = np.load(path, allow_pickle=True)

        if len(vol.shape) == 3:
            vol = vol[np.newaxis, ...]

        return vol

    def _load_volume_and_header(self, file):
        vol = self._load_volume(file)
        header = load_pickle(file[: -len(".npy")] + ".pkl")
        return vol, header


# Alternative implementation with more control over pair sampling
class CombinedPretrainingV2(CombinedPretraining):
    """
    Alternative version that gives more control over how pairs are sampled.
    Useful if you want to ensure all modalities are seen during training.
    """

    def __init__(self, *args, pairs_per_epoch_multiplier: int = 1, crop=True, **kwargs):
        """
        pairs_per_epoch_multiplier: How many times to go through all possible pairs
        """
        super().__init__(*args, **kwargs)
        self.pairs_per_epoch_multiplier = pairs_per_epoch_multiplier
        self.crop = crop
        self._create_all_pairs()

    def _create_all_pairs(self):
        self.all_pairs = []      # list of (patient, path1, path2)
        self.index_to_patient = []  # keep for your sampler

        if self.mode == "multimodal":
            # Pairs of DIFFERENT modalities, but SAME session
            for patient, sess_dict in self.patients_sessions.items():
                for sess, paths in sess_dict.items():
                    if len(paths) < 2:
                        continue
                    # all unique pairs within this session
                    for i in range(len(paths)):
                        for j in range(i + 1, len(paths)):
                            self.all_pairs.append((patient, paths[i], paths[j]))
                            self.index_to_patient.append((patient, None))
        else:
            # Augmentation mode: same modality twice, but keep it within a single session item
            for patient, sess_dict in self.patients_sessions.items():
                for sess, paths in sess_dict.items():
                    for p in paths:
                        self.all_pairs.append((patient, p, p))
                        self.index_to_patient.append((patient, None))

        # multiplier & shuffle
        self.all_pairs = self.all_pairs * self.pairs_per_epoch_multiplier
        random.shuffle(self.all_pairs)


    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        patient_name, path1, path2 = self.all_pairs[idx]  # ✅ use all_pairs

        def _load_raw(p):
            if p.endswith(".npy"): p = p[:-4]
            x = self._load_volume(p)
            if np.isnan(x).any() or np.isinf(x).any():
                x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0, copy=True)
            return x

        v1 = _load_raw(path1)
        v2 = _load_raw(path2)

        if self.crop:
            v1c, v2c = self._shared_random_crop(v1, v2, patch_size=(96, 96, 96))
        else:
            v1c, v2c = v1, v2

        d1 = {"image": v1c}
        d2 = {"image": v2c}

        d1 = self.to_torch(d1)
        d2 = self.to_torch(d2)

        return {
            "view1": d1["image"],
            "view2": d2["image"],
            "patient": patient_name,
            "path1": path1,
            "path2": path2,
        }


    def on_epoch_end(self):
        """Reshuffle pairs at the end of each epoch"""
        random.shuffle(self.all_pairs)


import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List, Iterator

class UniquePatientBatchSampler(Sampler):
    """
    Custom sampler that ensures each batch contains unique patients.
    Each epoch sees every patient exactly once.
    """

    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Create patient groups - FIXED to handle the actual dataset structure
        self.patient_to_indices = {}

        # Check if dataset has index_to_patient attribute
        if hasattr(dataset, 'index_to_patient'):
            # For each valid index in the dataset
            for idx in range(len(dataset.index_to_patient)):
                patient_name, _ = dataset.index_to_patient[idx]
                if patient_name not in self.patient_to_indices:
                    self.patient_to_indices[patient_name] = []
                self.patient_to_indices[patient_name].append(idx)
        else:
            # Fallback: iterate through dataset to build mapping
            for idx in range(len(dataset)):
                sample = dataset[idx]
                patient_name = sample['patient']
                if patient_name not in self.patient_to_indices:
                    self.patient_to_indices[patient_name] = []
                self.patient_to_indices[patient_name].append(idx)

        self.patients = list(self.patient_to_indices.keys())
        self.num_patients = len(self.patients)

        print(f"UniquePatientBatchSampler initialized:")
        print(f"  Total patients with valid samples: {self.num_patients}")
        print(f"  Total dataset indices: {len(dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {len(self)}")
        print(f"  Samples per epoch: {len(self) * batch_size if drop_last else min(len(dataset), self.num_patients)}")

    def __iter__(self) -> Iterator[List[int]]:
        # Each epoch processes all patients exactly once
        if self.shuffle:
            patients_order = self.patients.copy()
            random.shuffle(patients_order)
        else:
            patients_order = self.patients

        batch = []
        for patient in patients_order:
            # For each patient, randomly select one of their samples
            idx = random.choice(self.patient_to_indices[patient])
            batch.append(idx)

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Handle last incomplete batch
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        """Number of batches per epoch"""
        if self.drop_last:
            return self.num_patients // self.batch_size
        else:
            return (self.num_patients + self.batch_size - 1) // self.batch_size


# Let's also create a diagnostic function to understand your dataset structure
def diagnose_dataset(dataset):
    """Diagnose the dataset structure to understand the issue"""
    print("Dataset Diagnosis:")
    print("-" * 50)

    # Check dataset length
    print(f"len(dataset): {len(dataset)}")

    # Check if index_to_patient exists and its length
    if hasattr(dataset, 'index_to_patient'):
        print(f"len(dataset.index_to_patient): {len(dataset.index_to_patient)}")
        print(f"First few entries in index_to_patient:")
        for i in range(min(5, len(dataset.index_to_patient))):
            print(f"  [{i}]: {dataset.index_to_patient[i]}")
    else:
        print("dataset.index_to_patient not found")

    # Check patients dict
    if hasattr(dataset, 'patients'):
        print(f"\nNumber of patients in dataset.patients: {len(dataset.patients)}")
        # Count patients by number of modalities
        modality_counts = {}
        for patient, modalities in dataset.patients.items():
            n_mod = len(modalities)
            modality_counts[n_mod] = modality_counts.get(n_mod, 0) + 1

        print("Patient distribution by modality count:")
        for n_mod, count in sorted(modality_counts.items()):
            print(f"  {n_mod} modalities: {count} patients")

    # Check mode
    if hasattr(dataset, 'mode'):
        print(f"\nDataset mode: {dataset.mode}")

    # Try to access a sample
    print("\nTrying to access first sample...")
    try:
        sample = dataset[0]
        print(f"Success! Sample keys: {sample.keys()}")
        print(f"Patient: {sample.get('patient', 'N/A')}")
    except Exception as e:
        print(f"Error: {e}")

    print("-" * 50)




class TrackedUniquePatientBatchSampler(UniquePatientBatchSampler):
    """Version that tracks which pairs were selected for each patient"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_count = 0
        self.selection_history = {}

    def __iter__(self) -> Iterator[List[int]]:
        self.epoch_count += 1
        epoch_selections = {}

        # Shuffle patients for this epoch
        if self.shuffle:
            patients_order = self.patients.copy()
            random.shuffle(patients_order)
        else:
            patients_order = self.patients

        batch = []
        for patient in patients_order:
            # Select index for this patient
            available_indices = self.patient_to_indices[patient]

            # If patient has multiple modalities, try to select different pairs across epochs
            if len(available_indices) > 1:
                # Get history for this patient
                if patient not in self.selection_history:
                    self.selection_history[patient] = []

                # Try to pick an index not recently used
                recent_indices = self.selection_history[patient][-3:]  # Last 3 epochs
                unused_indices = [idx for idx in available_indices if idx not in recent_indices]

                if unused_indices:
                    idx = random.choice(unused_indices)
                else:
                    idx = random.choice(available_indices)

                self.selection_history[patient].append(idx)
            else:
                idx = available_indices[0]

            batch.append(idx)
            epoch_selections[patient] = idx

            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Handle last batch
        if len(batch) > 0 and not self.drop_last:
            yield batch

        # Print epoch summary (optional)
        if self.epoch_count % 10 == 0:
            print(f"Epoch {self.epoch_count}: Processed {len(epoch_selections)} unique patients")

            
import random
from typing import Iterator, List, Dict
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedUniquePatientBatchSampler(Sampler):
    """
    Distributed version that maintains the same interface as UniquePatientBatchSampler
    """
    
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        self.patient_to_indices = {}
        for idx in range(len(dataset)):
            sample = dataset[idx]
            patient_name = sample['patient']
            if patient_name not in self.patient_to_indices:
                self.patient_to_indices[patient_name] = []
            self.patient_to_indices[patient_name].append(idx)
        
        self.patients = list(self.patient_to_indices.keys())
        self.num_patients = len(self.patients)
        
        # Distributed settings
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
            
        self.epoch = 0
        self.selection_history = {}
        
        # Calculate how many patients each GPU should handle
        self.num_patients_per_replica = self.num_patients // self.num_replicas
        if self.rank == self.num_replicas - 1:
            # Last GPU gets any remaining patients
            self.num_patients_per_replica = self.num_patients - (self.num_replicas - 1) * self.num_patients_per_replica
        
        print(f"[Rank {self.rank}] DistributedUniquePatientBatchSampler initialized:")
        print(f"  Total patients: {self.num_patients}")
        print(f"  Patients for this GPU: {self.num_patients_per_replica}")
        print(f"  World size: {self.num_replicas}")
        print(f"  Batch size: {batch_size}")

    def set_epoch(self, epoch: int):
        """Set the epoch for proper shuffling across GPUs"""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        # Create deterministic shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Shuffle patients deterministically (all GPUs get same order)
        if self.shuffle:
            indices = torch.randperm(len(self.patients), generator=g).tolist()
            patients_order = [self.patients[i] for i in indices]
        else:
            patients_order = self.patients.copy()
        
        # Split patients among GPUs
        patients_per_replica = len(patients_order) // self.num_replicas
        start_idx = self.rank * patients_per_replica
        end_idx = start_idx + patients_per_replica
        
        # Last GPU gets remaining patients
        if self.rank == self.num_replicas - 1:
            end_idx = len(patients_order)
            
        my_patients = patients_order[start_idx:end_idx]
        
        # Generate batches for this GPU
        batch = []
        for patient in my_patients:
            # Select index for this patient
            available_indices = self.patient_to_indices[patient]
            
            # If patient has multiple samples, try to select different ones
            if len(available_indices) > 1:
                if patient not in self.selection_history:
                    self.selection_history[patient] = []
                
                recent_indices = self.selection_history[patient][-3:]
                unused_indices = [idx for idx in available_indices if idx not in recent_indices]
                
                if unused_indices:
                    idx_pos = int(torch.randint(len(unused_indices), (1,), generator=g))
                    idx = unused_indices[idx_pos]
                else:
                    idx_pos = int(torch.randint(len(available_indices), (1,), generator=g))
                    idx = available_indices[idx_pos]
                
                self.selection_history[patient].append(idx)
            else:
                idx = available_indices[0]
            
            batch.append(idx)
            
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        # Handle last batch
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        """Returns the number of batches per epoch for this replica"""
        my_patients = self.num_patients_per_replica
        if self.drop_last:
            return my_patients // self.batch_size
        else:
            return (my_patients + self.batch_size - 1) // self.batch_size