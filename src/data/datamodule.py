import pytorch_lightning as pl
from torchvision.transforms import Compose
import logging
import torch
import os
from typing import Literal, Optional, Tuple
from torch.utils.data import DataLoader, Sampler
from yucca.pipeline.configuration.split_data import SplitConfig
from yucca.functional.array_operations.matrix_ops import get_max_rotated_size
from yucca.modules.data.augmentation.transforms.Spatial import Spatial

from data.dataset import PretrainDataset, PretrainDatasetCombinedPatient, TrackedUniquePatientBatchSampler, CombinedPretrainingV2, DistributedUniquePatientBatchSampler
from sklearn.model_selection import train_test_split

class PretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        batch_size: int,
        num_workers: int,
        splits_config: SplitConfig,
        split_idx: int,
        train_data_dir: str,
        train_sampler: Optional[Sampler] = None,
        val_sampler: Optional[Sampler] = None,
        composed_train_transforms: Optional[Compose] = None,
        composed_val_transforms: Optional[Compose] = None,
        dataset="default",
        crop: bool = False,
    ):
        super().__init__()

        self.dataset = dataset
        # extract parameters
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.crop = crop

        self.split_idx = split_idx
        self.splits_config = splits_config
        self.train_data_dir = train_data_dir

        self.composed_train_transforms = composed_train_transforms
        self.composed_val_transforms = composed_val_transforms
        self.pre_aug_patch_size = (
            get_max_rotated_size(patch_size)
            if augmentations_include_spatial(composed_train_transforms)
            else None
        )
        assert self.pre_aug_patch_size is None or isinstance(
            self.pre_aug_patch_size, tuple
        )

        self.num_workers = (
            max(0, int(torch.get_num_threads() - 1))
            if num_workers is None
            else num_workers
        )
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        logging.info(f"Using {self.num_workers} workers for data loading (Data Module)")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        assert stage == "fit"

        # Assign train/val datasets for use in dataloaders
        assert self.train_data_dir is not None
        assert self.split_idx is not None
        assert self.splits_config is not None

        self.train_samples = self.splits_config.train(self.split_idx)
        self.val_samples = self.splits_config.val(self.split_idx)

        logging.info(f"Train samples: {len(self.train_samples)}")
        logging.info(f"Val samples: {len(self.val_samples)}")

        if self.dataset == 'default':
            self.train_dataset = PretrainDataset(
                self.train_samples,
                data_dir=self.train_data_dir,
                composed_transforms=self.composed_train_transforms,
                pre_aug_patch_size=self.pre_aug_patch_size,  # type: ignore
                patch_size=self.patch_size,
            )

            self.train_batch_sampler = (self.train_sampler(self.train_dataset) if self.train_sampler is not None else None)

            self.val_dataset = PretrainDataset(
                self.val_samples,
                data_dir=self.train_data_dir,
                composed_transforms=self.composed_val_transforms,
                patch_size=self.patch_size,
            )
            self.val_batch_sampler = (self.val_sampler(self.val_dataset) if self.val_sampler is not None else None)

        elif self.dataset == 'contrastive':
            all_studies = os.listdir(self.train_data_dir)
            all_studies = set([study.split('_')[1] for study in all_studies if study.endswith('.npy')])
            self.train_patients, self.val_patients = train_test_split(
                list(all_studies),
                train_size=0.9,
                random_state=42
            )

            # Create datasets
            self.train_dataset = CombinedPretrainingV2(
                self.train_patients,
                data_dir=self.train_data_dir,
                composed_transforms=self.composed_train_transforms,
                pre_aug_patch_size=self.pre_aug_patch_size,
                patch_size=self.patch_size,
                crop=self.crop,
            )

            self.val_dataset = CombinedPretrainingV2(
                self.val_patients,
                data_dir=self.train_data_dir,
                composed_transforms=self.composed_val_transforms,
                patch_size=self.patch_size,
            )

            # Create samplers with the mapping
            self.train_batch_sampler = DistributedUniquePatientBatchSampler(
                self.train_dataset,
                batch_size=10,
                drop_last=True,
                shuffle=True
            )

            self.val_batch_sampler = DistributedUniquePatientBatchSampler(
                self.val_dataset,
                batch_size=10,
                drop_last=True,
                shuffle=True
            )

    def _build_patient_to_indices(self, dataset):
        """Build patient_to_indices mapping from CombinedPretrainingV2 dataset"""
        patient_to_indices = {}
        
        # The dataset has index_to_patient attribute
        if hasattr(dataset, 'index_to_patient'):
            for idx in range(len(dataset.index_to_patient)):
                patient_name, _ = dataset.index_to_patient[idx]
                if patient_name not in patient_to_indices:
                    patient_to_indices[patient_name] = []
                patient_to_indices[patient_name].append(idx)
        else:
            # Fallback: iterate through dataset
            for idx in range(len(dataset)):
                try:
                    sample = dataset[idx]
                    patient_name = sample['patient']
                    if patient_name not in patient_to_indices:
                        patient_to_indices[patient_name] = []
                    patient_to_indices[patient_name].append(idx)
                except Exception as e:
                    print(f"Warning: Could not get patient for index {idx}: {e}")
                    continue
        
        print(f"Built patient_to_indices mapping with {len(patient_to_indices)} patients")
        return patient_to_indices


    def train_dataloader(self):
        logging.info(f"Starting training with data from: {self.train_data_dir}")
        if not self.dataset == 'contrastive':
            return DataLoader(
                self.train_dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=False,
                sampler=self.train_batch_sampler,
            )
        else:
            return DataLoader(
                self.train_dataset,
                num_workers=self.num_workers,
                pin_memory=False,
                batch_sampler=self.train_batch_sampler,
            )

    def val_dataloader(self):
        if not self.dataset == 'contrastive':
            return DataLoader(
                self.val_dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=False,
                sampler=self.val_batch_sampler,
            )
        else:
            return DataLoader(
                self.val_dataset,
                num_workers=self.num_workers,
                pin_memory=False,
                batch_sampler=self.val_batch_sampler,
            )
def augmentations_include_spatial(augmentations):
    if augmentations is None:
        return False

    for augmentation in augmentations.transforms:
        if isinstance(augmentation, Spatial):
            return True

    return False