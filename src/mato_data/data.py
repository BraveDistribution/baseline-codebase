from typing import Callable

import os
import pytorch_lightning as pl

from pathlib import Path

from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from mato_data.dataset import ContrastivePatientDataset
from torch.utils.data import DataLoader
import re


class ContrastiveDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str | Path, transforms: Callable | None = None, batch_size=int):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.setup(None)
        

    def setup(self, stage):
        SESSION_RE = re.compile(r'sub_(?P<patient>\d+)_ses_(?P<session>\d+)_(?P<scan_type>.+)\.npy')
        patient_ids = {
            int(match.group('patient'))
            for filename in os.listdir(self.data_dir)
            if (match := SESSION_RE.match(filename))
        }

        patient_ids = [str(i) for i in patient_ids]
        train_patients, val_patients = train_test_split(patient_ids, test_size=0.2, random_state=42)
        self.train_dataset = ContrastivePatientDataset(data_dir=self.data_dir, patients_included=set(train_patients),
                                                transforms=self.transforms)
        self.val_dataset = ContrastivePatientDataset(data_dir=self.data_dir, patients_included=set(val_patients),
                                                transforms=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=32)


if __name__ == "__main__":
    datamodule = ContrastiveDataModule("/home/mg873uh/Projects_kb/data/pretrain_preproc/FOMO60k", None)
    print(datamodule.train_dataset)
    print(datamodule.val_dataset)
    print(len(datamodule.train_dataset))
    print(len(datamodule.val_dataset))