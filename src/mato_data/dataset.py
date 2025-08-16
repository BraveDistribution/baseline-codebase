from typing import Callable
import os
import random
from typing import Dict, List, Tuple, Optional, Set, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from math import comb
from pathlib import Path
from itertools import combinations
import re


SESSION_RE = re.compile(r'sub_(?P<patient>\d+)_ses_(?P<session>\d+)_(?P<scan_type>.+)\.npy')

class ContrastivePatientDataset(Dataset):
    def __init__(self, data_dir: str | Path, patients_included: Set[str], transforms: Callable = None):
        self.data_dir = data_dir
        self.patients_included = patients_included
        self.populate_paths()
    
    def _load_volume(self, file):
        path = os.path.join(self.data_dir, file) if not file.startswith(self.data_dir) else file
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

    def populate_paths(self):
        """patients_sessions: {patient: {session: [full_paths...]}}"""
        self.patients_sessions = {}
        
        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".npy"):
                continue
                
            # Extract patient ID for filtering
            split_parts = filename.split('_')
            if len(split_parts) <= 1:
                continue
                
            patient_id = split_parts[1]
            if patient_id not in self.patients_included:
                continue
                
            m = SESSION_RE.match(filename)
            if not m:
                print(f"Warning: Unexpected filename: {filename}")
                continue
                
            patient = m.group("patient")
            session = m.group("session")
            scan_type = m.group("scan_type")
            
            full = os.path.join(self.data_dir, filename)
            self.patients_sessions.setdefault(patient, {}).setdefault(session, []).append(full)
        
        self.pairs = []
        for patient, sessions in self.patients_sessions.items():
            for session, paths in sessions.items():
                if len(paths) >= 2:
                    for path1, path2 in combinations(paths, 2):
                        self.pairs.append({
                            'path1': path1,
                            'path2': path2,
                            'patient': patient,
                            'session': session
                        })

    def __len__(self):
        return len(self.pairs)

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

    def __getitem__(self, idx):
        pair_info = self.pairs[idx]
        vol1, header1 = self._load_volume_and_header(pair_info['path1'])
        vol2, header2 = self._load_volume_and_header(pair_info['path2'])
        vol1, vol2 = self._shared_random_crop(vol1, vol2)
        
        return {
            'vol1': vol1,
            'vol2': vol2,
        }

if __name__ == "__main__":
    dataset = ContrastivePatientDataset("/home/mg873uh/Projects_kb/data/pretrain_preproc/FOMO60k", set(["4972"]))
    print(dataset.pairs)
    print(len(dataset))
    print(dataset[0]['vol1'].shape)