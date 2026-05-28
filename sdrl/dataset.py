"""
dataset.py
----------
PyTorch Dataset classes for SDRL gravity super-resolution.

Upload to: /content/drive/MyDrive/sdrl_gravity/dataset.py
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# ════════════════════════════════════════════════════════════════════════════
# Paired dataset  (satellite LR  +  shipborne HR ground truth)
# ════════════════════════════════════════════════════════════════════════════
class PairedGravityDataset(Dataset):
    """
    Loads lr_patches.npy and hr_patches.npy from a directory.
    Each sample returns (lr_tensor, hr_tensor) both in [0, 1].
    Shapes: lr (1,50,50), hr (1,200,200).
    """
    def __init__(self, data_dir: str, augment: bool = True):
        self.augment = augment
        lr_path = os.path.join(data_dir, 'lr_patches.npy')
        hr_path = os.path.join(data_dir, 'hr_patches.npy')

        if not os.path.exists(lr_path):
            raise FileNotFoundError(f"LR patches not found: {lr_path}\n"
                                    "Run preprocess.run_preprocessing() first.")

        self.lr = np.load(lr_path, mmap_mode='r')  # (N, 1, 50, 50)
        self.hr = np.load(hr_path, mmap_mode='r')  # (N, 1, 200, 200)
        assert len(self.lr) == len(self.hr), "LR/HR length mismatch"
        print(f"PairedDataset: {len(self.lr):,} samples from {data_dir}")

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr[idx].copy()).float()
        hr = torch.from_numpy(self.hr[idx].copy()).float()

        if self.augment:
            lr, hr = self._augment(lr, hr)

        return lr, hr

    @staticmethod
    def _augment(lr, hr):
        """Consistent horizontal/vertical flips + 90° rotations."""
        if torch.rand(1) > 0.5:
            lr = torch.flip(lr, dims=[-1])
            hr = torch.flip(hr, dims=[-1])
        if torch.rand(1) > 0.5:
            lr = torch.flip(lr, dims=[-2])
            hr = torch.flip(hr, dims=[-2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            lr = torch.rot90(lr, k, dims=[-2, -1])
            hr = torch.rot90(hr, k, dims=[-2, -1])
        return lr, hr


# ════════════════════════════════════════════════════════════════════════════
# Unpaired dataset  (satellite LR only — no HR ground truth)
# ════════════════════════════════════════════════════════════════════════════
class UnpairedGravityDataset(Dataset):
    """
    Returns only LR satellite patches (no HR label).
    Used for the unsupervised cycle-consistency arm of SDRL.
    """
    def __init__(self, data_dir: str, augment: bool = True):
        self.augment = augment
        lr_path = os.path.join(data_dir, 'lr_patches.npy')
        if not os.path.exists(lr_path):
            raise FileNotFoundError(f"Unpaired LR patches not found: {lr_path}")
        self.lr = np.load(lr_path, mmap_mode='r')
        print(f"UnpairedDataset: {len(self.lr):,} samples from {data_dir}")

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr[idx].copy()).float()
        if self.augment and torch.rand(1) > 0.5:
            lr = torch.flip(lr, dims=[-1])
        return lr


# ════════════════════════════════════════════════════════════════════════════
# Semi-supervised batch sampler
# ════════════════════════════════════════════════════════════════════════════
class SDRLDataLoader:
    """
    Yields batches that contain both paired and unpaired samples
    at a 1:1 ratio (paper Section 4.1).

    Each batch is a dict:
      'lr_paired'   : (B/2, 1, 50, 50)
      'hr_paired'   : (B/2, 1, 200, 200)
      'lr_unpaired' : (B/2, 1, 50, 50)

    Use batch_size = total samples per step (split evenly between paired/unpaired).
    """
    def __init__(self,
                 paired_dir:  str,
                 unpair_dir:  str,
                 batch_size:  int  = 8,
                 num_workers: int  = 2,
                 augment:     bool = True,
                 val_split:   float = 0.1,
                 seed:        int  = 42):

        half = batch_size // 2

        # full datasets
        paired_full   = PairedGravityDataset(paired_dir,  augment=augment)
        unpaired_full = UnpairedGravityDataset(unpair_dir, augment=augment)

        # train / val split for paired data
        n_total  = len(paired_full)
        n_val    = max(1, int(n_total * val_split))
        n_train  = n_total - n_val
        gen      = torch.Generator().manual_seed(seed)
        train_p, val_p = torch.utils.data.random_split(
            paired_full, [n_train, n_val], generator=gen)

        self.train_paired_loader = DataLoader(
            train_p, batch_size=half, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True)

        self.val_loader = DataLoader(
            val_p, batch_size=half * 2, shuffle=False,
            num_workers=num_workers, pin_memory=True)

        self.train_unpaired_loader = DataLoader(
            unpaired_full, batch_size=half, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True)

        print(f"\nDataLoader summary:")
        print(f"  Train paired   : {n_train:,}  |  Val: {n_val:,}")
        print(f"  Train unpaired : {len(unpaired_full):,}")
        print(f"  Batch size     : {batch_size}  (half paired / half unpaired)")

    def train_batches(self):
        """Infinite generator that yields mixed batches."""
        paired_iter   = iter(self.train_paired_loader)
        unpaired_iter = iter(self.train_unpaired_loader)

        while True:
            # get next paired batch (restart if exhausted)
            try:
                lr_p, hr_p = next(paired_iter)
            except StopIteration:
                paired_iter = iter(self.train_paired_loader)
                lr_p, hr_p = next(paired_iter)

            # get next unpaired batch
            try:
                lr_u = next(unpaired_iter)
            except StopIteration:
                unpaired_iter = iter(self.train_unpaired_loader)
                lr_u = next(unpaired_iter)

            yield {
                'lr_paired':   lr_p,
                'hr_paired':   hr_p,
                'lr_unpaired': lr_u,
            }

    def steps_per_epoch(self):
        return len(self.train_paired_loader)


# ════════════════════════════════════════════════════════════════════════════
# Quick test
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    BASE = '/content/drive/MyDrive/sdrl_gravity'
    dl = SDRLDataLoader(
        paired_dir  = f'{BASE}/data/processed/paired',
        unpair_dir  = f'{BASE}/data/processed/unpaired',
        batch_size  = 8,
        num_workers = 0,
    )
    gen = dl.train_batches()
    batch = next(gen)
    print("lr_paired  :", batch['lr_paired'].shape)
    print("hr_paired  :", batch['hr_paired'].shape)
    print("lr_unpaired:", batch['lr_unpaired'].shape)
    print("Dataset test passed ✓")
