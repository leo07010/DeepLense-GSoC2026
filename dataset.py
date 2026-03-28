"""Dataset loader for DeepLense gravitational lensing images."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

CLASS_NAMES = ['no', 'sphere', 'vort']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class LensingDataset(Dataset):
    """Strong gravitational lensing image dataset.

    Each sample is a (1, 150, 150) single-channel image stored as .npy,
    already min-max normalized to [0, 1].
    """

    def __init__(self, root_dir, augment=False, downsample_size=None):
        self.samples = []
        self.labels = []
        self.augment = augment
        self.downsample_size = downsample_size

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith('.npy'):
                    self.samples.append(os.path.join(class_dir, fname))
                    self.labels.append(CLASS_TO_IDX[class_name])
        self.labels = np.array(self.labels)
        print(f'Loaded {len(self.samples)} samples from {root_dir}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = np.load(self.samples[idx]).astype(np.float32)  # (1, 150, 150)
        label = self.labels[idx]

        if self.augment:
            if np.random.rand() > 0.5:
                img = img[:, :, ::-1].copy()
            if np.random.rand() > 0.5:
                img = img[:, ::-1, :].copy()
            k = np.random.randint(0, 4)
            img = np.rot90(img, k, axes=(1, 2)).copy()

        img_t = torch.from_numpy(img)

        if self.downsample_size is not None:
            img_t = F.adaptive_avg_pool2d(img_t, self.downsample_size)

        return img_t, label


def get_dataloaders(data_dir, batch_size=64, downsample_size=None,
                    num_workers=4, train_subset=None):
    """Create train and val DataLoaders.

    Args:
        data_dir: path to data/ folder containing train/ and val/
        batch_size: batch size
        downsample_size: if set, downsample images to this spatial size
        num_workers: number of dataloader workers
        train_subset: if set, subsample training data to this total count
    """
    train_dataset = LensingDataset(
        os.path.join(data_dir, 'train'),
        augment=True, downsample_size=downsample_size)
    val_dataset = LensingDataset(
        os.path.join(data_dir, 'val'),
        augment=False, downsample_size=downsample_size)

    if train_subset is not None:
        per_class = train_subset // 3
        indices = []
        for c in range(3):
            class_indices = np.where(train_dataset.labels == c)[0]
            chosen = np.random.choice(class_indices, per_class, replace=False)
            indices.extend(chosen.tolist())
        np.random.shuffle(indices)
        train_dataset = Subset(train_dataset, indices)
        print(f'Using subset: {len(train_dataset)} training samples')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
