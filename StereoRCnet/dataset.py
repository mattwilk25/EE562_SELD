import os
import glob
import torch
from torch.utils.data import Dataset

from augment import apply_augmentation


def adpit_to_single_accdoa(labels):
    """Convert ADPIT labels to single ACCDOA format.

    ADPIT tracks: A0(idx=0), B0(idx=1), B1(idx=2), C0(idx=3), C1(idx=4), C2(idx=5)
    For single ACCDOA, take one representative source per class:
      - A0 when exactly 1 source of a class
      - B0 when exactly 2 overlapping sources of the same class
      - C0 when exactly 3 overlapping sources of the same class
    These categories are mutually exclusive per class per frame.

    Args:
        labels: (T, 6, 5, C) ADPIT tensor where dim2 = [SED, x, y, dist, onscreen]
    Returns:
        accdoa: (T, 3*C) single ACCDOA tensor [act*x, act*y, act*dist] per class
    """
    # Sum over representative tracks (A0=0, B0=1, C0=3) - mutually exclusive
    sed = labels[:, 0, 0, :] + labels[:, 1, 0, :] + labels[:, 3, 0, :]
    x   = labels[:, 0, 1, :] + labels[:, 1, 1, :] + labels[:, 3, 1, :]
    y   = labels[:, 0, 2, :] + labels[:, 1, 2, :] + labels[:, 3, 2, :]
    dist = labels[:, 0, 3, :] + labels[:, 1, 3, :] + labels[:, 3, 3, :]

    # Activity-coupled representation
    accdoa_x = sed * x
    accdoa_y = sed * y
    accdoa_dist = sed * dist

    return torch.cat([accdoa_x, accdoa_y, accdoa_dist], dim=1)


class SELDDataset(Dataset):
    """Dataset for stereo SELD with single ACCDOA labels.

    Loads pre-extracted log-mel features and ADPIT labels,
    converts labels to single ACCDOA format, and optionally
    applies data augmentation.
    """

    def __init__(self, cfg, mode='dev_train'):
        """
        Args:
            cfg: configuration dictionary
            mode: 'dev_train' or 'dev_test'
        """
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.augment = (mode == 'dev_train') and cfg.get('use_augmentation', False)

        folds = cfg['dev_train_folds'] if mode == 'dev_train' else cfg['dev_test_folds']

        # Collect feature and label file paths
        self.audio_files = []
        self.label_files = []
        for fold in folds:
            audio_pattern = os.path.join(cfg['feat_dir'], f'stereo_dev/{fold}*.pt')
            label_pattern = os.path.join(cfg['feat_dir'], f'metadata_dev_adpit/{fold}*.pt')
            self.audio_files.extend(glob.glob(audio_pattern))
            self.label_files.extend(glob.glob(label_pattern))

        self.audio_files = sorted(self.audio_files, key=lambda p: os.path.basename(p))
        self.label_files = sorted(self.label_files, key=lambda p: os.path.basename(p))

        assert len(self.audio_files) == len(self.label_files), \
            f'Mismatch: {len(self.audio_files)} audio vs {len(self.label_files)} label files'

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio = torch.load(self.audio_files[idx], weights_only=True)  # (2, 251, 64)
        labels = torch.load(self.label_files[idx], weights_only=True)  # (50, 6, 5, 13)

        # Apply augmentation to audio features during training
        if self.augment:
            audio = apply_augmentation(audio, self.cfg)

        # Convert ADPIT labels to single ACCDOA
        accdoa = adpit_to_single_accdoa(labels)  # (50, 39)

        return audio, accdoa

    def get_filenames(self):
        """Return list of filenames (without extension) for output mapping."""
        return [os.path.splitext(os.path.basename(f))[0] for f in self.audio_files]


if __name__ == '__main__':
    from config import config
    from torch.utils.data import DataLoader

    train_set = SELDDataset(config, mode='dev_train')
    test_set = SELDDataset(config, mode='dev_test')
    print(f'Train samples: {len(train_set)}')
    print(f'Test samples:  {len(test_set)}')

    loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=0)
    for audio, labels in loader:
        print(f'Audio shape:  {audio.shape}')
        print(f'Labels shape: {labels.shape}')
        print(f'Label range:  [{labels.min():.3f}, {labels.max():.3f}]')
        break
