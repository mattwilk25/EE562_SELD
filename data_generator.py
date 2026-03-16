import os
import glob
import torch
from torch.utils.data import Dataset


class SELDDataset(Dataset):
    def __init__(self, params, split='train'):
        folds = params['dev_train_folds'] if split == 'train' else params['dev_test_folds']
        feat  = params['feat_dir']

        audio, labels = [], []
        for fold in folds:
            audio  += glob.glob(os.path.join(feat, f'stereo_dev/{fold}*.pt'))
            labels += glob.glob(os.path.join(feat, f'metadata_dev_adpit/{fold}*.pt'))

        self.audio_files = sorted(audio,  key=os.path.basename)
        self.label_files = sorted(labels, key=os.path.basename)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio  = torch.load(self.audio_files[idx])
        labels = torch.load(self.label_files[idx])
        labels = labels[:, :, :-1, :]  # drop on-screen dim (audio-only)
        return audio, labels


# Alias kept for compatibility
DataGenerator = SELDDataset
