"""
extract_features.py

Extracts and caches log-mel spectrogram features and Multi-ACCDOA labels for
the DCASE 2025 Task 3 audio-only stereo SELD dataset.

Usage:
    extractor = AudioFeatureExtractor(params)
    extractor.extract_spectrograms(split='dev')
    extractor.process_and_save_labels(split='dev')
"""

import os
import glob
import torch
from tqdm import tqdm
import utils


class AudioFeatureExtractor:
    """
    Computes and caches pre-processed features for the SELD dataset.

    Audio:  stereo log-mel spectrograms  →  (2, T, nb_mels) tensors
    Labels: Multi-ACCDOA or single-ACCDOA format  →  tensor per clip

    Features are written to disk the first time and skipped on subsequent
    runs, so re-running on a machine with cached features is instant.
    """

    def __init__(self, params):
        self.params   = params
        self.root_dir = params['root_dir']
        self.feat_dir = params['feat_dir']

        sr = params['sampling_rate']
        hop = int(sr * params['hop_length_s'])   # samples per frame hop
        win = 2 * hop                              # window length = 2 × hop
        self.sr      = sr
        self.hop     = hop
        self.win     = win
        self.n_fft   = 2 ** (win - 1).bit_length()  # next power-of-2 ≥ win
        self.nb_mels = params['nb_mels']

        self.nb_frames   = params['label_sequence_length']
        self.nb_classes  = params['nb_classes']
        self.multi_accdoa = params['multiACCDOA']

    # ---------------------------------------------------------------------- #
    # Spectrogram extraction
    # ---------------------------------------------------------------------- #

    def extract_spectrograms(self, split='dev'):
        """
        Compute stereo log-mel spectrograms for all WAV files in `split` and
        save each as a .pt file alongside the original audio path structure.

        Args:
            split: 'dev' or 'eval'
        """
        wav_pattern = {
            'dev':  os.path.join(self.root_dir, 'stereo_dev',  'dev-*', '*.wav'),
            'eval': os.path.join(self.root_dir, 'stereo_eval', 'eval',  '*.wav'),
        }
        if split not in wav_pattern:
            raise ValueError(f"split must be 'dev' or 'eval', got '{split}'")

        wav_files = glob.glob(wav_pattern[split])
        out_dir   = os.path.join(self.feat_dir, f'stereo_{split}')
        os.makedirs(out_dir, exist_ok=True)

        for wav in tqdm(wav_files, desc=f'Spectrograms ({split})', unit='file'):
            out_path = os.path.join(out_dir,
                                    os.path.splitext(os.path.basename(wav))[0] + '.pt')
            if os.path.exists(out_path):
                continue

            audio, sr = utils.load_audio(wav, self.sr)
            spec = utils.extract_log_mel_spectrogram(
                audio, sr, self.n_fft, self.hop, self.win, self.nb_mels
            )
            torch.save(torch.tensor(spec, dtype=torch.float32), out_path)

    # ---------------------------------------------------------------------- #
    # Label processing
    # ---------------------------------------------------------------------- #

    def process_and_save_labels(self, split='dev'):
        """
        Load raw CSV labels, convert to Multi-ACCDOA or single-ACCDOA tensors,
        and save as .pt files.

        Args:
            split: 'dev' or 'eval'
        """
        csv_pattern = {
            'dev':  os.path.join(self.root_dir, 'metadata_dev',  'dev-*', '*.csv'),
            'eval': os.path.join(self.root_dir, 'metadata_eval', 'eval',  '*.csv'),
        }
        if split not in csv_pattern:
            raise ValueError(f"split must be 'dev' or 'eval', got '{split}'")

        suffix  = '_adpit' if self.multi_accdoa else ''
        out_dir = os.path.join(self.feat_dir, f'metadata_{split}{suffix}')
        os.makedirs(out_dir, exist_ok=True)

        csv_files = glob.glob(csv_pattern[split])
        for csv_path in tqdm(csv_files, desc=f'Labels ({split})', unit='file'):
            out_path = os.path.join(out_dir,
                                    os.path.splitext(os.path.basename(csv_path))[0] + '.pt')
            if os.path.exists(out_path):
                continue

            raw = utils.load_labels(csv_path)
            label_tensor = (
                utils.process_labels_adpit(raw, self.nb_frames, self.nb_classes)
                if self.multi_accdoa
                else utils.process_labels(raw, self.nb_frames, self.nb_classes)
            )
            torch.save(label_tensor, out_path)

    # ---------------------------------------------------------------------- #
    # Convenience wrapper (called from main.py / main_improved.py)
    # ---------------------------------------------------------------------- #

    def extract_features(self, split='dev'):
        """Run spectrogram extraction (audio-only track)."""
        os.makedirs(self.feat_dir, exist_ok=True)
        self.extract_spectrograms(split)

    def extract_labels(self, split='dev'):
        """Alias kept for compatibility with main.py call signature."""
        self.process_and_save_labels(split)
