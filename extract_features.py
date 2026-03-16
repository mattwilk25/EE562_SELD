import os
import glob
import torch
from tqdm import tqdm
import utils


class AudioFeatureExtractor:
    def __init__(self, params):
        self.params   = params
        self.root_dir = params['root_dir']
        self.feat_dir = params['feat_dir']

        sr  = params['sampling_rate']
        hop = int(sr * params['hop_length_s'])
        win = 2 * hop
        self.sr      = sr
        self.hop     = hop
        self.win     = win
        self.n_fft   = 2 ** (win - 1).bit_length()
        self.nb_mels = params['nb_mels']

        self.nb_frames  = params['label_sequence_length']
        self.nb_classes = params['nb_classes']

    def extract_features(self, split='dev'):
        os.makedirs(self.feat_dir, exist_ok=True)
        wav_pattern = {
            'dev':  os.path.join(self.root_dir, 'stereo_dev',  'dev-*', '*.wav'),
            'eval': os.path.join(self.root_dir, 'stereo_eval', 'eval',  '*.wav'),
        }
        out_dir = os.path.join(self.feat_dir, f'stereo_{split}')
        os.makedirs(out_dir, exist_ok=True)

        for wav in tqdm(glob.glob(wav_pattern[split]), desc=f'Spectrograms ({split})', unit='file'):
            out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(wav))[0] + '.pt')
            if os.path.exists(out_path):
                continue
            audio, sr = utils.load_audio(wav, self.sr)
            spec = utils.extract_log_mel_spectrogram(audio, sr, self.n_fft, self.hop, self.win, self.nb_mels)
            torch.save(torch.tensor(spec, dtype=torch.float32), out_path)

    def extract_labels(self, split='dev'):
        csv_pattern = {
            'dev':  os.path.join(self.root_dir, 'metadata_dev',  'dev-*', '*.csv'),
            'eval': os.path.join(self.root_dir, 'metadata_eval', 'eval',  '*.csv'),
        }
        out_dir = os.path.join(self.feat_dir, f'metadata_{split}_adpit')
        os.makedirs(out_dir, exist_ok=True)

        for csv_path in tqdm(glob.glob(csv_pattern[split]), desc=f'Labels ({split})', unit='file'):
            out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(csv_path))[0] + '.pt')
            if os.path.exists(out_path):
                continue
            raw = utils.load_labels(csv_path)
            torch.save(utils.process_labels_adpit(raw, self.nb_frames, self.nb_classes), out_path)
