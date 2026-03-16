"""Generate predictions on evaluation set (no labels)."""

import os
import sys
import glob
import pickle
import argparse

import torch
from torch.utils.data import Dataset, DataLoader

from config import config
from model import StereoRCnet
from seld_utils import decode_single_accdoa, write_output_csv


class EvalDataset(Dataset):
    """Dataset for evaluation set (audio features only, no labels)."""

    def __init__(self, feat_dir):
        eval_dir = os.path.join(feat_dir, 'stereo_eval')
        if not os.path.isdir(eval_dir):
            raise FileNotFoundError(
                f'Evaluation features not found at {eval_dir}. '
                f'Extract features from stereo_eval.zip first.'
            )
        self.files = sorted(glob.glob(os.path.join(eval_dir, '*.pt')))
        if len(self.files) == 0:
            raise FileNotFoundError(f'No .pt files found in {eval_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = torch.load(self.files[idx], weights_only=True)
        filename = os.path.splitext(os.path.basename(self.files[idx]))[0]
        return audio, filename


def inference(ckpt_path, output_dir=None, threshold=None):
    """Run inference on the evaluation set and write CSV predictions.

    Args:
        ckpt_path: path to checkpoint directory or .pth file
        output_dir: where to write prediction CSVs
        threshold: SED threshold (overrides config if provided)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint and config
    if os.path.isdir(ckpt_path):
        model_path = os.path.join(ckpt_path, 'best_model.pth')
        config_path = os.path.join(ckpt_path, 'config.pkl')
    else:
        model_path = ckpt_path
        config_path = os.path.join(os.path.dirname(ckpt_path), 'config.pkl')

    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            cfg = pickle.load(f)
    else:
        cfg = config.copy()

    if threshold is not None:
        cfg['sed_threshold'] = threshold

    model = StereoRCnet(cfg).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    print(f'Loaded model from epoch {state.get("epoch", "?")}')

    # Setup output
    if output_dir is None:
        output_dir = os.path.join(cfg['output_dir'], 'eval_predictions')
    os.makedirs(output_dir, exist_ok=True)

    # Load evaluation data
    try:
        eval_set = EvalDataset(cfg['feat_dir'])
    except FileNotFoundError as e:
        print(f'Error: {e}')
        return

    eval_loader = DataLoader(
        eval_set, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['nb_workers'], pin_memory=True,
    )

    print(f'Running inference on {len(eval_set)} evaluation clips...')

    # We need to collect filenames alongside batched predictions
    all_filenames = []
    with torch.no_grad():
        for audio, filenames in eval_loader:
            audio = audio.to(device)
            pred = model(audio)

            for i in range(pred.size(0)):
                output_dict = decode_single_accdoa(
                    pred[i], nb_classes=cfg['nb_classes'],
                    threshold=cfg['sed_threshold'],
                )
                csv_path = os.path.join(output_dir, filenames[i] + '.csv')
                write_output_csv(output_dict, csv_path)
                all_filenames.append(filenames[i])

    print(f'Wrote {len(all_filenames)} prediction files to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on evaluation set')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint dir or .pth file')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    args = parser.parse_args()

    inference(args.checkpoint, args.output_dir, args.threshold)
