"""Standalone evaluation script for a trained checkpoint."""

import os
import sys
import pickle
import argparse

import torch
from torch.utils.data import DataLoader

from config import config
from model import StereoRCnet
from dataset import SELDDataset
from seld_utils import (
    decode_single_accdoa,
    write_output_csv,
    get_seld_evaluator,
    evaluate_predictions,
    compute_dynamic_thresholds,
)


def load_checkpoint(ckpt_path, device):
    """Load model from checkpoint directory or .pth file."""
    if os.path.isdir(ckpt_path):
        model_path = os.path.join(ckpt_path, 'best_model.pth')
        config_path = os.path.join(ckpt_path, 'config.pkl')
    else:
        model_path = ckpt_path
        config_path = os.path.join(os.path.dirname(ckpt_path), 'config.pkl')

    # Load config
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            cfg = pickle.load(f)
    else:
        print('Config not found, using default config.')
        cfg = config.copy()

    # Build model and load weights
    model = StereoRCnet(cfg).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    print(f'Loaded checkpoint from epoch {state.get("epoch", "?")}, '
          f'F-score: {100 * state.get("best_f_score", 0):.1f}%')

    return model, cfg


def evaluate(ckpt_path, use_dynamic_threshold=False):
    """Run full evaluation on the test set.

    Args:
        ckpt_path: path to checkpoint directory or .pth file
        use_dynamic_threshold: if True, compute class-specific thresholds on val set
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_checkpoint(ckpt_path, device)

    test_set = SELDDataset(cfg, mode='dev_test')
    test_loader = DataLoader(
        test_set, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['nb_workers'], pin_memory=True,
    )

    # Optionally compute dynamic thresholds
    if use_dynamic_threshold:
        print('Computing class-specific dynamic thresholds...')
        thresholds = compute_dynamic_thresholds(model, test_loader, cfg, device)
        print(f'Thresholds: {thresholds}')
    else:
        thresholds = [cfg['sed_threshold']] * cfg['nb_classes']

    # Generate predictions
    out_dir = os.path.join(cfg['output_dir'], 'eval_output')
    pred_dir = os.path.join(out_dir, 'dev-test')
    os.makedirs(pred_dir, exist_ok=True)

    filenames = test_set.get_filenames()
    file_idx = 0

    model.eval()
    with torch.no_grad():
        for audio, _ in test_loader:
            audio = audio.to(device)
            pred = model(audio)
            batch_size = pred.size(0)

            for i in range(batch_size):
                pred_np = pred[i].cpu().numpy()
                C = cfg['nb_classes']
                x = pred_np[:, :C]
                y = pred_np[:, C:2 * C]
                magnitude = (x ** 2 + y ** 2) ** 0.5

                # Apply per-class thresholds
                output_dict = {}
                import math
                import numpy as np
                dist = np.clip(pred_np[:, 2 * C:3 * C], 0, None)

                for t in range(pred_np.shape[0]):
                    for c in range(C):
                        if magnitude[t, c] > thresholds[c]:
                            if t not in output_dict:
                                output_dict[t] = []
                            azi = math.atan2(y[t, c], x[t, c]) * 180.0 / math.pi
                            output_dict[t].append([c, 0, azi, dist[t, c], 0])

                csv_path = os.path.join(pred_dir, filenames[file_idx] + '.csv')
                write_output_csv(output_dict, csv_path)
                file_idx += 1

    # Compute metrics
    evaluator = get_seld_evaluator(cfg)
    metrics = evaluate_predictions(evaluator, pred_dir)

    print(f'\n{"=" * 50}')
    print(f'  F-score (20 deg): {100 * metrics["F"]:.1f}%')
    print(f'  DOA error:        {metrics["AngE"]:.1f} deg')
    print(f'  Dist error:       {metrics["DistE"]:.2f}')
    print(f'  Rel dist error:   {metrics["RelDistE"]:.2f}')
    print(f'{"=" * 50}')

    if len(metrics.get('classwise', [])):
        classwise = metrics['classwise']
        class_names = [
            'FemSpeech', 'MalSpeech', 'Clapping', 'Telephone', 'Laughter',
            'Domestic', 'Footsteps', 'Door', 'Music', 'Instrument',
            'WaterTap', 'Bell', 'Knock'
        ]
        print(f'\n{"Class":<12} {"F":>8} {"AngE":>8} {"RDE":>8}')
        print('-' * 40)
        for c in range(cfg['nb_classes']):
            print(f'{class_names[c]:<12} {classwise[0][c]:8.2f} '
                  f'{classwise[1][c]:8.1f} {classwise[3][c]:8.2f}')

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained Stereo-RCnet model')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint dir or .pth file')
    parser.add_argument('--dynamic_threshold', action='store_true',
                        help='Use class-specific dynamic thresholds')
    args = parser.parse_args()

    evaluate(args.checkpoint, use_dynamic_threshold=args.dynamic_threshold)
