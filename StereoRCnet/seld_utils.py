"""Utility functions for SELD prediction decoding, output formatting, and evaluation."""

import os
import sys
import math
import numpy as np
import torch


def decode_single_accdoa(pred, nb_classes=13, threshold=0.5):
    """Decode single ACCDOA output to event detections per frame.

    Args:
        pred: (T, 3*C) numpy array or torch tensor
        nb_classes: number of sound classes
        threshold: SED activity threshold on ACCDOA magnitude
    Returns:
        output_dict: {frame_idx: [[class, src_id, azimuth_deg, dist, onscreen], ...]}
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    T = pred.shape[0]
    C = nb_classes

    x = pred[:, :C]
    y = pred[:, C:2 * C]
    dist = pred[:, 2 * C:3 * C]

    magnitude = np.sqrt(x ** 2 + y ** 2)
    sed = magnitude > threshold

    dist = np.clip(dist, 0.0, None)

    output_dict = {}
    for t in range(T):
        for c in range(C):
            if sed[t, c]:
                if t not in output_dict:
                    output_dict[t] = []
                azi = math.atan2(y[t, c], x[t, c]) * 180.0 / math.pi
                output_dict[t].append([
                    c,                  # class index
                    0,                  # source id (single ACCDOA: always 0)
                    azi,                # azimuth in degrees
                    dist[t, c],         # distance in model scale (0-1)
                    0,                  # onscreen (N/A for audio-only)
                ])
    return output_dict


def write_output_csv(output_dict, filepath, convert_dist_to_cm=True):
    """Write predictions to DCASE output CSV format.

    Args:
        output_dict: {frame_idx: [[class, src_id, azimuth, dist, onscreen], ...]}
        filepath: output CSV file path
        convert_dist_to_cm: if True, multiply distance by 100
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write('frame,class,source,azimuth,distance,onscreen\n')
        for frame_idx in sorted(output_dict.keys()):
            for event in output_dict[frame_idx]:
                cls, src, azi, dist, onscreen = event
                dist_out = round(float(dist) * 100) if convert_dist_to_cm else round(float(dist))
                f.write(f'{int(frame_idx)},{int(cls)},{int(src)},'
                        f'{round(float(azi))},{dist_out},{int(onscreen)}\n')


def get_seld_evaluator(cfg):
    """Create a SELD evaluation object using the baseline metrics.

    Adds the baseline code directory to sys.path and imports ComputeSELDResults.

    Args:
        cfg: configuration dictionary
    Returns:
        ComputeSELDResults instance
    """
    baseline_dir = os.path.join(cfg['root_dir'], 'DCASE2025_seld_baseline')
    if baseline_dir not in sys.path:
        sys.path.insert(0, baseline_dir)

    from metrics import ComputeSELDResults

    eval_params = {
        'root_dir': cfg['root_dir'],
        'modality': 'audio',
        'lad_doa_thresh': cfg['lad_doa_thresh'],
        'lad_dist_thresh': cfg['lad_dist_thresh'],
        'lad_reldist_thresh': cfg['lad_reldist_thresh'],
        'lad_req_onscreen': False,
        'average': cfg['average'],
        'nb_classes': cfg['nb_classes'],
    }
    return ComputeSELDResults(eval_params, ref_files_folder=cfg['metadata_dir'])


def evaluate_predictions(evaluator, pred_dir):
    """Evaluate predictions using the SELD evaluator.

    Args:
        evaluator: ComputeSELDResults instance
        pred_dir: directory containing prediction CSV files (in a split subfolder)
    Returns:
        dict with F, AngE, DistE, RelDistE, OnscreenAcc
    """
    F, AngE, DistE, RelDistE, OnscreenAcc, classwise = evaluator.get_SELD_Results(pred_dir)
    return {
        'F': F,
        'AngE': AngE,
        'DistE': DistE,
        'RelDistE': RelDistE,
        'OnscreenAcc': OnscreenAcc,
        'classwise': classwise,
    }


def compute_dynamic_thresholds(model, dataloader, cfg, device,
                               threshold_range=None, nb_classes=13):
    """Compute class-specific detection thresholds on validation data.

    For each class, finds the threshold (from a grid) that maximizes the
    per-class F-score on the given data.

    Args:
        model: trained model
        dataloader: validation DataLoader
        cfg: configuration dict
        device: torch device
        threshold_range: list of thresholds to try; defaults to [0.3, 0.35, ..., 0.7]
        nb_classes: number of classes
    Returns:
        thresholds: list of length nb_classes with optimal threshold per class
    """
    if threshold_range is None:
        threshold_range = [round(0.3 + 0.05 * i, 2) for i in range(9)]

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for audio, labels in dataloader:
            audio = audio.to(device)
            pred = model(audio)
            all_preds.append(pred.cpu())
            all_targets.append(labels)

    all_preds = torch.cat(all_preds, dim=0).numpy()    # (N, 50, 39)
    all_targets = torch.cat(all_targets, dim=0).numpy()  # (N, 50, 39)

    C = nb_classes
    target_x = all_targets[:, :, :C]
    target_y = all_targets[:, :, C:2 * C]
    target_activity = np.sqrt(target_x ** 2 + target_y ** 2) > 0.5

    pred_x = all_preds[:, :, :C]
    pred_y = all_preds[:, :, C:2 * C]
    pred_magnitude = np.sqrt(pred_x ** 2 + pred_y ** 2)

    best_thresholds = [0.5] * C
    for c in range(C):
        best_f = -1.0
        for thresh in threshold_range:
            pred_activity = pred_magnitude[:, :, c] > thresh
            gt_activity = target_activity[:, :, c]

            tp = np.sum(pred_activity & gt_activity)
            fp = np.sum(pred_activity & ~gt_activity)
            fn = np.sum(~pred_activity & gt_activity)

            f_score = tp / (tp + 0.5 * (fp + fn) + 1e-8)
            if f_score > best_f:
                best_f = f_score
                best_thresholds[c] = thresh

    return best_thresholds
