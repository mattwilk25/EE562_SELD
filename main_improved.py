"""main_improved.py — ResNet-Conformer training pipeline with ACS augmentation."""

import os
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from model_improved import SELDModelImproved
from loss import ADPITLoss
from metrics import ComputeSELDResults
from data_generator import SELDDataset
from extract_features import AudioFeatureExtractor
import utils

params = {
    # paths
    'net_type':        'SELDnet',
    'root_dir':        '../DCASE2025_SELD_dataset',
    'feat_dir':        '../DCASE2025_SELD_dataset/features',
    'log_dir':         'logs',
    'checkpoints_dir': 'checkpoints',
    'output_dir':      'outputs',
    # audio features
    'sampling_rate':   24000,
    'hop_length_s':    0.02,
    'nb_mels':         64,
    # ResNet encoder
    'f_pool_size':     [4, 4, 2],
    't_pool_size':     [5, 1, 1],
    'dropout':         0.05,
    # Conformer
    'rc_d_model':      256,
    'rc_n_heads':      8,
    'rc_ff_dim':       512,
    'rc_nb_conformer': 4,
    'rc_conv_kernel':  31,
    # output
    'max_polyphony':   3,
    'nb_classes':      13,
    'label_sequence_length': 50,
    'thresh_unify':    15,
    # training
    'nb_epochs':       200,
    'batch_size':      256,
    'nb_workers':      0,
    'shuffle':         True,
    'learning_rate':   1e-3,
    'weight_decay':    0,
    'dev_train_folds': ['fold3'],
    'dev_test_folds':  ['fold4'],
    # ACS augmentation
    'acs_prob':        0.5,
    # eval
    'average':         'macro',
    'lad_doa_thresh':  20,
    'lad_dist_thresh': float('inf'),
    'lad_reldist_thresh': 1.0,
    'lad_req_onscreen': False,
    'use_jackknife':   False,
}

VAL_INTERVAL = 5


def apply_acs(audio, labels):
    """Audio Channel Swapping: flip L/R channels and negate sin(azimuth)."""
    B = audio.shape[0]
    mask = torch.rand(B, device=audio.device) < params['acs_prob']
    if not mask.any():
        return audio, labels
    audio  = audio.clone()
    labels = labels.clone()
    audio[mask]                   = audio[mask].flip(1)   # swap stereo channels
    labels[mask, :, :, 2, :]      = -labels[mask, :, :, 2, :]  # negate y=sin(az)
    return audio, labels


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for audio, labels in tqdm(loader, desc='Train', leave=False):
        audio, labels = audio.to(device), labels.to(device)
        audio, labels = apply_acs(audio, labels)
        optimizer.zero_grad()
        loss = criterion(model(audio), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def val_epoch(model, loader, criterion, scorer, output_dir):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for j, (audio, labels) in enumerate(tqdm(loader, desc='Val', leave=False)):
            audio, labels = audio.to(device), labels.to(device)
            logits = model(audio)
            total_loss += criterion(logits, labels).item()
            utils.write_logits_to_dcase_format(
                logits, params, output_dir,
                loader.dataset.label_files[j * params['batch_size']: (j + 1) * params['batch_size']]
            )
    scores = scorer.get_SELD_Results(os.path.join(output_dir, 'dev-test'))
    return total_loss / len(loader), scores


def main():
    global params
    if restore_from_checkpoint:
        with open(os.path.join(checkpoint_path, 'config.pkl'), 'rb') as f:
            params = pickle.load(f)

    ckpt_dir, out_dir, _ = utils.setup(params)

    extractor = AudioFeatureExtractor(params)
    extractor.extract_features('dev')
    extractor.extract_labels('dev')

    print('Building datasets...')
    train_loader = DataLoader(SELDDataset(params, 'train'), batch_size=params['batch_size'],
                              num_workers=params['nb_workers'], shuffle=params['shuffle'], drop_last=True)
    val_loader   = DataLoader(SELDDataset(params, 'test'),  batch_size=params['batch_size'],
                              num_workers=params['nb_workers'], shuffle=False, drop_last=False)
    print(f'  Train: {len(train_loader.dataset)} clips | Val: {len(val_loader.dataset)} clips')

    model     = SELDModelImproved(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
    criterion = ADPITLoss().to(device)
    scorer    = ComputeSELDResults(params, ref_files_folder=os.path.join(params['root_dir'], 'metadata_dev'))

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,} | Device: {device}')

    start_epoch, best_f = 0, float('-inf')
    if restore_from_checkpoint:
        ckpt = torch.load(os.path.join(checkpoint_path, 'best_model.pth'),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        start_epoch, best_f = ckpt['epoch'] + 1, ckpt['best_f']

    for epoch in range(start_epoch, params['nb_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)

        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == params['nb_epochs'] - 1:
            val_loss, (f, ang, dist, *_) = val_epoch(model, val_loader, criterion, scorer, out_dir)
            print(f"Epoch {epoch+1}/{params['nb_epochs']} | "
                  f"Train: {train_loss:.3f} | Val: {val_loss:.3f} | "
                  f"F: {f*100:.1f}% | DOA: {ang:.1f}° | Dist: {dist:.2f}", flush=True)
        else:
            f = float('-inf')
            print(f"Epoch {epoch+1}/{params['nb_epochs']} | Train: {train_loss:.3f}", flush=True)

        if f > best_f:
            best_f = f
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                        'epoch': epoch, 'best_f': best_f},
                       os.path.join(ckpt_dir, 'best_model.pth'))

    ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    _, scores = val_epoch(model, val_loader, criterion, scorer, out_dir)
    utils.print_results(*scores, params)


if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    restore_from_checkpoint = False
    checkpoint_path = 'checkpoints/SELDnet_20250331_152343'
    main()
