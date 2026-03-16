"""Training script for Stereo-RCnet SELD model."""

import os
import sys
import time
import pickle
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from config import config
from model import StereoRCnet
from dataset import SELDDataset
from loss import SELDLoss
from seld_utils import (
    decode_single_accdoa,
    write_output_csv,
    get_seld_evaluator,
    evaluate_predictions,
)


def setup_experiment(cfg):
    """Create directories and logging for a training run."""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = f'StereoRCnet_{timestamp}'

    ckpt_dir = os.path.join(cfg['checkpoints_dir'], run_name)
    log_dir = os.path.join(cfg['log_dir'], run_name)
    out_dir = os.path.join(cfg['output_dir'], run_name)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(ckpt_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(cfg, f)

    writer = SummaryWriter(log_dir=log_dir)

    return ckpt_dir, out_dir, writer, run_name


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    """Linear warmup for the first few epochs."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def run_epoch(model, dataloader, criterion, optimizer, device, scaler,
              is_train=True, grad_accum_steps=1):
    """Run a single epoch of training or validation with AMP.

    Returns:
        Average loss for the epoch.
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step, (audio, labels) in enumerate(dataloader):
            audio = audio.to(device)
            labels = labels.to(device)

            with autocast('cuda', dtype=torch.bfloat16):
                pred = model(audio)
                loss = criterion(pred, labels)
                loss = loss / grad_accum_steps

            if is_train:
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps
            n_batches += 1

    return total_loss / max(n_batches, 1)


def run_evaluation(model, dataloader, cfg, device, out_dir, split='dev-test'):
    """Generate predictions and compute SELD metrics.

    Returns:
        metrics: dict with F, AngE, DistE, RelDistE
    """
    model.eval()
    filenames = dataloader.dataset.get_filenames()
    file_idx = 0

    pred_dir = os.path.join(out_dir, split)
    os.makedirs(pred_dir, exist_ok=True)

    with torch.no_grad():
        for audio, _ in dataloader:
            audio = audio.to(device)
            with autocast('cuda', dtype=torch.bfloat16):
                pred = model(audio)  # (B, 50, 39)
            pred = pred.float()

            batch_size = pred.size(0)
            for i in range(batch_size):
                output_dict = decode_single_accdoa(
                    pred[i], nb_classes=cfg['nb_classes'],
                    threshold=cfg['sed_threshold']
                )
                csv_path = os.path.join(pred_dir, filenames[file_idx] + '.csv')
                write_output_csv(output_dict, csv_path, convert_dist_to_cm=True)
                file_idx += 1

    evaluator = get_seld_evaluator(cfg)
    metrics = evaluate_predictions(evaluator, pred_dir)
    return metrics


def train(cfg):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Print configuration
    print('\nConfiguration:')
    for k, v in cfg.items():
        print(f'  {k}: {v}')

    # Setup experiment
    ckpt_dir, out_dir, writer, run_name = setup_experiment(cfg)
    print(f'\nExperiment: {run_name}')

    # Data loaders
    train_set = SELDDataset(cfg, mode='dev_train')
    test_set = SELDDataset(cfg, mode='dev_test')
    print(f'Train samples: {len(train_set)}, Test samples: {len(test_set)}')

    train_loader = DataLoader(
        train_set, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['nb_workers'], pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['nb_workers'], pin_memory=True, drop_last=False,
    )

    # Model
    model = StereoRCnet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')

    # Loss and optimizer
    criterion = SELDLoss(
        nb_classes=cfg['nb_classes'],
        alpha=cfg['loss_alpha'],
        beta=cfg['loss_beta'],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg['lr_factor'],
        patience=cfg['lr_patience'],
    )

    # Mixed precision scaler
    scaler = GradScaler('cuda')

    # Gradient accumulation: effective_batch = batch_size * grad_accum_steps
    grad_accum_steps = cfg.get('grad_accum_steps', 1)

    # Training state
    best_f_score = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    print(f'Batch size: {cfg["batch_size"]} x {grad_accum_steps} accum '
          f'= {cfg["batch_size"] * grad_accum_steps} effective')
    print(f'Mixed precision: bfloat16')
    print(f'\nStarting training for {cfg["nb_epochs"]} epochs...\n')

    for epoch in range(1, cfg['nb_epochs'] + 1):
        t_start = time.time()

        # Warmup learning rate
        warmup_lr(optimizer, epoch - 1, cfg['warmup_epochs'], cfg['learning_rate'])

        # Training
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            is_train=True, grad_accum_steps=grad_accum_steps,
        )

        # Validation loss
        val_loss = run_epoch(
            model, test_loader, criterion, None, device, scaler,
            is_train=False,
        )

        # Step scheduler after warmup
        if epoch > cfg['warmup_epochs']:
            scheduler.step(val_loss)

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t_start
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        print(f'Epoch {epoch:3d}/{cfg["nb_epochs"]} | '
              f'train: {train_loss:.4f} | val: {val_loss:.4f} | '
              f'lr: {current_lr:.2e} | {elapsed:.0f}s', end='')

        # Full SELD evaluation
        if epoch % cfg['eval_freq'] == 0 or epoch == 1:
            metrics = run_evaluation(model, test_loader, cfg, device, out_dir)
            F = metrics['F']
            AngE = metrics['AngE']
            RelDistE = metrics['RelDistE']

            writer.add_scalar('Metrics/F_20deg', F, epoch)
            writer.add_scalar('Metrics/AngE', AngE, epoch)
            writer.add_scalar('Metrics/RelDistE', RelDistE, epoch)

            print(f' | F:{100*F:.1f}% AngE:{AngE:.1f} RDE:{RelDistE:.2f}', end='')

            # Save best model based on F-score
            if F > best_f_score:
                best_f_score = F
                best_epoch = epoch
                epochs_no_improve = 0

                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f_score': best_f_score,
                    'metrics': metrics,
                }
                torch.save(state, os.path.join(ckpt_dir, 'best_model.pth'))
                print(' *BEST*', end='')
            else:
                epochs_no_improve += cfg['eval_freq']
        else:
            epochs_no_improve += 1

        print()  # newline

        # Early stopping
        if epochs_no_improve >= cfg['early_stop_patience']:
            print(f'\nEarly stopping at epoch {epoch}. '
                  f'Best F-score: {100*best_f_score:.1f}% at epoch {best_epoch}.')
            break

    # Final summary
    print(f'\nTraining complete. Best F-score: {100*best_f_score:.1f}% at epoch {best_epoch}.')
    print(f'Checkpoints: {ckpt_dir}')
    print(f'Predictions: {out_dir}')

    writer.close()
    return ckpt_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Stereo-RCnet SELD model')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--eval_freq', type=int, default=None)
    parser.add_argument('--grad_accum', type=int, default=None,
                        help='Gradient accumulation steps')
    args = parser.parse_args()

    cfg = config.copy()
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.lr is not None:
        cfg['learning_rate'] = args.lr
    if args.epochs is not None:
        cfg['nb_epochs'] = args.epochs
    if args.no_augment:
        cfg['use_augmentation'] = False
    if args.eval_freq is not None:
        cfg['eval_freq'] = args.eval_freq
    if args.grad_accum is not None:
        cfg['grad_accum_steps'] = args.grad_accum

    train(cfg)
