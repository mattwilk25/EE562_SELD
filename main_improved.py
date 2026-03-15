"""
main_improved.py

Training pipeline using the ResNet-Conformer model (model_improved.py).

Changes from main.py:
  1. Imports SELDModelImproved instead of SELDModel.
  2. Adds Audio Channel Swapping (ACS) augmentation in train_epoch:
       With probability 0.5, swap the stereo L/R channels and negate the
       y-component (sin-azimuth) of the multiACCDOA labels. This doubles
       effective training data and improves left/right symmetry.
       Reference: Du_NERCSLIP (Rank 1) and He_HIT (Rank 2), DCASE 2025 Task 3.
  3. Adds new ResNet-Conformer hyperparameters to params.

To run:
    cd DCASE2025_seld_baseline
    python main_improved.py
"""

import os.path
import torch
from parameters import params
from model_improved import SELDModelImproved as SELDModel
from loss import SELDLossADPIT, SELDLossSingleACCDOA
from metrics import ComputeSELDResults
from data_generator import DataGenerator
from torch.utils.data import DataLoader
from extract_features import SELDFeatureExtractor
from tqdm import tqdm
import utils
import pickle


# --------------------------------------------------------------------------- #
# New hyperparameters for ResNet-Conformer (extend the base params dict)
# --------------------------------------------------------------------------- #
params.update({
    'rc_d_model':      256,   # Conformer hidden dimension
    'rc_n_heads':      8,     # MHSA heads
    'rc_ff_dim':       512,   # Conformer feedforward inner dim
    'rc_nb_conformer': 4,     # Number of Conformer blocks
    'rc_conv_kernel':  31,    # Depthwise conv kernel size in Conformer

    'acs_prob': 0.5,          # Audio Channel Swap augmentation probability
})


# --------------------------------------------------------------------------- #
# ACS helper
# --------------------------------------------------------------------------- #
def apply_acs(audio_features, labels, params):
    """
    Audio Channel Swapping (ACS) augmentation.

    Swaps the stereo L/R channels and negates the y (sin-azimuth) component
    of the DOA labels, which is equivalent to mirroring the sound scene
    left-to-right.

    For multiACCDOA audio labels shape (B, 50, 6, 4, 13):
        dim3 ordering = [0=activity, 1=x=cos(az), 2=y=sin(az), 3=dist]
        Swapping L/R  →  azimuth negates  →  only y (sin) flips sign.

    Applied stochastically: each batch item is flipped with probability acs_prob.
    """
    if not params['multiACCDOA']:
        return audio_features, labels  # single-ACCDOA label format not handled

    B = audio_features.shape[0]
    mask = torch.rand(B, device=audio_features.device) < params['acs_prob']
    if not mask.any():
        return audio_features, labels

    audio_out  = audio_features.clone()
    labels_out = labels.clone()

    # Swap channels for selected samples
    audio_out[mask] = audio_features[mask].flip(1)  # flip channel dim (dim 1)

    # Negate y (sin-azimuth) component: labels[B, T, track, component, class]
    # component index 2 = y = sin(azimuth)
    labels_out[mask, :, :, 2, :] = -labels_out[mask, :, :, 2, :]

    return audio_out, labels_out


# --------------------------------------------------------------------------- #
# Training / validation loops
# --------------------------------------------------------------------------- #

def train_epoch(seld_model, dev_train_iterator, optimizer, seld_loss):

    seld_model.train()
    train_loss_per_epoch = 0

    for i, (input_features, labels) in enumerate(tqdm(dev_train_iterator, desc="Train", leave=False)):
        optimizer.zero_grad()
        labels = labels.to(device)

        if params['modality'] == 'audio':
            audio_features, video_features = input_features.to(device), None
        elif params['modality'] == 'audio_visual':
            audio_features, video_features = input_features[0].to(device), input_features[1].to(device)
        else:
            raise AssertionError("Modality should be one of 'audio' or 'audio_visual'.")

        # ACS augmentation (audio-only multiACCDOA)
        if params['modality'] == 'audio':
            audio_features, labels = apply_acs(audio_features, labels, params)

        logits = seld_model(audio_features, video_features)
        loss   = seld_loss(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss_per_epoch += loss.item()

    return train_loss_per_epoch / len(dev_train_iterator)


def val_epoch(seld_model, dev_test_iterator, seld_loss, seld_metrics, output_dir, is_jackknife=False):

    seld_model.eval()
    val_loss_per_epoch = 0
    with torch.no_grad():
        for j, (input_features, labels) in enumerate(tqdm(dev_test_iterator, desc="Val", leave=False)):
            labels = labels.to(device)

            if params['modality'] == 'audio':
                audio_features, video_features = input_features.to(device), None
            elif params['modality'] == 'audio_visual':
                audio_features, video_features = input_features[0].to(device), input_features[1].to(device)
            else:
                raise AssertionError("Modality should be one of 'audio' or 'audio_visual'.")

            logits = seld_model(audio_features, video_features)
            loss   = seld_loss(logits, labels)
            val_loss_per_epoch += loss.item()

            utils.write_logits_to_dcase_format(
                logits, params, output_dir,
                dev_test_iterator.dataset.label_files[j * params['batch_size']: (j + 1) * params['batch_size']]
            )

        avg_val_loss  = val_loss_per_epoch / len(dev_test_iterator)
        metric_scores = seld_metrics.get_SELD_Results(
            pred_files_path=os.path.join(output_dir, 'dev-test'),
            is_jackknife=is_jackknife
        )
        return avg_val_loss, metric_scores


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():

    if restore_from_checkpoint:
        print('Loading params from the initial checkpoint')
        with open(os.path.join(initial_checkpoint_path, 'config.pkl'), 'rb') as f:
            loaded_params = pickle.load(f)
        params.clear()
        params.update(loaded_params)

    checkpoints_folder, output_dir, summary_writer = utils.setup(params)

    feature_extractor = SELDFeatureExtractor(params)
    feature_extractor.extract_features(split='dev')
    feature_extractor.extract_labels(split='dev')

    print("Building train dataset...", flush=True)
    dev_train_dataset  = DataGenerator(params=params, mode='dev_train')
    dev_train_iterator = DataLoader(dev_train_dataset, batch_size=params['batch_size'],
                                    num_workers=params['nb_workers'], shuffle=params['shuffle'],
                                    drop_last=True)
    print(f"  Train: {len(dev_train_dataset)} clips, {len(dev_train_iterator)} batches", flush=True)

    print("Building test dataset...", flush=True)
    dev_test_dataset  = DataGenerator(params=params, mode='dev_test')
    dev_test_iterator = DataLoader(dev_test_dataset, batch_size=params['batch_size'],
                                   num_workers=params['nb_workers'], shuffle=False, drop_last=False)
    print(f"  Test:  {len(dev_test_dataset)} clips, {len(dev_test_iterator)} batches", flush=True)

    print("Building model...", flush=True)
    seld_model = SELDModel(params=params).to(device)
    total_params = sum(p.numel() for p in seld_model.parameters())
    print(f"  Device: {device}  |  Parameters: {total_params:,}", flush=True)

    optimizer = torch.optim.Adam(seld_model.parameters(),
                                 lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])

    seld_loss = SELDLossADPIT(params=params).to(device) if params['multiACCDOA'] \
                else SELDLossSingleACCDOA(params=params).to(device)

    seld_metrics = ComputeSELDResults(
        params=params,
        ref_files_folder=os.path.join(params['root_dir'], 'metadata_dev')
    )

    start_epoch  = 0
    best_f_score = float('-inf')

    if restore_from_checkpoint:
        print('Loading model weights and optimizer state from checkpoint...')
        ckpt = torch.load(os.path.join(initial_checkpoint_path, 'best_model.pth'),
                          map_location=device, weights_only=False)
        seld_model.load_state_dict(ckpt['seld_model'])
        optimizer.load_state_dict(ckpt['opt'])
        start_epoch  = ckpt['epoch'] + 1
        best_f_score = ckpt['best_f_score']

    VAL_INTERVAL = 5  # run full validation every N epochs

    for epoch in range(start_epoch, params['nb_epochs']):
        avg_train_loss = train_epoch(seld_model, dev_train_iterator, optimizer, seld_loss)

        if (epoch + 1) % VAL_INTERVAL == 0 or epoch == params['nb_epochs'] - 1:
            avg_val_loss, metric_scores = val_epoch(seld_model, dev_test_iterator,
                                                    seld_loss, seld_metrics, output_dir)
            val_f, val_ang_error, val_dist_error, val_rel_dist_error, val_onscreen_acc, class_wise_scr = metric_scores
            print(
                f"Epoch {epoch + 1}/{params['nb_epochs']} | "
                f"Train Loss: {avg_train_loss:.2f} | "
                f"Val Loss: {avg_val_loss:.2f} | "
                f"F-score: {val_f * 100:.2f} | "
                f"Ang Err: {val_ang_error:.2f} | "
                f"Dist Err: {val_dist_error:.2f} | "
                f"Rel Dist Err: {val_rel_dist_error:.2f}" +
                (f" | On-Screen Acc: {val_onscreen_acc:.2f}" if params['modality'] == 'audio_visual' else ""),
                flush=True
            )
        else:
            val_f = float('-inf')
            print(f"Epoch {epoch + 1}/{params['nb_epochs']} | Train Loss: {avg_train_loss:.2f}", flush=True)

        if val_f > best_f_score:
            best_f_score = val_f
            net_save = {
                'seld_model': seld_model.state_dict(),
                'opt': optimizer.state_dict(),
                'epoch': epoch,
                'best_f_score': best_f_score,
                'best_ang_err': val_ang_error,
                'best_rel_dist_err': val_rel_dist_error,
            }
            if params['modality'] == 'audio_visual':
                net_save['best_onscreen_acc'] = val_onscreen_acc
            torch.save(net_save, checkpoints_folder + '/best_model.pth')

    # Final evaluation on best model
    best_ckpt = torch.load(os.path.join(checkpoints_folder, 'best_model.pth'),
                           map_location=device, weights_only=False)
    seld_model.load_state_dict(best_ckpt['seld_model'])
    test_loss, test_scores = val_epoch(seld_model, dev_test_iterator, seld_loss,
                                       seld_metrics, output_dir,
                                       is_jackknife=params['use_jackknife'])
    test_f, test_ang, test_dist, test_rel, test_onsc, class_wise = test_scores
    utils.print_results(test_f, test_ang, test_dist, test_rel, test_onsc, class_wise, params)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    restore_from_checkpoint  = False
    initial_checkpoint_path  = 'checkpoints/SELDnet_audio_multiACCDOA_20250331_152343'

    main()
