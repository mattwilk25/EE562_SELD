import os

config = {
    # Data paths
    'root_dir': os.path.join(os.path.dirname(__file__), '..'),
    'feat_dir': os.path.join(os.path.dirname(__file__), '..', 'features'),
    'metadata_dir': os.path.join(os.path.dirname(__file__), '..', 'metadata_dev'),

    # Feature specifications (matching pre-extracted features)
    'sampling_rate': 24000,
    'nb_mels': 64,
    'nb_frames': 251,
    'nb_classes': 13,
    'label_sequence_length': 50,  # 5 seconds at 100ms resolution

    # MCSAnet backbone
    'mcsanet_channels': [64, 128, 256, 256],
    'f_pool_sizes': [4, 4, 2, 1],  # freq: 64 -> 16 -> 4 -> 2 -> 2
    'mcsa_reduction': 4,
    'dropout': 0.05,

    # Conformer encoder
    'conformer_dim': 256,
    'conformer_layers': 4,
    'conformer_heads': 4,
    'conformer_ffn_dim': 512,
    'conformer_depthwise_kernel': 31,
    'conformer_dropout': 0.15,

    # Loss function weights: L = alpha * L_doa + beta * L_dist
    'loss_alpha': 1.0,
    'loss_beta': 1.5,

    # Training
    'nb_epochs': 200,
    'batch_size': 32,
    'grad_accum_steps': 8,    # effective batch = 32 * 8 = 256
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'lr_patience': 15,
    'lr_factor': 0.5,
    'early_stop_patience': 40,
    'warmup_epochs': 5,
    'nb_workers': 8,

    # Data augmentation
    'use_augmentation': True,
    'augment_prob': 0.5,
    'spec_aug_time_masks': 2,
    'spec_aug_time_width': 20,
    'spec_aug_freq_masks': 2,
    'spec_aug_freq_width': 8,
    'freq_shift_max': 8,
    'cutout_num': 3,
    'cutout_max_time': 20,
    'cutout_max_freq': 8,

    # Data folds
    'dev_train_folds': ['fold3'],
    'dev_test_folds': ['fold4'],

    # Evaluation
    'eval_freq': 10,
    'sed_threshold': 0.5,
    'lad_doa_thresh': 20,
    'lad_dist_thresh': float('inf'),
    'lad_reldist_thresh': 1.0,
    'average': 'macro',

    # Output directories
    'checkpoints_dir': os.path.join(os.path.dirname(__file__), 'checkpoints'),
    'log_dir': os.path.join(os.path.dirname(__file__), 'logs'),
    'output_dir': os.path.join(os.path.dirname(__file__), 'outputs'),
}
