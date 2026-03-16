import random
import torch


def spec_augment(spec, num_time_masks=2, time_mask_width=20,
                 num_freq_masks=2, freq_mask_width=8):
    """Apply SpecAugment: time and frequency masking on log-mel spectrogram.

    Args:
        spec: (C, T, F) tensor
        num_time_masks: number of time mask bands
        time_mask_width: maximum width of each time mask
        num_freq_masks: number of frequency mask bands
        freq_mask_width: maximum width of each frequency mask
    Returns:
        Augmented spectrogram (C, T, F)
    """
    spec = spec.clone()
    _, T, F = spec.shape

    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_width, T - 1))
        t0 = random.randint(0, T - t)
        spec[:, t0:t0 + t, :] = 0.0

    for _ in range(num_freq_masks):
        f = random.randint(0, min(freq_mask_width, F - 1))
        f0 = random.randint(0, F - f)
        spec[:, :, f0:f0 + f] = 0.0

    return spec


def frequency_shift(spec, max_shift=8):
    """Shift spectrogram along the frequency axis by a random amount.

    Args:
        spec: (C, T, F) tensor
        max_shift: maximum shift in frequency bins (positive or negative)
    Returns:
        Shifted spectrogram (C, T, F)
    """
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return spec

    spec = spec.clone()
    if shift > 0:
        spec[:, :, shift:] = spec[:, :, :-shift].clone()
        spec[:, :, :shift] = 0.0
    else:
        spec[:, :, :shift] = spec[:, :, -shift:].clone()
        spec[:, :, shift:] = 0.0

    return spec


def random_cutout(spec, num_cutouts=3, max_time=20, max_freq=8):
    """Apply random rectangular cutouts (erasing) to the spectrogram.

    Args:
        spec: (C, T, F) tensor
        num_cutouts: number of rectangular regions to erase
        max_time: maximum height of each cutout (time dimension)
        max_freq: maximum width of each cutout (frequency dimension)
    Returns:
        Spectrogram with cutout regions zeroed (C, T, F)
    """
    spec = spec.clone()
    _, T, F = spec.shape

    for _ in range(num_cutouts):
        h = random.randint(1, min(max_time, T))
        w = random.randint(1, min(max_freq, F))
        t0 = random.randint(0, T - h)
        f0 = random.randint(0, F - w)
        spec[:, t0:t0 + h, f0:f0 + w] = 0.0

    return spec


def augmix(spec, cfg, num_chains=3, chain_depth_range=(1, 3), severity=1.0):
    """AugMix: blend multiple augmentation chains with the original.

    Generates several independently augmented versions of the input,
    mixes them with random Dirichlet-sampled weights, then blends
    with the original using a Beta-sampled coefficient.

    Args:
        spec: (C, T, F) tensor
        cfg: config dict with augmentation parameters
        num_chains: number of parallel augmentation chains
        chain_depth_range: (min, max) depth of each chain
        severity: scales augmentation intensity
    Returns:
        Mixed spectrogram (C, T, F)
    """
    augmentation_ops = [
        lambda s: spec_augment(
            s,
            num_time_masks=cfg['spec_aug_time_masks'],
            time_mask_width=int(cfg['spec_aug_time_width'] * severity),
            num_freq_masks=cfg['spec_aug_freq_masks'],
            freq_mask_width=int(cfg['spec_aug_freq_width'] * severity),
        ),
        lambda s: frequency_shift(s, max_shift=int(cfg['freq_shift_max'] * severity)),
        lambda s: random_cutout(
            s,
            num_cutouts=cfg['cutout_num'],
            max_time=int(cfg['cutout_max_time'] * severity),
            max_freq=int(cfg['cutout_max_freq'] * severity),
        ),
    ]

    # Sample mixing weights from Dirichlet distribution
    weights = torch.tensor(
        [random.random() for _ in range(num_chains)], dtype=torch.float32
    )
    weights = weights / weights.sum()

    # Build augmented mixture
    mixed = torch.zeros_like(spec)
    for i in range(num_chains):
        chain_input = spec.clone()
        depth = random.randint(*chain_depth_range)
        for _ in range(depth):
            op = random.choice(augmentation_ops)
            chain_input = op(chain_input)
        mixed = mixed + weights[i] * chain_input

    # Blend with original
    beta = random.uniform(0.5, 1.0)
    result = beta * spec + (1.0 - beta) * mixed

    return result


def apply_augmentation(spec, cfg):
    """Apply the full augmentation pipeline with configured probability.

    Each augmentation is applied independently with probability cfg['augment_prob'].
    AugMix is applied as an alternative to individual augmentations (50/50 chance).

    Args:
        spec: (C, T, F) tensor
        cfg: configuration dictionary
    Returns:
        Augmented spectrogram (C, T, F)
    """
    p = cfg['augment_prob']

    if random.random() < 0.5:
        # Individual augmentations
        if random.random() < p:
            spec = spec_augment(
                spec,
                num_time_masks=cfg['spec_aug_time_masks'],
                time_mask_width=cfg['spec_aug_time_width'],
                num_freq_masks=cfg['spec_aug_freq_masks'],
                freq_mask_width=cfg['spec_aug_freq_width'],
            )
        if random.random() < p:
            spec = frequency_shift(spec, max_shift=cfg['freq_shift_max'])
        if random.random() < p:
            spec = random_cutout(
                spec,
                num_cutouts=cfg['cutout_num'],
                max_time=cfg['cutout_max_time'],
                max_freq=cfg['cutout_max_freq'],
            )
    else:
        # AugMix
        spec = augmix(spec, cfg)

    return spec
