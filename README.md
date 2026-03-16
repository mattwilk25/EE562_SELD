# EE562 Final Project — Stereo Sound Event Localization and Detection

DCASE 2025 Task 3 (audio-only track): detect sound events in stereo recordings, predict their azimuth direction and distance.
Dataset: [DCASE 2025 Stereo SELD](https://doi.org/10.5281/zenodo.15087603) — 30,000 five-second clips, 13 classes, 16 rooms.

---

## Packages Used

| Package | Version | Purpose |
|---|---|---|
| `torch` / `torchvision` | 2.6 | Model definition, training loop, tensor operations |
| `librosa` | ≥ 0.10 | Audio loading and log-mel spectrogram extraction |
| `numpy` | ≥ 1.24 | Array math, metric accumulation |
| `scipy` | ≥ 1.11 | Hungarian algorithm (`linear_sum_assignment`) for GT/pred track matching; jackknife statistics (`scipy.stats`) |
| `tqdm` | ≥ 4.0 | Progress bars during feature extraction and training |
| `torch.utils.tensorboard` | — | Loss/metric logging to TensorBoard |

Install with:
```bash
pip install torch librosa numpy scipy tqdm tensorboard
```

---

## Models / Classifiers

### Model 1 — Baseline CRNN (`model.py`)
A Convolutional Recurrent Neural Network following the original SELDnet architecture
([Adavanne et al., JSTSP 2018](https://arxiv.org/pdf/1807.00129.pdf)):

```
Stereo log-mel input (2 × T × 64)
  → 3 × ConvBlock (Conv2d → BN → ReLU → MaxPool → Dropout)
  → BiGRU (bidirectional, 2 layers, hidden=128)
  → 2 × Multi-Head Self-Attention (8 heads)
  → Linear head → Multi-ACCDOA output (3 tracks × 3 values × 13 classes = 117)
```

### Model 2 — ResNet-Conformer (`model_improved.py`)
Replaces plain ConvBlocks with residual blocks and the BiGRU+MHSA stack with Conformer blocks
([Gulati et al., Interspeech 2020](https://arxiv.org/abs/2005.08100)), motivated by top DCASE 2025 Task 3 submissions:

```
Stereo log-mel input (2 × T × 64)
  → 3 × ResidualBlock (Conv → BN → ReLU → Conv → BN → skip-add → MaxPool)
  → 4 × ConformerBlock (MHSA + depthwise Conv + Feed-Forward, d_model=256)
  → Linear head → Multi-ACCDOA output (117 values)
```

Residual connections improve gradient flow; Conformer blocks capture both local (convolution) and global (attention) temporal patterns in a single stage.

### Model 3 — Dual-Branch ResNet-Conformer, StereoRCnet (`StereoRCnet/`)
A dual-branch architecture that processes left and right stereo channels independently through shared-weight MCSANet encoders before fusing with Attentional Feature Fusion (AFF):

```
Left channel  →  MCSANet (ResNet + Multi-Scale Channel-Spatial Attention)  ─┐
                                                                              AFF → Conformer → output
Right channel →  MCSANet (shared weights)                                  ─┘
```

- **Dual-branch stereo processing**: left and right channels each get their own ResNet encoder with MCSA attention
- **Frequency-only pooling**: all 251 time frames preserved through the CNN; temporal compression (251→50) happens only after the Conformer
- **Attentional Feature Fusion**: learns position-wise weights to combine left/right features, capturing interaural level differences

### Output Format — Multi-ACCDOA
All models use the **Multi-ACCDOA** output representation
([Shimada et al., ICASSP 2022](https://arxiv.org/pdf/2110.07124.pdf)):
Each of the 3 output tracks predicts `(x, y, dist)` per class per frame.
Activity is encoded implicitly in the vector magnitude — a near-zero vector means the class is inactive.

### Loss — ADPIT (`loss.py`)
**Auxiliary Duplicating Permutation Invariant Training** — generates all 13 valid permutations of the 6 ground-truth dummy tracks, picks the assignment with minimum MSE, and backpropagates only through that assignment. This avoids the order ambiguity inherent in multi-source detection.

---

## File Overview

```
DCASE2025_seld_baseline/
├── main.py               # Training pipeline for the baseline CRNN
├── main_improved.py      # Training pipeline for the ResNet-Conformer model
├── model.py              # Baseline CRNN architecture (CRNNBaseline)
├── model_improved.py     # Improved ResNet-Conformer architecture (SELDModelImproved)
├── loss.py               # ADPIT loss function (ADPITLoss)
├── data_generator.py     # PyTorch Dataset — loads pre-extracted .pt feature files
├── extract_features.py   # Extracts log-mel spectrograms and ADPIT labels to disk
├── metrics.py            # Location-aware F-score, DOA error, distance error (SELDEvaluator)
├── utils.py              # Audio loading, spectrogram, label processing, CSV writing helpers
└── StereoRCnet/          # Dual-branch model (Model 3)
    ├── model.py          # StereoRCnet architecture (MCSANet + AFF + Conformer)
    ├── config.py         # All hyperparameters
    ├── train.py          # Training loop with AMP, gradient accumulation, early stopping
    ├── dataset.py        # Data loading with ACCDOA label conversion
    ├── augment.py        # SpecAugment, frequency shift, random cutout, AugMix
    ├── loss.py           # SELD loss (DOA MSE + distance MSPE)
    ├── evaluate.py       # Evaluation with optional dynamic thresholds
    └── inference.py      # Inference on evaluation set
```

### `main.py` / `main_improved.py`
Entry points. Define all hyperparameters inline as a `params` dict (no separate config file).
Run the full pipeline: feature extraction → dataset build → model training → validation every 5 epochs → save best checkpoint.
`main_improved.py` also applies **Audio Channel Swapping (ACS)** augmentation — randomly flip L/R channels and negate the sin(azimuth) label with probability 0.5.

### `model.py`
`CRNNBaseline`: convolutional encoder (3 blocks) → BiGRU → 2× MHSA → output head.
The BiGRU output is split in half and the two halves are multiplied (gated output), then passed through attention.

### `model_improved.py`
`SELDModelImproved`: ResNet encoder (3 residual blocks) → 4× Conformer → output head.
`ResidualBlock` adds a 1×1 projection shortcut when channel count changes.
`ConformerBlock` follows the standard Conformer structure: MHSA with relative positional encoding, depthwise separable conv, two half-step feed-forward modules.

### `loss.py`
`ADPITLoss`: MSE-based permutation-invariant loss for Multi-ACCDOA.
Activity is folded into the target by multiplying the one-hot activity indicator into the `(x, y, dist)` vector before computing MSE, so inactive tracks contribute zero loss regardless of predicted location.

### `data_generator.py`
`SELDDataset`: a simple `torch.utils.data.Dataset` that loads pre-saved `.pt` tensor files from disk.
The on-screen label dimension is dropped at load time (audio-only track doesn't use it).

### `extract_features.py`
`AudioFeatureExtractor`: reads raw WAV files, computes stereo log-mel spectrograms via `librosa`, and saves them as `.pt` tensors. Also reads raw CSV label files, converts them to ADPIT format tensors, and saves those. Both operations skip files that already exist on disk.

### `metrics.py`
`LocationAwareMetrics`: accumulates TP/FP/FN and localization errors frame-by-frame.
Uses the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) to optimally match predicted tracks to ground-truth tracks per class per frame.
A prediction counts as a TP only if DOA error < 20° **and** relative distance error < 1.0.
`SELDEvaluator` (`ComputeSELDResults`): loads reference CSV files once, then scores any prediction directory against them.

### `utils.py`
- `load_audio`: reads WAV via `librosa`, resamples if needed
- `extract_log_mel_spectrogram`: STFT → mel filterbank → log compression
- `load_labels`: parses DCASE-format CSV into a dict keyed by frame index
- `process_labels_adpit`: converts raw label dict into a `(T, 6, 4, C)` ADPIT tensor
- `write_logits_to_dcase_format`: decodes model output back to per-clip prediction CSVs
- `least_distance_between_gt_pred`: wraps `linear_sum_assignment` for azimuth-based GT/pred matching
- `jackknife_estimation`: leave-one-out confidence interval estimation via `scipy.stats`

---

## How to Run

```bash
# Train baseline CRNN (features extracted automatically on first run)
python main.py

# Train ResNet-Conformer (improved model)
python main_improved.py

# Train StereoRCnet (dual-branch model)
cd StereoRCnet && python train.py
```

Checkpoints are saved to `checkpoints/` and predictions to `outputs/`.
Adjust `dev_train_folds` / `dev_test_folds` in the `params` dict at the top of either main file to change the train/test split.

---

## Results (development set, fold4 test)

| Model | F₂₀° | DOA Error | Dist Error | RDE |
|---|---|---|---|---|
| Official DCASE Baseline | 22.8% | 24.5° | — | 0.41 |
| Our CRNN Baseline | 21.8% | 21.1° | 58.5 cm | 0.33 |
| Our ResNet-Conformer | **31.0%** | **18.3°** | 70.0 cm | 0.37 |
| StereoRCnet | 20.2% | 24.6° | 54.4 cm | 0.30 |

F₂₀° requires both DOA error < 20° and RDE < 1.0 to count a detection as correct.

---

## License

MIT License.
