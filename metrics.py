"""
metrics.py

Location-aware SELD evaluation metrics following the DCASE 2025 Task 3
evaluation protocol.

Two classes are provided:
  - LocationAwareMetrics  : accumulates per-frame TP/FP/FN counts and
                            localization errors across a dataset, then
                            computes F-score, DOA error, distance error,
                            relative distance error, and on-screen accuracy.
  - SELDEvaluator         : loads prediction and reference CSV files from
                            disk and orchestrates the scoring pipeline.

Evaluation thresholds (audio-only defaults):
  DOA threshold:       20 degrees
  Rel-dist threshold:  1.0
  On-screen check:     disabled (audio-only track)

The Hungarian algorithm is used to match predicted tracks to ground-truth
tracks per class per frame, minimising azimuth error.
"""

import os
import warnings
import numpy as np
from utils import least_distance_between_gt_pred, jackknife_estimation, \
    load_labels, organize_labels


# --------------------------------------------------------------------------- #
# Per-frame metric accumulator
# --------------------------------------------------------------------------- #

class LocationAwareMetrics:
    """
    Incrementally accumulates TP/FP/FN statistics and localization errors
    over a set of (prediction, ground-truth) frame pairs.

    Call `accumulate(pred, gt)` for each clip, then `compute_scores()` once
    at the end to get macro-averaged F-score, angular error, distance error,
    relative distance error, and on-screen accuracy.

    Args:
        doa_threshold   : maximum azimuth error (°) to count as a correct loc.
        dist_threshold  : maximum absolute distance error to count as correct.
        reldist_threshold: maximum relative distance error to count as correct.
        req_onscreen    : if True, on-screen label must match for TP.
        nb_classes      : number of sound event classes.
        average         : 'macro' (class-wise then average) or 'micro' (global).
    """

    def __init__(self, doa_threshold=20, dist_threshold=np.inf,
                 reldist_threshold=np.inf, req_onscreen=False,
                 nb_classes=13, average='macro'):
        assert average in ('macro', 'micro')
        self.nb_classes  = nb_classes
        self.average     = average

        # Detection thresholds
        self.doa_thresh     = doa_threshold
        self.dist_thresh    = dist_threshold
        self.reldist_thresh = reldist_threshold
        self.req_onscreen   = req_onscreen

        self._reset()

    def _reset(self):
        C = self.nb_classes
        # Location-sensitive detection counters
        self.true_pos       = np.zeros(C)
        self.false_pos      = np.zeros(C)
        self.false_pos_loc  = np.zeros(C)   # FP due to bad localisation
        self.false_neg      = np.zeros(C)
        self.n_ref          = np.zeros(C)   # total reference events per class

        # Class-sensitive localization accumulators
        self.sum_ang_err        = np.zeros(C)
        self.sum_dist_err       = np.zeros(C)
        self.sum_reldist_err    = np.zeros(C)
        self.sum_onscreen_corr  = np.zeros(C)

        # TP denominator for localization metrics
        self.loc_tp     = np.zeros(C)
        self.loc_fp     = np.zeros(C)
        self.loc_fn     = np.zeros(C)

    # ------------------------------------------------------------------ #
    # Per-frame update
    # ------------------------------------------------------------------ #

    def accumulate(self, pred, gt):
        """
        Process one clip's worth of frame-level predictions vs. ground truth.

        Args:
            pred: dict  frame → {class → {track → [az, dist, onscreen]}}
            gt:   dict  frame → {class → {track → [az, dist, onscreen]}}
        """
        eps = np.finfo(float).eps

        for frame in range(len(gt)):
            for cls in range(self.nb_classes):
                n_gt   = len(gt[frame][cls])   if cls in gt[frame]   else None
                n_pred = len(pred[frame][cls]) if cls in pred[frame] else None

                if n_gt is not None:
                    self.n_ref[cls] += n_gt

                if cls in gt[frame] and cls in pred[frame]:
                    self._update_matched(frame, cls, gt, pred, eps)
                elif cls in gt[frame]:          # missed detection
                    self.false_neg[cls] += n_gt
                    self.loc_fn[cls]    += n_gt
                elif cls in pred[frame]:        # spurious detection
                    self.false_pos[cls] += n_pred
                    self.loc_fp[cls]    += n_pred
                # else: true negative — nothing to count

    def _update_matched(self, frame, cls, gt, pred, eps):
        """Handle a frame/class pair where both GT and pred are present."""
        gt_vals   = np.array(list(gt[frame][cls].values()))    # (n_gt,  3)
        pred_vals = np.array(list(pred[frame][cls].values()))  # (n_pred, 3)

        gt_az,   gt_dist,   gt_on   = gt_vals[:, 0],   gt_vals[:, 1],   gt_vals[:, 2]
        pred_az, pred_dist, pred_on = pred_vals[:, 0], pred_vals[:, 1], pred_vals[:, 2]

        # Match predicted to GT tracks via Hungarian on azimuth error
        ang_errs, row_idx, col_idx = least_distance_between_gt_pred(gt_az, pred_az)
        dist_errs    = np.abs(gt_dist[row_idx] - pred_dist[col_idx])
        reldist_errs = dist_errs / (gt_dist[row_idx] + eps)
        on_correct   = (gt_on[row_idx] == pred_on[col_idx])

        n_pred, n_gt = len(pred_az), len(gt_az)
        K   = min(n_pred, n_gt)          # matched pairs
        FNc = max(0, n_gt   - n_pred)    # unmatched GT
        FPc_inf = max(0, n_pred - n_gt)  # excess predictions

        # Localisation failures among matched pairs
        loc_fail = np.sum(np.any([
            ang_errs    > self.doa_thresh,
            dist_errs   > self.dist_thresh,
            reldist_errs > self.reldist_thresh,
            np.logical_and(~on_correct, self.req_onscreen),
        ], axis=0))

        FPc_loc = loc_fail
        FPc     = FPc_inf + FPc_loc
        TPct    = K - FPc_loc

        assert n_pred == TPct + FPc
        assert n_gt   == TPct + FPc_loc + FNc

        # Accumulate localization errors
        self.sum_ang_err[cls]       += ang_errs.sum()
        self.sum_dist_err[cls]      += dist_errs.sum()
        self.sum_reldist_err[cls]   += reldist_errs.sum()
        self.sum_onscreen_corr[cls] += on_correct.sum()

        # Update detection counters
        self.true_pos[cls]      += TPct
        self.loc_tp[cls]        += K
        self.false_pos[cls]     += FPc_inf
        self.false_pos_loc[cls] += FPc_loc
        self.loc_fp[cls]        += FPc_inf
        self.false_neg[cls]     += FNc
        self.loc_fn[cls]        += FNc

    # ------------------------------------------------------------------ #
    # Score computation
    # ------------------------------------------------------------------ #

    def compute_scores(self):
        """
        Compute final F-score and localization errors from accumulated stats.

        Returns:
            (F, ang_err, dist_err, reldist_err, onscreen_acc, classwise)
            All values are scalars (micro) or class-wise arrays averaged to
            scalars (macro).  classwise is a (5, nb_classes) array when using
            macro averaging, [] otherwise.
        """
        eps = np.finfo(float).eps
        classwise = []

        if self.average == 'micro':
            TP  = self.true_pos.sum()
            FP  = self.false_pos.sum()
            FPs = self.false_pos_loc.sum()
            FN  = self.false_neg.sum()
            DE_TP = self.loc_tp.sum()

            F        = TP / (eps + TP + FPs + 0.5 * (FP + FN))
            ang_err  = self.sum_ang_err.sum()     / (DE_TP + eps) if DE_TP else np.nan
            dist_err = self.sum_dist_err.sum()    / (DE_TP + eps) if DE_TP else np.nan
            rel_err  = self.sum_reldist_err.sum() / (DE_TP + eps) if DE_TP else np.nan
            on_acc   = self.sum_onscreen_corr.sum() / (DE_TP + eps) if DE_TP else np.nan

        else:  # macro
            denom = eps + self.true_pos + self.false_pos_loc + \
                    0.5 * (self.false_pos + self.false_neg)
            F = self.true_pos / denom

            ang_err  = np.where(self.loc_tp > 0,
                                self.sum_ang_err     / (self.loc_tp + eps), np.nan)
            dist_err = np.where(self.loc_tp > 0,
                                self.sum_dist_err    / (self.loc_tp + eps), np.nan)
            rel_err  = np.where(self.loc_tp > 0,
                                self.sum_reldist_err / (self.loc_tp + eps), np.nan)
            on_acc   = np.where(self.loc_tp > 0,
                                self.sum_onscreen_corr / (self.loc_tp + eps), np.nan)

            classwise = np.array([F, ang_err, dist_err, rel_err, on_acc])
            F        = F.mean()
            ang_err  = np.nanmean(ang_err)
            dist_err = np.nanmean(dist_err)
            rel_err  = np.nanmean(rel_err)
            on_acc   = np.nanmean(on_acc)

        return F, ang_err, dist_err, rel_err, on_acc, classwise


# --------------------------------------------------------------------------- #
# File-based evaluator
# --------------------------------------------------------------------------- #

class SELDEvaluator:
    """
    Evaluates SELD performance by reading predicted and reference CSV files.

    Reference labels are loaded once at construction time from `ref_files_folder`.
    Call `evaluate(pred_files_path)` to score a prediction directory.

    Args:
        params           : parameter dict (must contain lad_* threshold keys).
        ref_files_folder : path to the ground-truth CSV directory
                           (defaults to params['root_dir']/metadata_dev).
    """

    def __init__(self, params, ref_files_folder=None):
        gt_root = ref_files_folder or os.path.join(params['root_dir'], 'metadata_dev')

        self.doa_thresh     = params['lad_doa_thresh']
        self.dist_thresh    = params['lad_dist_thresh']
        self.reldist_thresh = params['lad_reldist_thresh']
        self.req_onscreen   = params['lad_req_onscreen']
        self.average        = params['average']
        self.nb_classes     = params['nb_classes']

        if params['modality'] == 'audio' and self.req_onscreen:
            warnings.warn(
                "lad_req_onscreen=True has no effect for the audio-only track "
                "(modality='audio'). Resetting to False."
            )
            self.req_onscreen = False

        # Pre-load all reference files
        self.ref_labels = {}
        for fold in os.listdir(gt_root):
            for fname in os.listdir(os.path.join(gt_root, fold)):
                gt_dict = load_labels(
                    os.path.join(gt_root, fold, fname),
                    convert_to_cartesian=False
                )
                max_frame = max(gt_dict) if gt_dict else 0
                self.ref_labels[fname] = (organize_labels(gt_dict, max_frame), max_frame)

    def _build_scorer(self):
        return LocationAwareMetrics(
            doa_threshold=self.doa_thresh,
            dist_threshold=self.dist_thresh,
            reldist_threshold=self.reldist_thresh,
            req_onscreen=self.req_onscreen,
            nb_classes=self.nb_classes,
            average=self.average,
        )

    def evaluate(self, pred_files_path, is_jackknife=False):
        """
        Score all prediction CSVs in `pred_files_path` against the reference
        labels loaded at construction.

        Args:
            pred_files_path : directory containing per-clip prediction CSVs.
            is_jackknife    : if True, also compute jackknife confidence intervals.

        Returns:
            (F, ang_err, dist_err, rel_err, onscreen_acc, classwise_results)
            Each scalar is a float; with jackknife=True each is a
            [estimate, conf_interval] pair.
        """
        pred_files = os.listdir(pred_files_path)
        scorer = self._build_scorer()
        cached_preds = {}

        for fname in pred_files:
            pred_dict   = load_labels(
                os.path.join(pred_files_path, fname),
                convert_to_cartesian=False
            )
            max_pred    = max(pred_dict) if pred_dict else 0
            max_ref     = self.ref_labels[fname][1]
            pred_labels = organize_labels(pred_dict, max(max_pred, max_ref))

            scorer.accumulate(pred_labels, self.ref_labels[fname][0])
            if is_jackknife:
                cached_preds[fname] = pred_labels

        scores = scorer.compute_scores()   # (F, ang, dist, rel, on, classwise)

        if not is_jackknife:
            return scores

        # ---- Jackknife confidence intervals --------------------------------
        global_vals = list(scores[:5])
        if len(scores[5]):
            global_vals.extend(scores[5].reshape(-1).tolist())

        partial = []
        for leave_out in pred_files:
            s = self._build_scorer()
            for fname in pred_files:
                if fname == leave_out:
                    continue
                s.accumulate(cached_preds[fname], self.ref_labels[fname][0])
            loo_scores = s.compute_scores()
            entry = list(loo_scores[:5])
            if len(loo_scores[5]):
                entry.extend(loo_scores[5].reshape(-1).tolist())
            partial.append(entry)
        partial = np.array(partial)

        estimates, conf_intervals = [], []
        for i, gv in enumerate(global_vals):
            est, _, _, ci = jackknife_estimation(gv, partial[:, i])
            estimates.append(est)
            conf_intervals.append(ci)

        F, ang, dist, rel, on = scores[:5]
        cw = scores[5]
        ci = np.array(conf_intervals)
        return (
            [F,   ci[0]], [ang,  ci[1]], [dist, ci[2]],
            [rel, ci[3]], [on,   ci[4]],
            [cw,  ci[5:].reshape(5, self.nb_classes, 2) if len(cw) else []],
        )

    # Keep the old method name so main.py doesn't need to change
    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        return self.evaluate(pred_files_path, is_jackknife)


# Alias used by main.py imports
ComputeSELDResults = SELDEvaluator
