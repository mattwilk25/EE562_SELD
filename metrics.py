import os
import numpy as np
from utils import least_distance_between_gt_pred, jackknife_estimation, load_labels, organize_labels


class LocationAwareMetrics:
    def __init__(self, doa_threshold=20, dist_threshold=np.inf,
                 reldist_threshold=np.inf, req_onscreen=False,
                 nb_classes=13, average='macro'):
        self.nb_classes      = nb_classes
        self.average         = average
        self.doa_thresh      = doa_threshold
        self.dist_thresh     = dist_threshold
        self.reldist_thresh  = reldist_threshold
        self.req_onscreen    = req_onscreen
        self._reset()

    def _reset(self):
        C = self.nb_classes
        self.true_pos       = np.zeros(C)
        self.false_pos      = np.zeros(C)
        self.false_pos_loc  = np.zeros(C)
        self.false_neg      = np.zeros(C)
        self.n_ref          = np.zeros(C)
        self.sum_ang_err        = np.zeros(C)
        self.sum_dist_err       = np.zeros(C)
        self.sum_reldist_err    = np.zeros(C)
        self.sum_onscreen_corr  = np.zeros(C)
        self.loc_tp  = np.zeros(C)
        self.loc_fp  = np.zeros(C)
        self.loc_fn  = np.zeros(C)

    def accumulate(self, pred, gt):
        eps = np.finfo(float).eps
        for frame in range(len(gt)):
            for cls in range(self.nb_classes):
                n_gt   = len(gt[frame][cls])   if cls in gt[frame]   else None
                n_pred = len(pred[frame][cls]) if cls in pred[frame] else None

                if n_gt is not None:
                    self.n_ref[cls] += n_gt

                if cls in gt[frame] and cls in pred[frame]:
                    self._update_matched(frame, cls, gt, pred, eps)
                elif cls in gt[frame]:
                    self.false_neg[cls] += n_gt
                    self.loc_fn[cls]    += n_gt
                elif cls in pred[frame]:
                    self.false_pos[cls] += n_pred
                    self.loc_fp[cls]    += n_pred

    def _update_matched(self, frame, cls, gt, pred, eps):
        gt_vals   = np.array(list(gt[frame][cls].values()))
        pred_vals = np.array(list(pred[frame][cls].values()))

        gt_az,   gt_dist,   gt_on   = gt_vals[:, 0],   gt_vals[:, 1],   gt_vals[:, 2]
        pred_az, pred_dist, pred_on = pred_vals[:, 0], pred_vals[:, 1], pred_vals[:, 2]

        ang_errs, row_idx, col_idx = least_distance_between_gt_pred(gt_az, pred_az)
        dist_errs    = np.abs(gt_dist[row_idx] - pred_dist[col_idx])
        reldist_errs = dist_errs / (gt_dist[row_idx] + eps)
        on_correct   = (gt_on[row_idx] == pred_on[col_idx])

        n_pred, n_gt = len(pred_az), len(gt_az)
        K       = min(n_pred, n_gt)
        FNc     = max(0, n_gt   - n_pred)
        FPc_inf = max(0, n_pred - n_gt)

        loc_fail = np.sum(np.any([
            ang_errs     > self.doa_thresh,
            dist_errs    > self.dist_thresh,
            reldist_errs > self.reldist_thresh,
            np.logical_and(~on_correct, self.req_onscreen),
        ], axis=0))

        FPc  = FPc_inf + loc_fail
        TPct = K - loc_fail

        self.sum_ang_err[cls]       += ang_errs.sum()
        self.sum_dist_err[cls]      += dist_errs.sum()
        self.sum_reldist_err[cls]   += reldist_errs.sum()
        self.sum_onscreen_corr[cls] += on_correct.sum()

        self.true_pos[cls]      += TPct
        self.loc_tp[cls]        += K
        self.false_pos[cls]     += FPc_inf
        self.false_pos_loc[cls] += loc_fail
        self.loc_fp[cls]        += FPc_inf
        self.false_neg[cls]     += FNc
        self.loc_fn[cls]        += FNc

    def compute_scores(self):
        eps = np.finfo(float).eps
        classwise = []

        if self.average == 'micro':
            TP    = self.true_pos.sum()
            FP    = self.false_pos.sum()
            FPs   = self.false_pos_loc.sum()
            FN    = self.false_neg.sum()
            DE_TP = self.loc_tp.sum()

            F        = TP / (eps + TP + FPs + 0.5 * (FP + FN))
            ang_err  = self.sum_ang_err.sum()     / (DE_TP + eps) if DE_TP else np.nan
            dist_err = self.sum_dist_err.sum()    / (DE_TP + eps) if DE_TP else np.nan
            rel_err  = self.sum_reldist_err.sum() / (DE_TP + eps) if DE_TP else np.nan
            on_acc   = self.sum_onscreen_corr.sum() / (DE_TP + eps) if DE_TP else np.nan
        else:
            denom    = eps + self.true_pos + self.false_pos_loc + 0.5 * (self.false_pos + self.false_neg)
            F        = self.true_pos / denom
            ang_err  = np.where(self.loc_tp > 0, self.sum_ang_err     / (self.loc_tp + eps), np.nan)
            dist_err = np.where(self.loc_tp > 0, self.sum_dist_err    / (self.loc_tp + eps), np.nan)
            rel_err  = np.where(self.loc_tp > 0, self.sum_reldist_err / (self.loc_tp + eps), np.nan)
            on_acc   = np.where(self.loc_tp > 0, self.sum_onscreen_corr / (self.loc_tp + eps), np.nan)

            classwise = np.array([F, ang_err, dist_err, rel_err, on_acc])
            F        = F.mean()
            ang_err  = np.nanmean(ang_err)
            dist_err = np.nanmean(dist_err)
            rel_err  = np.nanmean(rel_err)
            on_acc   = np.nanmean(on_acc)

        return F, ang_err, dist_err, rel_err, on_acc, classwise


class SELDEvaluator:
    def __init__(self, params, ref_files_folder=None):
        gt_root = ref_files_folder or os.path.join(params['root_dir'], 'metadata_dev')

        self.doa_thresh     = params['lad_doa_thresh']
        self.dist_thresh    = params['lad_dist_thresh']
        self.reldist_thresh = params['lad_reldist_thresh']
        self.req_onscreen   = params['lad_req_onscreen']
        self.average        = params['average']
        self.nb_classes     = params['nb_classes']

        self.ref_labels = {}
        for fold in os.listdir(gt_root):
            for fname in os.listdir(os.path.join(gt_root, fold)):
                gt_dict   = load_labels(os.path.join(gt_root, fold, fname), convert_to_cartesian=False)
                max_frame = max(gt_dict) if gt_dict else 0
                self.ref_labels[fname] = (organize_labels(gt_dict, max_frame), max_frame)

    def _build_scorer(self):
        return LocationAwareMetrics(
            doa_threshold=self.doa_thresh, dist_threshold=self.dist_thresh,
            reldist_threshold=self.reldist_thresh, req_onscreen=self.req_onscreen,
            nb_classes=self.nb_classes, average=self.average,
        )

    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        pred_files   = os.listdir(pred_files_path)
        scorer       = self._build_scorer()
        cached_preds = {}

        for fname in pred_files:
            pred_dict   = load_labels(os.path.join(pred_files_path, fname), convert_to_cartesian=False)
            max_pred    = max(pred_dict) if pred_dict else 0
            max_ref     = self.ref_labels[fname][1]
            pred_labels = organize_labels(pred_dict, max(max_pred, max_ref))

            scorer.accumulate(pred_labels, self.ref_labels[fname][0])
            if is_jackknife:
                cached_preds[fname] = pred_labels

        scores = scorer.compute_scores()

        if not is_jackknife:
            return scores

        global_vals = list(scores[:5])
        if len(scores[5]):
            global_vals.extend(scores[5].reshape(-1).tolist())

        partial = []
        for leave_out in pred_files:
            s = self._build_scorer()
            for fname in pred_files:
                if fname != leave_out:
                    s.accumulate(cached_preds[fname], self.ref_labels[fname][0])
            loo = s.compute_scores()
            entry = list(loo[:5])
            if len(loo[5]):
                entry.extend(loo[5].reshape(-1).tolist())
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


ComputeSELDResults = SELDEvaluator
