import os
import time
import pickle
import warnings
import librosa
import librosa.feature
import numpy as np
import torch
from scipy import stats
from scipy.optimize import linear_sum_assignment
from torch.utils.tensorboard import SummaryWriter


# --------------------------------------------------------------------------- #
# Training setup
# --------------------------------------------------------------------------- #

def setup(params):
    name = f"{params['net_type']}{time.strftime('_%Y%m%d_%H%M%S')}"
    ckpt_dir = os.path.join(params['checkpoints_dir'], name)
    out_dir  = os.path.join(params['output_dir'], name)
    log_dir  = os.path.join(params['log_dir'], name)
    for d in (ckpt_dir, out_dir, log_dir):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(params, f)
    return ckpt_dir, out_dir, SummaryWriter(log_dir=log_dir)


# --------------------------------------------------------------------------- #
# Audio feature extraction
# --------------------------------------------------------------------------- #

def load_audio(path, sr):
    return librosa.load(path, sr=sr, mono=False)


def extract_log_mel_spectrogram(audio, sr, n_fft, hop, win, nb_mels):
    stft    = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop, win_length=win).T
    mel     = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=sr, n_mels=nb_mels)
    log_mel = librosa.power_to_db(mel).transpose((2, 0, 1))  # (2, T, F)
    return log_mel


# --------------------------------------------------------------------------- #
# Label loading and processing
# --------------------------------------------------------------------------- #

def load_labels(path, convert_to_cartesian=True):
    data = {}
    with open(path) as f:
        for line in f.readlines()[1:]:
            p = line.strip().split(',')
            frame = int(p[0])
            data.setdefault(frame, []).append(
                [int(p[1]), int(p[2]), float(p[3]), float(p[4]), int(p[5])]
            )
    if convert_to_cartesian:
        data = _polar_to_cartesian(data)
    return data


def _polar_to_cartesian(data):
    out = {}
    for frame, events in data.items():
        out[frame] = []
        for e in events:
            az = e[2] * np.pi / 180
            out[frame].append(e[:2] + [np.cos(az), np.sin(az)] + e[3:])
    return out


def convert_cartesian_to_polar(data):
    out = {}
    for frame, events in data.items():
        out[frame] = []
        for e in events:
            az = np.arctan2(e[3], e[2]) * 180 / np.pi
            out[frame].append(e[:2] + [az] + e[4:])
    return out


def _fill_adpit_slot(se, x_l, y_l, dist_l, on_l, frame, slot, ev):
    cls = ev[0]
    se[frame, slot, cls]    = 1
    x_l[frame, slot, cls]   = ev[2]
    y_l[frame, slot, cls]   = ev[3]
    dist_l[frame, slot, cls] = ev[4] / 100.0
    on_l[frame, slot, cls]  = ev[5]


def process_labels_adpit(label_data, nb_frames, nb_classes):
    se   = torch.zeros(nb_frames, 6, nb_classes)
    x_l  = torch.zeros(nb_frames, 6, nb_classes)
    y_l  = torch.zeros(nb_frames, 6, nb_classes)
    d_l  = torch.zeros(nb_frames, 6, nb_classes)
    on_l = torch.zeros(nb_frames, 6, nb_classes)

    for frame, events in label_data.items():
        if frame >= nb_frames:
            continue
        events = sorted(events, key=lambda e: e[0])
        group = []
        for i, ev in enumerate(events):
            group.append(ev)
            at_end     = (i == len(events) - 1)
            class_done = at_end or (ev[0] != events[i + 1][0])
            if class_done:
                n = len(group)
                if n == 1:
                    _fill_adpit_slot(se, x_l, y_l, d_l, on_l, frame, 0, group[0])
                elif n == 2:
                    _fill_adpit_slot(se, x_l, y_l, d_l, on_l, frame, 1, group[0])
                    _fill_adpit_slot(se, x_l, y_l, d_l, on_l, frame, 2, group[1])
                else:
                    _fill_adpit_slot(se, x_l, y_l, d_l, on_l, frame, 3, group[0])
                    _fill_adpit_slot(se, x_l, y_l, d_l, on_l, frame, 4, group[1])
                    _fill_adpit_slot(se, x_l, y_l, d_l, on_l, frame, 5, group[2])
                group = []

    return torch.stack((se, x_l, y_l, d_l, on_l), dim=2)


def organize_labels(data, max_frames, max_tracks=10):
    out    = {f: {} for f in range(max_frames)}
    tracks = set(range(max_tracks))
    for frame in range(max_frames):
        if frame not in data:
            continue
        for cls, src, az, dist, on in data[frame]:
            out[frame].setdefault(cls, {})
            if src not in out[frame][cls] and src < max_tracks:
                tid = src
            else:
                free = list(tracks - set(out[frame][cls].keys()))
                if not free:
                    warnings.warn("More sources than tracks; dropping event.")
                    tid = 0
                else:
                    tid = free[0]
            out[frame][cls][tid] = [az, dist, on]
    return out


# --------------------------------------------------------------------------- #
# Prediction decoding and output writing
# --------------------------------------------------------------------------- #

def _decode_multiaccdoa(logits, nb_cls):
    # logits: (B, T, 117) = 3 tracks × (x, y, dist) × 13 classes
    results = []
    for t in range(3):
        b = t * 3 * nb_cls
        x   = logits[:, :, b          : b +   nb_cls]
        y   = logits[:, :, b + nb_cls  : b + 2*nb_cls]
        d   = logits[:, :, b + 2*nb_cls: b + 3*nb_cls].clamp(min=0)
        sed = (x**2 + y**2).sqrt() > 0.5
        doa = logits[:, :, b: b + 2*nb_cls]
        src = torch.zeros_like(d)
        on  = torch.zeros_like(d)
        results.append((sed, src, doa, d, on))
    return results


def _similar_location(sed0, sed1, doa0, doa1, cls, thresh, nb_cls):
    if sed0 == 1 and sed1 == 1:
        x1, y1 = doa0[cls], doa0[cls + nb_cls]
        x2, y2 = doa1[cls], doa1[cls + nb_cls]
        n1 = np.sqrt(x1**2 + y1**2 + 1e-10)
        n2 = np.sqrt(x2**2 + y2**2 + 1e-10)
        dot = np.clip(x1/n1*x2/n2 + y1/n1*y2/n2, -1, 1)
        return int(np.arccos(dot) * 180 / np.pi < thresh)
    return 0


def _build_output_dict(sed0, sid0, doa0, dist0, os0,
                       sed1, sid1, doa1, dist1, os1,
                       sed2, sid2, doa2, dist2, os2,
                       thresh, nb_cls):
    out = {}
    for f in range(sed0.shape[0]):
        for c in range(sed0.shape[1]):
            s01 = _similar_location(sed0[f][c], sed1[f][c], doa0[f], doa1[f], c, thresh, nb_cls)
            s12 = _similar_location(sed1[f][c], sed2[f][c], doa1[f], doa2[f], c, thresh, nb_cls)
            s20 = _similar_location(sed2[f][c], sed0[f][c], doa2[f], doa0[f], c, thresh, nb_cls)
            total = s01 + s12 + s20
            out.setdefault(f, [])

            def _add(sed, sid, doa, dist, on):
                if sed[f][c] > 0.5:
                    out[f].append([c, sid[f][c], doa[f][c], doa[f][c + nb_cls], dist[f][c], on[f][c]])

            def _add_avg(doa_a, doa_b, dist_a, dist_b, sid, on):
                avg_doa  = (doa_a[f] + doa_b[f]) / 2
                avg_dist = (dist_a[f] + dist_b[f]) / 2
                out[f].append([c, sid[f][c], avg_doa[c], avg_doa[c + nb_cls], avg_dist[c], on[f][c]])

            if total == 0:
                _add(sed0, sid0, doa0, dist0, os0)
                _add(sed1, sid1, doa1, dist1, os1)
                _add(sed2, sid2, doa2, dist2, os2)
            elif total == 1:
                if s01:
                    _add(sed2, sid2, doa2, dist2, os2)
                    _add_avg(doa0, doa1, dist0, dist1, sid0, os0)
                elif s12:
                    _add(sed0, sid0, doa0, dist0, os0)
                    _add_avg(doa1, doa2, dist1, dist2, sid1, os1)
                else:
                    _add(sed1, sid1, doa1, dist1, os1)
                    _add_avg(doa2, doa0, dist2, dist0, sid2, os2)
            else:
                avg_doa  = (doa0[f] + doa1[f] + doa2[f]) / 3
                avg_dist = (dist0[f] + dist1[f] + dist2[f]) / 3
                out[f].append([c, sid0[f][c], avg_doa[c], avg_doa[c + nb_cls], avg_dist[c], os0[f][c]])

    return convert_cartesian_to_polar(out)


def _write_csv(output_dict, output_dir, filename, split):
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    with open(os.path.join(output_dir, split, filename), 'w') as f:
        f.write('frame,class,source,azimuth,distance,onscreen\n')
        for frame, events in output_dict.items():
            for v in events:
                f.write(f"{int(frame)},{int(v[0])},{int(v[1])},"
                        f"{round(float(v[2]))},{round(float(v[3])*100)},{int(v[4])}\n")


def write_logits_to_dcase_format(logits, params, output_dir, filelist, split='dev-test'):
    nb_cls = params['nb_classes']
    tracks = _decode_multiaccdoa(logits, nb_cls)
    for i in range(logits.shape[0]):
        t = [tuple(x[i].cpu().numpy() for x in trk) for trk in tracks]
        out = _build_output_dict(
            *t[0], *t[1], *t[2],
            params['thresh_unify'], nb_cls
        )
        _write_csv(out, output_dir, os.path.basename(filelist[i])[:-3] + '.csv', split)


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #

def fold_az_angle(az):
    az = (az + 180) % 360 - 180
    folded = az.copy()
    folded[az < -90] = -180 - az[az < -90]
    folded[az >  90] =  180 - az[az >  90]
    return folded


def least_distance_between_gt_pred(gt_az, pred_az):
    gt_len, pred_len = len(gt_az), len(pred_az)
    cost = np.zeros((gt_len, pred_len))
    if gt_len and pred_len:
        pairs = np.array([[g, p] for p in range(pred_len) for g in range(gt_len)])
        cost[pairs[:, 0], pairs[:, 1]] = np.abs(
            fold_az_angle(gt_az[pairs[:, 0]]) - fold_az_angle(pred_az[pairs[:, 1]])
        )
    rows, cols = linear_sum_assignment(cost)
    return cost[rows, cols], rows, cols


def jackknife_estimation(global_value, partial_estimates, significance_level=0.05):
    n    = len(partial_estimates)
    mean = np.mean(partial_estimates)
    bias = (n - 1) * (mean - global_value)
    std_err = np.sqrt((n - 1) * np.mean((partial_estimates - mean)**2))
    estimate = global_value - bias
    t = stats.t.ppf(1 - significance_level / 2, n - 1)
    return estimate, bias, std_err, estimate + t * np.array([-std_err, std_err])


# --------------------------------------------------------------------------- #
# Results printing
# --------------------------------------------------------------------------- #

def print_results(f, ang, dist, rel_dist, onscreen, class_wise, params):
    jk = params['use_jackknife']
    def fmt(v, ci): return f"{v[0]:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if jk else f"{v:.2f}"

    print(f"\nF-score:              {fmt(f,        f[1]        if jk else 0) if jk else f'{100*f:.1f}%'}")
    print(f"DOA error:            {fmt(ang,      ang[1]      if jk else 0) if jk else f'{ang:.1f}°'}")
    print(f"Distance error:       {fmt(dist,     dist[1]     if jk else 0) if jk else f'{dist:.2f} cm'}")
    print(f"Rel distance error:   {fmt(rel_dist, rel_dist[1] if jk else 0) if jk else f'{rel_dist:.2f}'}")

    if params['average'] == 'macro' and len(class_wise):
        cw = class_wise[0] if jk else class_wise
        print("\nClass  F-score  DOA-Err  Dist-Err  RelDist-Err")
        for c in range(params['nb_classes']):
            print(f"  {c:2d}   {cw[0][c]:.2f}    {cw[1][c]:.1f}°    {cw[2][c]:.2f}     {cw[3][c]:.2f}")
