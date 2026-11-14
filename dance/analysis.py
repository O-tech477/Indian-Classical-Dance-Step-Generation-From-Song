"""Stage 1: the song -> beat times, tala position, sections and pitch."""
import sys
from pathlib import Path

import librosa
import numpy as np

from .common import HOP, SR, log


def get_beats(path, y, sr, backend):
    for b in (["beat_this", "librosa"] if backend == "auto" else [backend]):
        try:
            if b == "beat_this":
                from beat_this.inference import File2Beats
                bt, db = File2Beats(device="cpu", dbn=False)(str(path))
                return np.asarray(bt, float), np.asarray(db, float), b
            _, fr = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP, units="frames")
            return librosa.frames_to_time(fr, sr=sr, hop_length=HOP), None, "librosa"
        except Exception as e:
            log(f"backend {b} failed: {type(e).__name__}: {e}")
    sys.exit("beat tracking failed")



def fill_beat_gaps(bt, factor=1.8):
    """Put back beats the tracker skipped, at the song's own tempo.

    A 30 second hole is not a slow step. Playing one adavu across it moves the body in
    slow motion. The hole is filled with the median beat so every step stays at the pace
    of the rest of the song.
    """
    bt = [float(t) for t in bt]
    if len(bt) < 4:
        return np.asarray(bt), 0
    med = float(np.median(np.diff(bt)))
    if med <= 0.05:
        return np.asarray(bt), 0
    out, added = [bt[0]], 0
    for t in bt[1:]:
        start, gap = out[-1], t - out[-1]
        if gap > factor * med:
            n = max(2, int(round(gap / med)))
            step = gap / n
            for k in range(1, n):
                out.append(start + k * step)
                added += 1
        out.append(t)
    # A beat a fraction of a second after the last one is a tracker glitch, and it makes
    # one count of the step snap. Drop it, but keep the last beat so the song still ends.
    med = float(np.median(np.diff(out))) or med
    kept = [out[0]]
    for t in out[1:-1]:
        if t - kept[-1] >= 0.55 * med:
            kept.append(t)
    if out[-1] - kept[-1] >= 0.55 * med:
        kept.append(out[-1])
    return np.asarray(kept), added


def guess_sam_phase(y, sr, bt, tala):
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    fr = np.clip(librosa.time_to_frames(bt, sr=sr, hop_length=HOP), 0, len(env) - 1)
    vals = np.array([env[max(0, f - 4):f + 5].max() for f in fr])
    return int(np.argmax([vals[p::tala].mean() if len(vals[p::tala]) else 0.0 for p in range(tala)]))


def extract_audio_features(y, sr, bt, tala, first_sam):
    """Extract harmonic/percussive stems, continuous pitch contour, and macro-sections.

    100% computed on the fly from the audio waveform with librosa/scipy (no JSON lookup needed).
    """
    # 1. Harmonic vs Percussive separation (HPSS)
    y_harm, y_perc = librosa.effects.hpss(y)
    perc_onsets = librosa.onset.onset_detect(y=y_perc, sr=sr, hop_length=HOP, units="time")
    harm_rms = librosa.feature.rms(y=y_harm, hop_length=HOP)[0]
    perc_rms = librosa.feature.rms(y=y_perc, hop_length=HOP)[0]

    # 2. Continuous Pitch Tracking (f0)
    try:
        f0 = librosa.yin(y_harm, fmin=65, fmax=800, sr=sr, hop_length=HOP)
        valid = np.isfinite(f0) & (f0 > 70) & (f0 < 750)
        if np.any(valid):
            med = float(np.median(f0[valid]))
            std = float(np.std(f0[valid])) or 50.0
            norm_f0 = np.where(valid, (f0 - med) / std, 0.0)
            norm_f0 = np.clip(norm_f0, -1.8, 1.8)
        else:
            norm_f0 = np.zeros(len(f0))
    except Exception as e:
        log("pitch tracking fallback:", e)
        norm_f0 = np.zeros(max(1, int(len(y) / HOP)))

    # Sample normalized pitch at each beat
    beat_frames = np.clip(librosa.time_to_frames(bt, sr=sr, hop_length=HOP), 0, len(norm_f0) - 1)
    pitch_per_beat = [round(float(norm_f0[f]), 3) for f in beat_frames]

    # 3. Macro-Sectioning (Unsupervised Pallavi / Anupallavi / Charanam)
    starts = list(range(first_sam, len(bt) - tala + 1, tala))
    sections = []
    if len(starts) >= 4:
        try:
            chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)
            feat = np.vstack([chroma, mfcc])
            cycle_feats = []
            for s in starts:
                t0 = bt[s]
                t1 = bt[s + tala] if s + tala < len(bt) else bt[-1]
                fr0 = int(librosa.time_to_frames(t0, sr=sr, hop_length=HOP))
                fr1 = max(fr0 + 1, int(librosa.time_to_frames(t1, sr=sr, hop_length=HOP)))
                cycle_feats.append(feat[:, fr0:min(feat.shape[1], fr1)].mean(axis=1))
            X_c = np.array(cycle_feats).T
            k_sec = max(2, min(4, len(starts) // 4))
            b_idx = [int(b) for b in librosa.segment.agglomerative(X_c, k=k_sec)]
            if 0 not in b_idx:
                b_idx = [0] + b_idx
            b_idx.append(len(starts))
            for si in range(len(b_idx) - 1):
                c0, c1 = b_idx[si], b_idx[si + 1]
                sb = starts[c0]
                eb = starts[c1 - 1] + tala if c1 < len(starts) else len(bt)
                label = "pallavi" if si == 0 else ("anupallavi" if si == 1 else "charanam")
                sections.append({"start_beat": sb, "end_beat": eb, "label": label})
        except Exception as e:
            log("section segmentation fallback:", e)
            sections = [{"start_beat": first_sam, "end_beat": len(bt), "label": "pallavi"}]
    else:
        sections = [{"start_beat": first_sam, "end_beat": len(bt), "label": "pallavi"}]

    return {
        "perc_onsets": perc_onsets,
        "harm_rms": harm_rms,
        "perc_rms": perc_rms,
        "norm_f0": norm_f0,
        "pitch_per_beat": pitch_per_beat,
        "sections": sections,
    }

def analyze(path, tala, backend, sam_beat):
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    bt, db, used = get_beats(path, y, sr, backend)
    if len(bt) < 8:
        sys.exit("too few beats detected")
    bt, filled = fill_beat_gaps(bt)
    warnings, conf = [], None
    if filled:
        warnings.append(f"the tracker skipped {filled} beats: filled at the song's tempo so no step is stretched")
    idx = np.arange(len(bt))
    if sam_beat is not None:
        first = sam_beat % tala
        warnings.append(f"sam set manually from beat {sam_beat}")
    elif db is not None and len(db):
        dbi = np.unique([int(np.argmin(np.abs(bt - d))) for d in db])
        first = int(dbi[0] % tala)
        conf = float(np.mean(((dbi - dbi[0]) % tala) == 0))
        if conf < 0.8:
            warnings.append(f"only {conf:.0%} of detected downbeats fit a regular {tala}-beat cycle")
    else:
        first = guess_sam_phase(y, sr, bt, tala)
        warnings.append("tracker gave no downbeats: the sam position is a weak guess, check it with the click option")
    pos = ((idx - first) % tala) + 1
    bpm = 60.0 / np.diff(bt)
    bpm = np.append(bpm, bpm[-1])
    med = float(np.median(bpm))
    drift = float(np.max(np.abs(bpm - med)) / med)
    if drift > 0.10:
        warnings.append(f"tempo drifts {drift:.0%} from the median: the plan follows the real beats, but check the clicks")

    extra_features = extract_audio_features(y, sr, bt, tala, first)
    # Prefer percussive transients for sharper rhythmic footwork, fallback to general onsets
    onsets = extra_features["perc_onsets"] if len(extra_features["perc_onsets"]) > 8 else librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="time")

    return {
        "file": Path(path).name, "backend": used, "tala_beats": int(tala), "first_sam_index": int(first),
        "grid_confidence": None if conf is None else round(conf, 3), "bpm_median": round(med, 2),
        "duration": round(len(y) / sr, 2), "warnings": warnings,
        "sections": extra_features["sections"],
        "beats": [{"t": round(float(t), 3), "pos": int(p), "bpm": round(float(b), 1)} for t, p, b in zip(bt, pos, bpm)],
    }, bt, onsets, extra_features

