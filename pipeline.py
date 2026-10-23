"""
pipeline.py - song + adavu vocabulary -> beats/tala JSON -> adavu plan -> stick-figure player.

  1. analysis: beat times and tala position of every beat (beat_this if installed, else librosa)
  2. planning: fills whole tala cycles with adavu families (one step per phrase, teermanam at the end)
  3. output:   <out>/analysis.json   beats + tala
               <out>/timeline.json   the plan (which adavu, which speed, which beats/seconds)
               <out>/index.html      stick-figure player with the song; open it in a browser (double-click)

The figure is not glued to the centre of the canvas. Bone angles shape the pose; the path across
the stage comes from the image landmarks (hip sideways, lower foot for jumps). World landmarks are
centred on the hips and are not used for travel.

Inputs: the song, and EITHER the folder of landmark JSONs written by extract_landmarks.py (the adavu vocabulary is
built from them automatically, into <out>/adavus/) OR a folder that already has vocabulary.json (from build_adavu.py).

Usage:
  uv run pipeline.py GarudaTavaTest.mp3 --adavus landmarks_json -o output --tala 8 --default-beats 4
  uv run pipeline.py song.mp3 --adavus landmarks_json --tala 6 --sam-beat 7 --clip-bpm 60

Every adavu needs a length in beats. Pass --default-beats N (same for all) or --clip-bpm (beats = period x bpm / 60,
a rough estimate: check the printed periods).

Install: uv add numpy scipy librosa soundfile scikit-learn   (+ beat_this from GitHub for good downbeats, see analyze_song.py)

Planning rules (hard): every adavu is placed whole and tiles the phrase; the piece ends on a teermanam
(if the vocabulary has one); a step may not be faster than --min-seconds. One full adavu needs about
1.25 seconds, which is one avartanam of this song at first speed. A phrase holds one adavu family
and repeats it the way a tala cycle is filled in class. The next phrase changes family. Speed follows how
busy that phrase is. The first phrase prefers tatta, then natta. Clips that share a sollukattu
(ta tei tei ta and paraval, for example) count as one family, so they cannot pretend to be variety.
"""
import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, savgol_filter

BONES = ["sp", "lua", "lfa", "rua", "rfa", "lt", "ls", "rt", "rs"]
LSH, RSH, LHP, RHP, LAN, RAN = 11, 12, 23, 24, 27, 28
Q = 4          # planning resolution: quarter beats
N = 200        # samples per adavu cycle
SR = 22050
HOP = 512
STAGE_X = 3.2


def log(*a):
    print(*a, file=sys.stderr)


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


def guess_sam_phase(y, sr, bt, tala):
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    fr = np.clip(librosa.time_to_frames(bt, sr=sr, hop_length=HOP), 0, len(env) - 1)
    vals = np.array([env[max(0, f - 4):f + 5].max() for f in fr])
    return int(np.argmax([vals[p::tala].mean() if len(vals[p::tala]) else 0.0 for p in range(tala)]))


def analyze(path, tala, backend, sam_beat):
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    bt, db, used = get_beats(path, y, sr, backend)
    if len(bt) < 8:
        sys.exit("too few beats detected")
    warnings, conf = [], None
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
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="time")
    return {
        "file": Path(path).name, "backend": used, "tala_beats": int(tala), "first_sam_index": int(first),
        "grid_confidence": None if conf is None else round(conf, 3), "bpm_median": round(med, 2),
        "duration": round(len(y) / sr, 2), "warnings": warnings,
        "beats": [{"t": round(float(t), 3), "pos": int(p), "bpm": round(float(b), 1)} for t, p, b in zip(bt, pos, bpm)],
    }, bt, onsets


def load(path):
    d = json.loads(Path(path).read_text())
    A = np.array([d["angles"][b] for b in BONES], float).T                       # (T, 9)
    F = np.array([d["foot_height_norm"]["foot_l"], d["foot_height_norm"]["foot_r"]], float).T  # (T, 2)
    return d, np.array(d["t"], float), A, F


def fill_series(series, smooth):
    """Interpolate gaps, then a short moving average. Not for angles."""
    x = np.asarray(series, float)
    good = np.isfinite(x)
    if good.sum() < 2:
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x = np.interp(idx, idx[good], x[good])
    if smooth > 1 and len(x) > smooth:
        pad = smooth // 2
        xp = np.pad(x, (pad, pad), mode="edge")
        x = np.convolve(xp, np.ones(smooth) / smooth, mode="valid")[:len(x)]
    return x


def image_root(d, smooth=5):
    """Where the dancer travels, in stick-figure units (x right, y down).

    Image landmarks are normalised to the frame, so x,y still say where the body is.
    World landmarks are centred on the hips and cannot. Horizontal travel follows the
    hip. Vertical travel follows the lower foot, so a knee bend (already in the bone
    angles) is not applied a second time; a jump, where both feet leave the floor, is.
    """
    lms = d.get("landmarks") or []
    n = len(d["angles"]["sp"])
    out = np.zeros((n, 2))
    if len(lms) != n or not d.get("width") or not d.get("height"):
        return out
    w, h = float(d["width"]), float(d["height"])
    hip_x = np.full(n, np.nan)
    foot_y = np.full(n, np.nan)
    scales = []
    for i, lm in enumerate(lms):
        if not lm or len(lm) <= RAN:
            continue

        def xy(j):
            return lm[j][0] * w, lm[j][1] * h

        def vis(j):
            return lm[j][3] if len(lm[j]) > 3 else 1.0

        if min(vis(LSH), vis(RSH), vis(LHP), vis(RHP)) < 0.5:
            continue
        hx = (xy(LHP)[0] + xy(RHP)[0]) / 2
        hy = (xy(LHP)[1] + xy(RHP)[1]) / 2
        sx = (xy(LSH)[0] + xy(RSH)[0]) / 2
        sy = (xy(LSH)[1] + xy(RSH)[1]) / 2
        torso = float(np.hypot(sx - hx, sy - hy))
        if torso < 1:
            continue
        scales.append(torso / 1.5)
        hip_x[i] = hx
        feet = [xy(j)[1] for j in (LAN, RAN) if vis(j) >= 0.5]
        if not feet:
            continue
        foot = max(feet)
        drop = foot - hy
        if 0.15 * torso <= drop <= 4.2 * torso:
            foot_y[i] = foot
    if len(scales) < 2:
        return out
    scale = float(np.median(scales)) or 1.0
    out[:, 0] = fill_series(hip_x / scale, smooth)
    out[:, 1] = fill_series(foot_y / scale, smooth)
    return out


def motion_speed(A, fps, win=7):
    s = np.sqrt(((np.gradient(A, axis=0) * fps) ** 2).sum(axis=1))
    return savgol_filter(s, win, 2) if len(s) > win else s


def active_span(s, fps, frac=0.15, pad_s=0.3):
    idx = np.where(s > frac * np.percentile(s, 90))[0]
    if len(idx) == 0:
        return 0, len(s)
    p = int(pad_s * fps)
    return max(0, idx[0] - p), min(len(s), idx[-1] + p)


def estimate_period(A, fps, min_s=1.6, max_s=6.5, max_frac=0.5):
    """Period of one adavu, via autocorrelation. Returns (seconds, confidence).

    The search stops at 6.5 seconds. A longer lag is several steps, or the whole
    practice video, and playing that back as one repetition is a blur.
    """
    X = (A - A.mean(0)) / np.maximum(A.std(0), 5.0)
    n = len(X)
    f = np.fft.rfft(X, 2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:n].sum(1)
    acf = acf / (n - np.arange(n))
    acf = acf / acf[0]
    lo = int(min_s * fps)
    hi = min(len(acf) - 1, int(n * max_frac), int(max_s * fps))
    if hi <= lo + 2:
        return None, 0.0
    peaks, _ = find_peaks(acf[lo:hi])
    peaks = peaks + lo
    if len(peaks) == 0:
        return None, 0.0
    vals = acf[peaks]
    k = peaks[np.argmax(vals >= 0.85 * vals.max())]
    if float(acf[k]) < 0.08:
        return None, float(acf[k])
    return k / fps, float(acf[k])


def framed_span(d, fps):
    """Longest stretch where the shoulders and hips stay in the frame.

    A walk into the lens, or a zoom that crops the body, is not part of the step.
    Gaps shorter than 0.4 seconds are blinks and stay inside the stretch.
    """
    lms = d.get("landmarks") or []
    n = len(d["angles"]["sp"])
    if len(lms) != n or n == 0:
        return 0, n
    ok = np.zeros(n, dtype=bool)
    for i, lm in enumerate(lms):
        if not lm or len(lm) <= RHP:
            continue
        if min(lm[LSH][3], lm[RSH][3], lm[LHP][3], lm[RHP][3]) < 0.5:
            continue
        hy = (lm[LHP][1] + lm[RHP][1]) / 2.0
        head = lm[0][1] if lm[0] else 0.4
        if head < 0.02 or head > 0.85 or hy < 0.15 or hy > 0.92:
            continue
        ok[i] = True
    gap = int(0.4 * fps)
    i = 0
    while i < n:
        if ok[i]:
            i += 1
            continue
        j = i
        while j < n and not ok[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= gap:
            ok[i:j] = True
        i = j
    best_len, b0, b1, cur, c0 = 0, 0, n, 0, 0
    for i, flag in enumerate(list(ok) + [False]):
        if flag:
            if cur == 0:
                c0 = i
            cur += 1
        else:
            if cur > best_len:
                best_len, b0, b1 = cur, c0, i
            cur = 0
    if best_len < int(1.2 * fps):
        return 0, n
    return b0, b1


def closest_rep(X, starts, Pf):
    """How far the repetitions sit from their median, and which rep is the typical one."""
    samples = [sample_cycle(X, float(s), float(Pf)) for s in starts]
    samples = [same_turn(s, samples[0]) for s in samples]
    med = np.median(np.stack(samples, 0), axis=0)
    costs = [float(np.abs(same_turn(s, med) - med).mean()) for s in samples]
    return float(np.median(costs)), int(np.argmin(costs))


def bone_travel(window):
    """Degrees each bone actually moves, after unwrapping. A spin past 180 is a lost track."""
    out = []
    for j in range(window.shape[1]):
        a = np.unwrap(np.radians(window[:, j]))
        out.append(float(np.degrees(a.max() - a.min())))
    return np.array(out)


def clean_pass(A, fps, seconds=4.0, limit=170.0, root=None):
    """A few seconds where the legs move, no limb spins, and the dancer stays put.

    Used when the clip has no repeated period. A walk to the next mark in the
    practice room is not part of the adavu, so a window that crosses the stage
    loses to one that dances in place.
    """
    n = int(seconds * fps)
    if len(A) <= n + 2:
        return 0, len(A)
    step = max(1, int(0.35 * fps))
    scored = []
    for s in range(0, len(A) - n, step):
        ranges = bone_travel(A[s:s + n])
        legs = float(ranges[5:9].max()) if ranges.size >= 9 else float(ranges.max())
        arms = float(ranges[1:5].max()) if ranges.size >= 5 else 0.0
        hip = float(np.ptp(root[s:s + n, 0])) if root is not None and len(root) >= s + n else 0.0
        spin = float(ranges.max())
        score = legs - 0.25 * max(0.0, arms - 120.0) - 40.0 * max(0.0, hip - 1.2)
        scored.append((score, spin, s))
    calm = [row for row in scored if row[1] <= limit]
    pool = calm or scored
    pool.sort(key=lambda row: (-row[0], row[1]))
    start = pool[0][2]
    return start, start + n


def fold_spins(C, limit=200.0):
    """Drop an extra full turn from a bone that cannot actually wind that far.

    A cycle ends on its first pose, so a forearm that finishes a full turn away
    from where it started has accumulated tracker drift. That drift is removed.
    A sweep that still covers more than a half-turn after that is clipped around
    its middle, which keeps the gesture and stops the windmill.
    """
    out = np.array(C, float, copy=True)
    names = []
    for j in range(out.shape[1]):
        x = np.degrees(np.unwrap(np.radians(out[:, j])))
        if float(np.ptp(x)) <= limit:
            continue
        x = x - np.linspace(0.0, x[-1] - x[0], len(x), endpoint=False)
        if float(np.ptp(x)) > limit:
            mid = float(np.median(x))
            half = limit / 2.0
            x = mid + np.clip(x - mid, -half, half)
        out[:, j] = x
        names.append(BONES[j])
    return out, names


def sample_cycle(X, start_f, Pf, n=N):
    fr = start_f + np.arange(n) / n * Pf
    return np.stack([np.interp(fr, np.arange(len(X)), X[:, j]) for j in range(X.shape[1])], 1)


def same_turn(cyc, ref):
    """Shift each angle column by whole turns so it sits next to ref.

    Two repetitions of an upright spine can be stored as -178 and +182. The
    arithmetic median of those is 2, which points the torso at the floor.
    """
    turn = np.round(np.median(cyc - ref, axis=0) / 360.0)
    return cyc - 360.0 * turn


def mean_cycle(X, s0, Pf, jitter=0.1, iters=2, circular=False):
    """Cut repetitions starting at frame s0 with period Pf (frames) and align them.

    Returns (median cycle, n_reps, start frame of each aligned rep, period actually sampled).
    circular=True averages bone angles, which wrap every 360 degrees.
    """
    J = max(1, int(jitter * Pf))
    starts = []
    k = 0
    while s0 + (k + 1) * Pf + J <= len(X) - 1:
        starts.append(s0 + k * Pf)
        k += 1

    if not starts:
        used = min(Pf, max(1e-6, len(X) - 1 - s0))
        return sample_cycle(X, s0, used), 0, np.array([float(s0)]), used
    starts = np.array(starts, float)
    samples = [sample_cycle(X, s, Pf) for s in starts]
    if circular:
        samples = [same_turn(s, samples[0]) for s in samples]
    cyc = np.median(np.stack(samples, 0), axis=0)
    used = starts
    for _ in range(iters):
        new, used_list = [], []
        for s in starts:
            best_cost, best_sh, best_samp = 1e18, 0.0, None
            for sh in range(-J, J + 1):
                if s + sh < 0 or s + sh + Pf > len(X) - 1:
                    continue
                samp = sample_cycle(X, s + sh, Pf)
                if circular:
                    samp = same_turn(samp, cyc)
                cost = float(np.abs(samp - cyc).mean())
                if cost < best_cost:
                    best_cost, best_sh, best_samp = cost, sh, samp
            used_list.append(s + best_sh)
            new.append(best_samp)
        if circular:
            new = [same_turn(s, cyc) for s in new]
        cyc = np.median(np.stack(new, 0), axis=0)
        used = np.array(used_list, float)
    return cyc, len(starts), used, Pf


def settle_angles(C):
    """Make one cycle of bone angles continuous and periodic.

    The averaged window does not always end on its first pose. Rolling that cut
    into the middle of the cycle spins a limb the long way through the join.
    """
    out = np.array(C, float, copy=True)
    n = len(out)
    if n < 5:
        return out
    for j in range(out.shape[1]):
        x = np.degrees(np.unwrap(np.radians(out[:, j])))
        x = uniform_filter1d(x, size=5, mode="nearest")
        target = x[0] + 360.0 * np.round((x[-1] - x[0]) / 360.0)
        out[:, j] = x + np.linspace(0.0, target - x[-1], n, endpoint=False)
    return out


def aligned_root(R, starts, Pf):
    """Stage path for one cycle, phase-locked to the angle repetitions.

    Each repetition is measured from its own start, so a dancer who begins on the
    left of the frame is not left there. Repetitions that travel the other way are
    flipped before the median: a right-then-left practice clip would otherwise
    cancel, and the figure would never leave the spot.
    """
    paths = []
    for s in np.atleast_1d(starts):
        p = sample_cycle(R, float(s), float(Pf))
        paths.append(p - p[0])
    if not paths:
        return np.zeros((N, 2))
    spans = [float(p[:, 0].max() - p[:, 0].min()) for p in paths]
    oriented = [p.copy() for p in paths]
    if len(paths) >= 2 and float(np.median(spans)) >= 0.3:
        signs = []
        for p in oriented:
            i = int(np.argmax(np.abs(p[:, 0])))
            signs.append(1.0 if p[i, 0] >= 0 else -1.0)
        dom = 1.0 if sum(signs) >= 0 else -1.0
        for p, sg in zip(oriented, signs):
            if sg != dom:
                p[:, 0] *= -1
    med = np.median(np.stack(oriented, 0), axis=0)
    med_span = float(med[:, 0].max() - med[:, 0].min())
    best_i = int(np.argmax(spans))
    best_span = float(spans[best_i])
    if med_span < 0.75 and best_span > max(0.45, med_span * 1.35):
        best = paths[best_i].copy()
        if med_span >= 0.3:
            i = int(np.argmax(np.abs(best[:, 0])))
            j = int(np.argmax(np.abs(med[:, 0])))
            if best[i, 0] * med[j, 0] < 0:
                best[:, 0] *= -1
        return best
    return med


def periodic_root(Rc):
    """Split a cycle path into a loop plus the net sideways shift.

    The pose cycle repeats, so the stored root samples have to meet at the ends.
    A step that finishes to the side of where it started keeps that offset as
    `shift`; the player adds it in proportion to time instead of snapping home
    on the last frame. Vertical drift is removed so the dancer lands.
    """
    Rc = np.array(Rc, float, copy=True)
    if len(Rc) > 5:
        Rc = uniform_filter1d(Rc, size=5, axis=0, mode="nearest")
    Rc -= Rc[0]
    peak = float(np.max(np.abs(Rc[:, 0]))) if len(Rc) else 0.0
    if peak > STAGE_X:
        Rc[:, 0] *= STAGE_X / peak
    if len(Rc):
        Rc[:, 1] = np.clip(Rc[:, 1], -1.2, 0.35)
    u = np.linspace(0, 1, len(Rc))
    shift_x = float(Rc[-1, 0])
    end_y = float(Rc[-1, 1])
    cyc = np.column_stack([Rc[:, 0] - u * shift_x, Rc[:, 1] - u * end_y])
    return cyc, shift_x


def circ_speed(C, P):
    """Speed (deg/s) along a circular cycle of N samples spanning P seconds."""
    d = (np.roll(C, -1, 0) - np.roll(C, 1, 0)) / 2 * (len(C) / P)
    return uniform_filter1d(np.sqrt((d ** 2).sum(1)), size=max(3, len(C) // 40), mode="wrap")


def circ_peaks(y, **kw):
    n, pad = len(y), len(y) // 4
    yy = np.concatenate([y[-pad:], y, y[:pad]])
    p, _ = find_peaks(yy, **kw)
    return sorted({int(i - pad) for i in p if pad <= i < pad + n})


def choose_start(C, Fc, P):
    """Start the cycle at a pause where both feet are down (the base stance); fall back to the stillest frame."""
    s = circ_speed(C, P)
    pauses = circ_peaks(-s, prominence=0.1 * s.max(), distance=max(2, int(0.04 * len(C))))
    if not pauses:
        return int(np.argmin(s))
    return max(pauses, key=lambda i: (round(float(min(Fc[i, 0], Fc[i, 1])), 1), -s[i]))


def find_keys(C, Fc, P, min_gap_s=0.2):
    """Key poses = pauses in the motion (always kept) plus foot strikes that fall in between."""
    n = len(C)
    gap = max(2, int(min_gap_s / P * n))
    s = circ_speed(C, P)
    far = lambda i, keys: all(min(abs(i - k), n - abs(i - k)) >= gap for k in keys)
    keys = [0]
    for i in circ_peaks(-s, prominence=0.1 * s.max(), distance=max(2, int(0.04 * n))):
        if far(i, keys):
            keys.append(i)
    for j in range(Fc.shape[1]):                      # foot strikes (foot at its lowest)
        rng = Fc[:, j].max() - Fc[:, j].min()
        for i in circ_peaks(Fc[:, j], prominence=max(0.02, 0.15 * rng)):
            if far(i, keys):
                keys.append(i)
    return sorted(keys)


def key_fit_rms(C, keys):
    u = np.arange(len(C)) / len(C)
    ku = np.array([k / len(C) for k in keys] + [1.0])
    rec = np.stack([np.interp(u, ku, np.append(C[keys, j], C[keys[0], j])) for j in range(C.shape[1])], 1)
    return float(np.sqrt(((rec - C) ** 2).mean()))


def build(path, bpm, ov, min_gap=0.2):
    d, t, A, F = load(path)
    fps = float(d["fps"])
    warns = []
    if d.get("detected_ratio", 1) < 0.8:
        warns.append(f"pose found in only {d['detected_ratio']:.0%} of frames")
    a0, a1 = active_span(motion_speed(A, fps), fps)
    f0, f1 = framed_span(d, fps)
    if f1 - f0 < a1 - a0 - int(0.5 * fps):
        warns.append(f"trimmed to the in-frame stretch {f0 / fps:.1f}s–{f1 / fps:.1f}s")
    a0, a1 = max(a0, f0), min(a1, f1)
    if a1 - a0 < int(1.2 * fps):
        a0, a1 = f0, f1
    if "start_s" in ov:
        a0 = int(ov["start_s"] * fps)
    if "end_s" in ov:
        a1 = min(len(A), int(ov["end_s"] * fps))
    Aa, Fa = A[a0:a1], F[a0:a1]
    Ra = image_root(d)[a0:a1]
    if "period_s" in ov:
        P, conf = float(ov["period_s"]), 1.0
    else:
        P, conf = estimate_period(Aa, fps)
    if P is None:
        span_s = (len(Aa) - 1) / fps
        if span_s > 5.5:
            lo, hi = clean_pass(Aa, fps, 4.0, root=Ra)
            Aa, Fa, Ra = Aa[lo:hi], Fa[lo:hi], Ra[lo:hi]
            P, conf = (len(Aa) - 1) / fps, 0.0
            warns.append("no clear repetition: using one clean 4s pass, not the whole video")
        else:
            P, conf = max(span_s, 1e-3), 0.0
            warns.append("no repetition period found: the in-frame clip is one cycle")
    elif conf < 0.5:
        warns.append(f"weak period (confidence {conf:.2f}): check it, or set period_s in --overrides")
    C, nreps, starts, Pf = mean_cycle(Aa, 0, P * fps, circular=True)
    Fc, _, _, _ = mean_cycle(Fa, 0, P * fps)
    if nreps >= 2:
        spread, bi = closest_rep(Aa, starts, Pf)
        if spread > 14:
            s = float(starts[bi])
            C = same_turn(sample_cycle(Aa, s, Pf), C)
            Fc = sample_cycle(Fa, s, Pf)
            starts = np.array([s])
            nreps = 1
            warns.append(f"repetitions do not match (spread {spread:.0f}°): keeping one pass")
        elif nreps >= 2:
            foot = np.array([float(np.ptp(sample_cycle(Fa, float(s), Pf))) for s in starts])
            legs = np.array([float(bone_travel(sample_cycle(Aa, float(s), Pf))[5:9].max()) for s in starts])
            hops = np.array([float(np.ptp(sample_cycle(Ra, float(s), Pf)[:, 1])) for s in starts])
            med_root = np.median(np.stack([sample_cycle(Ra, float(s), Pf) for s in starts], 0), axis=0)
            med_foot, med_leg = float(np.ptp(Fc)), float(bone_travel(C)[5:9].max())
            med_hop = float(np.ptp(med_root[:, 1]))

            def washed(reps, med, abs_floor, ratio):
                return float(np.median(reps)) > med + abs_floor and float(np.median(reps)) > med * ratio

            if washed(foot, med_foot, 0.05, 1.45) or washed(legs, med_leg, 4.0, 1.45) or washed(hops, med_hop, 0.05, 1.45):
                bi = int(np.argmin(np.abs(hops - np.median(hops))))
                s = float(starts[bi])
                C = same_turn(sample_cycle(Aa, s, Pf), C)
                Fc = sample_cycle(Fa, s, Pf)
                starts = np.array([s])
                nreps = 1
                warns.append("averaging the repetitions cancelled the step: keeping one pass")
    if bone_travel(C).max() > 200:
        lo, hi = clean_pass(Aa, fps, min(4.0, max(1.6, (len(Aa) - 1) / fps)), root=Ra)
        Aa, Fa, Ra = Aa[lo:hi], Fa[lo:hi], Ra[lo:hi]
        P = (len(Aa) - 1) / fps
        C, nreps, starts, Pf = mean_cycle(Aa, 0, P * fps, circular=True)
        Fc, _, _, _ = mean_cycle(Fa, 0, P * fps)
        warns.append("a limb spun in the repeated cycle: replaced it with one clean pass")
    C = settle_angles(C)
    C, spun = fold_spins(C)
    if spun:
        warns.append("folded a spun track on " + ", ".join(spun))
    Rc = aligned_root(Ra, starts, Pf)
    if nreps < 2:
        warns.append(f"only {nreps} full repetition(s): the mean cycle is not averaged")
    r = choose_start(C, Fc, P)
    C, Fc, Rc = np.roll(C, -r, 0), np.roll(Fc, -r, 0), np.roll(Rc, -r, 0)
    root_cyc, shift_x = periodic_root(Rc)
    u = np.linspace(0, 1, len(root_cyc))
    root_path = np.column_stack([root_cyc[:, 0] + u * shift_x, root_cyc[:, 1]])
    keys = find_keys(C, Fc, P, min_gap)
    rms = key_fit_rms(C, keys)
    if rms > 8:
        warns.append(f"key poses reproduce the motion only to {rms:.1f} deg RMS: more keys may be needed")
    beats = ov.get("beats")

    def pose_at(i, rx, ry):
        pose = {b: round(float(C[i, j]), 1) for j, b in enumerate(BONES)}
        pose["rootx"] = round(float(rx), 3)
        pose["rooty"] = round(float(ry), 3)
        return pose

    end_pose = pose_at(keys[0], shift_x, 0.0)
    out = {
        "name": Path(path).stem, "label": d.get("label"), "source": d.get("video"), "fps": fps,
        "period_s": round(P, 3), "period_confidence": round(conf, 3), "n_reps": int(nreps),
        "active_span_s": [round(a0 / fps, 2), round(a1 / fps, 2)],
        "cycle_beats": beats if beats is not None else (round(P * bpm / 60, 2) if bpm else None),
        "cycle_beats_is_estimate": beats is None and bool(bpm),
        "keys": [[round(i / N, 4), pose_at(i, root_path[i, 0], root_path[i, 1])] for i in keys] + [[1.0, end_pose]],
        "n_keys": len(keys), "key_fit_rms_deg": round(rms, 2),
        "cycle": {b: [round(float(v), 1) for v in C[:, j]] for j, b in enumerate(BONES)},
        "root": {
            "x": [round(float(v), 3) for v in root_cyc[:, 0]],
            "y": [round(float(v), 3) for v in root_cyc[:, 1]],
            "shift": [round(shift_x, 3), 0.0],
        },
        "root_span": [round(float(np.ptp(root_path[:, 0])), 3), round(float(np.ptp(root_path[:, 1])), 3)],
        "warnings": warns,
    }
    return out



def build_vocab_from_landmarks(lm_dir, out_dir, clip_bpm):
    files = [p for p in sorted(Path(lm_dir).glob("*.json")) if p.name != "manifest.json" and ".adavu" not in p.name]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = {}
    for p in files:
        try:
            o = build(p, clip_bpm, {})
        except Exception as e:
            log(f"    {p.name}: FAILED ({type(e).__name__}: {e})")
            continue
        (out_dir / f"{o['name']}.adavu.json").write_text(json.dumps(o, separators=(",", ":")))
        vocab[o["name"]] = {k: o[k] for k in ("label", "period_s", "cycle_beats", "cycle_beats_is_estimate", "keys", "warnings")}
        log(f"    {o['name']}: period {o['period_s']}s (confidence {o['period_confidence']}), {o['n_reps']} reps, "
            f"{o['n_keys']} keys, fit {o['key_fit_rms_deg']} deg, travel x {o['root_span'][0]:.2f} y {o['root_span'][1]:.2f}"
            + (f"  [{'; '.join(o['warnings'])}]" if o["warnings"] else ""))
    (out_dir / "vocabulary.json").write_text(json.dumps({"bones": BONES, "adavus": vocab}, indent=1))
    return out_dir / "vocabulary.json"


def resolve_adavus(path, out_dir, clip_bpm):
    """A folder with vocabulary.json, or a folder of landmark JSONs (the vocabulary is then built from them)."""
    p = Path(path)
    if p.is_file():
        return p
    if (p / "vocabulary.json").exists():
        return p / "vocabulary.json"
    if p.is_dir() and any(f.name != "manifest.json" for f in p.glob("*.json")):
        log(f"    no vocabulary.json in {p}: building it from the landmark files ...")
        return build_vocab_from_landmarks(p, Path(out_dir) / "adavus", clip_bpm)
    sys.exit(f"nothing usable in {p}: expected vocabulary.json or landmark JSONs from extract_landmarks.py")


def load_vocab(adavu_dir, default_beats):
    d = Path(adavu_dir)
    vf = d / "vocabulary.json" if d.is_dir() else d
    base = vf.parent
    raw = json.loads(vf.read_text())["adavus"]
    vocab, warns = {}, []
    for name, v in raw.items():
        cb = v.get("cycle_beats")
        if not cb:
            if default_beats:
                cb = default_beats
                warns.append(f"{name}: no cycle_beats, assuming {default_beats}")
            else:
                warns.append(f"{name}: skipped, no length in beats (pass --default-beats or --clip-bpm)")
                continue
        entry = {"cycle_beats": float(cb), "keys": v["keys"]}
        full = base / f"{name}.adavu.json"
        if full.exists():
            full_o = json.loads(full.read_text())
            entry["cycle"] = full_o.get("cycle")
            if full_o.get("root"):
                entry["root"] = full_o["root"]
        vocab[name] = entry
    return vocab, warns


_FAMILIES = [
    ("teermanam", r"theer|teerman|thirman|tirman"),
    ("tattimettu", r"tattimettu|thattimettu"),
    ("tatta", r"thattadavu|tattadavu"),
    ("natta", r"nattadavu|natta"),
    ("paraval", r"paraval|tateiteita|thatheitheitha|thathaihaitha"),
    ("kuditta", r"kudhit|kuditta|kudittu"),
    ("sarukkal", r"sarukkal|sarikal|thaiyathaihi"),
    ("kuthu", r"thathaithaha|kuthadavu|ettadavu"),
    ("sutral", r"thaithaithatham|thathaitham"),
    ("kartari", r"karthari|kartari"),
    ("utplavana", r"uthplavan|utplavan"),
    ("mandi", r"mandi"),
    ("thahatha", r"thahatha|jamthari"),
    ("korvai", r"korvai"),
    ("dhalanku", r"dhalank|dhalang"),
    ("katti", r"katti"),
]
# Where in a piece this family usually sits, 0 = opening steps, 1 = later nritta.
_ARC = {
    "tatta": 0.00, "natta": 0.08, "tattimettu": 0.18, "kuditta": 0.30, "kuthu": 0.38,
    "paraval": 0.48, "sutral": 0.56, "kartari": 0.62, "sarukkal": 0.70, "thahatha": 0.76,
    "korvai": 0.82, "katti": 0.86, "dhalanku": 0.90, "utplavana": 0.94, "mandi": 0.98,
    "other": 0.60,
}
# Floor work and long travelling steps are not danced at third speed.
# Tatta, natta, mettu and the smaller rhythmic steps are.
_MAX_SPEED = {
    "mandi": 1, "utplavana": 2, "korvai": 2, "sarukkal": 2, "thahatha": 2,
    "kartari": 2, "dhalanku": 2, "katti": 2,
}


def family_of(name):
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    for fam, pat in _FAMILIES:
        if re.search(pat, key):
            return fam
    return "other"


def adavu_motion(entry):
    """How large the joint motion is, in degrees per sample. Missing cycles count as average."""
    cyc = entry.get("cycle")
    if not cyc or "sp" not in cyc:
        return None
    A = np.array([cyc[b] for b in BONES if b in cyc], float)
    if A.size == 0 or A.shape[1] < 2:
        return None
    d = np.abs(np.diff(A, axis=1))
    d = np.minimum(d, 360.0 - d)
    return float(d.mean())


def _tiles(cycle_beats, speed, phrase_beats):
    d = float(cycle_beats) / float(speed)
    if d <= 0:
        return None
    reps = phrase_beats / d
    if reps < 1 - 1e-6 or abs(reps - round(reps)) > 0.08:
        return None
    dq = int(round(d * Q))
    if dq < 1 or abs(dq / Q - d) > 0.1:
        return None
    return int(round(reps)), dq


def _phrase_beats(tala, cycle_beats, speeds):
    """Shortest whole number of avartanams that can hold the slowest speed.

    A 4-beat step does not fit in one 6-beat cycle, but it fits in two.
    """
    slow = min(speeds)
    fallback = None
    for mult in range(1, 9):
        pb = tala * mult
        tiles = [_tiles(cycle_beats, k, pb) for k in speeds]
        if not any(tiles):
            continue
        if fallback is None:
            fallback = pb
        if _tiles(cycle_beats, slow, pb):
            return pb
    return fallback or tala


def plan(vocab, first_sam, total_beats, bt, onsets, speeds, min_sec, warnings, tala):
    """One adavu family per tala phrase, repeated to fill it. The next phrase changes family.

    Resting poses are not compared: every clip is stored so it ends where it starts, and that
    comparison is what made two similar stances alternate for the whole song.
    """
    families = {}
    for name in sorted(vocab):
        families.setdefault(family_of(name), []).append(name)
    motions = {}
    for fam, names in families.items():
        vals = [m for m in (adavu_motion(vocab[n]) for n in names) if m is not None]
        motions[fam] = float(np.mean(vals)) if vals else None
    known = [m for m in motions.values() if m is not None]
    lo, hi = (min(known), max(known)) if known else (0.0, 1.0)
    span = hi - lo or 1.0
    energy = {f: 0.5 if motions[f] is None else (motions[f] - lo) / span for f in families}

    cbs = [v["cycle_beats"] for v in vocab.values()]
    common_cb = max(set(cbs), key=cbs.count)
    phrase_len = _phrase_beats(tala, common_cb, speeds)
    body = [f for f in families if f != "teermanam"]
    if not body:
        sys.exit("no adavus to plan")

    nidx = np.arange(len(bt))
    total_q = int(round(total_beats * Q))
    Tq = np.interp(first_sam + np.arange(total_q + 1) / Q, nidx, bt)

    def beat_time(b):
        return float(np.interp(first_sam + b, nidx, bt))

    def density(b0, b1):
        t0, t1 = beat_time(b0), beat_time(b1)
        if t1 <= t0:
            return 0.0
        cnt = int(np.searchsorted(onsets, t1) - np.searchsorted(onsets, t0))
        return cnt / (b1 - b0)

    # Whole phrases, then a shorter tail that is still whole avartanams.
    chunks = []
    beat = 0
    while beat + phrase_len <= total_beats + 1e-6:
        chunks.append(phrase_len)
        beat += phrase_len
    if beat < total_beats - 1e-6:
        chunks.append(total_beats - beat)
    # Drop a tail that no clip can tile; those beats stay on the final pose.
    trimmed = 0
    while chunks:
        pb = chunks[-1]
        if any(_tiles(vocab[n]["cycle_beats"], k, pb) for n in vocab for k in speeds):
            break
        trimmed += chunks.pop()
    if trimmed:
        warnings.append(f"the beats do not divide exactly: the last {trimmed:g} beat(s) hold the final pose")
    if not chunks:
        sys.exit("no phrase of this tala fits an adavu: try other --speeds, --default-beats or --min-seconds")

    dens = []
    bcur = 0
    for pb in chunks:
        dens.append(density(bcur, bcur + pb))
        bcur += pb
    dlo, dhi = min(dens), max(dens)
    dspan = dhi - dlo or 1.0

    recent = []
    used_n = {f: 0 for f in families}
    cursor = {f: 0 for f in families}
    steps = []
    summary = []
    beat = 0
    for pi, pb in enumerate(chunks):
        t0, t1 = beat_time(beat), beat_time(beat + pb)
        dens_n = (dens[pi] - dlo) / dspan
        pref = 1 if dens[pi] < 1.2 else 2 if dens[pi] < 3.0 else 4
        pos = pi / max(1, len(chunks) - 1)
        last = pi == len(chunks) - 1

        def feasible(name):
            found = []
            for k in speeds:
                tiled = _tiles(vocab[name]["cycle_beats"], k, pb)
                if not tiled:
                    continue
                reps, dq = tiled
                if k > _MAX_SPEED.get(family_of(name), max(speeds)):
                    continue
                # The phrase average can hide one repetition crushed between two close beats.
                step = dq / Q
                shorts = [beat_time(beat + (i + 1) * step) - beat_time(beat + i * step) for i in range(reps)]
                if min(shorts) < min_sec:
                    continue
                found.append((k, reps, dq))
            return found

        def choose_clip(fam):
            names = families[fam]
            start = cursor[fam]
            best = None
            for j in range(len(names)):
                name = names[(start + j) % len(names)]
                opts = feasible(name)
                if not opts:
                    continue
                k, reps, dq = min(opts, key=lambda o: (abs(np.log2(o[0] / pref)), o[0]))
                best = (name, k, reps, dq, (start + j + 1) % len(names))
                break
            return best

        if last and "teermanam" in families:
            order = ["teermanam"]
        elif pi == 0 and "tatta" in families:
            order = ["tatta"] + [f for f in body if f != "tatta"]
        elif pi == 1 and "natta" in families and recent[-1:] != ["natta"]:
            order = ["natta"] + [f for f in body if f != "natta"]
        else:
            def cost(fam):
                if recent and fam == recent[-1]:
                    return 1e6
                # Do not return to a family until the others have been used.
                # The last few phrases also stay out of the pool, so two steps cannot trade off.
                recency = 2.5 if fam in recent[-3:] else 0.0
                arc = _ARC.get(fam, _ARC["other"])
                return recency + 1.8 * used_n[fam] + abs(energy[fam] - dens_n) + 0.7 * abs(arc - pos)
            order = sorted(body, key=cost)

        picked = None
        for fam in order:
            picked = choose_clip(fam)
            if picked:
                break
        if picked is None and recent:
            # A phrase that nothing new fits still gets the previous family, not a hole.
            picked = choose_clip(recent[-1])
        if picked is None:
            sys.exit("a tala phrase could not be filled: try other --speeds, --default-beats or --min-seconds")
        name, k, reps, dq, nxt = picked
        fam = family_of(name)
        cursor[fam] = nxt
        recent.append(fam)
        used_n[fam] += 1
        pos_q = int(round(beat * Q))
        for _ in range(reps):
            steps.append((pos_q, {"n": name, "k": k, "dq": dq}))
            pos_q += dq
        beat = pos_q / Q
        summary.append(f"{name} x{k:g}×{reps}")

    if "teermanam" in families and recent[-1] != "teermanam":
        warnings.append("could not end on a teermanam: no speed of it fits the closing phrase")
    log("    sequence: " + " | ".join(summary))
    return steps, Tq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("song")
    ap.add_argument("--adavus", required=True, help="folder of landmark JSONs (extract_landmarks.py) OR a folder with vocabulary.json")
    ap.add_argument("-o", "--out-dir", default="output")
    ap.add_argument("--tala", type=int, default=8, help="beats per cycle (adi = 8)")
    ap.add_argument("--backend", default="auto", choices=["auto", "beat_this", "librosa"])
    ap.add_argument("--sam-beat", type=int, default=None, help="index of any beat that is a sam (fix the grid)")
    ap.add_argument("--speeds", default="1,2,4", help="allowed speed factors")
    ap.add_argument("--min-seconds", type=float, default=1.25,
                    help="shortest allowed adavu repetition (a full step read at a human pace)")
    ap.add_argument("--clip-bpm", type=float, default=0, help="tempo of the adavu clips: estimates beats per adavu from its period")
    ap.add_argument("--default-beats", type=float, default=0, help="beats per cycle for adavus that have none")
    a = ap.parse_args()

    song = Path(a.song)
    if not song.exists():
        sys.exit(f"song not found: {song}")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    speeds = [float(s) for s in a.speeds.split(",")]

    log("1/3 analysing the song ...")
    ana, bt, onsets = analyze(song, a.tala, a.backend, a.sam_beat)
    (out / "analysis.json").write_text(json.dumps(ana, indent=1))
    log(f"    {len(bt)} beats, ~{ana['bpm_median']} bpm, backend {ana['backend']}, first sam = beat {ana['first_sam_index']}")

    log("2/3 planning adavus ...")
    vocab, vwarn = load_vocab(resolve_adavus(a.adavus, out, a.clip_bpm), a.default_beats)
    if not vocab:
        sys.exit("no usable adavus in the vocabulary:\n  " + "\n  ".join(vwarn) +
                 "\nGive the adavus a length in beats: --default-beats 4 (every adavu), or --clip-bpm <tempo of your clips>.")
    first = ana["first_sam_index"]
    total_beats = ((len(bt) - 1 - first) // a.tala) * a.tala
    if total_beats < a.tala:
        sys.exit("the song has less than one full cycle after the first sam")
    warnings = ana["warnings"] + vwarn
    steps, Tq = plan(vocab, first, total_beats, bt, onsets, speeds, a.min_seconds, warnings, a.tala)

    segments = []
    for pos, o in steps:
        e = pos + o["dq"]
        segments.append({"adavu": o["n"], "speed": o["k"], "start_beat": round(first + pos / Q, 3),
                         "n_beats": o["dq"] / Q, "start_time": round(float(Tq[pos]), 3), "end_time": round(float(Tq[e]), 3)})
    used = sorted({s["adavu"] for s in segments})
    timeline = {
        "audio_file": song.name, "title": song.stem, "tala_beats": a.tala, "first_sam_index": first,
        "duration": ana["duration"], "beats": [{"t": b["t"], "pos": b["pos"]} for b in ana["beats"]],
        "segments": segments, "warnings": warnings,
        "vocabulary": {n: {k: v for k, v in vocab[n].items() if k in ("keys", "cycle", "root")} for n in used},
    }
    (out / "timeline.json").write_text(json.dumps(timeline, separators=(",", ":")))
    for w in warnings:
        log("WARNING:", w)
    log(f"    {len(segments)} steps from {len(used)} adavus, covering {total_beats} beats")

    log("3/3 writing the player ...")
    if song.resolve() != (out / song.name).resolve():
        shutil.copy(song, out / song.name)
    page = PLAYER.replace("__TIMELINE__", json.dumps(timeline, separators=(",", ":")).replace("</", "<\\/"))
    page = page.replace("__TITLE__", html.escape(song.stem))
    (out / "index.html").write_text(page)
    log(f"done: open {out / 'index.html'} in a browser")


PLAYER = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ - adavu player</title>
<style>
:root{--bg:#fafaf7;--fg:#1c1c1a;--mut:#6b6b66;--line:#d8d8d0;--acc:#c2410c;--card:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecece6;--mut:#9a9a92;--line:#34342f;--acc:#fb923c;--card:#1f1f1c;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;padding:16px}
main{max-width:720px;margin:0 auto}
h1{font-size:18px;margin:0 0 8px}
canvas{width:100%;display:block;background:var(--card);border:1px solid var(--line);border-radius:10px}
#fig{height:auto;aspect-ratio:3/2}
#bar{height:44px;margin-top:8px;cursor:pointer}
audio{width:100%;margin:10px 0}
.row{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:8px 0;color:var(--mut);font-size:13px}
#now{font-size:16px;color:var(--fg)}
#cnt{font-variant-numeric:tabular-nums}
.sam{color:var(--acc);font-weight:600}
details{margin-top:10px}summary{cursor:pointer;color:var(--mut)}
#list{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}
#list button{font:12px system-ui;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:3px 7px;cursor:pointer}
.warn{color:var(--acc);font-size:13px}
</style></head><body><main>
<h1>__TITLE__</h1>
<canvas id="fig"></canvas>
<canvas id="bar"></canvas>
<audio id="au" controls preload="auto"></audio>
<div class="row"><span id="now"></span><span id="cnt"></span></div>
<div class="row">
<label><input type="checkbox" id="clk"> beat clicks (accent on the sam)</label>
<label><input type="checkbox" id="dense" checked> smooth motion (uses the recorded cycle when available)</label>
</div>
<div id="warns" class="warn"></div>
<details><summary>All steps (click to jump)</summary><div id="list"></div></details>
<script>
const T = __TIMELINE__;
const BONES = ["sp","lua","lfa","rua","rfa","lt","ls","rt","rs"];
const BASE = {sp:180,lua:-100,lfa:-100,rua:100,rfa:100,lt:-40,ls:30,rt:40,rs:-30};
const bt = T.beats.map(b => b.t), nb = bt.length, segs = T.segments, tala = T.tala_beats;
const ease = x => x*x*(3-2*x);
// Shortest step around the circle. 179 -> -179 is a 2 degree move, not a 358 degree flip.
function angDelta(a, b) {
  let d = b - a;
  d = d % 360;
  if (d > 180) d -= 360;
  if (d < -180) d += 360;
  return d;
}
const lerpPose = (a,b,w) => {
  const r = {};
  for (const k of BONES) r[k] = a[k] + angDelta(a[k], b[k])*w;
  r.rootx = (a.rootx||0) + ((b.rootx||0)-(a.rootx||0))*w;
  r.rooty = (a.rooty||0) + ((b.rooty||0)-(a.rooty||0))*w;
  return r;
};
// Place each step on the stage. A step that finishes off to one side continues from there
// and turns back at the edge. A step that travels out and comes home is mirrored the next
// time, so the dance uses both sides of the stage instead of always leaving to screen right.
function layoutStage(){
  const limit = 3.2;
  let x = 0, sign = 1;
  const reach = root => {
    if (!root || !root.x || !root.x.length) return 0;
    let m = Math.abs((root.shift && root.shift[0]) || 0);
    for (const v of root.x) m = Math.max(m, Math.abs(v));
    return m;
  };
  for (const s of segs) {
    const root = T.vocabulary[s.adavu] && T.vocabulary[s.adavu].root;
    const shift = root && root.shift ? (root.shift[0] || 0) : 0;
    let sg = sign;
    if (Math.abs(x + sg*shift) > limit && Math.abs(x - sg*shift) < Math.abs(x + sg*shift)) sg = -sg;
    s.xSign = sg;
    s.x0 = x;
    x += sg * shift;
    if (x > limit) x = limit; else if (x < -limit) x = -limit;
    if (reach(root) > 0.45) sign = -sg;
  }
}

function beatAt(t){                                   // fractional beat index at time t
  if (t <= bt[0]) return (t-bt[0])/(bt[1]-bt[0]);
  if (t >= bt[nb-1]) return nb-1 + (t-bt[nb-1])/(bt[nb-1]-bt[nb-2]);
  let lo = 0, hi = nb-1;
  while (hi-lo > 1) { const m = (lo+hi) >> 1; if (bt[m] <= t) lo = m; else hi = m; }
  return lo + (t-bt[lo])/(bt[hi]-bt[lo]);
}
let useDense = true;
function poseAtU(seg, u){
  const v = T.vocabulary[seg.adavu]; u = Math.min(Math.max(u,0),1);
  const sign = seg.xSign || 1, x0 = seg.x0 || 0;
  let r;
  if (useDense && v.cycle) {
    const N = v.cycle.sp.length, x = u*N, i = Math.floor(x) % N, j = (i+1) % N, w = x - Math.floor(x);
    r = {};
    for (const k of BONES) r[k] = v.cycle[k][i] + angDelta(v.cycle[k][i], v.cycle[k][j])*w;
    const root = v.root;
    if (root) {
      const sh = root.shift || [0, 0];
      r.rootx = root.x[i] + (root.x[j]-root.x[i])*w + u*(sh[0]||0);
      r.rooty = root.y[i] + (root.y[j]-root.y[i])*w + u*(sh[1]||0);
    }
  } else {
    const ks = v.keys;
    let a = ks[ks.length-1], b = a, w = 0;
    for (let i = 0; i < ks.length-1; i++) if (u <= ks[i+1][0]) {
      a = ks[i]; b = ks[i+1];
      w = ease((u-a[0])/Math.max(1e-6, b[0]-a[0]));
      break;
    }
    r = lerpPose(a[1], b[1], w);
  }
  r.rootx = x0 + (r.rootx || 0) * sign;
  r.rooty = r.rooty || 0;
  return r;
}
function segIndexAt(b){
  if (!segs.length || b < segs[0].start_beat) return -1;
  let lo = 0, hi = segs.length-1;
  while (lo < hi) { const m = (lo+hi+1) >> 1; if (segs[m].start_beat <= b) lo = m; else hi = m-1; }
  return lo;
}
function poseFor(b){
  const i = segIndexAt(b);
  if (i < 0) return segs.length ? poseAtU(segs[0],0) : BASE;
  const s = segs[i], u = (b - s.start_beat)/s.n_beats;
  if (u >= 1) return poseAtU(s,1);
  let p = poseAtU(s,u);
  if (i > 0) {                                        // soften the join if the previous step ended in another pose
    const bl = Math.min(0.5, 0.25*s.n_beats), d = b - s.start_beat;
    if (d < bl) p = lerpPose(poseAtU(segs[i-1],1), p, ease(d/bl));
  }
  return p;
}
function fk(p){
  const dir = a => [Math.sin(a*Math.PI/180), Math.cos(a*Math.PI/180)];
  const add = (a,v,l) => [a[0]+v[0]*l, a[1]+v[1]*l];
  const pel = [0,0], neck = add(pel, dir(p.sp), 1.5), head = add(neck, dir(p.sp), .55);
  const across = dir(p.sp - 90);                         // screen-right when the spine is upright
  const ls = add(neck, across, -.55), rs = add(neck, across, .55);
  const lel = add(ls, dir(p.lua), .95), lwr = add(lel, dir(p.lfa), .85);
  const rel = add(rs, dir(p.rua), .95), rwr = add(rel, dir(p.rfa), .85);
  const lh = add(pel, across, -.34), rh = add(pel, across, .34);
  const lk = add(lh, dir(p.lt), 1.15), la = add(lk, dir(p.ls), 1.15);
  const rk = add(rh, dir(p.rt), 1.15), ra = add(rk, dir(p.rs), 1.15);
  // Turned-out foot, folded toward the floor from the shin. side -1 is screen left.
  const foot = (ankle, shin, side) => add(ankle, dir(shin + side * 65), .42);
  const lf = foot(la, p.ls, -1), rf = foot(ra, p.rs, 1);
  return {pel,neck,head,ls,rs,lel,lwr,rel,rwr,lh,rh,lk,la,rk,ra,lf,rf};
}

// ---------------------------------------------------------------- drawing and UI
const fig = document.getElementById('fig'), bar = document.getElementById('bar'), au = document.getElementById('au');
const fx = fig.getContext('2d'), bx = bar.getContext('2d');
const dpr = window.devicePixelRatio || 1;
function fit(c){ c.width = Math.max(1, c.clientWidth*dpr); c.height = Math.max(1, c.clientHeight*dpr); }
function css(n){ return getComputedStyle(document.documentElement).getPropertyValue(n); }
function drawFigure(p){
  const W = fig.width, H = fig.height, S = H/6.4, f = fk(p);
  const ry = p.rooty || 0;
  const my = Math.max(f.la[1], f.ra[1], f.lf[1], f.rf[1]), gy = H*0.9;
  let cx = W/2 + (p.rootx || 0)*S;
  const margin = Math.max(1.3*S, 0.06*W);
  if (cx < margin) cx = margin;
  if (cx > W - margin) cx = W - margin;
  const P = q => [cx + q[0]*S, gy + (q[1]-my+ry)*S];
  fx.clearRect(0,0,W,H);
  fx.strokeStyle = css('--line'); fx.lineWidth = 2*dpr; fx.beginPath(); fx.moveTo(W*.05,gy); fx.lineTo(W*.95,gy); fx.stroke();
  fx.strokeStyle = css('--fg'); fx.lineWidth = 5*dpr; fx.lineCap = 'round'; fx.lineJoin = 'round';
  const line = pts => { fx.beginPath(); pts.forEach((q,i) => { const t = P(q); i ? fx.lineTo(t[0],t[1]) : fx.moveTo(t[0],t[1]); }); fx.stroke(); };
  line([f.pel,f.neck]); line([f.ls,f.rs]); line([f.lh,f.rh]);
  line([f.ls,f.lel,f.lwr]); line([f.rs,f.rel,f.rwr]);
  line([f.lh,f.lk,f.la,f.lf]); line([f.rh,f.rk,f.ra,f.rf]);
  const h = P(f.head); fx.beginPath(); fx.arc(h[0],h[1],.28*S,0,7); fx.stroke();
  fx.lineWidth = 3*dpr;
  for (const q of [f.ls,f.rs,f.lel,f.rel,f.lh,f.rh,f.lk,f.rk]) {
    const t = P(q); fx.beginPath(); fx.arc(t[0],t[1],.07*S,0,7); fx.stroke();
  }
}
const names = [...new Set(segs.map(s => s.adavu))];
const hue = n => (names.indexOf(n)*57) % 360;
function drawBar(t){
  const W = bar.width, H = bar.height, D = T.duration;
  bx.clearRect(0,0,W,H);
  for (const s of segs) {
    bx.fillStyle = 'hsl(' + hue(s.adavu) + ',55%,55%)';
    bx.fillRect(s.start_time/D*W, H*.15, Math.max(1,(s.end_time-s.start_time)/D*W-1), H*.7);
  }
  bx.fillStyle = css('--fg'); bx.fillRect(Math.min(au.currentTime,D)/D*W - dpr, 0, 2*dpr, H);
}
let ac = null, lastInt = -1;
function click(accent){
  if (!ac) ac = new (window.AudioContext || window.webkitAudioContext)();
  const o = ac.createOscillator(), g = ac.createGain();
  o.frequency.value = accent ? 1500 : 900; g.gain.value = .12;
  g.gain.exponentialRampToValueAtTime(.001, ac.currentTime+.06);
  o.connect(g); g.connect(ac.destination); o.start(); o.stop(ac.currentTime+.07);
}
function frame(){
  const t = au.currentTime, b = beatAt(t), bi = Math.floor(b);
  drawFigure(poseFor(b)); drawBar(t);
  const i = segIndexAt(b), s = i >= 0 ? segs[i] : null;
  document.getElementById('now').textContent = s && b < s.start_beat + s.n_beats ? s.adavu + '  x' + s.speed : '-';
  const rel = bi - T.first_sam_index, pos = ((rel % tala) + tala) % tala + 1;
  const cnt = document.getElementById('cnt');
  cnt.textContent = 'beat count ' + pos + ' / ' + tala; cnt.className = pos === 1 ? 'sam' : '';
  if (bi !== lastInt) { lastInt = bi; if (!au.paused && document.getElementById('clk').checked) click(pos === 1); }
  requestAnimationFrame(frame);
}
function init(){
  layoutStage();
  au.src = encodeURI(T.audio_file);
  fit(fig); fit(bar);
  document.getElementById('dense').onchange = e => { useDense = e.target.checked; };
  bar.onclick = e => { au.currentTime = (e.offsetX/bar.clientWidth)*T.duration; };
  const list = document.getElementById('list');
  segs.forEach(s => {
    const btn = document.createElement('button');
    btn.textContent = s.start_time.toFixed(1) + 's  ' + s.adavu + ' x' + s.speed;
    btn.style.borderLeft = '4px solid hsl(' + hue(s.adavu) + ',55%,55%)';
    btn.onclick = () => { au.currentTime = s.start_time; };
    list.appendChild(btn);
  });
  document.getElementById('warns').textContent = T.warnings.join('  |  ');
  window.__player = {beatAt, poseFor, segIndexAt, fk, drawFigure};
  requestAnimationFrame(frame);
}
init();
</script></main></body></html>
"""

if __name__ == "__main__":
    main()
