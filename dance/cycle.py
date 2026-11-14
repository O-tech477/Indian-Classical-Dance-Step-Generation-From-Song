"""One adavu clip (landmark JSON) -> one clean, periodic cycle of bone angles and stage travel."""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, savgol_filter

from .common import BONES, LAN, LHP, LSH, N, RAN, RHP, RSH, STAGE_X
from .pose import enforce_anatomical_limits


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
    C = enforce_anatomical_limits(C)
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
