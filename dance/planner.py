"""Stage 2: beats + vocabulary -> which adavu, at which speed, on which beats."""
import re
import sys

import librosa
import numpy as np

from .common import BONES, HOP, Q, SR, STAGE_X, log


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


def pose_trans_distance(name1, name2, vocab):
    """Angular mismatch in degrees between end pose of name1 and start pose of name2."""
    if not name1 or not name2 or name1 not in vocab or name2 not in vocab:
        return 0.0
    cyc1 = vocab[name1].get("cycle")
    cyc2 = vocab[name2].get("cycle")
    if not cyc1 or not cyc2:
        return 0.0
    try:
        p1 = np.array([cyc1[b][-1] for b in BONES if b in cyc1], float)
        p2 = np.array([cyc2[b][0] for b in BONES if b in cyc2], float)
        if len(p1) == 0 or len(p1) != len(p2):
            return 0.0
        d = np.abs(p1 - p2)
        d = np.minimum(d, 360.0 - d)
        return float(d.mean())
    except Exception:
        return 0.0


def spatial_drift_cost(name, reps, current_x, vocab):
    """Penalize steps that push dancer past stage boundary; reward returning toward center."""
    root = vocab[name].get("root")
    if not root or not root.get("shift"):
        return 0.0
    shift_x = float(root["shift"][0]) * reps
    proj_x = current_x + shift_x
    if abs(proj_x) > 1.2:
        return 2.5 * (abs(proj_x) - 1.2)
    return -0.2 * (abs(current_x) - abs(proj_x))


def plan(vocab, first_sam, total_beats, bt, onsets, speeds, min_sec, warnings, tala, extra_features=None):
    """One adavu family per tala phrase, repeated to fill it. The next phrase changes family.

    Resting poses are not compared: every clip is stored so it ends where it starts, and that
    comparison is what made two similar stances alternate for the whole song.
    """
    extra_features = extra_features or {}
    harm_rms = extra_features.get("harm_rms")
    perc_rms = extra_features.get("perc_rms")
    sections = extra_features.get("sections") or []

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
    stage_x = 0.0
    last_name = None

    for pi, pb in enumerate(chunks):
        t0, t1 = beat_time(beat), beat_time(beat + pb)
        dens_n = (dens[pi] - dlo) / dspan
        pref = 1 if dens[pi] < 1.2 else 2 if dens[pi] < 3.0 else 4
        pos = pi / max(1, len(chunks) - 1)
        last = pi == len(chunks) - 1

        # Determine current section (Pallavi / Anupallavi / Charanam)
        curr_sec = "pallavi"
        curr_beat_abs = first_sam + beat
        for sec in sections:
            if sec["start_beat"] <= curr_beat_abs <= sec["end_beat"]:
                curr_sec = sec["label"]
                break

        # Compute harmonic vs percussive ratio for this phrase
        perc_ratio = 0.5
        if harm_rms is not None and perc_rms is not None and len(harm_rms) > 0 and len(perc_rms) > 0:
            fr0 = int(librosa.time_to_frames(t0, sr=SR, hop_length=HOP))
            fr1 = max(fr0 + 1, int(librosa.time_to_frames(t1, sr=SR, hop_length=HOP)))
            h_val = float(harm_rms[fr0:min(len(harm_rms), fr1)].mean()) if fr0 < len(harm_rms) else 0.0
            p_val = float(perc_rms[fr0:min(len(perc_rms), fr1)].mean()) if fr0 < len(perc_rms) else 0.0
            if (h_val + p_val) > 1e-6:
                perc_ratio = p_val / (h_val + p_val)

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
            candidates = []
            for j in range(len(names)):
                name = names[(start + j) % len(names)]
                opts = feasible(name)
                if not opts:
                    continue
                for k, reps, dq in opts:
                    tempo_fit = abs(np.log2(k / pref))
                    trans_fit = 0.03 * pose_trans_distance(last_name, name, vocab)
                    spat_fit = spatial_drift_cost(name, reps, stage_x, vocab)
                    total_fit = tempo_fit + trans_fit + spat_fit
                    candidates.append((total_fit, name, k, reps, dq, (start + j + 1) % len(names)))
            if candidates:
                candidates.sort(key=lambda c: c[0])
                _, b_name, b_k, b_reps, b_dq, b_nxt = candidates[0]
                return (b_name, b_k, b_reps, b_dq, b_nxt)
            return None

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
                recency = 2.5 if fam in recent[-3:] else 0.0
                arc = _ARC.get(fam, _ARC["other"])
                base = recency + 1.8 * used_n[fam] + abs(energy[fam] - dens_n) + 0.7 * abs(arc - pos)

                # Section-aware preference
                if curr_sec == "pallavi" and fam in ("tatta", "natta", "tattimettu"):
                    base -= 0.6
                elif curr_sec == "anupallavi" and fam in ("kuditta", "sarukkal", "paraval", "utplavana"):
                    base -= 0.6

                # Percussive vs harmonic stem preference
                if perc_ratio > 0.55 and fam in ("tattimettu", "kuthu", "dhalanku", "thahatha"):
                    base -= 0.4
                elif perc_ratio < 0.45 and fam in ("natta", "paraval", "sarukkal"):
                    base -= 0.4

                return base
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

        root_data = vocab[name].get("root")
        if root_data and root_data.get("shift"):
            stage_x = np.clip(stage_x + float(root_data["shift"][0]) * reps, -STAGE_X, STAGE_X)
        last_name = name

        summary.append(f"{name} x{k:g}×{reps}")

    if "teermanam" in families and recent[-1] != "teermanam":
        warnings.append("could not end on a teermanam: no speed of it fits the closing phrase")
    log("    sequence: " + " | ".join(summary))
    return steps, Tq

