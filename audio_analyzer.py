"""
analyze_song.py - audio file -> rhythm-analysis JSON for the adavu planner.

Outputs: beat times with position in the tala cycle (1 = sam), local tempo,
sections snapped to the sam, a nritta/vocal label per section, onset density
and energy per section. Optionally writes a WAV with clicks laid over the song
so you can CHECK THE BEAT GRID BY EAR (accent on the sam).

Install:
  pip install numpy librosa soundfile scikit-learn
  pip install demucs                      # stem separation (optional, needs torch)
  pip install https://github.com/CPJKU/beat_this/archive/main.zip   # beat/downbeat tracker
  Without beat_this the script falls back to librosa (beats only, sam is a weak guess).

Usage:
  python analyze_song.py song.mp3 -o song.json --tala 8 --check song_clicks.wav
  (listen to the click file; if the accent is on the wrong beat, rerun with --sam-beat N)
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

SR = 22050
HOP = 512


def log(*a):
    print(*a, file=sys.stderr)


# --------------------------------------------------------------------------- stems
def separate_stems(path, workdir):
    """Demucs two-stem split. Returns (vocals_wav, rest_wav) or (None, None)."""
    try:
        subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals",
                        "-o", str(workdir), str(path)],
                       check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log("demucs unavailable or failed, skipping stems:", e)
        err = getattr(e, "stderr", None)
        if err:
            log("--- last lines of demucs error ---")
            log("\n".join(err.strip().splitlines()[-12:]))
        return None, None
    hits = list(Path(workdir).rglob("vocals.wav"))
    if not hits:
        return None, None
    return hits[0], hits[0].parent / "no_vocals.wav"


def beats_beat_this(path):
    from beat_this.inference import File2Beats
    beats, downbeats = File2Beats(device="cpu", dbn=False)(str(path))
    return np.asarray(beats, float), np.asarray(downbeats, float)


def beats_librosa(y, sr):
    _, frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP, units="frames")
    return librosa.frames_to_time(frames, sr=sr, hop_length=HOP), None


def guess_sam_phase(y, sr, beat_times, tala):
    """Weak fallback: the beat position (mod tala) with the strongest average onset."""
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    fr = np.clip(librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP), 0, len(env) - 1)
    vals = np.array([env[max(0, f - 4):f + 5].max() for f in fr])  # tolerate small beat-time offsets
    scores = [vals[p::tala].mean() if len(vals[p::tala]) else 0.0 for p in range(tala)]
    return int(np.argmax(scores))


def build_grid(beat_times, downbeat_times, tala, sam_phase):
    """Assign cycle positions 1..tala to every beat. Returns pos, first_sam, tala, confidence."""
    n = len(beat_times)
    idx = np.arange(n)
    if downbeat_times is not None and len(downbeat_times):
        db = np.unique([int(np.argmin(np.abs(beat_times - d))) for d in downbeat_times])
        if tala is None:
            gaps = np.diff(db)
            tala = int(np.median(gaps)) if len(gaps) else 8
        pos = ((idx - db[0]) % tala) + 1
        conf = float(np.mean(((db - db[0]) % tala) == 0))
        first_sam = int(db[0] % tala)
        return pos, first_sam, tala, conf
    tala = tala or 8
    pos = ((idx - sam_phase) % tala) + 1
    return pos, int(sam_phase), tala, None


def local_tempo(beat_times, win=5):
    if len(beat_times) < 2:
        return np.array([0.0] * len(beat_times))
    bpm = 60.0 / np.diff(beat_times)
    bpm = np.append(bpm, bpm[-1])
    k = win // 2
    p = np.pad(bpm, (k, k), mode="edge")
    return np.array([np.median(p[i:i + win]) for i in range(len(bpm))])


def beat_features(y, sr, beat_times):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)
    f = np.vstack([chroma, mfcc])
    f = (f - f.mean(axis=1, keepdims=True)) / (f.std(axis=1, keepdims=True) + 1e-8)
    fr = np.clip(librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP), 0, f.shape[1] - 1)
    last = min(f.shape[1], fr[-1] + max(1, (fr[-1] - fr[-2]) if len(fr) > 1 else 1))
    edges = np.append(fr, last)
    cols = [f[:, edges[i]:max(edges[i + 1], edges[i] + 1)].mean(axis=1) for i in range(len(fr))]
    return np.array(cols).T  # (d, n_beats)


def segment_cycles(bf, starts, tala, k):
    X = np.stack([bf[:, s:s + tala].mean(axis=1) for s in starts], axis=1)  # (d, n_cycles)
    n = X.shape[1]
    if n < 2 or k < 2:
        return [0, n], X
    k = min(k, n)
    b = [int(v) for v in librosa.segment.agglomerative(X, k)]
    if b[0] != 0:
        b = [0] + b
    return b + [n], X


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def rms_curve(path, sr):
    y, _ = librosa.load(str(path), sr=sr)
    return librosa.feature.rms(y=y, hop_length=HOP)[0]


def span_mean(curve, t0, t1, sr):
    a = int(librosa.time_to_frames(t0, sr=sr, hop_length=HOP))
    b = max(a + 1, int(librosa.time_to_frames(t1, sr=sr, hop_length=HOP)))
    return float(curve[a:b].mean()) if a < len(curve) else 0.0


def write_click_check(y, sr, beat_times, pos, out):
    def clicks(times, freq, dur):
        if len(times) == 0:
            return np.zeros(len(y))
        return librosa.clicks(times=times, sr=sr, length=len(y), click_freq=freq, click_duration=dur)
    mix = 0.8 * y + 0.5 * (clicks(beat_times[pos == 1], 1500, 0.12) + clicks(beat_times[pos != 1], 900, 0.05))
    mix /= max(1.0, float(np.abs(mix).max()))
    sf.write(out, mix, sr)
    log("wrote click-check audio:", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("-o", "--out", default=None, help="output JSON (default: <audio>.json)")
    ap.add_argument("--tala", default="8", help="beats per cycle (8 = adi), or 'auto' (needs downbeats)")
    ap.add_argument("--backend", default="auto", choices=["auto", "beat_this", "librosa"])
    ap.add_argument("--sections", type=int, default=0, help="number of sections (0 = auto)")
    ap.add_argument("--no-stems", action="store_true", help="skip Demucs")
    ap.add_argument("--force-type", choices=["nritta", "vocal"], help="override the nritta/vocal label")
    ap.add_argument("--vocal-thresh", type=float, default=0.45, help="vocal_ratio above this = vocal")
    ap.add_argument("--repeat-thresh", type=float, default=0.9, help="cosine similarity to call a repeat")
    ap.add_argument("--sam-beat", type=int, default=None,
                    help="index of any beat that is a sam (fix the grid after listening to the --check file)")
    ap.add_argument("--check", default=None, help="write WAV with clicks over the song")
    a = ap.parse_args()

    tala = None if a.tala == "auto" else int(a.tala)
    path = Path(a.audio)
    out_json = a.out or str(path.with_suffix(".json"))
    warnings = []

    y, sr = librosa.load(str(path), sr=SR, mono=True)

    # stems
    vocals_rms = rest_rms = None
    y_perc = None
    if not a.no_stems:
        with tempfile.TemporaryDirectory() as td:
            v, r = separate_stems(path, td)
            if v is not None:
                vocals_rms, rest_rms = rms_curve(v, sr), rms_curve(r, sr)
                y_perc, _ = librosa.load(str(r), sr=sr)
    if y_perc is None:
        y_perc = librosa.effects.percussive(y)
        if not a.no_stems:
            warnings.append("stems unavailable: section types are 'unknown'")

    # beats
    beat_times = downbeats = None
    backend = None
    order = ["beat_this", "librosa"] if a.backend == "auto" else [a.backend]
    for b in order:
        try:
            if b == "beat_this":
                beat_times, downbeats = beats_beat_this(path)
            else:
                beat_times, downbeats = beats_librosa(y, sr)
            backend = b
            break
        except Exception as e:  # missing package, missing weights, etc.
            log(f"backend {b} failed: {type(e).__name__}: {e}")
    if beat_times is None or len(beat_times) < 4:
        sys.exit("beat tracking failed")
    log("beat backend:", backend)

    sam_phase = 0
    if a.sam_beat is not None:
        downbeats = None
        sam_phase = a.sam_beat % (tala or 8)
        warnings.append(f"sam set manually from beat {a.sam_beat}")
    elif downbeats is None or len(downbeats) == 0:
        sam_phase = guess_sam_phase(y, sr, beat_times, tala or 8)
        warnings.append("no downbeats from tracker: sam position is a weak guess, check the click file")
    pos, first_sam, tala, conf = build_grid(beat_times, downbeats, tala, sam_phase)
    if conf is not None and conf < 0.8:
        warnings.append(f"only {conf:.0%} of detected downbeats fit a regular {tala}-beat cycle")

    tempo = local_tempo(beat_times)
    med = float(np.median(tempo))
    drift = float(np.max(np.abs(tempo - med)) / med) if med else 0.0
    if drift > 0.10:
        warnings.append(f"tempo drifts {drift:.0%} from the median: do not assume a fixed BPM")

    # cycles and sections
    n_beats = len(beat_times)
    starts = list(range(first_sam, n_beats - tala + 1, tala))
    sections = []
    if len(starts) >= 1:
        bf = beat_features(y, sr, beat_times)
        k = a.sections or max(2, min(8, round(len(starts) / 4)))
        bounds, X = segment_cycles(bf, starts, tala, k)
        means = []
        ibi = float(np.median(np.diff(beat_times)))
        env = librosa.feature.rms(y=y, hop_length=HOP)[0]
        onsets = librosa.onset.onset_detect(y=y_perc, sr=sr, hop_length=HOP, units="time")
        for i in range(len(bounds) - 1):
            c0, c1 = bounds[i], bounds[i + 1]
            sb = starts[c0]
            eb = n_beats if i == len(bounds) - 2 else starts[c1]
            t0 = float(beat_times[sb])
            t1 = float(beat_times[eb]) if eb < n_beats else float(beat_times[-1] + ibi)
            m = X[:, c0:c1].mean(axis=1)
            rep = None
            for j, pm in enumerate(means):
                if cosine(m, pm) >= a.repeat_thresh and abs(sections[j]["n_beats"] - (eb - sb)) <= tala:
                    rep = j
                    break
            means.append(m)
            stype, vr = "unknown", None
            if vocals_rms is not None:
                v, r = span_mean(vocals_rms, t0, t1, sr), span_mean(rest_rms, t0, t1, sr)
                vr = v / (v + r + 1e-9)
                stype = "vocal" if vr > a.vocal_thresh else "nritta"
            if a.force_type:
                stype = a.force_type
            sections.append({
                "index": i, "start_beat": int(sb), "n_beats": int(eb - sb), "n_cycles": int(c1 - c0),
                "start_time": round(t0, 3), "end_time": round(t1, 3), "type": stype,
                "vocal_ratio": None if vr is None else round(vr, 3),
                "onset_density": round(float(((onsets >= t0) & (onsets < t1)).sum()) / max(1, eb - sb), 3),
                "energy": span_mean(env, t0, t1, sr), "repeat_of": rep,
            })
        emax = max(s["energy"] for s in sections) or 1.0
        for s in sections:
            s["energy"] = round(s["energy"] / emax, 3)
    else:
        warnings.append("fewer than one full cycle detected: no sections")

    result = {
        "file": path.name, "backend": backend, "tala_beats": int(tala),
        "first_sam_index": int(first_sam), "pickup_beats": int(first_sam),
        "grid_confidence": None if conf is None else round(conf, 3),
        "bpm_median": round(med, 2), "tempo_drift": round(drift, 3),
        "duration": round(len(y) / sr, 2), "warnings": warnings,
        "beats": [{"t": round(float(t), 3), "pos": int(p), "bpm": round(float(b), 1)}
                  for t, p, b in zip(beat_times, pos, tempo)],
        "sections": sections,
    }
    Path(out_json).write_text(json.dumps(result, indent=1))
    log("wrote", out_json)
    for w in warnings:
        log("WARNING:", w)

    if a.check:
        write_click_check(y, sr, np.asarray(beat_times), pos, a.check)


if __name__ == "__main__":
    main()
