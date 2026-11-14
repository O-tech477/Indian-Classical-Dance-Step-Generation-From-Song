"""The adavu vocabulary: built from landmark clips and adavu_meta.json, cached, loaded for planning."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from .common import BONES, log
from .cycle import build
from .pose import enforce_anatomical_limits, enforce_pose_limits


META_FILE = "adavu_meta.json"
NOT_LANDMARKS = {"manifest.json", META_FILE}


def landmark_files(lm_dir):
    return [p for p in sorted(Path(lm_dir).glob("*.json")) if p.name not in NOT_LANDMARKS and ".adavu" not in p.name]


def read_meta(lm_dir):
    """Hand-checked facts per clip from adavu_meta.json: beats, period_s, start_s, end_s, checked."""
    p = Path(lm_dir) / META_FILE
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, dict)}


def build_vocab_from_landmarks(lm_dir, out_dir, clip_bpm):
    files = landmark_files(lm_dir)
    meta = read_meta(lm_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = {}
    for p in files:
        m = meta.get(p.stem, {})
        ov = {k: m[k] for k in ("beats", "period_s", "start_s", "end_s") if m.get(k) is not None}
        try:
            o = build(p, clip_bpm, ov)
        except Exception as e:
            log(f"    {p.name}: FAILED ({type(e).__name__}: {e})")
            continue
        o["beats_checked"] = bool(m.get("checked")) and "beats" in ov
        (out_dir / f"{o['name']}.adavu.json").write_text(json.dumps(o, separators=(",", ":")))
        vocab[o["name"]] = {k: o[k] for k in ("label", "period_s", "cycle_beats", "cycle_beats_is_estimate",
                                              "beats_checked", "keys", "warnings")}
        log(f"    {o['name']}: period {o['period_s']}s (confidence {o['period_confidence']}), {o['n_reps']} reps, "
            f"{o['n_keys']} keys, fit {o['key_fit_rms_deg']} deg, travel x {o['root_span'][0]:.2f} y {o['root_span'][1]:.2f}"
            + (f"  [{'; '.join(o['warnings'])}]" if o["warnings"] else ""))
    (out_dir / "vocabulary.json").write_text(json.dumps({"bones": BONES, "adavus": vocab}, indent=1))
    return out_dir / "vocabulary.json"


def resolve_adavus(path, clip_bpm):
    """A folder with vocabulary.json, or a folder of landmark JSONs.

    A vocabulary built from landmarks is cached in <folder>/.vocab and rebuilt only when a
    landmark file, adavu_meta.json, the vocabulary code or --clip-bpm changes.
    """
    p = Path(path)
    if p.is_file():
        return p
    if (p / "vocabulary.json").exists():
        return p / "vocabulary.json"
    files = landmark_files(p) if p.is_dir() else []
    if not files:
        sys.exit(f"nothing usable in {p}: expected vocabulary.json or landmark JSONs from extract_landmarks.py")
    cache = p / ".vocab"
    vf, stamp = cache / "vocabulary.json", cache / "built_from.json"
    code = [Path(__file__).with_name(n) for n in ("vocabulary.py", "cycle.py", "pose.py", "common.py")]
    inputs = files + [f for f in (p / META_FILE, *code) if f.exists()]
    key = {"clip_bpm": clip_bpm, "files": {f.name: f.stat().st_mtime for f in inputs}}
    if vf.exists() and stamp.exists() and json.loads(stamp.read_text()) == key:
        return vf
    log(f"    building the adavu vocabulary from {len(files)} landmark files ...")
    tmp = Path(tempfile.mkdtemp(prefix=".vocab-", dir=p))
    try:
        build_vocab_from_landmarks(p, tmp, clip_bpm)
        (tmp / "built_from.json").write_text(json.dumps(key))
        old = None
        if cache.exists():
            old = cache.rename(p / f".vocab-old-{tmp.name}")
        tmp.rename(cache)
        if old:
            shutil.rmtree(old, ignore_errors=True)
    except OSError:
        # Another run swapped in its build first; use that one.
        shutil.rmtree(tmp, ignore_errors=True)
        if not vf.exists():
            raise
    return vf


def load_vocab(adavu_dir, default_beats):
    d = Path(adavu_dir)
    vf = d / "vocabulary.json" if d.is_dir() else d
    base = vf.parent
    raw = json.loads(vf.read_text())["adavus"]
    vocab, warns, assumed, unchecked = {}, [], [], []
    for name, v in raw.items():
        cb = v.get("cycle_beats")
        if not cb:
            if default_beats:
                cb = default_beats
                assumed.append(name)
            else:
                warns.append(f"{name}: skipped, no length in beats (pass --default-beats or --clip-bpm)")
                continue
        elif v.get("beats_checked") is False:
            unchecked.append(name)
        keys_clean = [[u, enforce_pose_limits(pose)] for u, pose in v["keys"]]
        entry = {"cycle_beats": float(cb), "keys": keys_clean}
        full = base / f"{name}.adavu.json"
        if full.exists():
            full_o = json.loads(full.read_text())
            cyc = full_o.get("cycle")
            if cyc and "sp" in cyc:
                arr = np.column_stack([cyc[b] for b in BONES])
                arr = enforce_anatomical_limits(arr)
                entry["cycle"] = {b: [round(float(x), 2) for x in arr[:, idx]] for idx, b in enumerate(BONES)}
            if full_o.get("root"):
                entry["root"] = full_o["root"]
        vocab[name] = entry
    if assumed:
        warns.append(f"{len(assumed)} adavu(s) have no length in beats, assuming {default_beats:g}: "
                     f"set \"beats\" in {META_FILE} ({', '.join(assumed)})")
    if unchecked:
        warns.append(f"{len(unchecked)} adavu length(s) in {META_FILE} are not checked yet "
                     f"(set \"checked\": true once verified)")
    return vocab, warns
