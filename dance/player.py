"""Stage 3: the plan -> timeline.json and the stick-figure player (index.html, next to the song)."""
import html
import json
import shutil
from pathlib import Path

from .common import Q

TEMPLATE = Path(__file__).with_name("player.html")


def build_timeline(song, ana, extra_features, vocab, steps, Tq, first, tala, warnings):
    segments = []
    for pos, o in steps:
        e = pos + o["dq"]
        segments.append({"adavu": o["n"], "speed": o["k"], "start_beat": round(first + pos / Q, 3),
                         "n_beats": o["dq"] / Q, "start_time": round(float(Tq[pos]), 3), "end_time": round(float(Tq[e]), 3)})
    used = sorted({s["adavu"] for s in segments})
    return {
        "audio_file": song.name, "title": song.stem, "tala_beats": tala, "first_sam_index": first,
        "duration": ana["duration"], "beats": [{"t": b["t"], "pos": b["pos"]} for b in ana["beats"]],
        "pitch_track": extra_features.get("pitch_per_beat", []),
        "sections": extra_features.get("sections", []),
        "segments": segments, "warnings": warnings,
        "vocabulary": {n: {k: v for k, v in vocab[n].items() if k in ("keys", "cycle", "root")} for n in used},
    }


def write_player(out, song, timeline, mudhras=None):
    """Write index.html with the timeline inlined, and copy the song next to it."""
    if song.resolve() != (out / song.name).resolve():
        shutil.copy(song, out / song.name)
    mudhra_payload = "null"
    segments = (mudhras or {}).get("segments") or []
    if segments:
        mudhra_payload = json.dumps({"segments": segments}, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.read_text().replace("__TIMELINE__", json.dumps(timeline, separators=(",", ":")).replace("</", "<\\/"))
    page = page.replace("__MUDHRAS__", mudhra_payload)
    page = page.replace("__TITLE__", html.escape(song.stem))
    (out / "index.html").write_text(page)
    return out / "index.html"
