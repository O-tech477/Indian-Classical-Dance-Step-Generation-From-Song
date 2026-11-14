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
built from them once and cached in <folder>/.vocab/) OR a folder that already has vocabulary.json (from build_adavu.py).

Usage:
  uv run pipeline.py GarudaTavaTest.mp3 --adavus landmarks_json -o output --tala 8
  uv run pipeline.py song.mp3 --adavus landmarks_json --tala 6 --sam-beat 7 --clip-bpm 60

Every adavu needs a length in beats. Set "beats" per clip in <landmarks folder>/adavu_meta.json (and "checked": true
once counted). Clips without one use --default-beats (4), or --clip-bpm (beats = period x bpm / 60, a rough estimate).

Install: uv add numpy scipy librosa soundfile scikit-learn   (+ beat_this from GitHub for good downbeats, see analyze_song.py)

Planning rules (hard): every adavu is placed whole and tiles the phrase; the piece ends on a teermanam
(if the vocabulary has one); a step may not be faster than --min-seconds. One full adavu needs about
1.25 seconds, which is one avartanam of this song at first speed. A phrase holds one adavu family
and repeats it the way a tala cycle is filled in class. The next phrase changes family. Speed follows how
busy that phrase is. The first phrase prefers tatta, then natta. Clips that share a sollukattu
(ta tei tei ta and paraval, for example) count as one family, so they cannot pretend to be variety.

Code: dance/analysis.py (stage 1), dance/cycle.py + dance/vocabulary.py (the adavus), dance/planner.py (stage 2),
dance/player.py + dance/player.html (stage 3). This file is only the command line.
"""
import argparse
import json
import sys
from pathlib import Path

from dance.analysis import analyze
from dance.common import log
from dance.planner import plan
from dance.player import build_timeline, write_player
from dance.vocabulary import META_FILE, load_vocab, resolve_adavus
from modules.audio_service import MAX_SONG_SECONDS, cap_song


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
    ap.add_argument("--default-beats", type=float, default=4,
                    help=f"beats per cycle for adavus with no \"beats\" in {META_FILE} (0 = skip them)")
    a = ap.parse_args()

    song = Path(a.song)
    if not song.exists():
        sys.exit(f"song not found: {song}")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    speeds = [float(s) for s in a.speeds.split(",")]

    log("1/3 analysing the song ...")
    song = cap_song(song, out / song.name, MAX_SONG_SECONDS)
    ana, bt, onsets, extra_features = analyze(song, a.tala, a.backend, a.sam_beat)
    if ana["duration"] > MAX_SONG_SECONDS:
        ana["duration"] = MAX_SONG_SECONDS
        ana["warnings"].append("trimmed to the first 2 minutes")
    (out / "analysis.json").write_text(json.dumps(ana, indent=1))
    log(f"    {len(bt)} beats, ~{ana['bpm_median']} bpm, backend {ana['backend']}, first sam = beat {ana['first_sam_index']}")
    if ana.get("sections"):
        sec_str = " | ".join(f"{s['label']} (beats {s['start_beat']}-{s['end_beat']})" for s in ana["sections"])
        log(f"    detected sections: {sec_str}")

    log("2/3 planning adavus ...")
    vocab, vwarn = load_vocab(resolve_adavus(a.adavus, a.clip_bpm), a.default_beats)
    if not vocab:
        sys.exit("no usable adavus in the vocabulary:\n  " + "\n  ".join(vwarn) +
                 "\nGive the adavus a length in beats: --default-beats 4 (every adavu), or --clip-bpm <tempo of your clips>.")
    first = ana["first_sam_index"]
    total_beats = ((len(bt) - 1 - first) // a.tala) * a.tala
    if total_beats < a.tala:
        sys.exit("the song has less than one full cycle after the first sam")
    warnings = ana["warnings"] + vwarn
    steps, Tq = plan(vocab, first, total_beats, bt, onsets, speeds, a.min_seconds, warnings, a.tala, extra_features=extra_features)

    timeline = build_timeline(song, ana, extra_features, vocab, steps, Tq, first, a.tala, warnings)
    (out / "timeline.json").write_text(json.dumps(timeline, separators=(",", ":")))
    for w in warnings:
        log("WARNING:", w)
    used = {s["adavu"] for s in timeline["segments"]}
    log(f"    {len(timeline['segments'])} steps from {len(used)} adavus, covering {total_beats} beats")

    log("3/3 writing the player ...")
    page = write_player(out, song, timeline)
    log(f"done: open {page} in a browser")


if __name__ == "__main__":
    main()
