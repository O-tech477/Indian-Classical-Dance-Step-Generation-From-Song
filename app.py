"""
app.py - Flask web application connecting the user prompt to a Groq recommendation,
YouTube audio downloading, and local music storage.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory

load_dotenv(Path(__file__).resolve().parent / ".env")

from modules.audio_service import download_song_audio
from modules.llm_service import recommend_song
from modules.storage_service import MusicLibrary

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
DOWNLOADS_DIR = ROOT / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
DANCE_ROOT = ROOT / "output"
ADAVUS_DIR = ROOT / "landmarks_json"
library = MusicLibrary(storage_dir=str(DOWNLOADS_DIR))


def run_dance_pipeline(audio_file: Path, tala_beats: int) -> str:
    """Run pipeline.py and return the URL of the stick-figure player."""
    dance_id = re.sub(r"[^A-Za-z0-9._-]", "_", audio_file.stem).strip("._") or "dance"
    out_dir = DANCE_ROOT / dance_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tala = int(tala_beats or 8)
    if tala < 1:
        tala = 8
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pipeline.py"),
            str(audio_file.resolve()),
            "--adavus",
            str(ADAVUS_DIR),
            "-o",
            str(out_dir),
            "--tala",
            str(tala),
            "--default-beats",
            "4",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode != 0 or not (out_dir / "index.html").exists():
        detail = (completed.stderr or completed.stdout or "dance pipeline failed").strip()
        raise RuntimeError(detail[-1500:])
    return f"/dance/{dance_id}/index.html"


@app.route("/")
def index():
    """Render the main UI page."""
    return render_template("index.html")


@app.route("/api/songs", methods=["GET"])
def get_songs():
    """List all downloaded songs in the library."""
    songs = library.list_songs()
    return jsonify({"success": True, "songs": songs})


@app.route("/api/recommend-and-download", methods=["POST"])
def recommend_and_download():
    """Process user prompt: query Gemini LLM for recommendation,

    download audio from YouTube via yt-dlp, and catalog the track.
    """
    data = request.get_json() or {}
    user_prompt = data.get("prompt", "").strip()
    model = data.get("model", "openai/gpt-oss-20b")

    if not user_prompt:
        return jsonify({"success": False, "error": "Prompt cannot be empty"}), 400

    try:
        # Step 1: Query Groq with browser search
        recommendation = recommend_song(user_prompt=user_prompt, model=model)

        song_name = recommendation.get("song_name") or "Classical_Song"
        youtube_url = recommendation.get("youtube_url")
        search_query = (
            recommendation.get("search_query")
            or f"{song_name} {recommendation.get('raga', '')} {recommendation.get('composer', '')}".strip()
        )

        # Step 2: Download YouTube audio via yt-dlp to downloads/
        audio_file, audio_info = download_song_audio(
            youtube_url=youtube_url,
            search_query=search_query,
            song_name=song_name,
            output_dir=str(DOWNLOADS_DIR),
        )

        # Step 3: Catalog in library
        record = {
            **recommendation,
            "audio_path": str(audio_file.resolve()),
            "audio_url": f"/audio/{audio_file.name}",
            "filename": audio_file.name,
            "audio_info": audio_info,
        }
        saved_entry = library.add_song(record)
        dance_url = run_dance_pipeline(audio_file, saved_entry.get("tala_beats") or 8)
        saved_entry["dance_url"] = dance_url

        return jsonify({
            "success": True,
            "song": saved_entry,
            "dance_url": dance_url,
        })

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to process request: {str(e)}"}), 500


@app.route("/dance/<dance_id>/<path:filename>")
def serve_dance(dance_id, filename):
    """Serve a generated stick-figure player and its audio file."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", dance_id):
        return jsonify({"error": "Dance not found"}), 404
    folder = (DANCE_ROOT / dance_id).resolve()
    if not folder.is_relative_to(DANCE_ROOT.resolve()) or not folder.is_dir():
        return jsonify({"error": "Dance not found"}), 404
    return send_from_directory(folder, filename)


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    """Stream or serve downloaded audio files."""
    if (DOWNLOADS_DIR / filename).exists():
        return send_from_directory(str(DOWNLOADS_DIR), filename, mimetype="audio/mpeg")
    root_dir = Path(__file__).parent
    if (root_dir / filename).exists():
        return send_from_directory(str(root_dir), filename, mimetype="audio/mpeg")
    return jsonify({"error": "Audio file not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6767))
    app.run(host="0.0.0.0", port=port, debug=True)
