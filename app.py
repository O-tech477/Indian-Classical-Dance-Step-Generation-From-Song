"""
app.py - Flask web application connecting the user prompt to a Groq recommendation,
YouTube audio downloading, and local music storage.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory

load_dotenv(Path(__file__).resolve().parent / ".env")

from modules.audio_service import cap_song, download_song_audio, sanitize_filename, to_mp3
from modules.tala import tala_beats
from modules.chat_store import _title_from_prompt, build_chat_store
from modules.instrument_service import predict_instruments
from modules.llm_service import recommend_song
from modules.storage_service import MusicLibrary

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".webm"}

ROOT = Path(__file__).resolve().parent
DOWNLOADS_DIR = ROOT / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
DANCE_ROOT = ROOT / "output"
ADAVUS_DIR = ROOT / "landmarks_json"
library = MusicLibrary(storage_dir=str(DOWNLOADS_DIR))
chat_store, CHAT_BACKEND, CHAT_STATUS = build_chat_store(DOWNLOADS_DIR)


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


@app.route("/api/instruments/<path:filename>", methods=["GET"])
def get_instruments(filename):
    """Classify a short window around the current position in a downloaded song."""
    if Path(filename).name != filename:
        return jsonify({"success": False, "error": "Invalid audio filename"}), 400
    audio_path = (DOWNLOADS_DIR / filename).resolve()
    if not audio_path.is_relative_to(DOWNLOADS_DIR.resolve()) or not audio_path.is_file():
        return jsonify({"success": False, "error": "Audio file not found"}), 404
    try:
        result = predict_instruments(audio_path, request.args.get("time", 0, type=float))
        return jsonify({"success": True, **result})
    except Exception as exc:
        app.logger.exception("Instrument recognition failed")
        return jsonify({"success": False, "error": str(exc)}), 503


@app.route("/api/chats/status", methods=["GET"])
def chat_status():
    """Tell the UI whether chats are stored in Supabase or on this computer."""
    return jsonify({"success": True, "backend": CHAT_BACKEND, "message": CHAT_STATUS})


@app.route("/api/chats", methods=["GET"])
def list_chats():
    """List saved chats, newest activity first."""
    try:
        return jsonify({"success": True, "chats": chat_store.list_chats(), "backend": CHAT_BACKEND})
    except Exception as exc:
        return jsonify({"success": False, "error": f"Could not load chats: {exc}"}), 500


@app.route("/api/chats/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    """Return one chat and its prompts."""
    try:
        chat = chat_store.get_chat(chat_id)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Could not load chat: {exc}"}), 500
    if not chat:
        return jsonify({"success": False, "error": "Chat not found"}), 404
    return jsonify({"success": True, "chat": chat})


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    """Delete a chat and its saved prompts."""
    try:
        deleted = chat_store.delete_chat(chat_id)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Could not delete chat: {exc}"}), 500
    if not deleted:
        return jsonify({"success": False, "error": "Chat not found"}), 404
    return jsonify({"success": True})


def _save_chat_turn(chat_id, saved_entry, prompt):
    if not chat_id:
        created = chat_store.create_chat(saved_entry.get("song_name") or _title_from_prompt(prompt))
        chat_id = created["id"]
    turn = chat_store.add_turn(chat_id, {**saved_entry, "prompt": prompt, "dance_url": saved_entry.get("dance_url")})
    return chat_id, turn


@app.route("/api/upload", methods=["POST"])
def upload_song():
    """Save an uploaded song, catalog it, and build its dance. Adi tala unless the prompt names one."""
    uploaded = request.files.get("audio")
    if uploaded is None or not uploaded.filename:
        return jsonify({"success": False, "error": "Choose an audio file to upload"}), 400
    ext = Path(uploaded.filename).suffix.lower()
    if ext not in AUDIO_EXTENSIONS:
        return jsonify({"success": False, "error": "Upload an audio file (mp3, wav, m4a, ogg, flac)"}), 400

    prompt = (request.form.get("prompt") or "").strip()
    chat_id = (request.form.get("chat_id") or "").strip()
    song_name = sanitize_filename(Path(uploaded.filename).stem).replace("_", " ")
    beats = tala_beats(prompt, 8) if prompt else 8
    dest = DOWNLOADS_DIR / f"{sanitize_filename(Path(uploaded.filename).stem)}{ext}"
    if dest.exists() or dest.with_suffix(".mp3").exists():
        dest = dest.with_name(f"{dest.stem}_{int(time.time())}{dest.suffix}")

    try:
        if chat_id and not chat_store.get_chat(chat_id):
            return jsonify({"success": False, "error": "Chat not found"}), 404
        uploaded.save(dest)
        dest = cap_song(to_mp3(dest), dest.with_suffix(".mp3"))
        prompt = prompt or f"Uploaded {song_name}"
        dance_url = run_dance_pipeline(dest, beats)
        saved_entry = library.add_song({
            "song_name": song_name,
            "tala": "Adi Tala" if beats == 8 else f"{beats} beats",
            "tala_beats": beats,
            "description": prompt,
            "audio_path": str(dest.resolve()),
            "audio_url": f"/audio/{dest.name}",
            "filename": dest.name,
            "prompt": prompt,
            "dance_url": dance_url,
        })
        chat_id, turn = _save_chat_turn(chat_id, saved_entry, prompt)
        return jsonify({
            "success": True,
            "song": saved_entry,
            "dance_url": dance_url,
            "chat": chat_store.get_chat(chat_id),
            "turn": turn,
        })
    except Exception as exc:
        dest.unlink(missing_ok=True)
        detail = str(exc).strip().splitlines()[-1] or "the file could not be read"
        return jsonify({"success": False, "error": f"Could not use that file: {detail}"}), 500


@app.route("/api/recommend-and-download", methods=["POST"])
def recommend_and_download():
    """Process user prompt: query Gemini LLM for recommendation,

    download audio from YouTube via yt-dlp, and catalog the track.
    """
    data = request.get_json() or {}
    user_prompt = data.get("prompt", "").strip()
    model = data.get("model", "openai/gpt-oss-20b")
    chat_id = (data.get("chat_id") or "").strip()

    if not user_prompt:
        return jsonify({"success": False, "error": "Prompt cannot be empty"}), 400

    try:
        if chat_id and not chat_store.get_chat(chat_id):
            return jsonify({"success": False, "error": "Chat not found"}), 404

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
        saved_entry["prompt"] = user_prompt
        saved_entry = library.add_song(saved_entry)

        if not chat_id:
            created = chat_store.create_chat(saved_entry.get("song_name") or _title_from_prompt(user_prompt))
            chat_id = created["id"]

        turn = chat_store.add_turn(chat_id, {**saved_entry, "prompt": user_prompt, "dance_url": dance_url})

        return jsonify({
            "success": True,
            "song": saved_entry,
            "dance_url": dance_url,
            "chat": chat_store.get_chat(chat_id),
            "turn": turn,
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
