"""
app.py - Flask web application connecting the user prompt to a Groq recommendation,
YouTube audio downloading, and local music storage.
"""

import os
import re
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, g, jsonify, redirect, render_template, request, send_from_directory, session, url_for

load_dotenv(Path(__file__).resolve().parent / ".env")

from modules.audio_service import cap_song, download_song_audio, sanitize_filename, to_mp3
from modules.auth_service import (
    account_folder,
    finish_google_sign_in,
    google_provider_status,
    start_google_sign_in,
    supabase_configured,
    write_profile_file,
)
from modules.tala import tala_beats
from modules.chat_store import _title_from_prompt, build_chat_store
from modules.instrument_service import predict_instruments
from modules.mudhra_service import load_saved_mudhra_timeline, write_mudhra_timeline
from modules.llm_service import recommend_song
from modules.storage_service import MusicLibrary

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.permanent_session_lifetime = timedelta(days=30)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".webm"}

ROOT = Path(__file__).resolve().parent
DOWNLOADS_DIR = ROOT / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
DANCE_ROOT = ROOT / "output"
MUDHRA_DIR = ROOT / "modules" / "mudhra_dataset"
ADAVUS_DIR = ROOT / "landmarks_json"
chat_store, CHAT_BACKEND, CHAT_STATUS = build_chat_store(DOWNLOADS_DIR)
PUBLIC_ENDPOINTS = {"index", "google_start", "google_callback", "logout", "static"}


def _flask_secret() -> str:
    configured = os.environ.get("FLASK_SECRET_KEY", "").strip()
    if configured:
        return configured
    secret_path = ROOT / "instance" / "flask_secret"
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    if secret_path.is_file():
        saved = secret_path.read_text(encoding="utf-8").strip()
        if saved:
            return saved
    saved = os.urandom(32).hex()
    secret_path.write_text(saved, encoding="utf-8")
    return saved


app.secret_key = _flask_secret()


def current_user() -> dict | None:
    user = session.get("user")
    if not isinstance(user, dict) or not user.get("id"):
        return None
    return user


def user_library() -> MusicLibrary:
    folder = account_folder(DOWNLOADS_DIR, g.user["id"])
    return MusicLibrary(storage_dir=str(folder))


def callback_url() -> str:
    return request.url_root.rstrip("/") + "/auth/callback"


def run_dance_pipeline(audio_file: Path, tala_beats: int, user_id: str) -> str:
    """Run pipeline.py and return the URL of the stick-figure player."""
    dance_id = re.sub(r"[^A-Za-z0-9._-]", "_", audio_file.stem).strip("._") or "dance"
    out_dir = DANCE_ROOT / user_id / dance_id
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
    return f"/dance/{user_id}/{dance_id}/index.html"


@app.before_request
def require_sign_in():
    g.user = current_user()
    if request.endpoint in PUBLIC_ENDPOINTS or (request.endpoint or "").startswith("static"):
        return None
    if g.user:
        return None
    if request.path.startswith("/api"):
        return jsonify({"success": False, "error": "Sign in with Google to continue."}), 401
    return redirect(url_for("index"))


@app.route("/")
def index():
    """Show Google sign-in, or the app once an account is signed in."""
    if not g.user:
        ready, message = google_provider_status(callback_url()) if supabase_configured() else (
            False,
            CHAT_STATUS,
        )
        return render_template(
            "login.html",
            google_ready=ready and supabase_configured(),
            google_message=message,
            auth_error=request.args.get("auth_error") or "",
        )
    account_folder(DOWNLOADS_DIR, g.user["id"])
    return render_template("index.html", user=g.user)


@app.route("/auth/google")
def google_start():
    """Send the browser to Supabase, which sends it on to Google."""
    ready, message = google_provider_status(callback_url())
    if not ready:
        return redirect(url_for("index", auth_error=message))
    try:
        target, verifier = start_google_sign_in(callback_url())
    except Exception as exc:
        return redirect(url_for("index", auth_error=str(exc)))
    session["oauth_verifier"] = verifier
    return redirect(target)


@app.route("/auth/callback")
def google_callback():
    """Supabase returns here with the Google sign-in code."""
    error = request.args.get("error_description") or request.args.get("error")
    if error:
        return redirect(url_for("index", auth_error=error))
    code = request.args.get("code") or ""
    verifier = session.pop("oauth_verifier", "")
    if not code or not verifier:
        return redirect(url_for("index", auth_error="Google sign-in did not finish. Try again."))
    try:
        profile = finish_google_sign_in(code, verifier)
        write_profile_file(account_folder(DOWNLOADS_DIR, profile["id"]), profile)
    except Exception as exc:
        return redirect(url_for("index", auth_error=f"Google sign-in failed: {exc}"))
    session.clear()
    session["user"] = profile
    session.permanent = True
    return redirect(url_for("index"))


@app.route("/auth/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/api/songs", methods=["GET"])
def get_songs():
    """List songs saved for the signed-in account."""
    songs = user_library().list_songs()
    return jsonify({"success": True, "songs": songs})


@app.route("/api/mudhra/<path:filename>", methods=["GET"])
def get_mudhra_timeline(filename):
    """Return the saved mudra segments for one song in downloads/."""
    if Path(filename).name != filename:
        return jsonify({"success": False, "error": "Invalid audio filename"}), 400
    data = load_saved_mudhra_timeline(filename, account_folder(DOWNLOADS_DIR, g.user["id"]))
    if not data:
        return jsonify({"success": True, "segments": []})
    segments = []
    for segment in data.get("segments") or []:
        number = segment.get("number")
        if number is None:
            continue
        segments.append({
            "start": segment.get("start"),
            "end": segment.get("end"),
            "number": number,
            "name": segment.get("name"),
        })
    return jsonify({
        "success": True,
        "song_name": data.get("song_name"),
        "audio_file": data.get("audio_file") or filename,
        "segments": segments,
    })


@app.route("/mudhras/<int:number>.png")
def serve_mudhra_image(number):
    """Serve one hasta photograph from the numbered mudra set."""
    if not 1 <= number <= 52:
        return jsonify({"error": "Mudra not found"}), 404
    image = MUDHRA_DIR / f"{number}.png"
    if not image.is_file():
        return jsonify({"error": "Mudra not found"}), 404
    return send_from_directory(MUDHRA_DIR, image.name)


@app.route("/api/instruments/<path:filename>", methods=["GET"])
def get_instruments(filename):
    """Classify a short window around the current position in a downloaded song."""
    if Path(filename).name != filename:
        return jsonify({"success": False, "error": "Invalid audio filename"}), 400
    folder = account_folder(DOWNLOADS_DIR, g.user["id"])
    audio_path = (folder / filename).resolve()
    if not audio_path.is_relative_to(folder) or not audio_path.is_file():
        return jsonify({"success": False, "error": "Audio file not found"}), 404
    try:
        result = predict_instruments(audio_path, request.args.get("time", 0, type=float))
        return jsonify({"success": True, **result})
    except Exception as exc:
        app.logger.exception("Instrument recognition failed")
        return jsonify({"success": False, "error": str(exc)}), 503


@app.route("/api/chats/status", methods=["GET"])
def chat_status():
    """Tell the UI whether this account's chats are in Supabase."""
    if getattr(chat_store, "remote_ready", lambda: False)():
        return jsonify({
            "success": True,
            "backend": "supabase",
            "message": "Chats are saved in Supabase for this Google account.",
        })
    message = CHAT_STATUS
    if CHAT_BACKEND == "supabase":
        message = "Chats for this account are in its downloads folder until you run supabase/schema.sql in the Supabase SQL editor."
    return jsonify({"success": True, "backend": "account", "message": message})


@app.route("/api/chats", methods=["GET"])
def list_chats():
    """List this account's chats, newest activity first."""
    try:
        return jsonify({
            "success": True,
            "chats": chat_store.list_chats(g.user["id"]),
            "backend": "supabase" if chat_store.remote_ready() else "account",
        })
    except Exception as exc:
        return jsonify({"success": False, "error": f"Could not load chats: {exc}"}), 500


@app.route("/api/chats/<chat_id>", methods=["GET"])
def get_chat(chat_id):
    """Return one chat and its prompts."""
    try:
        chat = chat_store.get_chat(g.user["id"], chat_id)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Could not load chat: {exc}"}), 500
    if not chat:
        return jsonify({"success": False, "error": "Chat not found"}), 404
    return jsonify({"success": True, "chat": chat})


@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    """Delete a chat and its saved prompts."""
    try:
        deleted = chat_store.delete_chat(g.user["id"], chat_id)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Could not delete chat: {exc}"}), 500
    if not deleted:
        return jsonify({"success": False, "error": "Chat not found"}), 404
    return jsonify({"success": True})


def _save_chat_turn(chat_id, saved_entry, prompt):
    user_id = g.user["id"]
    if not chat_id:
        created = chat_store.create_chat(user_id, saved_entry.get("song_name") or _title_from_prompt(prompt))
        chat_id = created["id"]
    turn = chat_store.add_turn(
        user_id,
        chat_id,
        {**saved_entry, "prompt": prompt, "dance_url": saved_entry.get("dance_url")},
    )
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
    folder = account_folder(DOWNLOADS_DIR, g.user["id"])
    dest = folder / f"{sanitize_filename(Path(uploaded.filename).stem)}{ext}"
    if dest.exists() or dest.with_suffix(".mp3").exists():
        dest = dest.with_name(f"{dest.stem}_{int(time.time())}{dest.suffix}")

    try:
        if chat_id and not chat_store.get_chat(g.user["id"], chat_id):
            return jsonify({"success": False, "error": "Chat not found"}), 404
        uploaded.save(dest)
        dest = cap_song(to_mp3(dest), dest.with_suffix(".mp3"))
        prompt = prompt or f"Uploaded {song_name}"
        mudhra_timeline, mudhra_json_path = write_mudhra_timeline(dest, song_name)
        dance_url = run_dance_pipeline(dest, beats, g.user["id"])
        saved_entry = user_library().add_song({
            "song_name": song_name,
            "tala": "Adi Tala" if beats == 8 else f"{beats} beats",
            "tala_beats": beats,
            "description": prompt,
            "audio_path": str(dest.resolve()),
            "audio_url": f"/audio/{g.user['id']}/{dest.name}",
            "filename": dest.name,
            "prompt": prompt,
            "dance_url": dance_url,
        })
        chat_id, turn = _save_chat_turn(chat_id, saved_entry, prompt)
        return jsonify({
            "success": True,
            "song": saved_entry,
            "dance_url": dance_url,
            "mudhra_timeline": mudhra_timeline,
            "mudhra_json_path": str(mudhra_json_path.relative_to(ROOT)),
            "chat": chat_store.get_chat(g.user["id"], chat_id),
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
        if chat_id and not chat_store.get_chat(g.user["id"], chat_id):
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
            output_dir=str(account_folder(DOWNLOADS_DIR, g.user["id"])),
        )
        mudhra_timeline, mudhra_json_path = write_mudhra_timeline(audio_file, song_name)

        # Step 3: Catalog in this account's library
        record = {
            **recommendation,
            "audio_path": str(audio_file.resolve()),
            "audio_url": f"/audio/{g.user['id']}/{audio_file.name}",
            "filename": audio_file.name,
            "audio_info": audio_info,
        }
        saved_entry = user_library().add_song(record)
        dance_url = run_dance_pipeline(audio_file, saved_entry.get("tala_beats") or 8, g.user["id"])
        saved_entry["dance_url"] = dance_url
        saved_entry["prompt"] = user_prompt
        saved_entry = user_library().add_song(saved_entry)

        if not chat_id:
            created = chat_store.create_chat(
                g.user["id"],
                saved_entry.get("song_name") or _title_from_prompt(user_prompt),
            )
            chat_id = created["id"]

        turn = chat_store.add_turn(
            g.user["id"],
            chat_id,
            {**saved_entry, "prompt": user_prompt, "dance_url": dance_url},
        )

        return jsonify({
            "success": True,
            "song": saved_entry,
            "dance_url": dance_url,
            "mudhra_timeline": mudhra_timeline,
            "mudhra_json_path": str(mudhra_json_path.relative_to(ROOT)),
            "chat": chat_store.get_chat(g.user["id"], chat_id),
            "turn": turn,
        })

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to process request: {str(e)}"}), 500


@app.route("/dance/<user_id>/<dance_id>/<path:filename>")
def serve_dance(user_id, dance_id, filename):
    """Serve a stick-figure player that belongs to the signed-in account."""
    if user_id != g.user["id"] or not re.fullmatch(r"[A-Za-z0-9._-]+", dance_id):
        return jsonify({"error": "Dance not found"}), 404
    folder = (DANCE_ROOT / user_id / dance_id).resolve()
    root = (DANCE_ROOT / user_id).resolve()
    if not folder.is_relative_to(root) or not folder.is_dir():
        return jsonify({"error": "Dance not found"}), 404
    return send_from_directory(folder, filename)


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    """Stream a song from the signed-in account's downloads folder."""
    folder = account_folder(DOWNLOADS_DIR, g.user["id"])
    parts = Path(filename).parts
    if len(parts) == 2 and parts[0] == g.user["id"]:
        name = parts[1]
    elif len(parts) == 1:
        name = parts[0]
    else:
        return jsonify({"error": "Audio file not found"}), 404
    if Path(name).suffix.lower() not in AUDIO_EXTENSIONS:
        return jsonify({"error": "Audio file not found"}), 404
    audio_path = (folder / name).resolve()
    if not audio_path.is_relative_to(folder) or not audio_path.is_file():
        return jsonify({"error": "Audio file not found"}), 404
    return send_from_directory(folder, name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6767))
    app.run(host="0.0.0.0", port=port, debug=True)
