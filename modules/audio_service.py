"""
modules/audio_service.py - Downloads YouTube audio and converts to MP3 using yt-dlp.
"""

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yt_dlp


def log(*args: Any) -> None:
    print(*args, file=sys.stderr)


def sanitize_filename(name: str) -> str:
    """Sanitize string for safe filesystem usage."""
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = re.sub(r"[\s_]+", "_", s).strip("_")
    return s or "audio_track"


def download_song_audio(
    youtube_url: Optional[str],
    search_query: str,
    song_name: str,
    output_dir: str = "downloads",
) -> Tuple[Path, Dict[str, Any]]:
    """Download audio from YouTube and convert to MP3.

    Returns:
        (file_path, audio_info_dict)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_name = sanitize_filename(song_name)
    target_tmpl = str(out_path / f"{base_name}.%(ext)s")
    final_mp3 = out_path / f"{base_name}.mp3"

    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": target_tmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
        "no_warnings": False,
    }

    # Enable node js-runtime if available for challenge solving
    node_path = shutil.which("node")
    if node_path:
        ydl_opts["js_runtimes"] = {"node": {"path": node_path}}

    targets = []
    if youtube_url and ("youtube.com" in youtube_url or "youtu.be" in youtube_url):
        targets.append((youtube_url, "direct YouTube URL"))
    if search_query:
        targets.append((f"ytsearch1:{search_query}", f"search query '{search_query}'"))
    targets.append((f"ytsearch1:{song_name} Carnatic classical audio", f"fallback search for '{song_name}'"))

    last_error = None
    collected_info: Dict[str, Any] = {}

    for target, desc in targets:
        log(f"[Audio Service] Downloading via {desc}...")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(target, download=True)
                entry = info["entries"][0] if ("entries" in info and info["entries"]) else info
                collected_info = {
                    "title": entry.get("title"),
                    "duration": entry.get("duration"),
                    "uploader": entry.get("uploader"),
                    "webpage_url": entry.get("webpage_url") or youtube_url,
                }

                if final_mp3.exists():
                    log(f"[Audio Service] Saved MP3 to {final_mp3}")
                    return final_mp3, collected_info

                matches = list(out_path.glob(f"{base_name}*.mp3"))
                if matches:
                    log(f"[Audio Service] Saved MP3 to {matches[0]}")
                    return matches[0], collected_info

                if entry and "title" in entry:
                    entry_clean = sanitize_filename(entry["title"])
                    matches = list(out_path.glob(f"{entry_clean}*.mp3"))
                    if matches:
                        return matches[0], collected_info

        except Exception as err:
            log(f"[Audio Service] Download via {desc} failed: {err}")
            last_error = err
            continue

    if final_mp3.exists():
        return final_mp3, collected_info

    raise RuntimeError(f"Could not download audio for '{song_name}'. Last error: {last_error}")
