"""
modules/audio_service.py - Downloads YouTube audio and converts to MP3 using yt-dlp.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yt_dlp

# Dance playback always uses the opening stretch of a track.
MAX_SONG_SECONDS = 120.0


def log(*args: Any) -> None:
    print(*args, file=sys.stderr)


def sanitize_filename(name: str) -> str:
    """Sanitize string for safe filesystem usage."""
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    s = re.sub(r"[\s_]+", "_", s).strip("_")
    return s or "audio_track"


def probe_duration(path: Path) -> Optional[float]:
    """Return the media duration in seconds, or None if it cannot be read."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    probe = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return None
    try:
        return float(probe.stdout.strip())
    except ValueError:
        return None


def to_mp3(src: Path) -> Path:
    """Return src as an MP3, converting and removing the original when it is not one."""
    if src.suffix.lower() == ".mp3":
        return src
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to convert uploaded audio")
    out = src.with_suffix(".mp3")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-vn", "-acodec", "libmp3lame", "-q:a", "2", str(out)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as err:
        out.unlink(missing_ok=True)
        detail = (err.stderr or "ffmpeg failed").strip()
        raise RuntimeError(f"Could not read '{src.name}': {detail[-300:]}") from err
    src.unlink(missing_ok=True)
    return out


def cap_song(src: Path, dest: Path, max_seconds: float = MAX_SONG_SECONDS) -> Path:
    """Write the first two minutes of src to dest.

    Tracks that are already at or under the cap are copied unchanged.
    Longer tracks keep only the opening stretch. When src and dest are
    the same file, a longer track is replaced in place.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to keep songs under 2 minutes")

    dest.parent.mkdir(parents=True, exist_ok=True)
    same = src.resolve() == dest.resolve()
    duration = probe_duration(src)
    if duration is not None and duration <= max_seconds + 0.05:
        if not same:
            shutil.copy(src, dest)
        return dest

    dest = dest.with_suffix(".mp3")
    same = src.resolve() == dest.resolve()
    tmp = dest.with_name(f".{dest.stem}.capping.mp3")
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(src),
                "-t", str(max_seconds),
                "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                str(tmp),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as err:
        tmp.unlink(missing_ok=True)
        detail = (err.stderr or err.stdout or "ffmpeg failed").strip()
        raise RuntimeError(f"Could not trim '{src.name}' to 2 minutes: {detail[-500:]}") from err

    if same:
        tmp.replace(src)
        log(f"[Audio Service] Trimmed {src.name} to the first 2 minutes")
        return src
    tmp.replace(dest)
    log(f"[Audio Service] Trimmed {src.name} to the first 2 minutes")
    return dest


def _with_capped_duration(path: Path, info: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    capped = cap_song(path, path)
    updated = dict(info)
    duration = probe_duration(capped)
    if duration is not None:
        updated["duration"] = int(duration) if duration >= 1 else duration
    return capped, updated


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
                    return _with_capped_duration(final_mp3, collected_info)

                matches = list(out_path.glob(f"{base_name}*.mp3"))
                if matches:
                    log(f"[Audio Service] Saved MP3 to {matches[0]}")
                    return _with_capped_duration(matches[0], collected_info)

                if entry and "title" in entry:
                    entry_clean = sanitize_filename(entry["title"])
                    matches = list(out_path.glob(f"{entry_clean}*.mp3"))
                    if matches:
                        return _with_capped_duration(matches[0], collected_info)

        except Exception as err:
            log(f"[Audio Service] Download via {desc} failed: {err}")
            last_error = err
            continue

    if final_mp3.exists():
        return _with_capped_duration(final_mp3, collected_info)

    raise RuntimeError(f"Could not download audio for '{song_name}'. Last error: {last_error}")
