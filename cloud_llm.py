"""
cloud_llm.py - Facade & CLI for song recommendation and downloading.
Delegates to modular services in `modules/`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from modules.audio_service import download_song_audio
from modules.llm_service import log, recommend_song
from modules.storage_service import MusicLibrary
from modules.tala import tala_beats


def find_and_download_song(
    user_prompt: str,
    output_dir: str = "downloads",
    api_key: Optional[str] = None,
    model: str = "openai/gpt-oss-20b",
) -> Dict[str, Any]:
    """High-level function: prompts Gemini, receives song recommendation & URL,

    downloads audio, and logs to local music library.
    """
    meta = recommend_song(user_prompt=user_prompt, api_key=api_key, model=model)

    song_name = meta.get("song_name") or "Indian_Classical_Song"
    yt_url = meta.get("youtube_url")
    search_query = (
        meta.get("search_query")
        or f"{song_name} {meta.get('raga', '')} {meta.get('composer', '')}".strip()
    )

    log("\n--- Song Recommendation ---")
    log(f"Title:       {song_name}")
    log(f"Composer:    {meta.get('composer', 'Traditional')}")
    log(f"Raga:        {meta.get('raga', 'N/A')}")
    log(f"Tala:        {meta.get('tala', 'N/A')} (Beats: {meta.get('tala_beats', 8)})")
    log(f"YouTube URL: {yt_url}")
    log(f"Description: {meta.get('description', '')}\n")

    audio_file, audio_info = download_song_audio(
        youtube_url=yt_url,
        search_query=search_query,
        song_name=song_name,
        output_dir=output_dir,
    )

    meta["tala_beats"] = tala_beats(meta.get("tala"), meta.get("tala_beats"))
    meta["audio_path"] = str(audio_file.resolve())
    meta["audio_info"] = audio_info

    # Save to local library
    library = MusicLibrary(storage_dir=output_dir)
    library.add_song(meta)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend an Indian classical dance song via Groq browser search and download audio from YouTube."
    )
    parser.add_argument("prompt", help="Song description, deity, raga, tala, or adavu style")
    parser.add_argument("-o", "--out-dir", default="downloads", help="Directory to save downloaded audio")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Groq model name")
    parser.add_argument("--api-key", default=None, help="Groq API key (or set GROQ_API_KEY)")
    args = parser.parse_args()

    try:
        result = find_and_download_song(
            user_prompt=args.prompt,
            output_dir=args.out_dir,
            api_key=args.api_key,
            model=args.model,
        )
        print(json.dumps(result, indent=2))
        log("\nReady to run with pipeline.py:")
        tala = result.get("tala_beats") or 8
        log(f"  uv run pipeline.py \"{result['audio_path']}\" --adavus landmarks_json -o output --tala {tala}")
    except Exception as e:
        log(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
