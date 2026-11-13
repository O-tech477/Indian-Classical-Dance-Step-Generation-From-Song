"""
modules/storage_service.py - Manages local catalog and metadata of downloaded music.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class MusicLibrary:
    def __init__(self, storage_dir: str = "downloads"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_file = self.storage_dir / "library.json"
        self._ensure_catalog()

    def _ensure_catalog(self) -> None:
        if not self.catalog_file.exists():
            self._save_catalog([])

    def _read_catalog(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.catalog_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_catalog(self, items: List[Dict[str, Any]]) -> None:
        self.catalog_file.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def add_song(self, song_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Add or update a song record in the local library."""
        catalog = self._read_catalog()
        filename = Path(song_meta.get("audio_path", "")).name

        entry = {
            "id": song_meta.get("id") or str(int(time.time() * 1000)),
            "song_name": song_meta.get("song_name"),
            "composer": song_meta.get("composer"),
            "raga": song_meta.get("raga"),
            "tala": song_meta.get("tala"),
            "tala_beats": song_meta.get("tala_beats", 8),
            "youtube_url": song_meta.get("youtube_url"),
            "description": song_meta.get("description"),
            "filename": filename,
            "audio_path": song_meta.get("audio_path"),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Replace existing entry if filename matches, else prepend
        updated = False
        for i, item in enumerate(catalog):
            if item.get("filename") == filename:
                catalog[i] = entry
                updated = True
                break

        if not updated:
            catalog.insert(0, entry)

        self._save_catalog(catalog)
        return entry

    def list_songs(self) -> List[Dict[str, Any]]:
        """Return all cataloged songs, checking file existence."""
        catalog = self._read_catalog()
        # Verify if the audio file actually exists
        for item in catalog:
            p = Path(item.get("audio_path", ""))
            item["exists"] = p.exists()
            if p.exists():
                item["file_size_mb"] = round(p.stat().st_size / (1024 * 1024), 2)
        return catalog

    def get_song(self, song_id: str) -> Optional[Dict[str, Any]]:
        catalog = self._read_catalog()
        for item in catalog:
            if item.get("id") == song_id or item.get("filename") == song_id:
                return item
        return None
