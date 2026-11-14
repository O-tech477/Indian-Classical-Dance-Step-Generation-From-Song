"""Persistent prompt chats. Uses Supabase when configured, otherwise a local JSON file."""

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _title_from_prompt(prompt: str) -> str:
    text = " ".join(prompt.split())
    if len(text) <= 48:
        return text or "New chat"
    return text[:45].rstrip() + "..."


def _song_title(turn: dict[str, Any], fallback: str = "") -> str:
    name = " ".join(str(turn.get("song_name") or "").split())
    if name:
        return name
    if fallback:
        return fallback
    return _title_from_prompt(str(turn.get("prompt") or ""))


def _account_id(user_id: str) -> str:
    try:
        return str(uuid.UUID(str(user_id)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid account") from exc


class LocalChatStore:
    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self._lock = threading.Lock()

    def _path(self, user_id: str) -> Path:
        folder = self.storage_dir / _account_id(user_id)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "chats.json"

    def _read(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"chats": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"chats": []}
        if not isinstance(data, dict) or not isinstance(data.get("chats"), list):
            return {"chats": []}
        return data

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_chats(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            chats = self._read(self._path(user_id))["chats"]
        summary = []
        for chat in chats:
            title = chat.get("title") or "New chat"
            for turn in reversed(chat.get("turns") or []):
                named = _song_title(turn, "")
                if turn.get("song_name"):
                    title = named
                    break
            summary.append({
                "id": chat["id"],
                "title": title,
                "created_at": chat.get("created_at"),
                "updated_at": chat.get("updated_at"),
            })
        summary.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return summary

    def get_chat(self, user_id: str, chat_id: str) -> dict[str, Any] | None:
        with self._lock:
            for chat in self._read(self._path(user_id))["chats"]:
                if chat.get("id") == chat_id:
                    turns = list(chat.get("turns") or [])
                    turns.sort(key=lambda item: item.get("created_at") or "")
                    return {
                        "id": chat["id"],
                        "title": chat.get("title") or "New chat",
                        "created_at": chat.get("created_at"),
                        "updated_at": chat.get("updated_at"),
                        "turns": turns,
                    }
        return None

    def create_chat(self, user_id: str, title: str) -> dict[str, Any]:
        now = _now()
        chat = {
            "id": str(uuid.uuid4()),
            "title": title or "New chat",
            "created_at": now,
            "updated_at": now,
            "turns": [],
        }
        with self._lock:
            path = self._path(user_id)
            data = self._read(path)
            data["chats"].insert(0, chat)
            self._write(path, data)
        return {key: chat[key] for key in ("id", "title", "created_at", "updated_at")}

    def add_turn(self, user_id: str, chat_id: str, turn: dict[str, Any]) -> dict[str, Any]:
        saved = {
            "id": str(uuid.uuid4()),
            "prompt": turn.get("prompt") or "",
            "song_name": turn.get("song_name"),
            "composer": turn.get("composer"),
            "raga": turn.get("raga"),
            "tala": turn.get("tala"),
            "tala_beats": turn.get("tala_beats"),
            "youtube_url": turn.get("youtube_url"),
            "description": turn.get("description"),
            "filename": turn.get("filename"),
            "audio_url": turn.get("audio_url"),
            "audio_path": turn.get("audio_path"),
            "dance_url": turn.get("dance_url"),
            "created_at": _now(),
        }
        with self._lock:
            path = self._path(user_id)
            data = self._read(path)
            for chat in data["chats"]:
                if chat.get("id") != chat_id:
                    continue
                chat.setdefault("turns", []).append(saved)
                chat["updated_at"] = saved["created_at"]
                chat["title"] = _song_title(saved, chat.get("title") or "New chat")
                self._write(path, data)
                return saved
        raise KeyError("Chat not found")

    def delete_chat(self, user_id: str, chat_id: str) -> bool:
        with self._lock:
            path = self._path(user_id)
            data = self._read(path)
            kept = [chat for chat in data["chats"] if chat.get("id") != chat_id]
            if len(kept) == len(data["chats"]):
                return False
            data["chats"] = kept
            self._write(path, data)
        return True


class SupabaseChatStore:
    def __init__(self, url: str, service_role_key: str):
        from supabase import create_client

        self.client = create_client(url, service_role_key)

    def list_chats(self, user_id: str) -> list[dict[str, Any]]:
        result = (
            self.client.table("chats")
            .select("id,title,created_at,updated_at,chat_turns(song_name,created_at)")
            .eq("user_id", _account_id(user_id))
            .order("updated_at", desc=True)
            .execute()
        )
        chats = []
        for row in result.data or []:
            turns = row.pop("chat_turns", None) or []
            turns.sort(key=lambda item: item.get("created_at") or "")
            title = row.get("title") or "New chat"
            for turn in reversed(turns):
                if (turn.get("song_name") or "").strip():
                    title = turn["song_name"].strip()
                    break
            row["title"] = title
            chats.append(row)
        return chats

    def get_chat(self, user_id: str, chat_id: str) -> dict[str, Any] | None:
        chat_result = (
            self.client.table("chats")
            .select("id,title,created_at,updated_at")
            .eq("id", chat_id)
            .eq("user_id", _account_id(user_id))
            .limit(1)
            .execute()
        )
        rows = chat_result.data or []
        if not rows:
            return None
        turns = (
            self.client.table("chat_turns")
            .select("*")
            .eq("chat_id", chat_id)
            .order("created_at")
            .execute()
        )
        chat = rows[0]
        chat["turns"] = turns.data or []
        return chat

    def create_chat(self, user_id: str, title: str) -> dict[str, Any]:
        result = (
            self.client.table("chats")
            .insert({"title": title or "New chat", "user_id": _account_id(user_id)})
            .execute()
        )
        rows = result.data or []
        if not rows:
            raise RuntimeError("Supabase did not return the new chat")
        return rows[0]

    def add_turn(self, user_id: str, chat_id: str, turn: dict[str, Any]) -> dict[str, Any]:
        if not self.get_chat(user_id, chat_id):
            raise KeyError("Chat not found")
        payload = {
            "chat_id": chat_id,
            "prompt": turn.get("prompt") or "",
            "song_name": turn.get("song_name"),
            "composer": turn.get("composer"),
            "raga": turn.get("raga"),
            "tala": turn.get("tala"),
            "tala_beats": turn.get("tala_beats"),
            "youtube_url": turn.get("youtube_url"),
            "description": turn.get("description"),
            "filename": turn.get("filename"),
            "audio_url": turn.get("audio_url"),
            "audio_path": turn.get("audio_path"),
            "dance_url": turn.get("dance_url"),
        }
        result = self.client.table("chat_turns").insert(payload).execute()
        rows = result.data or []
        if not rows:
            raise RuntimeError("Supabase did not return the saved turn")
        saved = rows[0]
        self.client.table("chats").update({
            "title": _song_title(saved),
            "updated_at": _now(),
        }).eq("id", chat_id).eq("user_id", _account_id(user_id)).execute()
        return saved

    def delete_chat(self, user_id: str, chat_id: str) -> bool:
        existing = (
            self.client.table("chats")
            .select("id")
            .eq("id", chat_id)
            .eq("user_id", _account_id(user_id))
            .limit(1)
            .execute()
        )
        if not existing.data:
            return False
        self.client.table("chats").delete().eq("id", chat_id).eq("user_id", _account_id(user_id)).execute()
        return True


class AccountChatStore:
    """Chats for one Google account. Uses Supabase once chats.user_id exists."""

    def __init__(self, remote: SupabaseChatStore | None, storage_dir: Path):
        self.remote = remote
        self.local = LocalChatStore(storage_dir)
        self.storage_dir = Path(storage_dir)
        self._ready = False
        self._ready_until = 0.0

    def remote_ready(self) -> bool:
        if self.remote is None:
            return False
        now = time.time()
        if now < self._ready_until:
            return self._ready
        try:
            self.remote.client.table("chats").select("user_id").limit(1).execute()
            self._ready = True
        except Exception:
            self._ready = False
        self._ready_until = now + 15
        return self._ready

    def _import_local(self, user_id: str) -> None:
        if self.remote is None:
            return
        path = self.storage_dir / _account_id(user_id) / "chats.json"
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        chats = data.get("chats") if isinstance(data, dict) else None
        if not isinstance(chats, list):
            return
        if not chats:
            path.rename(path.with_name("chats.json.imported"))
            return
        remaining = []
        for chat in chats:
            try:
                created = self.remote.create_chat(user_id, chat.get("title") or "New chat")
                for turn in chat.get("turns") or []:
                    self.remote.add_turn(user_id, created["id"], turn)
            except Exception:
                remaining.append(chat)
        if remaining:
            path.write_text(json.dumps({"chats": remaining}, indent=2), encoding="utf-8")
            return
        path.rename(path.with_name("chats.json.imported"))

    def list_chats(self, user_id: str) -> list[dict[str, Any]]:
        if self.remote_ready():
            self._import_local(user_id)
            return self.remote.list_chats(user_id)
        return self.local.list_chats(user_id)

    def get_chat(self, user_id: str, chat_id: str) -> dict[str, Any] | None:
        if self.remote_ready():
            return self.remote.get_chat(user_id, chat_id)
        return self.local.get_chat(user_id, chat_id)

    def create_chat(self, user_id: str, title: str) -> dict[str, Any]:
        if self.remote_ready():
            return self.remote.create_chat(user_id, title)
        return self.local.create_chat(user_id, title)

    def add_turn(self, user_id: str, chat_id: str, turn: dict[str, Any]) -> dict[str, Any]:
        if self.remote_ready():
            return self.remote.add_turn(user_id, chat_id, turn)
        return self.local.add_turn(user_id, chat_id, turn)

    def delete_chat(self, user_id: str, chat_id: str) -> bool:
        if self.remote_ready():
            return self.remote.delete_chat(user_id, chat_id)
        return self.local.delete_chat(user_id, chat_id)


def build_chat_store(storage_dir: Path) -> tuple[Any, str, str]:
    """Return (store, backend name, status message)."""
    url = os.environ.get("SUPABASE_URL", "").strip().strip('"').strip("'")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip().strip('"').strip("'")
    if key.startswith("sb_publishable_"):
        return (
            AccountChatStore(None, storage_dir),
            "misconfigured",
            "SUPABASE_SERVICE_ROLE_KEY is the publishable key. In Supabase, open Project Settings → API Keys and copy the secret key instead, then restart the app.",
        )
    if url and key:
        return (
            AccountChatStore(SupabaseChatStore(url, key), storage_dir),
            "supabase",
            "Chats are saved in Supabase for this Google account.",
        )
    if url or key:
        return (
            AccountChatStore(None, storage_dir),
            "misconfigured",
            "Add both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to .env.",
        )
    return (
        AccountChatStore(None, storage_dir),
        "local",
        "Supabase is not connected yet. Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to .env.",
    )
