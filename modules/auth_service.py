"""Google sign-in through Supabase Auth, and one downloads folder per account."""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


class _MemoryStorage:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get_item(self, key: str) -> str | None:
        return self.data.get(key)

    def set_item(self, key: str, value: str) -> None:
        self.data[key] = value

    def remove_item(self, key: str) -> None:
        self.data.pop(key, None)


def supabase_credentials() -> tuple[str, str]:
    url = os.environ.get("SUPABASE_URL", "").strip().strip('"').strip("'").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip().strip('"').strip("'")
    return url, key


def supabase_configured() -> bool:
    url, key = supabase_credentials()
    return bool(url and key and not key.startswith("sb_publishable_"))


def account_id(user_id: str) -> str:
    try:
        return str(uuid.UUID(str(user_id)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid account") from exc


def account_folder(downloads: Path, user_id: str) -> Path:
    folder = (Path(downloads) / account_id(user_id)).resolve()
    if not folder.is_relative_to(Path(downloads).resolve()):
        raise ValueError("Invalid account")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_profile_file(folder: Path, profile: dict[str, Any]) -> None:
    payload = {
        "id": profile.get("id"),
        "email": profile.get("email") or "",
        "full_name": profile.get("full_name") or "",
        "avatar_url": profile.get("avatar_url") or "",
    }
    (folder / "profile.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _client(storage: _MemoryStorage | None = None):
    from supabase import create_client
    from supabase.lib.client_options import SyncClientOptions

    url, key = supabase_credentials()
    if not url or not key:
        raise RuntimeError("Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to .env, then restart the app.")
    options = SyncClientOptions(
        storage=storage or _MemoryStorage(),
        flow_type="pkce",
        persist_session=False,
        auto_refresh_token=False,
    )
    return create_client(url, key, options=options)


def profile_from_user(user: Any) -> dict[str, str]:
    meta = getattr(user, "user_metadata", None) or {}
    if not isinstance(meta, dict):
        try:
            meta = dict(meta)
        except Exception:
            meta = {}
    email = getattr(user, "email", None) or ""
    name = meta.get("full_name") or meta.get("name") or email or "Google account"
    avatar = meta.get("avatar_url") or meta.get("picture") or ""
    return {
        "id": account_id(str(user.id)),
        "email": email,
        "full_name": str(name),
        "avatar_url": str(avatar),
    }


def start_google_sign_in(redirect_to: str) -> tuple[str, str]:
    storage = _MemoryStorage()
    client = _client(storage)
    response = client.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {"redirect_to": redirect_to},
    })
    verifier = next((value for value in storage.data.values() if value), "")
    if not response.url or not verifier:
        raise RuntimeError("Could not start Google sign-in.")
    return response.url, verifier


def finish_google_sign_in(code: str, verifier: str) -> dict[str, str]:
    client = _client()
    result = client.auth.exchange_code_for_session({
        "auth_code": code,
        "code_verifier": verifier,
    })
    user = result.user
    if user is None:
        raise RuntimeError("Google did not return an account.")
    profile = profile_from_user(user)
    try:
        client.table("profiles").upsert({
            "id": profile["id"],
            "email": profile["email"],
            "full_name": profile["full_name"],
            "avatar_url": profile["avatar_url"],
        }).execute()
    except Exception:
        pass
    return profile


_provider_cache: dict[str, tuple[float, bool, str]] = {}


def google_provider_status(redirect_to: str) -> tuple[bool, str]:
    """Return whether Supabase will send the browser to Google."""
    url, key = supabase_credentials()
    if not url or not key:
        return False, "Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to .env, then restart the app."
    if key.startswith("sb_publishable_"):
        return False, "SUPABASE_SERVICE_ROLE_KEY is the publishable key. Copy the secret key from Supabase Project Settings → API Keys."
    cached = _provider_cache.get(redirect_to)
    if cached and cached[0] > time.time():
        return cached[1], cached[2]
    probe = f"{url}/auth/v1/authorize?provider=google&redirect_to={quote(redirect_to, safe='')}"
    try:
        response = httpx.get(probe, follow_redirects=False, timeout=10)
    except Exception:
        return False, "Could not reach Supabase to start Google sign-in."
    if response.status_code in {301, 302, 303, 307, 308}:
        _provider_cache[redirect_to] = (time.time() + 30, True, "")
        return True, ""
    message = "Google sign-in is not available on this Supabase project."
    try:
        payload = response.json()
    except Exception:
        payload = {}
    detail = str(payload.get("msg") or payload.get("message") or "")
    if "not enabled" in detail.lower() or payload.get("error_code") == "validation_failed":
        message = (
            "Google sign-in is not enabled for this Supabase project yet. "
            "Open Authentication → Sign In / Providers → Google and turn it on. "
            f"In Google Cloud, set the redirect URI to {url}/auth/v1/callback. "
            f"In Supabase URL configuration, allow {redirect_to}."
        )
    elif detail:
        message = detail
    _provider_cache[redirect_to] = (time.time() + 15, False, message)
    return False, message
