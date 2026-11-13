"""
modules/llm_service.py - Prompts Groq with browser search to find an Indian
classical composition and a public YouTube link.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import dotenv

dotenv.load_dotenv(Path(__file__).resolve().parents[1] / ".env")

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"

SYSTEM_INSTRUCTION = """\
You are an expert musicologist and accompanist in Indian Classical Music (Carnatic and Hindustani) and Bharatanatyam dance (Nritta).
Given a user prompt describing a song, dance style, raga, tala, tempo, mood, or deity:
1. Recommend an authentic, well-suited Indian classical composition.
2. Use browser search to find an authentic, publicly available YouTube video for this composition.
3. Determine the traditional Tala and estimated beat cycle length (e.g., Adi Tala = 8 beats, Rupaka Tala = 3 or 6 beats, Misra Chapu = 7 beats).

Remember that only the first 2 minutes of the song is used for the dance. Please conisder this while recommending the song.
Return your response strictly as a valid JSON object matching this structure:   
{
  "song_name": "<Title of the composition>",
  "composer": "<Composer / Vaggeyakara or artist>",
  "raga": "<Raga name>",
  "tala": "<Tala name and beat count, e.g. Adi Tala (8 beats)>",
  "tala_beats": 8,
  "youtube_url": "<Full valid YouTube URL, e.g. https://www.youtube.com/watch?v=...>",
  "search_query": "<Precise YouTube search keywords to locate this exact recording if URL fails>",
  "description": "<Brief 1-2 sentence explanation of why this song fits the dance prompt>"
}
Do not include any conversational preamble or markdown code blocks if possible; just the raw JSON object.
"""


def log(*args: Any) -> None:
    print(*args, file=sys.stderr)


def extract_json(text: str) -> Dict[str, Any]:
    """Robustly extract a JSON object from LLM response text."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def _groq_chat(api_key: str, model: str, user_prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {
                "role": "user",
                "content": (
                    f"User request: {user_prompt}\n\n"
                    "Search for a real public YouTube recording, then return the composition, "
                    "tala, and YouTube link in the specified JSON format."
                ),
            },
        ],
        "temperature": 0.2,
        "max_completion_tokens": 2048,
        "reasoning_effort": "low",
        "tool_choice": "required",
        "tools": [{"type": "browser_search"}],
    }
    request = urllib.request.Request(
        GROQ_CHAT_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "groq-python/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        raise RuntimeError(f"Groq request failed ({exc.code}): {detail}") from exc

    message = body["choices"][0]["message"]
    content = (message.get("content") or "").strip()
    if not content:
        raise RuntimeError("No response received from Groq.")
    return content


def recommend_song(
    user_prompt: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Ask Groq, with browser search, for a song recommendation and YouTube link."""
    resolved_key = api_key or os.environ.get("GROQ_API_KEY")
    if not resolved_key:
        raise ValueError(
            "Groq API key not found. Please configure GROQ_API_KEY in your environment or a .env file."
        )

    log(f"[LLM Service] Consulting Groq ({model}) with browser search for: '{user_prompt}'")
    text = _groq_chat(resolved_key, model, user_prompt)
    return extract_json(text)
