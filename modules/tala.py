"""
modules/tala.py - beats per cycle for a tala name.

The recommender returns a tala name and a beat count, and the count is often wrong
("Khanda Chapu (7 beats)", "Ata (4 beats)"). The name is more reliable, so a known
name wins over the count it came with.
"""

import re
from typing import Any, Optional

# Checked in order: the more specific names come before the ones they contain.
_TALAS = [
    ("misrachapu", 7),
    ("khandachapu kandachapu khantachapu kantachapu", 5),
    ("tisrachapu", 3),
    ("sankeernachapu sankirnachapu", 9),
    ("ata", 14),
    ("jhampa", 10),
    ("tisratriputa", 7),
    ("khandatriputa", 9),
    ("matya", 10),
    ("dhruva", 14),
    ("rupaka rupakam", 6),
    ("desadi madhyadi", 4),
    ("adi chatusratriputa chatusrajatitriputa", 8),
    ("eka", 4),
    # Hindustani
    ("teental tintal trital", 16),
    ("ektal", 12),
    ("chautal chartal", 12),
    ("jhaptal", 10),
    ("rupak", 7),
    ("dadra", 6),
    ("keherwa kaharwa kehrwa", 8),
    ("deepchandi dipchandi jhoomra jhumra dhamar", 14),
]


def _key(name: str) -> str:
    key = re.sub(r"\(.*?\)", "", name.lower())
    key = re.sub(r"[^a-z]", "", key)
    key = key.replace("chappu", "chapu").replace("kh", "k").replace("th", "t").replace("dh", "d")
    key = re.sub(r"(.)\1+", r"\1", key)
    for suffix in ("talam", "tala", "tal"):
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)]
    return key


_PATTERNS = [([_key(w) for w in words.split()], beats) for words, beats in _TALAS]


def tala_beats(name: Optional[str], fallback: Any = None, default: int = 8) -> int:
    """Beats per cycle for `name`; the recommender's count only when the name is unknown."""
    key = _key(name) if name else ""
    if key:
        for words, beats in _PATTERNS:
            if any(key == w or (len(w) > 3 and w in key) or key.endswith(w) for w in words):
                return beats
    try:
        n = int(fallback)
    except (TypeError, ValueError):
        return default
    return n if 2 <= n <= 32 else default
