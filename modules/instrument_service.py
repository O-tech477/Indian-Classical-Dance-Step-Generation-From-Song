"""Continuous instrument tagging for Indian classical music playback.

Uses the multi-label AudioSet AST model. Instrument tags are mapped to names
familiar in Carnatic/Hindustani music where the source label permits it.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 10.0
MIN_SCORE = 0.08

# AudioSet has broad categories for some instruments, rather than separate
# labels for every Indian instrument.
INSTRUMENT_LABELS = {
    "tabla": "Tabla",
    "sitar": "Sitar",
    "violin": "Violin",
    "flute": "Flute / venu / bansuri",
    "singing": "Carnatic / Indian classical vocals",
    "vocal music": "Carnatic / Indian classical vocals",
    "drone": "Tanpura / tambura / shruti box",
    "drum": "Mridangam / Indian hand percussion",
    "drum kit": "Mridangam / Indian hand percussion",
    "percussion": "Mridangam / ghatam / kanjira (percussion)",
    "cymbal": "Nattuvangam / cymbals",
    "gong": "Nattuvangam / cymbals",
    "harmonium": "Harmonium",
    "accordion": "Harmonium / reed instrument",
    "string instrument": "Veena / Indian string instrument",
    "plucked string instrument": "Veena / Indian string instrument",
    "harp": "Veena / Indian string instrument",
    "zither": "Veena / Indian string instrument",
}


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, Any]:
    """Load and cache model on first request, so the Flask app starts quickly."""
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    except ImportError as exc:
        raise RuntimeError(
            "Instrument recognition dependencies are missing. Install the project dependencies."
        ) from exc

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return torch, feature_extractor, model


def _indian_name(label: str) -> str | None:
    normalized = label.strip().lower()
    if normalized in INSTRUMENT_LABELS:
        return INSTRUMENT_LABELS[normalized]
    for key, name in INSTRUMENT_LABELS.items():
        if key in ("singing", "vocal music"):
            continue
        if key in normalized:
            return name
    return None


def predict_instruments(audio_path: Path, at_seconds: float) -> dict[str, Any]:
    """Predict audible instrument tags around a playback position."""
    audio_path = Path(audio_path)
    duration = librosa.get_duration(path=str(audio_path))
    center = min(max(0.0, float(at_seconds)), max(0.0, duration))
    start = min(max(0.0, center - WINDOW_SECONDS / 2), max(0.0, duration - WINDOW_SECONDS))
    waveform, _ = librosa.load(
        str(audio_path), sr=SAMPLE_RATE, mono=True,
        offset=start, duration=WINDOW_SECONDS,
    )
    expected = int(WINDOW_SECONDS * SAMPLE_RATE)
    if len(waveform) < expected:
        waveform = np.pad(waveform, (0, expected - len(waveform)))
    else:
        waveform = waveform[:expected]

    torch, feature_extractor, model = _load_model()
    inputs = feature_extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        scores = torch.sigmoid(logits).cpu().numpy()

    detected: dict[str, float] = {}
    for index, score in enumerate(scores):
        if float(score) < MIN_SCORE:
            continue
        label = model.config.id2label.get(index, "")
        name = _indian_name(label)
        if name:
            detected[name] = max(detected.get(name, 0.0), float(score))

    instruments = [
        {"name": name, "confidence": round(score, 2)}
        for name, score in sorted(detected.items(), key=lambda item: item[1], reverse=True)
    ]
    return {
        "instruments": instruments,
        "window_start": round(start, 2),
        "window_end": round(min(duration, start + WINDOW_SECONDS), 2),
        "model": MODEL_NAME,
    }
