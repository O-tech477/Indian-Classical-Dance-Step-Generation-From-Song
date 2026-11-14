"""Timestamped, Carnatic-aware semantic mudhra suggestions.

CultureMERT supplies Carnatic-adapted audio representations for change-aware
temporal smoothing and section boundaries. CLAP ranks semantic descriptions
for all mudhras. The JSON file remains the number/name source of truth.
"""

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

# Keep Hugging Face model downloads and Transformers initialization quiet in
# the app's terminal; prediction results are returned as JSON instead.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import librosa
import numpy as np

MODEL_NAME = "laion/clap-htsat-unfused"
CARNATIC_MODEL_NAME = "ntua-slp/CultureMERT-95M"
SAMPLE_RATE = 48_000
WINDOW_SECONDS = 10.0
HOP_SECONDS = 5.0
CARNATIC_CONTEXT_SECONDS = 5.0
MIN_SECTION_SIMILARITY = 0.72
NEIGHBOR_SCORE_WEIGHT = 0.20
KEYS_PATH = Path(__file__).with_name("mudhra_keys.json")

# Short viniyoga/theme summaries, grouped around the common meanings of these
# hastas. Multiple meanings are retained because one hasta can depict many ideas.
MUDHRA_THEMES = {
    "Pataka": "blessing, sky, cloud, rain, forest, river, stillness, or a firm stop",
    "Tripataka": "invocation, crown, tree, lightning, flame, arrow, or divine power",
    "Ardhapataka": "leaves, riverbank, separation, a flag, or a tender parting",
    "Kartarimukha": "separation, lightning, a falling object, opposition, or a sharp turn",
    "Mayura": "peacock, beauty, a delicate plume, adornment, or graceful nature",
    "Ardhachandra": "crescent moon, gentle blessing, calmness, an offering, or open sky",
    "Arala": "a sip, a delicate curve, wonder, or a subtle expressive gesture",
    "Shukatunda": "a parrot beak, an arrow, a pointed message, or focused intention",
    "Mushthi": "strength, resolve, holding, struggle, courage, or forceful energy",
    "Shikhara": "a bell, bow, vow, pillar, Shiva, or steadfast devotion",
    "Kapittha": "Lakshmi, Saraswati, worship, holding a sacred object, or reverence",
    "Katakamukha": "plucking flowers, offering a garland, adornment, or delicate grace",
    "Suchi": "one, pointing, a direction, a place, a question, or clear emphasis",
    "Chandrakala": "the moon, crescent light, night, serenity, or celestial beauty",
    "Padmakosha": "lotus bud, fruit, a rounded blossom, abundance, or nature",
    "Sarpashirsha": "serpent, water, sprinkling, flowing movement, or sacred nature",
    "Mrigashirsha": "deer, a gentle face, a beloved, playfulness, or tender longing",
    "Simhamukha": "lion, courage, majesty, fierce energy, or a powerful deity",
    "Kangula": "a small bell, fruit, a bud, delicacy, or gentle ornament",
    "Alapadma": "fully opened lotus, beauty, blossoming, moonlight, or joyful nature",
    "Chatura": "cleverness, subtlety, a small amount, distinction, or playful wit",
    "Bhramara": "bee, flower, attraction, humming, love, or lively nature",
    "Hamsasya": "swan, picking a pearl, fine detail, purity, or delicate grace",
    "Hamsapaksha": "swan wing, bridge, movement across, shelter, or expansive grace",
    "Sandamsha": "offering, grasping, a flower, ritual action, or focused devotion",
    "Mukula": "flower bud, eating, a seed, potential, or a compact offering",
    "Tamrachuda": "rooster, dawn, calling, alertness, or the start of a new day",
    "Trishula": "trident, three, Shiva, protection, power, or sacred strength",
    "Anjali": "salutation, prayer, worship, respect, surrender, or devotional offering",
    "Kapota": "respectful greeting, humility, gentle speech, or reverent listening",
    "Karkata": "gathering, togetherness, a crowd, branches, or opening outward",
    "Swastika": "auspiciousness, protection, crossed movement, or sacred completeness",
    "Dola": "relaxed hands, ease, a gentle sway, rest, or a calm transition",
    "Pushpaputa": "offering flowers, receiving a blessing, worship, or a sacred gift",
    "Utsanga": "embrace, affection, intimacy, comfort, or loving protection",
    "Shivalinga": "Shiva, linga worship, devotion, stillness, or sacred presence",
    "Katakavardhana": "coronation, auspicious ceremony, marriage, garland, or celebration",
    "Kartariswastika": "branches, crossed paths, separation, a bird, or a dramatic crossing",
    "Shakata": "cart, demon, obstacle, forceful action, or dramatic opposition",
    "Shankha": "conch, invocation, sacred sound, Vishnu, or ceremonial beginning",
    "Chakra": "wheel, Vishnu, cosmic order, protection, or circular movement",
    "Samputa": "container, secret, keeping something safe, or a concealed treasure",
    "Pasha": "bond, attachment, embrace, connection, or emotional entanglement",
    "Kilaka": "intimate affection, private bond, friendship, or loving connection",
    "Matsya": "fish, water, flowing movement, Vishnu, or aquatic life",
    "Kurma": "tortoise, patience, steadiness, support, or Vishnu",
    "Varaha": "boar, earth, rescue, protection, or Vishnu",
    "Garuda": "Garuda, flight, speed, courage, Vishnu, or triumphant movement",
    "Nagabandha": "serpents entwined, union, embrace, mystery, or sacred power",
    "Khatva": "a cot, ascetic life, rest, renunciation, or a ritual object",
    "Bherunda": "paired birds, duality, union, flight, or majestic power",
    "Avahittha": "hands held against the heart, modesty, shyness, concealed emotion, or guarded affection",
}


def _load_keys() -> dict[int, str]:
    raw = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
    keys = {int(number): name for number, name in raw.items()}
    if set(keys) != set(range(1, 53)):
        raise ValueError(f"{KEYS_PATH} must contain mudhra numbers 1 through 52")
    missing = set(keys.values()) - set(MUDHRA_THEMES)
    if missing:
        raise ValueError(f"Missing mudhra semantic descriptions: {', '.join(sorted(missing))}")
    return keys


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from transformers import ClapModel, ClapProcessor
        from transformers.utils import logging as transformers_logging
        from huggingface_hub.utils import disable_progress_bars
    except ImportError as exc:
        raise RuntimeError("Mudhra prediction needs torch and transformers installed.") from exc

    disable_progress_bars()
    transformers_logging.set_verbosity_error()
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME)
    model.eval()
    return torch, F, processor, model


@lru_cache(maxsize=1)
def _load_carnatic_model() -> tuple[Any, Any, Any, Any]:
    """Load Carnatic-adapted CultureMERT as a music representation encoder."""
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoFeatureExtractor, AutoModel
        from transformers.utils import logging as transformers_logging
        from huggingface_hub.utils import disable_progress_bars
    except (ImportError, OSError, RuntimeError):
        # Preserve the semantic predictor when optional model weights cannot
        # be downloaded or loaded (for example, in an offline run).
        return None, None, None, None

    disable_progress_bars()
    transformers_logging.set_verbosity_error()
    try:
        processor = AutoFeatureExtractor.from_pretrained(CARNATIC_MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(CARNATIC_MODEL_NAME, trust_remote_code=True)
        model.eval()
        return torch, F, processor, model
    except (OSError, RuntimeError):
        return None, None, None, None


def _fallback_audio_embedding(waveform: np.ndarray) -> np.ndarray:
    """Compute a small acoustic descriptor if CultureMERT is unavailable."""
    chroma = librosa.feature.chroma_stft(y=waveform, sr=SAMPLE_RATE, n_fft=4096)
    energy = librosa.feature.rms(y=waveform)[0]
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=SAMPLE_RATE)[0]
    features = np.concatenate((chroma.mean(axis=1), [energy.mean(), centroid.mean() / SAMPLE_RATE]))
    return (features / (np.linalg.norm(features) + 1e-8)).astype(np.float32)


@lru_cache(maxsize=16)
def _text_features(song_name: str) -> tuple[Any, list[tuple[int, str]]]:
    torch, F, processor, model = _load_model()
    keys = sorted(_load_keys().items())
    captions = []
    for number, name in keys:
        theme = MUDHRA_THEMES[name]
        title = f" The Carnatic song is titled {song_name}." if song_name else ""
        captions.append(
            f"A Carnatic classical music passage with a mood or expressive meaning of "
            f"{theme}. This music suits the Bharatanatyam hasta {name}.{title}"
        )
    encoded = processor(text=captions, return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        output = model.get_text_features(**encoded)
        features = _embedding_tensor(output)
        features = F.normalize(features, dim=-1)
    return features, keys


def _analyze_window(
    audio_path: Path, at_seconds: float, song_name: str = "", window_seconds: float = WINDOW_SECONDS,
) -> tuple[dict[str, Any], np.ndarray]:
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    duration = float(librosa.get_duration(path=str(audio_path)))
    center = min(max(0.0, float(at_seconds)), duration)
    start = min(max(0.0, center - window_seconds / 2), max(0.0, duration - window_seconds))
    waveform, _ = librosa.load(
        str(audio_path), sr=SAMPLE_RATE, mono=True, offset=start, duration=window_seconds,
    )
    expected = int(window_seconds * SAMPLE_RATE)
    waveform = np.pad(waveform[:expected], (0, max(0, expected - len(waveform))))

    # CultureMERT uses a 5-second context at 24 kHz. Its representation is
    # used for musical continuity/change detection, not as a text classifier.
    culture_torch, culture_F, culture_processor, culture_model = _load_carnatic_model()
    if culture_model is not None:
        try:
            culture_sample_rate = int(culture_processor.sampling_rate)
            context_samples = int(CARNATIC_CONTEXT_SECONDS * SAMPLE_RATE)
            context_start = max(0, (len(waveform) - context_samples) // 2)
            culture_waveform = waveform[context_start:context_start + context_samples]
            culture_waveform = np.pad(
                culture_waveform, (0, max(0, context_samples - len(culture_waveform))),
            )
            if culture_sample_rate != SAMPLE_RATE:
                culture_waveform = librosa.resample(
                    culture_waveform, orig_sr=SAMPLE_RATE, target_sr=culture_sample_rate,
                )
            culture_inputs = culture_processor(
                culture_waveform, sampling_rate=culture_sample_rate, return_tensors="pt",
            )
            with culture_torch.inference_mode():
                culture_output = culture_model(**culture_inputs, output_hidden_states=True)
                hidden_states = getattr(culture_output, "hidden_states", None)
                culture_features = hidden_states[-1] if hidden_states else getattr(
                    culture_output, "last_hidden_state", None,
                )
                if culture_features is None:
                    raise RuntimeError("CultureMERT did not return audio representations.")
                culture_embedding = culture_F.normalize(culture_features.mean(dim=1), dim=-1)[0]
            culture_embedding = culture_embedding.detach().cpu().numpy().astype(np.float32)
            culture_encoder_name = CARNATIC_MODEL_NAME
        except (OSError, RuntimeError, ValueError):
            culture_embedding = _fallback_audio_embedding(waveform)
            culture_encoder_name = "librosa acoustic-feature fallback"
    else:
        culture_embedding = _fallback_audio_embedding(waveform)
        culture_encoder_name = "librosa acoustic-feature fallback"

    torch, F, processor, model = _load_model()
    audio_inputs = processor(audio=waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    text_features, keys = _text_features(song_name.strip())
    with torch.inference_mode():
        output = model.get_audio_features(**audio_inputs)
        audio_features = _embedding_tensor(output)
        audio_features = F.normalize(audio_features, dim=-1)
        similarities = (text_features @ audio_features[0].unsqueeze(-1)).squeeze(-1)
    score_values = similarities.cpu().tolist()
    ranked = sorted(
        ({"number": number, "name": name, "score": round(float(score), 4)}
         for (number, name), score in zip(keys, score_values)),
        key=lambda item: item["score"],
        reverse=True,
    )
    winner = ranked[0]
    return ({
        "mudhra": winner,
        "alternatives": ranked[1:4],
        "window_start": round(start, 2),
        "window_end": round(min(duration, start + window_seconds), 2),
        "semantic_model": MODEL_NAME,
        "carnatic_audio_model": culture_encoder_name,
        "_all_scores": score_values,
    }, culture_embedding)


def _window_scores(
    audio_path: Path, at_seconds: float, song_name: str = "", window_seconds: float = WINDOW_SECONDS,
) -> dict[str, Any]:
    result, _ = _analyze_window(audio_path, at_seconds, song_name, window_seconds)
    result.pop("_all_scores", None)
    return result


def _embedding_tensor(output: Any) -> Any:
    """Unwrap CLAP embeddings across Transformers tensor/output-object APIs."""
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "audio_embeds") and output.audio_embeds is not None:
        return output.audio_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def predict_mudhra_at(
    audio_path: Path, at_seconds: float, song_name: str = "", window_seconds: float = WINDOW_SECONDS,
) -> dict[str, Any]:
    """Return a ranked mudhra suggestion for the audio around a playback time."""
    return _window_scores(audio_path, at_seconds, song_name, window_seconds)


def predict_mudhra_timeline(
    audio_path: Path,
    song_name: str = "",
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> dict[str, Any]:
    """Analyze the whole song and coalesce adjacent windows with the same winner."""
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    duration = float(librosa.get_duration(path=str(audio_path)))
    centers = np.arange(0, max(duration, 0.01), hop_seconds)
    analyzed = [_analyze_window(audio_path, float(t), song_name, window_seconds) for t in centers]
    raw = [item[0] for item in analyzed]
    embeddings = [item[1] for item in analyzed]
    semantic_scores = [np.asarray(item["_all_scores"], dtype=np.float32) for item in raw]
    # Smooth only across adjacent windows that remain similar in the
    # Carnatic-adapted representation; stronger changes stay responsive.
    for index, result in enumerate(raw):
        current = semantic_scores[index]
        weighted = current.copy()
        total_weight = 1.0
        for neighbor_index in (index - 1, index + 1):
            if not 0 <= neighbor_index < len(raw):
                continue
            similarity = float(np.dot(embeddings[index], embeddings[neighbor_index]))
            if similarity >= MIN_SECTION_SIMILARITY:
                weighted += NEIGHBOR_SCORE_WEIGHT * np.asarray(
                    semantic_scores[neighbor_index], dtype=np.float32,
                )
                total_weight += NEIGHBOR_SCORE_WEIGHT
        weighted /= total_weight
        keys = sorted(_load_keys().items())
        ranked = sorted(
            ({"number": number, "name": name, "score": round(float(score), 4)}
             for (number, name), score in zip(keys, weighted.tolist())),
            key=lambda item: item["score"], reverse=True,
        )
        result["mudhra"] = ranked[0]
        result["alternatives"] = ranked[1:4]
        result["_all_scores"] = weighted.tolist()
    spans: list[dict[str, Any]] = []
    for index, result in enumerate(raw):
        mudhra = result["mudhra"]
        start = 0.0 if index == 0 else float((centers[index - 1] + centers[index]) / 2)
        end = duration if index == len(centers) - 1 else float((centers[index] + centers[index + 1]) / 2)
        similarity_to_previous = (
            None if index == 0
            else round(float(np.dot(embeddings[index - 1], embeddings[index])), 4)
        )
        if (spans and spans[-1]["number"] == mudhra["number"]
                and similarity_to_previous is not None
                and similarity_to_previous >= MIN_SECTION_SIMILARITY):
            spans[-1]["end"] = round(end, 2)
            spans[-1]["score"] = round((spans[-1]["score"] + mudhra["score"]) / 2, 4)
        else:
            spans.append({
                "start": round(start, 2), "end": round(end, 2),
                "number": mudhra["number"], "name": mudhra["name"], "score": mudhra["score"],
                "music_similarity_to_previous": similarity_to_previous,
            })
    return {
        "song_name": song_name,
        "duration": round(duration, 2),
        "semantic_model": MODEL_NAME,
        "carnatic_audio_model": raw[0]["carnatic_audio_model"],
        "selection_basis": (
            "CLAP semantic ranking across all mudhra descriptions; CultureMERT similarity "
            "controls temporal smoothing and music-change boundaries."
            if raw[0]["carnatic_audio_model"] == CARNATIC_MODEL_NAME
            else "CLAP semantic ranking across all mudhra descriptions; librosa acoustic similarity "
            "controls temporal smoothing and music-change boundaries because CultureMERT was unavailable."
        ),
        "segments": spans,
    }


def write_mudhra_timeline(
    audio_path: Path, song_name: str = "", output_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    """Predict a full song and save it to downloads/mudhra_timeline.json by default."""
    audio_path = Path(audio_path)
    result = predict_mudhra_timeline(audio_path, song_name)
    destination = (
        Path(output_path)
        if output_path
        else Path(__file__).resolve().parents[1] / "downloads" / "mudhra_timeline.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result, destination


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict timestamped Carnatic-music mudhras")
    parser.add_argument("audio", type=Path, help="MP3 or other librosa-readable audio file")
    parser.add_argument("--song-name", default="", help="Song title used as additional text context")
    parser.add_argument("--out", type=Path, help="Write full timestamped predictions to this JSON file")
    parser.add_argument("--at", type=float, help="Predict only around this timestamp (seconds)")
    args = parser.parse_args()
    if args.at is not None:
        result = predict_mudhra_at(args.audio, args.at, args.song_name)
        rendered = json.dumps(result, indent=2, ensure_ascii=False)
        if args.out:
            args.out.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
    else:
        result, destination = write_mudhra_timeline(args.audio, args.song_name, args.out)
        print(f"Saved mudhra timeline: {destination}")
