## Abstract
We attempt to map Bharatanatyam steps to any Indian classical song on the basis of nritta only i.e. rhythm and tempo. We use librosa and Demucs to analyse the song and convert it into chunks with beat information and then map the adavus to the beats.

## Architecture & Modular Components
- **`app.py`**: Flask web server hosting the curation UI and music storage APIs.
- **`modules/`**:
  - `llm_service.py`: Google Gemini cloud LLM interface with Google Search grounding to discover Indian classical songs, tala, and YouTube links.
  - `audio_service.py`: YouTube extraction and MP3 audio download via `yt-dlp`.
  - `storage_service.py`: Local music catalog and metadata tracker (`downloads/library.json`).
- **`cloud_llm.py`**: CLI facade for LLM-based song recommendation and audio downloading.
- **`pipeline.py`**: End-to-end choreography generator (song analysis -> adavu planning -> interactive stick-figure HTML player).

## How to Initialize and Run

### 1. Environment Setup
Sync the dependencies with `uv`:
```bash
uv sync
```
Set your Google Gemini API key:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

### 2. Run the Web UI (Flask)
Start the web app:
```bash
uv run app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser. You can enter prompts, preview recommended songs, stream the audio, and view your stored music library.

### 3. CLI Music Curation & Download
```bash
uv run cloud_llm.py "Energetic Adi tala Carnatic song for Thattadavu" -o downloads
```

### 4. Run the Dance Choreography Pipeline
Generate the stick-figure player from any downloaded song:
```bash
uv run pipeline.py downloads/<your-song>.mp3 --adavus landmarks_json -o output --tala 8 --default-beats 4
```
Double-click `output/index.html` to open the stick-figure player in your browser.
