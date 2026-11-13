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

### Supabase chats
Prompts, the song they produced, and the dance link are stored as chats. Audio files and the stick-figure player stay in `downloads/` and `output/` on this computer. Until Supabase is configured, those chats are written to `downloads/chats.json`.

In the Supabase dashboard:

1. Create a project.
2. Open **SQL Editor**, paste `supabase/schema.sql`, and run it. That creates `chats` and `chat_turns`, and turns on row level security so the public anon key cannot read them.
3. Open **Project Settings → API** and copy the project URL and the **service_role** key. The service role key bypasses row level security. Keep it on the server only. Do not put it in the browser, and do not commit it.
4. Add both values to `.env` in this folder:

```bash
SUPABASE_URL="https://YOUR-PROJECT.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
```

Restart `uv run app.py`. The sidebar note disappears once the app is talking to Supabase. Each prompt is appended to the open chat. **New chat** starts another thread. Opening a dance keeps that prompt at the top of the dance screen.

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
