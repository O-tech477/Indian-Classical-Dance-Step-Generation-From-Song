## Abstract
We attempt to map Bharatanatyam steps to any Indian classical song on the basis of nritta only i.e. rhythm and tempo. We use librosa and Demucs to analyse the song and convert it into chunks with beat information and then map the adavus to the beats.

## Architecture & Modular Components
- **`app.py`**: Flask web server hosting the curation UI and music storage APIs.
- **`modules/`**:
  - `llm_service.py`: Google Gemini cloud LLM interface with Google Search grounding to discover Indian classical songs, tala, and YouTube links.
  - `audio_service.py`: YouTube extraction and MP3 audio download via `yt-dlp`.
  - `storage_service.py`: Local music catalog and metadata tracker (`downloads/library.json`).
- **`cloud_llm.py`**: CLI facade for LLM-based song recommendation and audio downloading.
- **`pipeline.py`**: Command line for the choreography generator (song analysis -> adavu planning -> interactive stick-figure HTML player). The stages live in `dance/`:
  - `analysis.py`: beats, tempo-octave fix, sam, gaps in the pulse, sections, pitch.
  - `cycle.py`: one landmark clip -> one clean periodic cycle of bone angles and stage travel.
  - `pose.py`: knee and elbow hinge limits.
  - `vocabulary.py`: builds, caches and loads the adavu vocabulary (with `adavu_meta.json`).
  - `planner.py`: picks the adavu, speed and beats for each tala phrase.
  - `player.py` + `player.html`: writes `timeline.json` and the stick-figure player.

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

### Supabase chats and Google sign-in
The first screen is Google sign-in through Supabase Auth. Each Google account has one profile, its own chats, and a folder at `downloads/<account-id>/` for that account's songs. The stick-figure player for those songs is written under `output/<account-id>/`.

In the Supabase dashboard:

1. Create a project.
2. Open **SQL Editor**, paste `supabase/schema.sql`, and run it. That creates `profiles`, `chats`, and `chat_turns`, adds `chats.user_id`, and turns on row level security. Run it again if the project was created before accounts existed.
3. Open **Project Settings → API** and copy the project URL and the **service_role** key. The service role key bypasses row level security. Keep it on the server only. Do not put it in the browser, and do not commit it.
4. Open **Authentication → Sign In / Providers → Google** and turn Google on. In Google Cloud, create a web OAuth client whose redirect URI is `https://YOUR-PROJECT.supabase.co/auth/v1/callback`, then paste that client id and secret into Supabase. Under **Authentication → URL Configuration**, allow `http://127.0.0.1:6767/auth/callback` (and the same URL with `localhost` if you open the app that way).
5. Add both Supabase values to `.env` in this folder:

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
Open [http://127.0.0.1:6767](http://127.0.0.1:6767) in your browser. Sign in with Google, then enter prompts, stream the audio, and view the songs stored for that account.

### 3. CLI Music Curation & Download
```bash
uv run cloud_llm.py "Energetic Adi tala Carnatic song for Thattadavu" -o downloads
```

### 4. Run the Dance Choreography Pipeline
Generate the stick-figure player from any downloaded song:
```bash
uv run pipeline.py downloads/<your-song>.mp3 --adavus landmarks_json -o output --tala 8
```
Double-click `output/index.html` to open the stick-figure player in your browser.

The length of each adavu in beats comes from `landmarks_json/adavu_meta.json`. Every clip starts at 4 beats with `"checked": false`. Count each step against its clip, set `"beats"`, and set `"checked": true`. `period_s`, `start_s` and `end_s` can also be set there when the automatic period or clean stretch is wrong. The vocabulary is cached in `landmarks_json/.vocab/` and rebuilt when a clip, the metadata file or the vocabulary code in `dance/` changes.

If the accent of the beat clicks in the player is not on the sam, rerun with `--sam-beat N`, where N is the index of a beat that is a sam.
