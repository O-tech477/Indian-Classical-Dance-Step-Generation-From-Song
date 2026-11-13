// static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('curate-form');
  const promptInput = document.getElementById('prompt-input');
  const modelSelect = document.getElementById('model-select');
  const submitBtn = document.getElementById('submit-btn');
  const btnSpinner = document.getElementById('btn-spinner');
  const statusBox = document.getElementById('status-box');
  const statusTitle = document.getElementById('status-title');
  const statusDetail = document.getElementById('status-detail');
  const errorBox = document.getElementById('error-box');

  const emptyState = document.getElementById('empty-state');
  const songCard = document.getElementById('song-card');
  const songTitle = document.getElementById('song-title');
  const songComposer = document.getElementById('song-composer');
  const ragaBadge = document.getElementById('raga-badge');
  const talaBadge = document.getElementById('tala-badge');
  const songDesc = document.getElementById('song-desc');
  const audioPlayer = document.getElementById('audio-player');
  const ytSourceLink = document.getElementById('yt-source-link');
  const fileSize = document.getElementById('file-size');
  const pipelineCode = document.getElementById('pipeline-code');
  const copyCmdBtn = document.getElementById('copy-cmd-btn');

  const libraryList = document.getElementById('library-list');
  const refreshLibBtn = document.getElementById('refresh-lib-btn');
  const promptView = document.getElementById('prompt-view');
  const danceView = document.getElementById('dance-view');
  const danceFrame = document.getElementById('dance-frame');
  const backBtn = document.getElementById('back-btn');

  // Load library on start
  loadLibrary();

  function syncComposer() {
    const ready = promptInput.value.trim().length > 0;
    submitBtn.classList.toggle('is-ready', ready);
    promptInput.style.height = 'auto';
    promptInput.style.height = Math.min(promptInput.scrollHeight, 220) + 'px';
  }

  promptInput.addEventListener('input', syncComposer);
  syncComposer();

  // Chips click handler
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      promptInput.value = chip.getAttribute('data-prompt');
      promptInput.focus();
      syncComposer();
    });
  });

  // Copy command button
  copyCmdBtn.addEventListener('click', () => {
    const text = pipelineCode.textContent;
    navigator.clipboard.writeText(text).then(() => {
      const orig = copyCmdBtn.textContent;
      copyCmdBtn.textContent = 'Copied!';
      setTimeout(() => { copyCmdBtn.textContent = orig; }, 2000);
    });
  });

  // Refresh library
  refreshLibBtn.addEventListener('click', loadLibrary);

  function showPrompt() {
    danceFrame.src = 'about:blank';
    danceView.hidden = true;
    promptView.hidden = false;
    document.body.classList.remove('showing-dance');
  }

  function showDance(url) {
    danceFrame.src = url;
    promptView.hidden = true;
    danceView.hidden = false;
    document.body.classList.add('showing-dance');
  }

  backBtn.addEventListener('click', showPrompt);

  // Form submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    // Reset error & show loading
    errorBox.hidden = true;
    errorBox.textContent = '';
    submitBtn.disabled = true;
    btnSpinner.hidden = false;
    statusBox.hidden = false;
    statusTitle.textContent = 'Searching with Groq...';
    statusDetail.textContent = 'Looking up an Indian classical recording with browser search...';

    const timer1 = setTimeout(() => {
      statusTitle.textContent = 'Extracting YouTube Audio via yt-dlp...';
      statusDetail.textContent = 'Downloading high-quality audio and converting to MP3 in downloads/...';
    }, 4500);
    const timer2 = setTimeout(() => {
      statusTitle.textContent = 'Mapping the dance...';
      statusDetail.textContent = 'Fitting adavus to the beats and drawing the stick figure.';
    }, 20000);

    try {
      const res = await fetch('/api/recommend-and-download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt,
          model: modelSelect.value,
        }),
      });

      clearTimeout(timer1);
      clearTimeout(timer2);
      const data = await res.json();

      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Server returned an error');
      }

      displaySong(data.song);
      loadLibrary();
      if (!data.dance_url) {
        throw new Error('The dance player was not created.');
      }
      showDance(data.dance_url);
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.hidden = false;
    } finally {
      clearTimeout(timer1);
      clearTimeout(timer2);
      submitBtn.disabled = false;
      btnSpinner.hidden = true;
      statusBox.hidden = true;
    }
  });

  function displaySong(song) {
    emptyState.hidden = true;
    songCard.hidden = false;

    songTitle.textContent = song.song_name || 'Indian Classical Song';
    songComposer.textContent = song.composer ? `Composer: ${song.composer}` : 'Traditional Composition';
    ragaBadge.textContent = `Raga: ${song.raga || 'N/A'}`;
    const talaName = song.tala || 'Adi Tala';
    const beatCount = song.tala_beats;
    const talaAlreadyCountsBeats = beatCount && String(talaName).includes(String(beatCount));
    talaBadge.textContent = talaAlreadyCountsBeats || !beatCount
      ? `Tala: ${talaName}`
      : `Tala: ${talaName} (${beatCount} beats)`;
    songDesc.textContent = song.description || '';

    if (song.youtube_url) {
      ytSourceLink.href = song.youtube_url;
      ytSourceLink.hidden = false;
    } else {
      ytSourceLink.hidden = true;
    }

    if (song.audio_url) {
      audioPlayer.src = song.audio_url;
      audioPlayer.play().catch(() => {});
    }

    if (song.file_size_mb) {
      fileSize.textContent = `${song.file_size_mb} MB`;
    } else {
      fileSize.textContent = '';
    }

    const talaBeats = song.tala_beats || 8;
    const audioPath = song.audio_path || `downloads/${song.filename}`;
    const cmd = `uv run pipeline.py "${audioPath}" --adavus landmarks_json -o output --tala ${talaBeats} --default-beats 4`;
    pipelineCode.textContent = cmd;
  }

  async function loadLibrary() {
    try {
      const res = await fetch('/api/songs');
      const data = await res.json();
      if (!data.success) return;

      const songs = data.songs || [];
      if (songs.length === 0) {
        libraryList.innerHTML = '<p class="empty-text">No music downloaded yet. Your downloaded tracks will be stored here.</p>';
        return;
      }

      libraryList.innerHTML = '';
      songs.forEach(song => {
        const item = document.createElement('div');
        item.className = 'library-item';
        item.innerHTML = `
          <div class="lib-title-row">
            <div>
              <div class="lib-title">${escapeHtml(song.song_name || song.filename)}</div>
              <div class="lib-composer">${escapeHtml(song.composer || 'Traditional')}</div>
            </div>
            <button class="btn btn-primary btn-sm play-btn">▶ Play</button>
          </div>
          <div class="lib-meta">
            <span class="badge badge-raga">${escapeHtml(song.raga || 'Raga: N/A')}</span>
            <span class="badge badge-tala">${escapeHtml(song.tala || 'Tala: N/A')}</span>
          </div>
          <div class="lib-footer">
            <span class="lib-date">${song.created_at || ''}</span>
            <span>${song.file_size_mb ? song.file_size_mb + ' MB' : ''}</span>
          </div>
        `;

        item.querySelector('.play-btn').addEventListener('click', () => {
          displaySong(song);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        });

        libraryList.appendChild(item);
      });
    } catch (e) {
      console.error('Failed to load library:', e);
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
});
