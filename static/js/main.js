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
  const dancePrompt = document.getElementById('dance-prompt');
  const backBtn = document.getElementById('back-btn');
  const thread = document.getElementById('thread');
  const welcome = document.getElementById('welcome');
  const chips = document.querySelector('.chips-container');
  const chatList = document.getElementById('chat-list');
  const newChatBtn = document.getElementById('new-chat-btn');
  const storageNote = document.getElementById('storage-note');
  const uploadBtn = document.getElementById('upload-btn');
  const uploadInput = document.getElementById('upload-input');

  let activeChatId = localStorage.getItem('bn-active-chat') || null;
  let turns = [];
  let savedChats = [];

  loadLibrary();
  loadStorageStatus();
  loadChats().then(() => {
    if (activeChatId) openChat(activeChatId);
  });

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

  function showDance(url, prompt) {
    audioPlayer.pause();
    dancePrompt.textContent = prompt || '';
    danceFrame.src = url;
    promptView.hidden = true;
    danceView.hidden = false;
    document.body.classList.add('showing-dance');
  }

  backBtn.addEventListener('click', showPrompt);
  newChatBtn.addEventListener('click', startNewChat);

  uploadBtn.addEventListener('click', () => uploadInput.click());
  uploadInput.addEventListener('change', () => {
    const file = uploadInput.files && uploadInput.files[0];
    uploadInput.value = '';
    if (file) uploadSong(file);
  });

  function setBusy(busy) {
    submitBtn.disabled = busy;
    uploadBtn.disabled = busy;
    btnSpinner.hidden = !busy;
    statusBox.hidden = !busy;
  }

  async function uploadSong(file) {
    errorBox.hidden = true;
    errorBox.textContent = '';
    setBusy(true);
    statusTitle.textContent = 'Uploading your song...';
    statusDetail.textContent = file.name;

    const timer = setTimeout(() => {
      statusTitle.textContent = 'Mapping the dance...';
      statusDetail.textContent = 'Fitting adavus to the beats and drawing the stick figure.';
    }, 1500);

    try {
      const body = new FormData();
      body.append('audio', file);
      body.append('prompt', promptInput.value.trim());
      body.append('chat_id', activeChatId || '');
      const res = await fetch('/api/upload', { method: 'POST', body });
      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || 'Upload failed');
      }
      displaySong(data.song);
      loadLibrary();
      if (data.chat && data.chat.id) {
        activeChatId = data.chat.id;
        localStorage.setItem('bn-active-chat', activeChatId);
        turns = data.chat.turns || [];
        renderThread();
        loadChats();
      }
      promptInput.value = '';
      syncComposer();
      if (!data.dance_url) throw new Error('The dance player was not created.');
      showDance(data.dance_url, data.song.prompt || file.name);
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.hidden = false;
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  }

  // Form submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = promptInput.value.trim();
    if (!prompt) return;

    // Reset error & show loading
    errorBox.hidden = true;
    errorBox.textContent = '';
    setBusy(true);
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
          chat_id: activeChatId,
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
      if (data.chat && data.chat.id) {
        activeChatId = data.chat.id;
        localStorage.setItem('bn-active-chat', activeChatId);
        turns = data.chat.turns || [];
        renderThread();
        loadChats();
      }
      promptInput.value = '';
      syncComposer();
      if (!data.dance_url) {
        throw new Error('The dance player was not created.');
      }
      showDance(data.dance_url, prompt);
    } catch (err) {
      errorBox.textContent = err.message;
      errorBox.hidden = false;
    } finally {
      clearTimeout(timer1);
      clearTimeout(timer2);
      setBusy(false);
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
          if (song.dance_url) {
            showDance(song.dance_url, song.prompt || song.song_name || '');
          } else {
            showPrompt();
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }
        });

        libraryList.appendChild(item);
      });
    } catch (e) {
      console.error('Failed to load library:', e);
    }
  }

  function startNewChat() {
    activeChatId = null;
    turns = [];
    localStorage.removeItem('bn-active-chat');
    renderThread();
    renderChatList(savedChats);
    showPrompt();
    promptInput.focus();
  }

  async function loadStorageStatus() {
    try {
      const res = await fetch('/api/chats/status');
      const data = await res.json();
      if (!data.success) return;
      if (data.backend === 'supabase') {
        storageNote.hidden = true;
        return;
      }
      storageNote.textContent = data.message || '';
      storageNote.hidden = !data.message;
    } catch (e) {
      console.error('Failed to read chat storage status:', e);
    }
  }

  async function loadChats() {
    try {
      const res = await fetch('/api/chats');
      const data = await res.json();
      if (!data.success) return;
      savedChats = data.chats || [];
      renderChatList(savedChats);
    } catch (e) {
      console.error('Failed to load chats:', e);
    }
  }

  function renderChatList(chats) {
    chatList.innerHTML = '';
    if (!chats.length) {
      chatList.innerHTML = '<p class="chat-empty">Your prompts will stay here.</p>';
      return;
    }
    chats.forEach(chat => {
      const row = document.createElement('div');
      row.className = 'chat-row' + (chat.id === activeChatId ? ' is-active' : '');
      row.innerHTML = `
        <button type="button" class="chat-item">
          <strong>${escapeHtml(chat.title || 'New chat')}</strong>
        </button>
        <button type="button" class="chat-delete" aria-label="Delete chat">&times;</button>
      `;
      row.querySelector('.chat-item').addEventListener('click', () => openChat(chat.id));
      row.querySelector('.chat-delete').addEventListener('click', () => removeChat(chat.id));
      chatList.appendChild(row);
    });
  }

  async function openChat(chatId) {
    try {
      const res = await fetch('/api/chats/' + encodeURIComponent(chatId));
      const data = await res.json();
      if (!res.ok || !data.success) {
        if (activeChatId === chatId) startNewChat();
        return;
      }
      activeChatId = data.chat.id;
      localStorage.setItem('bn-active-chat', activeChatId);
      turns = data.chat.turns || [];
      renderThread();
      renderChatList(savedChats);
      showPrompt();
    } catch (e) {
      console.error('Failed to open chat:', e);
    }
  }

  async function removeChat(chatId) {
    try {
      const res = await fetch('/api/chats/' + encodeURIComponent(chatId), { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok || !data.success) return;
      if (activeChatId === chatId) startNewChat();
      loadChats();
    } catch (e) {
      console.error('Failed to delete chat:', e);
    }
  }

  function renderThread() {
    thread.innerHTML = '';
    const empty = turns.length === 0;
    thread.hidden = empty;
    if (welcome) welcome.hidden = !empty;
    if (chips) chips.hidden = !empty;
    if (empty) return;
    turns.forEach(turn => {
      const user = document.createElement('div');
      user.className = 'bubble bubble-user';
      user.innerHTML = `<p>${escapeHtml(turn.prompt || '')}</p>`;
      thread.appendChild(user);

      const result = document.createElement('div');
      result.className = 'bubble bubble-result';
      const songName = turn.song_name || 'Indian classical song';
      const details = [turn.raga, turn.tala].filter(Boolean).join(' · ');
      result.innerHTML = `
        <p>${escapeHtml(songName)}</p>
        ${details ? `<div class="bubble-meta">${escapeHtml(details)}</div>` : ''}
      `;
      if (turn.dance_url) {
        const watch = document.createElement('button');
        watch.type = 'button';
        watch.className = 'watch-btn';
        watch.textContent = 'Watch dance';
        watch.addEventListener('click', () => showDance(turn.dance_url, turn.prompt || ''));
        result.appendChild(watch);
      }
      thread.appendChild(result);
    });
    thread.scrollTop = thread.scrollHeight;
  }

  function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
});
