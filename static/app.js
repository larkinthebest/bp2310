// ── State ────────────────────────────────────────────────────────

let isWaiting = false;
let chatHistory = []; // {role: 'user'|'assistant', content: string}

// ── Init ─────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    loadFiles();
    document.getElementById('chatInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    // File upload handler
    document.getElementById('fileUploadInput').addEventListener('change', handleFileUpload);
});

// ── File Upload ──────────────────────────────────────────────────

async function handleFileUpload(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const btn = document.getElementById('btnIngest');
    const status = document.getElementById('ingestStatus');
    btn.disabled = true;
    btn.textContent = '⏳ Uploading...';
    status.textContent = `Uploading ${files.length} file(s)...`;

    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (res.status === 409) {
            status.textContent = 'Ingestion already running, try again later';
            btn.disabled = false;
            btn.textContent = '⚡ Upload & Ingest Files';
            return;
        }

        status.textContent = `Uploaded ${data.uploaded.length} file(s). Ingesting ${data.ingesting.length}...`;
        loadFiles();

        if (data.ingesting && data.ingesting.length > 0) {
            btn.textContent = '⏳ Ingesting...';
            pollIngestion();
        } else {
            status.textContent = `Uploaded ${data.uploaded.length} file(s). Already processed.`;
            btn.disabled = false;
            btn.textContent = '⚡ Upload & Ingest Files';
        }
    } catch (err) {
        status.textContent = 'Upload failed';
        btn.disabled = false;
        btn.textContent = '⚡ Upload & Ingest Files';
    }

    // Reset input so same file can be re-selected
    e.target.value = '';
}

// ── File Selection ───────────────────────────────────────────────

function getSelectedFiles() {
    const checkboxes = document.querySelectorAll('.file-checkbox:checked');
    return Array.from(checkboxes).map(cb => cb.dataset.filename);
}

function toggleSelectAll() {
    const selectAll = document.getElementById('selectAll');
    document.querySelectorAll('.file-checkbox').forEach(cb => {
        cb.checked = selectAll.checked;
    });
}

function updateSelectAll() {
    const all = document.querySelectorAll('.file-checkbox');
    const checked = document.querySelectorAll('.file-checkbox:checked');
    const selectAll = document.getElementById('selectAll');
    selectAll.checked = all.length > 0 && all.length === checked.length;
    selectAll.indeterminate = checked.length > 0 && checked.length < all.length;
}

// ── Files ────────────────────────────────────────────────────────

async function loadFiles() {
    try {
        const res = await fetch('/api/files');
        const data = await res.json();
        const list = document.getElementById('fileList');

        if (!data.files || data.files.length === 0) {
            list.innerHTML = '<div class="file-item" style="color: var(--text-muted);">No files yet — upload some!</div>';
            return;
        }

        list.innerHTML = data.files.map(f => {
            const safeName = escapeHtml(f.name);
            const encodedName = encodeURIComponent(f.name);
            return `
            <div class="file-item">
                <input type="checkbox" class="file-checkbox" data-filename="${safeName}" 
                       checked onchange="updateSelectAll()">
                <div class="file-dot ${f.processed ? 'processed' : 'pending'}"></div>
                <div class="file-name" title="${safeName}">${safeName}</div>
                <button class="btn-delete-file" data-file="${encodedName}" onclick="deleteFile(decodeURIComponent(this.dataset.file))" title="Delete">🗑</button>
            </div>
            `;
        }).join('');
        updateSelectAll();
    } catch (e) {
        console.error('Failed to load files:', e);
    }
}

async function deleteFile(filename) {
    if (!confirm(`Delete "${filename}"? This removes the file from disk.`)) return;
    try {
        await fetch(`/api/files/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        loadFiles();
    } catch (e) {
        console.error('Delete failed:', e);
    }
}

// ── Ingestion Polling ────────────────────────────────────────────

function pollIngestion() {
    const btn = document.getElementById('btnIngest');
    const status = document.getElementById('ingestStatus');

    const interval = setInterval(async () => {
        try {
            const res = await fetch('/api/ingest/status');
            const data = await res.json();
            status.textContent = data.progress || '';

            if (!data.running) {
                clearInterval(interval);
                btn.disabled = false;
                btn.textContent = '⚡ Upload & Ingest Files';
                if (data.results) {
                    status.textContent = `✅ ${data.results.succeeded} succeeded, ${data.results.failed} failed`;
                }
                loadFiles();
            }
        } catch (e) {
            clearInterval(interval);
            btn.disabled = false;
            btn.textContent = '⚡ Upload & Ingest Files';
        }
    }, 2000);
}

// ── Chat ─────────────────────────────────────────────────────────

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();
    if (!question || isWaiting) return;

    // Hide welcome
    const welcome = document.getElementById('welcome');
    if (welcome) welcome.style.display = 'none';

    addUserMessage(question);
    chatHistory.push({ role: 'user', content: question });
    input.value = '';

    isWaiting = true;
    document.getElementById('btnSend').disabled = true;
    const loadingEl = addLoadingMessage();

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                selected_files: getSelectedFiles(),
                history: chatHistory.slice(-10)
            })
        });

        const data = await res.json();
        loadingEl.remove();
        addAIMessage(data);
        chatHistory.push({ role: 'assistant', content: data.answer });
    } catch (e) {
        loadingEl.remove();
        addAIMessage({ answer: 'Sorry, something went wrong. Please try again.', sources: [], frames: [] });
    } finally {
        isWaiting = false;
        document.getElementById('btnSend').disabled = false;
        input.focus();
    }
}

// ── Message Rendering ────────────────────────────────────────────

function addUserMessage(text) {
    const container = document.getElementById('chatContainer');
    const div = document.createElement('div');
    div.className = 'message message-user';
    div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    container.appendChild(div);
    scrollToBottom();
}

function addLoadingMessage() {
    const container = document.getElementById('chatContainer');
    const div = document.createElement('div');
    div.className = 'message message-ai';
    div.innerHTML = `
        <div class="ai-header">
            <div class="ai-avatar">⚽</div>
            <div class="ai-name">Sports AI</div>
        </div>
        <div class="loading-dots"><span></span><span></span><span></span></div>
    `;
    container.appendChild(div);
    scrollToBottom();
    return div;
}

function addAIMessage(data) {
    const container = document.getElementById('chatContainer');
    const div = document.createElement('div');
    div.className = 'message message-ai';
    const msgId = 'msg-' + Date.now();

    // Format answer — convert [Frame: Xs] into clickable badges
    let answerHtml = data.answer.split('\n').filter(p => p.trim()).map(p => {
        let escaped = escapeHtml(p);
        // Replace [Frame: 24s] with clickable badges that open video
        escaped = escaped.replace(/\[Frame:\s*(\d+(?:\.\d+)?)s?\]/gi, (match, ts) => {
            return `<a class="frame-badge" onclick="highlightFrame('${msgId}', ${ts})" title="Jump to frame">⏱ ${formatTimestamp(parseFloat(ts))}</a>`;
        });
        return `<p>${escaped}</p>`;
    }).join('');

    // Sources
    let sourcesHtml = '';
    if (data.sources && data.sources.length > 0) {
        const tags = data.sources.map(s => `<span>${escapeHtml(s)}</span>`).join('');
        sourcesHtml = `<div class="ai-sources">📎 Sources: ${tags}</div>`;
    }

    // Frames
    let framesHtml = '';
    if (data.frames && data.frames.length > 0) {
        const frameCards = data.frames
            .filter(f => f.image_base64)
            .map((f, i) => `
                <div class="frame-card" id="${msgId}-frame-${f.timestamp}" onclick="openVideoLightbox('${escapeAttr(f.source)}', ${f.timestamp}, '${escapeAttr(f.caption)}')" 
                     data-ts="${f.timestamp}" data-src="${escapeHtml(f.source)}" data-cap="${escapeHtml(f.caption)}">
                    <img src="data:image/jpeg;base64,${f.image_base64}" alt="Frame at ${f.timestamp}s" loading="lazy">
                    <div class="frame-card-info">
                        <div class="frame-timestamp">⏱ ${formatTimestamp(f.timestamp)}</div>
                        <div class="frame-source">${escapeHtml(f.source)}</div>
                        <div class="frame-caption">${escapeHtml(f.caption)}</div>
                    </div>
                    <div class="frame-play-icon">▶</div>
                </div>
            `).join('');

        if (frameCards) {
            framesHtml = `
                <div class="frame-gallery">
                    <div class="frame-gallery-title">🎬 Key Frames — click to play video</div>
                    <div class="frame-grid">${frameCards}</div>
                </div>
            `;
        }
    }

    div.innerHTML = `
        <div class="ai-header">
            <div class="ai-avatar">⚽</div>
            <div class="ai-name">Sports AI</div>
        </div>
        <div class="ai-answer">
            ${answerHtml}
            ${sourcesHtml}
        </div>
        ${framesHtml}
    `;
    div.id = msgId;

    // Store frames for badge click → video lookup
    if (data.frames) {
        div.dataset.frames = JSON.stringify(data.frames);
    }

    container.appendChild(div);
    scrollToBottom();
}

// ── Frame Interaction ────────────────────────────────────────────

function highlightFrame(msgId, timestamp) {
    const msg = document.getElementById(msgId);
    if (!msg) return;

    // Try to find matching frame and open video
    const framesData = msg.dataset.frames ? JSON.parse(msg.dataset.frames) : [];
    let closest = null;
    let minDiff = Infinity;
    for (const f of framesData) {
        const diff = Math.abs(f.timestamp - timestamp);
        if (diff < minDiff) { minDiff = diff; closest = f; }
    }

    if (closest && isVideoFile(closest.source)) {
        openVideoLightbox(closest.source, timestamp, closest.caption);
        return;
    }

    // Fallback: scroll to frame card
    const cards = msg.querySelectorAll('.frame-card');
    let closestCard = null;
    minDiff = Infinity;
    cards.forEach(card => {
        const ts = parseFloat(card.dataset.ts);
        const diff = Math.abs(ts - timestamp);
        if (diff < minDiff) { minDiff = diff; closestCard = card; }
    });
    if (closestCard) {
        closestCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        closestCard.classList.add('highlight');
        setTimeout(() => closestCard.classList.remove('highlight'), 2000);
    }
}

// ── Video Lightbox ───────────────────────────────────────────────

function openVideoLightbox(source, timestamp, caption) {
    const lightbox = document.getElementById('lightbox');
    const content = document.getElementById('lightboxContent');
    const videoUrl = `/api/video/${encodeURIComponent(source)}#t=${timestamp}`;

    content.innerHTML = `
        <video id="lightboxVideo" controls autoplay style="max-width:100%; max-height:70vh; border-radius:12px;">
            <source src="${videoUrl}" type="video/mp4">
            Your browser does not support video playback.
        </video>
        <div class="lightbox-caption">
            <div class="lightbox-meta">⏱ ${formatTimestamp(timestamp)} — ${escapeHtml(source)}</div>
            ${escapeHtml(caption)}
        </div>
    `;
    lightbox.classList.add('active');

    // Seek to exact timestamp after video loads
    const video = document.getElementById('lightboxVideo');
    video.addEventListener('loadedmetadata', () => {
        video.currentTime = timestamp;
    }, { once: true });
}

function openImageLightbox(imgB64, timestamp, source, caption) {
    const lightbox = document.getElementById('lightbox');
    const content = document.getElementById('lightboxContent');
    content.innerHTML = `
        <img src="data:image/jpeg;base64,${imgB64}" alt="Frame">
        <div class="lightbox-caption">
            <div class="lightbox-meta">⏱ ${formatTimestamp(timestamp)} — ${escapeHtml(source)}</div>
            ${escapeHtml(caption)}
        </div>
    `;
    lightbox.classList.add('active');
}

function closeLightbox(event) {
    if (event.target === document.getElementById('lightbox')) {
        // Pause video if playing
        const video = document.getElementById('lightboxVideo');
        if (video) video.pause();
        document.getElementById('lightbox').classList.remove('active');
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const video = document.getElementById('lightboxVideo');
        if (video) video.pause();
        document.getElementById('lightbox').classList.remove('active');
    }
});

// ── Helpers ──────────────────────────────────────────────────────

function isVideoFile(filename) {
    return /\.(mp4|avi|mov|mkv|webm)$/i.test(filename);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function escapeAttr(text) {
    return text.replace(/'/g, "\\'").replace(/"/g, '\\"');
}

function formatTimestamp(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function scrollToBottom() {
    const container = document.getElementById('chatContainer');
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}
