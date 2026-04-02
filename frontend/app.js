/**
 * Faqih.ai — Frontend Application
 * SSE streaming chat with citation support
 */

const API_BASE = 'http://localhost:8000';

// ─── State ─────────────────────────────────────────────────

let currentSessionId = null;
let sessions = [];
let isStreaming = false;
let currentController = null;

// ─── Initialization ────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('query-input');
    input.addEventListener('input', () => {
        document.getElementById('send-btn').disabled = !input.value.trim() || isStreaming;
    });
    createNewSession();
});

// ─── Session Management ────────────────────────────────────

async function createNewSession() {
    if (currentController) currentController.abort();

    try {
        const res = await fetch(`${API_BASE}/sessions`, { method: 'POST' });
        const data = await res.json();
        currentSessionId = data.session_id;

        sessions.unshift({
            id: data.session_id,
            title: 'محادثة جديدة',
            created: new Date().toISOString(),
        });

        renderSessions();
        clearMessages();
    } catch (err) {
        console.error('API Error:', err);
        showError('تعذر الاتصال بالخادم. تأكد من تشغيل النظام.');
    }
}

function renderSessions() {
    const list = document.getElementById('session-list');
    list.innerHTML = sessions.map(s => `
        <div class="session-item ${s.id === currentSessionId ? 'active' : ''}" onclick="switchSession('${s.id}')">
            <svg class="w-4 h-4 text-white/30 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-width="1.5" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z"/>
            </svg>
            <span class="text-sm text-white/60 truncate font-arabic">${s.title}</span>
        </div>
    `).join('');
}

function switchSession(id) {
    if (currentController) currentController.abort();
    currentSessionId = id;
    renderSessions();
    clearMessages();
}

// ─── Message Handling ──────────────────────────────────────

function clearMessages() {
    const container = document.getElementById('messages-container');
    const welcome = document.getElementById('welcome-screen');
    container.innerHTML = '';
    container.appendChild(welcome);
    welcome.classList.remove('hidden');
    welcome.classList.add('flex');
}

function hideWelcome() {
    const welcome = document.getElementById('welcome-screen');
    welcome.classList.add('hidden');
    welcome.classList.remove('flex');
}

function addUserMessage(text) {
    hideWelcome();
    const container = document.getElementById('messages-container');
    const div = document.createElement('div');
    div.className = 'flex justify-end';
    div.innerHTML = `
        <div class="message-user">
            <p class="font-arabic text-white/90 leading-relaxed">${escapeHtml(text)}</p>
        </div>
    `;
    container.appendChild(div);
    scrollToBottom();

    // Update session title with first message
    const session = sessions.find(s => s.id === currentSessionId);
    if (session && session.title === 'محادثة جديدة') {
        session.title = text.substring(0, 40) + (text.length > 40 ? '...' : '');
        renderSessions();
    }
}

function addAssistantMessage() {
    const container = document.getElementById('messages-container');
    const wrapper = document.createElement('div');
    wrapper.className = 'flex justify-start';
    wrapper.id = 'current-response';

    wrapper.innerHTML = `
        <div class="message-assistant w-full">
            <div class="flex items-center gap-2 mb-3">
                <div class="w-7 h-7 rounded-lg bg-gradient-to-br from-gold-400 to-gold-600 flex items-center justify-center shrink-0">
                    <svg class="w-4 h-4 text-navy-900" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                        <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
                    </svg>
                </div>
                <span class="text-xs text-gold-400/60 font-arabic">الفقيه</span>
            </div>
            <div id="intent-container"></div>
            <div id="response-text" class="response-text"></div>
            <div class="typing-indicator" id="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            <div id="citations-container" class="sources-container hidden"></div>
        </div>
    `;

    container.appendChild(wrapper);
    scrollToBottom();
}

// ─── Send Message ──────────────────────────────────────────

function sendSuggestion(text) {
    document.getElementById('query-input').value = text;
    sendMessage();
}

async function sendMessage() {
    const input = document.getElementById('query-input');
    const text = input.value.trim();
    if (!text || isStreaming) return;

    if (currentController) currentController.abort();
    currentController = new AbortController();

    input.value = '';
    autoResize(input);
    document.getElementById('send-btn').disabled = true;
    isStreaming = true;

    addUserMessage(text);
    addAssistantMessage();
    showStatus('جارٍ البحث في المصادر...');

    try {
        const response = await fetch(`${API_BASE}/sessions/${currentSessionId}/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text, session_id: currentSessionId }),
            signal: currentController.signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullAnswer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const payload = line.slice(6).trim();

                if (payload === '[DONE]') {
                    finishStreaming();
                    continue;
                }

                try {
                    const event = JSON.parse(payload);
                    handleSSEEvent(event);
                    if (event.type === 'token') {
                        fullAnswer += event.content;
                        const el = document.getElementById('response-text');
                        if (el) {
                            el.innerHTML = marked.parse(fullAnswer);
                            scrollToBottom();
                        }
                    }
                } catch (e) {
                    console.warn('SSE parse error:', e);
                }
            }
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            console.log('Stream aborted');
            return;
        } else {
            console.error('Query failed:', err);
            showError('تعذر الاتصال بالخادم. يرجى التفاعل مرة أخرى.');
        }
    }

    isStreaming = false;
    document.getElementById('send-btn').disabled = !input.value.trim();
}

function handleSSEEvent(event) {
    switch (event.type) {
        case 'intent':
            showIntent(event.content);
            showStatus('جارٍ الاسترجاع الهجين...');
            break;

        case 'retrieval':
            showStatus(`تم العثور على ${event.content.count} مصدر`);
            break;

        case 'token':
            hideTypingIndicator();
            showStatus('جارٍ التوليد...');
            break;

        case 'citations':
            showCitations(event.content);
            break;

        case 'cached':
            showStatus('نتيجة محفوظة ✓');
            break;

        case 'error':
            showError(event.content);
            break;
    }
}

// ─── SSE Event Handlers ────────────────────────────────────

// Removed appendToken in favor of marked.parse logic in sendMessage

function showIntent(intent) {
    const container = document.getElementById('intent-container');
    if (!container) return;

    const labels = {
        fatwa: 'فتوى',
        muqarana: 'مقارنة',
        tasil: 'تأصيل',
        tarikh: 'تاريخ',
    };

    const madhabs = {
        hanafi: 'حنفي',
        maliki: 'مالكي',
        shafii: 'شافعي',
        hanbali: 'حنبلي',
        all: 'جميع المذاهب',
    };

    container.innerHTML = `
        <div class="flex flex-wrap gap-2 mb-3">
            <span class="intent-badge">${labels[intent.question_type] || intent.question_type}</span>
            <span class="intent-badge" style="background:rgba(139,92,246,0.1);border-color:rgba(139,92,246,0.2);color:#a78bfa;">${madhabs[intent.madhab_preference] || intent.madhab_preference}</span>
        </div>
    `;
}

function showCitations(citations) {
    const container = document.getElementById('citations-container');
    if (!container || !citations.length) return;

    container.classList.remove('hidden');
    container.innerHTML = `
        <div class="w-full text-xs text-white/30 mb-2 font-arabic">المصادر:</div>
        ${citations.map((c, i) => `
            <div class="source-card" onclick="showCitationDetail('${c.chunk_id}')">
                <svg class="w-3.5 h-3.5 source-icon shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-width="1.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
                </svg>
                <span class="font-arabic truncate">${c.book_title} — ${(c.chapter_path || []).join(' > ')}</span>
            </div>
        `).join('')}
    `;
    scrollToBottom();
}

// ─── Citation Modal ────────────────────────────────────────

async function showCitationDetail(chunkId) {
    if (!chunkId) return;

    try {
        const res = await fetch(`${API_BASE}/chunks/${chunkId}`);
        const data = await res.json();

        document.getElementById('citation-title').textContent = data.book_title || 'المصدر';
        document.getElementById('citation-content').innerHTML = `
            <div class="space-y-4">
                <div class="flex flex-wrap gap-2 text-xs">
                    <span class="px-2 py-1 rounded-md bg-gold-500/10 text-gold-300">${data.madhab || ''}</span>
                    <span class="px-2 py-1 rounded-md bg-white/5 text-white/50">ص ${data.page_start}-${data.page_end}</span>
                    <span class="px-2 py-1 rounded-md bg-white/5 text-white/50">${data.chunk_type || ''}</span>
                </div>
                <div class="text-xs text-white/40 font-arabic">
                    ${(data.chapter_path || []).join(' ← ')}
                </div>
                <div class="font-arabic text-white/85 leading-[2.2] text-base border-t border-white/5 pt-4">
                    ${data.display_text || data.text || ''}
                </div>
                <div class="text-xs text-white/30 pt-2 border-t border-white/5">
                    المؤلف: ${data.author || ''} ● ${data.book_title || ''}
                </div>
            </div>
        `;

        const modal = document.getElementById('citation-modal');
        modal.classList.remove('hidden');
        modal.classList.add('flex');
    } catch (err) {
        console.error('Failed to fetch citation:', err);
    }
}

function closeCitationModal(event) {
    if (event && event.target !== event.currentTarget) return;
    const modal = document.getElementById('citation-modal');
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

// ─── Demo Fallback ─────────────────────────────────────────

// Demo response removed. Real network errors are now handled directly via showError.

// ─── UI Utilities ──────────────────────────────────────────

function showStatus(text) {
    const badge = document.getElementById('status-badge');
    const statusText = document.getElementById('status-text');
    badge.classList.remove('hidden');
    badge.classList.add('flex');
    statusText.textContent = text;
}

function hideStatus() {
    const badge = document.getElementById('status-badge');
    badge.classList.add('hidden');
    badge.classList.remove('flex');
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.style.display = 'none';
}

function finishStreaming() {
    hideStatus();
    hideTypingIndicator();
    isStreaming = false;
    const input = document.getElementById('query-input');
    document.getElementById('send-btn').disabled = !input.value.trim();
}

function showError(message) {
    const el = document.getElementById('response-text');
    if (el) {
        el.innerHTML = `<div class="text-red-400 text-sm">خطأ: ${escapeHtml(message)}</div>`;
    }
    finishStreaming();
}

function scrollToBottom() {
    const container = document.getElementById('messages-container');
    container.scrollTop = container.scrollHeight;
}

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
}

function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
