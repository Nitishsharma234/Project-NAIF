from flask import Flask, request, Response, render_template_string, jsonify
import ollama
import json
import os
import time
from datetime import datetime

app = Flask(__name__)

MODEL = "qwen2.5-coder:7b"

CHATS_FILE = "saved_chats.json"

SYSTEM_PROMPT = """
You are a senior full-stack developer.

Rules:
- Always generate COMPLETE working code.
- If HTML project, include HTML, CSS and JS clearly separated.
- If Python project, give full runnable code.
- Do NOT give explanations unless asked.
- Do NOT use markdown formatting.
- Just output clean code.
"""

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NAIF — AI Coder</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root {
    --bg: #080c10;
    --surface: #0d1318;
    --surface2: #121920;
    --border: rgba(99, 219, 184, 0.15);
    --accent: #63dbb8;
    --accent2: #f06a6a;
    --text: #c8e6df;
    --text-dim: #4a6b62;
    --glow: rgba(99, 219, 184, 0.2);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    background: var(--bg);
    font-family: 'Space Mono', monospace;
    color: var(--text);
    height: 100vh;
    display: flex;
    overflow: hidden;
}

/* ── SIDEBAR ── */
.sidebar {
    width: 280px;
    min-width: 280px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.sidebar-header {
    padding: 24px 20px 16px;
    border-bottom: 1px solid var(--border);
}

.logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 22px;
    letter-spacing: 6px;
    color: var(--accent);
    text-shadow: 0 0 20px var(--glow);
}

.logo-sub {
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: 3px;
    margin-top: 4px;
}

.new-chat-btn {
    margin: 16px;
    padding: 10px 14px;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    cursor: pointer;
    transition: all 0.2s;
    text-align: left;
    display: flex;
    align-items: center;
    gap: 10px;
}

.new-chat-btn:hover {
    background: var(--glow);
    border-color: var(--accent);
    box-shadow: 0 0 12px var(--glow);
}

.new-chat-btn span { font-size: 16px; }

.chats-label {
    padding: 8px 20px;
    font-size: 9px;
    letter-spacing: 3px;
    color: var(--text-dim);
    text-transform: uppercase;
}

.chat-list {
    flex: 1;
    overflow-y: auto;
    padding: 0 8px 16px;
}

.chat-list::-webkit-scrollbar { width: 3px; }
.chat-list::-webkit-scrollbar-track { background: transparent; }
.chat-list::-webkit-scrollbar-thumb { background: var(--border); }

.chat-item {
    padding: 10px 12px;
    margin-bottom: 2px;
    border-radius: 4px;
    cursor: pointer;
    border: 1px solid transparent;
    transition: all 0.15s;
    position: relative;
}

.chat-item:hover { background: var(--surface2); border-color: var(--border); }
.chat-item.active { background: rgba(99,219,184,0.08); border-color: rgba(99,219,184,0.3); }

.chat-item-title {
    font-size: 11px;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 3px;
}

.chat-item-meta {
    font-size: 9px;
    color: var(--text-dim);
    letter-spacing: 1px;
}

.chat-item-delete {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    background: none;
    border: none;
    color: var(--accent2);
    cursor: pointer;
    opacity: 0;
    font-size: 13px;
    transition: opacity 0.15s;
    padding: 4px;
}

.chat-item:hover .chat-item-delete { opacity: 1; }

/* ── MAIN ── */
.main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

.topbar {
    height: 54px;
    padding: 0 28px;
    display: flex;
    align-items: center;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    gap: 20px;
}

.topbar-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--text);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

.status-text { font-size: 10px; color: var(--text-dim); letter-spacing: 2px; }

/* ── CHAT MESSAGES ── */
.chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 32px 40px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.chat-area::-webkit-scrollbar { width: 4px; }
.chat-area::-webkit-scrollbar-track { background: transparent; }
.chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    color: var(--text-dim);
}

.empty-icon {
    font-size: 48px;
    opacity: 0.3;
}

.empty-state p {
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.message-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
    animation: fadeIn 0.25s ease;
}

@keyframes fadeIn { from { opacity:0; transform: translateY(6px); } to { opacity:1; transform: none; } }

.msg-label {
    font-size: 9px;
    letter-spacing: 3px;
    color: var(--text-dim);
    text-transform: uppercase;
    padding: 0 4px;
}

.msg-user .msg-label { color: rgba(240, 106, 106, 0.6); }
.msg-ai .msg-label { color: rgba(99, 219, 184, 0.6); }

.msg-bubble {
    padding: 14px 18px;
    border-radius: 2px;
    font-size: 13px;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
}

.msg-user .msg-bubble {
    background: rgba(240, 106, 106, 0.07);
    border-left: 2px solid var(--accent2);
    color: #e8c8c8;
}

.msg-ai .msg-bubble {
    background: rgba(99, 219, 184, 0.05);
    border-left: 2px solid var(--accent);
    color: var(--text);
    font-family: 'Space Mono', monospace;
}

.cursor-blink {
    display: inline-block;
    width: 2px;
    height: 14px;
    background: var(--accent);
    margin-left: 2px;
    vertical-align: middle;
    animation: blink 0.7s step-end infinite;
}

@keyframes blink { 50% { opacity: 0; } }

/* ── INPUT ── */
.input-zone {
    padding: 20px 40px 28px;
    background: var(--surface);
    border-top: 1px solid var(--border);
}

.input-row {
    display: flex;
    gap: 12px;
    align-items: flex-end;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 12px 16px;
    transition: border-color 0.2s, box-shadow 0.2s;
}

.input-row:focus-within {
    border-color: rgba(99,219,184,0.4);
    box-shadow: 0 0 20px rgba(99,219,184,0.08);
}

#prompt {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text);
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    resize: none;
    outline: none;
    line-height: 1.6;
    max-height: 160px;
    min-height: 22px;
}

#prompt::placeholder { color: var(--text-dim); }

#sendBtn {
    background: var(--accent);
    border: none;
    color: var(--bg);
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 2px;
    padding: 8px 16px;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}

#sendBtn:hover { box-shadow: 0 0 16px var(--glow); }
#sendBtn:disabled { opacity: 0.4; cursor: not-allowed; }

.input-hint {
    margin-top: 8px;
    font-size: 9px;
    color: var(--text-dim);
    letter-spacing: 2px;
    text-align: right;
}
</style>
</head>
<body>

<!-- SIDEBAR -->
<div class="sidebar">
    <div class="sidebar-header">
        <div class="logo">NAIF</div>
        <div class="logo-sub">AI CODE ASSISTANT</div>
    </div>

    <button class="new-chat-btn" onclick="newChat()">
        <span>+</span> NEW SESSION
    </button>

    <div class="chats-label">Saved Sessions</div>
    <div class="chat-list" id="chatList"></div>
</div>

<!-- MAIN -->
<div class="main">
    <div class="topbar">
        <div class="topbar-title" id="topbarTitle">New Session</div>
        <div class="status-dot"></div>
        <div class="status-text">MODEL ACTIVE</div>
    </div>

    <div class="chat-area" id="chatArea">
        <div class="empty-state" id="emptyState">
            <div class="empty-icon">⌥</div>
            <p>Start a session</p>
        </div>
    </div>

    <div class="input-zone">
        <div class="input-row">
            <textarea id="prompt" placeholder="What do you want to build?" rows="1"></textarea>
            <button id="sendBtn" onclick="sendMessage()">SEND ↵</button>
        </div>
        <div class="input-hint">SHIFT+ENTER for new line • ENTER to send</div>
    </div>
</div>

<script>
// ── STATE ──
let sessions = JSON.parse(localStorage.getItem('naif_sessions') || '[]');
let currentSessionId = null;
let isStreaming = false;

// ── UTILS ──
function saveToStorage() {
    localStorage.setItem('naif_sessions', JSON.stringify(sessions));
}

function genId() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2,6);
}

function formatDate(ts) {
    const d = new Date(ts);
    return d.toLocaleDateString('en-GB', { day:'2-digit', month:'short' }) + 
           ' ' + d.toLocaleTimeString('en-GB', { hour:'2-digit', minute:'2-digit' });
}

// ── SESSION MANAGEMENT ──
function newChat() {
    currentSessionId = genId();
    sessions.unshift({ id: currentSessionId, title: 'New Session', ts: Date.now(), messages: [] });
    saveToStorage();
    renderSidebar();
    renderChat();
}

function loadSession(id) {
    currentSessionId = id;
    renderSidebar();
    renderChat();
}

function deleteSession(id, e) {
    e.stopPropagation();
    sessions = sessions.filter(s => s.id !== id);
    if (currentSessionId === id) {
        currentSessionId = sessions[0]?.id || null;
    }
    saveToStorage();
    renderSidebar();
    renderChat();
}

function getCurrentSession() {
    return sessions.find(s => s.id === currentSessionId);
}

// ── RENDER ──
function renderSidebar() {
    const list = document.getElementById('chatList');
    list.innerHTML = sessions.map(s => `
        <div class="chat-item ${s.id === currentSessionId ? 'active' : ''}" onclick="loadSession('${s.id}')">
            <div class="chat-item-title">${s.title}</div>
            <div class="chat-item-meta">${formatDate(s.ts)}</div>
            <button class="chat-item-delete" onclick="deleteSession('${s.id}', event)">✕</button>
        </div>
    `).join('');
}

function renderChat() {
    const area = document.getElementById('chatArea');
    const session = getCurrentSession();
    const title = document.getElementById('topbarTitle');

    if (!session || session.messages.length === 0) {
        area.innerHTML = `
            <div class="empty-state" id="emptyState">
                <div class="empty-icon">⌥</div>
                <p>Start a session</p>
            </div>`;
        title.textContent = session?.title || 'New Session';
        return;
    }

    title.textContent = session.title;
    area.innerHTML = session.messages.map(m => `
        <div class="message-wrap msg-${m.role}">
            <div class="msg-label">${m.role === 'user' ? 'YOU' : 'NAIF'}</div>
            <div class="msg-bubble">${escHtml(m.content)}</div>
        </div>
    `).join('');

    area.scrollTop = area.scrollHeight;
}

function escHtml(t) {
    return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── MESSAGING ──
async function sendMessage() {
    if (isStreaming) return;

    const ta = document.getElementById('prompt');
    const prompt = ta.value.trim();
    if (!prompt) return;

    // Ensure a session exists
    if (!currentSessionId) newChat();

    const session = getCurrentSession();
    ta.value = '';
    ta.style.height = 'auto';
    isStreaming = true;
    document.getElementById('sendBtn').disabled = true;

    // Add user message
    session.messages.push({ role: 'user', content: prompt });

    // Auto-title from first message
    if (session.messages.length === 1) {
        session.title = prompt.slice(0, 42) + (prompt.length > 42 ? '…' : '');
    }

    saveToStorage();
    renderSidebar();

    // Render user bubble + empty AI bubble
    const area = document.getElementById('chatArea');
    area.innerHTML += `
        <div class="message-wrap msg-user">
            <div class="msg-label">YOU</div>
            <div class="msg-bubble">${escHtml(prompt)}</div>
        </div>
        <div class="message-wrap msg-ai" id="streamWrap">
            <div class="msg-label">NAIF</div>
            <div class="msg-bubble" id="streamBubble"><span class="cursor-blink"></span></div>
        </div>`;
    area.scrollTop = area.scrollHeight;

    // Streaming fetch
    try {
        const res = await fetch('/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ prompt, history: session.messages.slice(0, -1) })
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        const bubble = document.getElementById('streamBubble');

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            fullText += decoder.decode(value, { stream: true });
            bubble.innerHTML = escHtml(fullText) + '<span class="cursor-blink"></span>';
            area.scrollTop = area.scrollHeight;
        }

        // Remove cursor, save
        bubble.innerHTML = escHtml(fullText);
        session.messages.push({ role: 'ai', content: fullText });
        saveToStorage();

    } catch (err) {
        document.getElementById('streamBubble').innerHTML = '<span style="color:var(--accent2)">Connection error. Is the server running?</span>';
    }

    isStreaming = false;
    document.getElementById('sendBtn').disabled = false;
    document.getElementById('topbarTitle').textContent = getCurrentSession()?.title || '';
    renderSidebar();
}

// ── KEYBOARD ──
document.getElementById('prompt').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
document.getElementById('prompt').addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 160) + 'px';
});

// ── INIT ──
if (sessions.length === 0) {
    newChat();
} else {
    currentSessionId = sessions[0].id;
    renderSidebar();
    renderChat();
}
</script>
</body>
</html>
"""

def load_chats():
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_chats(chats):
    with open(CHATS_FILE, 'w') as f:
        json.dump(chats, f, indent=2)

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)

# BUG FIX: Was returning Response(stream()) but frontend expected JSON.
# Now properly streams text chunks and frontend reads them via ReadableStream.
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_prompt = data.get("prompt", "")
    history = data.get("history", [])  # optional conversation history

    # Build messages list with history for multi-turn context
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})

    def stream():
        try:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                stream=True
            )
            for chunk in response:
                # BUG FIX: original code checked chunk["message"] but the
                # correct key with stream=True is chunk["message"]["content"]
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as e:
            yield f"\n[ERROR: {str(e)}]"

    return Response(stream(), mimetype="text/plain")

# ── Chat persistence endpoints (server-side fallback) ──
@app.route("/chats", methods=["GET"])
def get_chats():
    return jsonify(load_chats())

@app.route("/chats", methods=["POST"])
def save_chat():
    chats = load_chats()
    new_chat = request.json
    chats = [c for c in chats if c.get("id") != new_chat.get("id")]
    chats.insert(0, new_chat)
    save_chats(chats)
    return jsonify({"ok": True})

@app.route("/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    chats = [c for c in load_chats() if c.get("id") != chat_id]
    save_chats(chats)
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("🚀 NAIF server running → http://localhost:5000")
    app.run(port=5000, debug=True)