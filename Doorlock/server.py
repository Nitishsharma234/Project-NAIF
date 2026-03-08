"""
server.py  —  Smart Door Security Dashboard
============================================
New features:
  - Enroll Face button → modal popup (name input → starts enrollment)
  - Stop Alarm button  → kills alarm immediately
  - Face Scan ON/OFF   → toggle button
  - Alarm flashing banner when active
  - Pending action queue polled by main.py every frame
"""

import os, time, threading, queue
from datetime import datetime
from flask import Flask, Response, jsonify, render_template_string, send_file, request

# ─────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────
door_state = {
    "locked":            True,
    "last_action":       "init",
    "last_action_time":  datetime.now().isoformat(),
    "alert_count":       0,
    "enrolled_people":   [],
    "system_status":     "Starting…",
    "alarm_active":      False,
    "face_scan_enabled": True,
}

_state_lock  = threading.Lock()
_frame_lock  = threading.Lock()
_log_lock    = threading.Lock()

latest_frame_bytes = None
activity_log       = []

# Action queue — main.py pops one action per frame
_action_queue = queue.Queue()

# Callbacks registered by main.py
_stop_alarm_cb   = None
_face_scan_cb    = None


def register_stop_alarm_callback(cb):    global _stop_alarm_cb;  _stop_alarm_cb  = cb
def register_face_scan_callback(cb):     global _face_scan_cb;   _face_scan_cb   = cb


def update_frame(jpeg_bytes):
    global latest_frame_bytes
    with _frame_lock:
        latest_frame_bytes = jpeg_bytes


def add_log(msg):
    with _log_lock:
        ts = datetime.now().strftime("%H:%M:%S")
        activity_log.append(f"[{ts}] {msg}")
        if len(activity_log) > 60:
            activity_log.pop(0)


def set_door_locked(locked: bool, reason: str = "remote"):
    with _state_lock:
        door_state["locked"]          = locked
        door_state["last_action"]     = "locked" if locked else "unlocked"
        door_state["last_action_time"]= datetime.now().isoformat()
    icon = "🔒" if locked else "🔓"
    add_log(f"Door {icon} {'LOCKED' if locked else 'UNLOCKED'} — {reason}")
    print(f"[DOOR] {icon} {'LOCKED' if locked else 'UNLOCKED'} ({reason})")


def get_door_locked():
    with _state_lock:
        return door_state["locked"]


def pop_pending_action():
    """main.py calls this every frame to get one queued action."""
    try:    return _action_queue.get_nowait()
    except: return None


def get_pending_enroll_name():
    return None   # legacy stub — enrollment now uses action queue


# ─────────────────────────────────────────────────────────────
# Dashboard HTML
# ─────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Door Security</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:     #0d1117; --card:   #161b22; --border: #30363d;
  --muted:  #8b949e; --green:  #238636; --red:    #da3633;
  --blue:   #1f6feb; --orange: #d29922; --text:   #e6edf3;
  --purple: #8957e5; --yellow: #e3b341;
}
body { font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }

/* HEADER */
header { background:var(--card); border-bottom:1px solid var(--border); padding:12px 22px;
  display:flex; align-items:center; gap:10px; position:sticky; top:0; z-index:200; }
header h1 { font-size:1.2rem; flex:1; }
.badge { padding:5px 13px; border-radius:20px; font-size:0.76rem; font-weight:700; }
.badge.locked   { background:var(--red);   color:#fff; }
.badge.unlocked { background:var(--green); color:#fff; }

/* ALARM BANNER */
#alarm-banner {
  display:none; background:#7d1a1a; border-bottom:2px solid var(--red);
  padding:10px 22px; text-align:center; font-weight:700; font-size:0.95rem;
  color:#fff; animation:flash 0.8s step-start infinite; position:sticky; top:57px; z-index:199;
}
@keyframes flash { 50%{ background:#da3633; } }

/* LAYOUT */
.layout { display:grid; grid-template-columns:1fr 330px; gap:16px; padding:16px 22px;
  max-width:1340px; margin:0 auto; }
@media(max-width:900px){ .layout{ grid-template-columns:1fr; } }

/* CARD */
.card { background:var(--card); border:1px solid var(--border); border-radius:12px;
  padding:16px; margin-bottom:14px; }
.card-title { font-size:0.7rem; color:var(--muted); text-transform:uppercase;
  letter-spacing:1px; margin-bottom:12px; display:flex; align-items:center; gap:6px; }

/* CAMERA */
#cam-wrap { position:relative; background:#000; border-radius:8px; overflow:hidden; aspect-ratio:16/9; }
#cam-canvas { width:100%; height:100%; display:block; object-fit:contain; }
#cam-overlay { position:absolute; inset:0; display:flex; flex-direction:column;
  align-items:center; justify-content:center; background:rgba(0,0,0,0.78);
  font-size:0.9rem; color:var(--muted); gap:10px; }
#cam-overlay .spinner { width:36px; height:36px; border:3px solid var(--border);
  border-top-color:var(--blue); border-radius:50%; animation:spin 0.8s linear infinite; }
@keyframes spin{ to{ transform:rotate(360deg); } }
#fullscreen-btn { position:absolute; top:8px; right:8px; background:rgba(0,0,0,0.5);
  border:none; color:#fff; border-radius:6px; padding:4px 8px; cursor:pointer; font-size:1rem; }
#cam-status { position:absolute; bottom:8px; left:8px; background:rgba(0,0,0,0.55);
  border-radius:4px; padding:2px 8px; font-size:0.7rem; color:#aaa; }
#face-scan-indicator { position:absolute; top:8px; left:8px; border-radius:5px;
  padding:3px 10px; font-size:0.72rem; font-weight:700; }
.face-on  { background:rgba(35,134,54,0.85); color:#fff; }
.face-off { background:rgba(60,60,60,0.85);  color:#aaa; }

/* BUTTONS */
.btn { display:flex; align-items:center; justify-content:center; gap:7px;
  width:100%; padding:10px; margin:6px 0; border:none; border-radius:8px;
  font-size:0.92rem; font-weight:600; cursor:pointer; transition:filter 0.15s, transform 0.1s; }
.btn:hover  { filter:brightness(1.15); }
.btn:active { transform:scale(0.97); }
.btn-unlock  { background:var(--green);  color:#fff; }
.btn-lock    { background:var(--red);    color:#fff; }
.btn-enroll  { background:var(--purple); color:#fff; }
.btn-alarm   { background:#b91c1c;       color:#fff; display:none; }
.btn-faceon  { background:var(--green);  color:#fff; }
.btn-faceoff { background:#374151;       color:#ccc; }
.btn-refresh { background:var(--blue);   color:#fff; }

/* STATUS */
.s-row { display:flex; justify-content:space-between; align-items:center;
  padding:7px 0; border-bottom:1px solid var(--border); font-size:0.86rem; }
.s-row:last-child { border-bottom:none; }
.s-key { color:var(--muted); }
.s-val { font-weight:600; }

/* ALERT COUNT */
.alert-big { font-size:2.6rem; font-weight:800; color:#f85149; text-align:center;
  padding:6px 0; text-shadow:0 0 18px rgba(248,81,73,0.4); }
.alert-sub { text-align:center; color:var(--muted); font-size:0.76rem; margin-top:2px; }

/* LOG */
#log { height:130px; overflow-y:auto; font-size:0.74rem; font-family:'Consolas',monospace;
  background:#090d12; padding:8px 10px; border-radius:6px;
  border:1px solid var(--border); color:#7ee787; line-height:1.65; }
#log::-webkit-scrollbar { width:4px; }
#log::-webkit-scrollbar-thumb { background:var(--border); border-radius:2px; }

/* ENROLLED */
#enrolled-list { display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; min-height:28px; }
.person-chip { background:rgba(35,134,54,0.16); border:1px solid var(--green);
  color:#7ee787; border-radius:20px; padding:3px 10px; font-size:0.76rem; font-weight:600; }
.no-people { color:var(--muted); font-size:0.8rem; }

/* ALERT PHOTOS */
#alerts-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(78px,1fr));
  gap:6px; margin-top:6px; max-height:155px; overflow-y:auto; }
.alert-thumb { width:100%; aspect-ratio:1; object-fit:cover; border-radius:5px;
  border:1px solid var(--border); cursor:pointer; transition:border-color 0.15s; }
.alert-thumb:hover { border-color:var(--orange); }

/* ── MODAL ── */
.modal-backdrop { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.72);
  z-index:500; align-items:center; justify-content:center; }
.modal-backdrop.open { display:flex; }
.modal { background:var(--card); border:1px solid var(--border); border-radius:14px;
  padding:28px 26px; width:100%; max-width:420px; position:relative;
  animation:pop 0.2s ease; }
@keyframes pop { from{ transform:scale(0.9); opacity:0; } to{ transform:scale(1); opacity:1; } }
.modal h2 { font-size:1.15rem; margin-bottom:6px; }
.modal p  { color:var(--muted); font-size:0.84rem; margin-bottom:18px; }
.modal-close { position:absolute; top:14px; right:16px; background:none; border:none;
  color:var(--muted); font-size:1.3rem; cursor:pointer; }
.modal-close:hover { color:var(--text); }
.field-label { font-size:0.8rem; color:var(--muted); margin-bottom:5px; display:block; }
.field-input { width:100%; padding:10px 12px; background:#0d1117; border:1px solid var(--border);
  border-radius:8px; color:var(--text); font-size:0.95rem; outline:none; }
.field-input:focus { border-color:var(--blue); }
.modal-hint { font-size:0.76rem; color:var(--muted); margin-top:8px; }
.modal-btns { display:flex; gap:10px; margin-top:18px; }
.modal-btns .btn { margin:0; }
.btn-cancel { background:#21262d; color:var(--text); }

/* progress inside modal */
#enroll-progress-wrap { display:none; margin-top:16px; }
#enroll-prog-bar-bg { background:#21262d; border-radius:6px; height:10px; overflow:hidden; }
#enroll-prog-bar    { background:var(--green); height:10px; width:0%; transition:width 0.3s; border-radius:6px; }
#enroll-prog-label  { font-size:0.78rem; color:var(--muted); margin-top:6px; text-align:center; }
#enroll-status-msg  { font-size:0.82rem; margin-top:10px; text-align:center; min-height:20px; }

footer { text-align:center; padding:12px; color:var(--muted); font-size:0.73rem; }
</style>
</head>
<body>

<!-- HEADER -->
<header>
  <span style="font-size:1.4rem">🔐</span>
  <h1>Smart Door Security System</h1>
  <span id="door-badge" class="badge locked">🔒 LOCKED</span>
</header>

<!-- ALARM BANNER (shown when alarm is active) -->
<div id="alarm-banner">
  🚨 ALARM ACTIVE — Intruder detected!
  <button onclick="stopAlarm()" style="margin-left:16px;padding:4px 14px;border:none;
    border-radius:6px;background:#fff;color:#b91c1c;font-weight:700;cursor:pointer;">
    🔕 Stop Alarm
  </button>
</div>

<!-- MAIN LAYOUT -->
<div class="layout">

  <!-- LEFT column -->
  <div>

    <!-- Camera feed -->
    <div class="card">
      <div class="card-title">📷 Live Camera Feed</div>
      <div id="cam-wrap">
        <canvas id="cam-canvas"></canvas>
        <div id="cam-overlay">
          <div class="spinner"></div>
          <span id="overlay-msg">Connecting to camera…</span>
        </div>
        <div id="face-scan-indicator" class="face-on">FACE: ON</div>
        <button id="fullscreen-btn" onclick="toggleFullscreen()" title="Fullscreen">⛶</button>
        <div id="cam-status">Waiting…</div>
      </div>
    </div>

    <!-- Enrolled people -->
    <div class="card">
      <div class="card-title">👤 Enrolled People</div>
      <div id="enrolled-list">
        <span class="no-people">No faces enrolled yet</span>
      </div>
    </div>

    <!-- Alert snapshots -->
    <div class="card">
      <div class="card-title">📸 Alert Snapshots</div>
      <div id="alerts-grid">
        <span class="no-people" style="font-size:0.78rem">No snapshots yet</span>
      </div>
    </div>

  </div>

  <!-- RIGHT column -->
  <div>

    <!-- Door control -->
    <div class="card">
      <div class="card-title">🎛️ Door Control</div>
      <button class="btn btn-unlock"  onclick="setDoor(false)">🔓 Unlock Door</button>
      <button class="btn btn-lock"    onclick="setDoor(true)">🔒 Lock Door</button>
    </div>

    <!-- Face recognition -->
    <div class="card">
      <div class="card-title">🧠 Face Recognition</div>
      <button class="btn btn-enroll"  onclick="openEnrollModal()">📸 Enroll New Face</button>
      <button id="btn-face-toggle" class="btn btn-faceon" onclick="toggleFaceScan()">
        👁️ Face Scan: ON
      </button>
    </div>

    <!-- Alarm control -->
    <div class="card">
      <div class="card-title">🚨 Alarm Control</div>
      <button id="btn-stop-alarm" class="btn btn-alarm" onclick="stopAlarm()">
        🔕 Stop Alarm
      </button>
      <p id="no-alarm-msg" style="color:var(--muted);font-size:0.82rem;text-align:center;padding:8px 0">
        ✅ No alarm active
      </p>
    </div>

    <!-- System status -->
    <div class="card">
      <div class="card-title">📊 System Status</div>
      <div class="s-row"><span class="s-key">Door</span>       <span class="s-val" id="s-door">—</span></div>
      <div class="s-row"><span class="s-key">Last Action</span><span class="s-val" id="s-action">—</span></div>
      <div class="s-row"><span class="s-key">Updated</span>    <span class="s-val" id="s-time">—</span></div>
      <div class="s-row"><span class="s-key">System</span>     <span class="s-val" id="s-sys">—</span></div>
    </div>

    <!-- Security alerts -->
    <div class="card">
      <div class="card-title">🚨 Security Alerts</div>
      <div class="alert-big" id="s-alerts">0</div>
      <div class="alert-sub">wrong password / gesture × 3</div>
    </div>

    <!-- Activity log -->
    <div class="card">
      <div class="card-title">📋 Activity Log</div>
      <div id="log">System starting…</div>
    </div>

  </div>
</div>

<footer>Smart Door Security &nbsp;·&nbsp; <span id="f-time"></span></footer>

<!-- ══════════════════════════════════════════
     ENROLL MODAL
══════════════════════════════════════════ -->
<div class="modal-backdrop" id="enroll-modal">
  <div class="modal">
    <button class="modal-close" onclick="closeEnrollModal()">✕</button>
    <h2>📸 Enroll New Face</h2>
    <p>Enter the person's name then click Start. The camera will automatically capture 300 images and train the model.</p>

    <label class="field-label">Full Name</label>
    <input  id="enroll-name-input" class="field-input"
            type="text" placeholder="e.g. John Smith"
            onkeydown="if(event.key==='Enter') startEnroll()" />
    <div class="modal-hint">
      💡 After clicking Start, look straight at the camera window on your PC.
      Training begins automatically after 300 captures.
    </div>

    <!-- Progress (shown after start) -->
    <div id="enroll-progress-wrap">
      <div id="enroll-prog-bar-bg">
        <div id="enroll-prog-bar"></div>
      </div>
      <div id="enroll-prog-label">0 / 300 images captured</div>
      <div id="enroll-status-msg"></div>
    </div>

    <div class="modal-btns" id="enroll-action-btns">
      <button class="btn btn-enroll" onclick="startEnroll()">▶ Start Enrollment</button>
      <button class="btn btn-cancel" onclick="closeEnrollModal()">Cancel</button>
    </div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════
// CAMERA STREAM
// ═══════════════════════════════════════════
const canvas    = document.getElementById('cam-canvas');
const ctx       = canvas.getContext('2d');
const overlay   = document.getElementById('cam-overlay');
const camStatus = document.getElementById('cam-status');
let frameCount = 0, lastFpsCheck = Date.now(), displayFps = 0, streamOk = false;

async function fetchFrame() {
  try {
    const res = await fetch('/snapshot?t=' + Date.now());
    if (!res.ok) throw new Error();
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const img  = new Image();
    img.onload = () => {
      canvas.width  = img.naturalWidth  || 1280;
      canvas.height = img.naturalHeight || 720;
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      frameCount++;
      if (!streamOk) { streamOk = true; overlay.style.display = 'none'; }
      const now = Date.now();
      if (now - lastFpsCheck >= 1000) {
        displayFps = Math.round(frameCount * 1000 / (now - lastFpsCheck));
        frameCount = 0; lastFpsCheck = now;
      }
      camStatus.textContent = `${canvas.width}×${canvas.height}  ${displayFps} fps`;
    };
    img.onerror = () => URL.revokeObjectURL(url);
    img.src = url;
  } catch(e) {
    if (streamOk) { streamOk=false; overlay.style.display='flex';
      document.getElementById('overlay-msg').textContent='Reconnecting…'; }
  }
}
function streamLoop() { fetchFrame().finally(() => setTimeout(streamLoop, 33)); }
streamLoop();

function toggleFullscreen() {
  const w = document.getElementById('cam-wrap');
  if (!document.fullscreenElement) w.requestFullscreen().catch(()=>{});
  else document.exitFullscreen();
}

// ═══════════════════════════════════════════
// DOOR
// ═══════════════════════════════════════════
async function setDoor(lock) {
  const res  = await fetch(lock ? '/lock' : '/unlock', {method:'POST'});
  const data = await res.json();
  addLog(data.message);
  refreshStatus();
}

// ═══════════════════════════════════════════
// ALARM
// ═══════════════════════════════════════════
async function stopAlarm() {
  await fetch('/stop_alarm', {method:'POST'});
  addLog('🔕 Alarm stopped via dashboard');
  setAlarmUI(false);
}

function setAlarmUI(active) {
  const banner   = document.getElementById('alarm-banner');
  const btn      = document.getElementById('btn-stop-alarm');
  const noAlarm  = document.getElementById('no-alarm-msg');
  if (active) {
    banner.style.display  = 'block';
    btn.style.display     = 'flex';
    noAlarm.style.display = 'none';
    document.title        = '🚨 ALARM — Smart Door';
  } else {
    banner.style.display  = 'none';
    btn.style.display     = 'none';
    noAlarm.style.display = 'block';
    document.title        = 'Smart Door Security';
  }
}

// ═══════════════════════════════════════════
// FACE SCAN TOGGLE
// ═══════════════════════════════════════════
let faceScanOn = true;
async function toggleFaceScan() {
  faceScanOn = !faceScanOn;
  await fetch('/face_scan', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({enabled: faceScanOn})});
  updateFaceScanUI(faceScanOn);
  addLog(faceScanOn ? '👁️ Face scan enabled' : '🚫 Face scan disabled');
}
function updateFaceScanUI(on) {
  const btn = document.getElementById('btn-face-toggle');
  const ind = document.getElementById('face-scan-indicator');
  btn.textContent = on ? '👁️ Face Scan: ON' : '🚫 Face Scan: OFF';
  btn.className   = 'btn ' + (on ? 'btn-faceon' : 'btn-faceoff');
  ind.textContent = 'FACE: ' + (on ? 'ON' : 'OFF');
  ind.className   = 'face-scan-indicator ' + (on ? 'face-on' : 'face-off');
}

// ═══════════════════════════════════════════
// ENROLL MODAL
// ═══════════════════════════════════════════
let enrollPolling = null;

function openEnrollModal() {
  document.getElementById('enroll-modal').classList.add('open');
  document.getElementById('enroll-name-input').value = '';
  document.getElementById('enroll-name-input').disabled = false;
  document.getElementById('enroll-progress-wrap').style.display = 'none';
  document.getElementById('enroll-action-btns').style.display   = 'flex';
  document.getElementById('enroll-status-msg').textContent = '';
  document.getElementById('enroll-prog-bar').style.width = '0%';
  setTimeout(() => document.getElementById('enroll-name-input').focus(), 80);
}

function closeEnrollModal() {
  document.getElementById('enroll-modal').classList.remove('open');
  if (enrollPolling) { clearInterval(enrollPolling); enrollPolling = null; }
}

// Close modal when clicking backdrop
document.getElementById('enroll-modal').addEventListener('click', function(e) {
  if (e.target === this) closeEnrollModal();
});

async function startEnroll() {
  const raw  = document.getElementById('enroll-name-input').value.trim();
  if (!raw) {
    document.getElementById('enroll-name-input').focus();
    return;
  }
  const name = raw.replace(/\s+/g, '_');

  // Send to server
  const res  = await fetch('/enroll', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name})});
  const data = await res.json();
  if (data.status !== 'ok') {
    alert('Error: ' + (data.message || 'Unknown error'));
    return;
  }

  // Switch modal to progress view
  document.getElementById('enroll-name-input').disabled = true;
  document.getElementById('enroll-action-btns').style.display   = 'none';
  document.getElementById('enroll-progress-wrap').style.display = 'block';
  document.getElementById('enroll-status-msg').textContent =
    `📸 Capturing images for "${raw}"…\nLook straight at the camera window on your PC!`;

  // Poll progress
  enrollPolling = setInterval(async () => {
    try {
      const pr   = await fetch('/enroll_progress');
      const pd   = await pr.json();
      const pct  = Math.round((pd.count / pd.total) * 100);
      document.getElementById('enroll-prog-bar').style.width   = pct + '%';
      document.getElementById('enroll-prog-label').textContent =
        `${pd.count} / ${pd.total} images captured`;

      if (pd.done) {
        clearInterval(enrollPolling); enrollPolling = null;
        if (pd.cancelled) {
          document.getElementById('enroll-status-msg').textContent = '❌ Enrollment cancelled.';
        } else {
          document.getElementById('enroll-prog-bar').style.width = '100%';
          document.getElementById('enroll-status-msg').textContent =
            `✅ "${raw}" enrolled successfully! Model trained.`;
          addLog(`✅ New face enrolled: ${raw}`);
          refreshStatus();
          setTimeout(closeEnrollModal, 2500);
        }
      }
    } catch(e) {}
  }, 600);
}

// ═══════════════════════════════════════════
// STATUS REFRESH
// ═══════════════════════════════════════════
async function refreshStatus() {
  try {
    const data = await (await fetch('/status')).json();
    document.getElementById('s-door').textContent   = data.locked ? '🔒 Locked' : '🔓 Unlocked';
    document.getElementById('s-action').textContent = data.last_action;
    document.getElementById('s-alerts').textContent = data.alert_count;
    document.getElementById('s-sys').textContent    = data.system_status || '—';
    const t = new Date(data.last_action_time);
    document.getElementById('s-time').textContent = isNaN(t) ? data.last_action_time : t.toLocaleTimeString();

    const badge = document.getElementById('door-badge');
    badge.textContent = data.locked ? '🔒 LOCKED' : '🔓 UNLOCKED';
    badge.className   = 'badge ' + (data.locked ? 'locked' : 'unlocked');

    // Alarm state
    setAlarmUI(data.alarm_active || false);

    // Face scan state
    const fse = data.face_scan_enabled !== false;
    if (fse !== faceScanOn) { faceScanOn = fse; updateFaceScanUI(fse); }

    // Enrolled chips
    const el = document.getElementById('enrolled-list');
    if (data.enrolled_people && data.enrolled_people.length) {
      el.innerHTML = data.enrolled_people
        .map(n => `<span class="person-chip">👤 ${n}</span>`).join('');
    } else {
      el.innerHTML = '<span class="no-people">No faces enrolled — click Enroll New Face</span>';
    }
  } catch(e) {}
}

// Alert snapshots
async function refreshAlerts() {
  try {
    const data = await (await fetch('/alert_images')).json();
    const grid = document.getElementById('alerts-grid');
    if (data.images && data.images.length) {
      grid.innerHTML = data.images.map(f =>
        `<img class="alert-thumb" src="/alert_image/${f}"
              title="${f}" onclick="window.open('/alert_image/${f}')">`).join('');
    }
  } catch(e) {}
}

// Log
const logEl = document.getElementById('log');
let lastLogLen = 0;
async function refreshLog() {
  try {
    const data = await (await fetch('/log')).json();
    if (data.log.length !== lastLogLen) {
      lastLogLen = data.log.length;
      logEl.textContent = data.log.join('\n');
      logEl.scrollTop   = logEl.scrollHeight;
    }
  } catch(e) {}
}
function addLog(msg) {
  const now = new Date().toLocaleTimeString();
  logEl.textContent += `\n[${now}] ${msg}`;
  logEl.scrollTop    = logEl.scrollHeight;
}

function updateFooter() {
  document.getElementById('f-time').textContent = new Date().toLocaleString();
}

refreshStatus(); refreshAlerts(); refreshLog();
setInterval(refreshStatus, 2000);
setInterval(refreshLog,    1500);
setInterval(refreshAlerts, 5000);
setInterval(updateFooter,  1000);
updateFooter();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────
# Enrollment progress tracking
# ─────────────────────────────────────────────────────────────
_enroll_progress = {"count": 0, "total": 300, "done": False, "cancelled": False}
_enroll_lock     = threading.Lock()

def update_enroll_progress(count, total, done=False, cancelled=False):
    with _enroll_lock:
        _enroll_progress.update(
            {"count": count, "total": total, "done": done, "cancelled": cancelled})

# ─────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "smart_door_2026"


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/snapshot")
def snapshot():
    with _frame_lock:
        frame = latest_frame_bytes
    if frame is None:
        placeholder = (
            b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
            b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
            b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a'
            b'\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\x1e\xef'
            b'A\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01'
            b'\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00'
            b'\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08'
            b'\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\x0f\xff\xd9'
        )
        return Response(placeholder, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    return Response(frame, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/unlock", methods=["POST"])
def unlock():
    set_door_locked(False, "web dashboard")
    return jsonify({"status":"ok","message":"🔓 Door UNLOCKED via dashboard","locked":False})


@app.route("/lock", methods=["POST"])
def lock():
    set_door_locked(True, "web dashboard")
    return jsonify({"status":"ok","message":"🔒 Door LOCKED via dashboard","locked":True})


@app.route("/stop_alarm", methods=["POST"])
def stop_alarm_route():
    _action_queue.put("stop_alarm")
    with _state_lock:
        door_state["alarm_active"] = False
    add_log("🔕 Stop alarm requested via dashboard")
    return jsonify({"status": "ok", "message": "Alarm stop requested"})


@app.route("/face_scan", methods=["POST"])
def face_scan_route():
    data    = request.get_json(silent=True) or {}
    enabled = data.get("enabled", True)
    _action_queue.put("face_scan_on" if enabled else "face_scan_off")
    with _state_lock:
        door_state["face_scan_enabled"] = enabled
    return jsonify({"status": "ok", "enabled": enabled})


@app.route("/enroll", methods=["POST"])
def enroll_route():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip().replace(" ", "_")
    if not name:
        return jsonify({"status": "error", "message": "Name is required"}), 400
    # Reset progress
    update_enroll_progress(0, 300, done=False, cancelled=False)
    # Queue action for main.py
    _action_queue.put(f"enroll:{name}")
    add_log(f"📸 Enrollment started: {name}")
    return jsonify({"status": "ok", "name": name})


@app.route("/enroll_progress")
def enroll_progress_route():
    with _enroll_lock:
        return jsonify(dict(_enroll_progress))


@app.route("/status")
def status():
    with _state_lock:
        return jsonify(dict(door_state))


@app.route("/log")
def get_log():
    with _log_lock:
        return jsonify({"log": list(activity_log)})


@app.route("/alert_images")
def alert_images():
    alerts_dir = "alerts"
    if not os.path.isdir(alerts_dir):
        return jsonify({"images": []})
    supported = (".jpg", ".jpeg", ".png")
    files = sorted(
        [f for f in os.listdir(alerts_dir) if f.lower().endswith(supported)],
        reverse=True)[:20]
    return jsonify({"images": files})


@app.route("/alert_image/<filename>")
def alert_image(filename):
    alerts_dir = os.path.abspath("alerts")
    filepath   = os.path.join(alerts_dir, filename)
    if not filepath.startswith(alerts_dir) or not os.path.isfile(filepath):
        return "Not found", 404
    return send_file(filepath, mimetype="image/jpeg")


@app.route("/trigger_alert", methods=["POST"])
def trigger_alert():
    with _state_lock:
        door_state["alert_count"] += 1
    add_log(f"🚨 Security alert #{door_state['alert_count']} triggered")
    return jsonify({"status": "ok", "alert_count": door_state["alert_count"]})


def start_server(host="0.0.0.0", port=5000):
    print(f"[SERVER] Dashboard → http://localhost:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    start_server()
