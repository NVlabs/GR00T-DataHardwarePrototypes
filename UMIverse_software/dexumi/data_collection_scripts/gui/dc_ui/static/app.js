// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* DexUMI data-collection GUI — vanilla JS front-end over gui/dc_ui/server.py */

const ROLE = 'recorder';
const SLOTS = [
  { id: 'right_iphone', kind: 'iphone', label: 'Right iPhone', key: 'right_tracking_camera', side: null },
  { id: 'left_iphone', kind: 'iphone', label: 'Left iPhone', key: 'left_tracking_camera', side: null },
  { id: 'right_oak', kind: 'oak', label: 'Right wrist OAK', key: 'right_umi_camera', side: null },
  { id: 'left_oak', kind: 'oak', label: 'Left wrist OAK', key: 'left_umi_camera', side: null },
  { id: 'head_oak', kind: 'oak', label: 'Head OAK', key: 'head_umi_camera', side: null, optional: true },
  { id: 'right_exo', kind: 'exo', label: 'Right exoskeleton', key: 'right_dexumi_exoskeleton', side: 'Right' },
  { id: 'left_exo', kind: 'exo', label: 'Left exoskeleton', key: 'left_dexumi_exoskeleton', side: 'Left' },
];
const LIVE_STATES = ['starting', 'ready', 'adopted', 'stopping'];
const state = {
  defaults: {}, detection: null, saved: {}, plan: null, blocked: true,
  live: false, procs: {}, procsSeq: 0, logSeq: 0, composeSeq: 0, composeTimer: null, detecting: false, busy: false,
  identifying: false, identifyRequest: false, identifyRemote: false, identifyUncertain: false,
  identifyEpoch: 0, pairing: null,
  cameraOperation: null, streamDiagnostics: [], startRequest: false, cancellingStream: false,
  preview: emptyPreview(),
  // The server owns scan validity. A ready launch consumes it only with
  // --scan-every-launch; device-binding refusal always invalidates it. Manual camera
  // selection remains available.



  identified: {}, scanEveryLaunch: false,
  identificationResync: false,
};
const IDENTIFIED_SLOTS = ['left_iphone', 'right_iphone', 'left_oak', 'right_oak'];
const isCameraSlot = (slot) => slot.kind !== 'exo';

const el = (id) => document.getElementById(id);
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function showToast(msg, kind) {
  const t = el('toast');
  t.textContent = msg;
  t.className = 'toast' + (kind === 'error' ? ' error' : '');
  t.hidden = false;
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => { t.hidden = true; }, 8000);
}

function detailText(body, fallback) {
  if (!body || body.detail === undefined) return fallback;
  const d = body.detail;
  if (typeof d === 'string') return d;
  if (d && typeof d.message === 'string') return d.message;
  return JSON.stringify(d);
}

async function api(path, opts) {
  const resp = await fetch(path, Object.assign({ headers: { 'Content-Type': 'application/json' } }, opts || {}));
  let body = null;
  try { body = await resp.json(); } catch (e) { body = null; }
  // 422 with a findings list = a blocked plan (rendered, not thrown)
  const blocked = resp.status === 422 && body && Array.isArray(body.findings);
  if (blocked) return { status: 422, body };
  if (!resp.ok) {
    const err = new Error(`${resp.status}: ${detailText(body, resp.statusText)}`);
    err.detail = detailText(body, resp.statusText);
    err.code = body && body.detail && body.detail.code;
    throw err;
  }
  return { status: resp.status, body };
}
const post = (path, payload) => api(path, { method: 'POST', body: JSON.stringify(payload || {}) });

// ─── form ──────────────────────────────────────────────────────────────────
function collectForm() {
  const form = {
    task_name: el('task-name').value.trim(),
    operator_name: el('operator-name').value.trim(),
    task_description: el('task-description').value.trim(),
  };
  SLOTS.forEach((s) => { const sel = el(`slot-${s.id}`); form[s.id] = sel ? sel.value : (state.saved[s.id] || ''); });
  return form;
}

function fillForm(form) {
  if (!form) return;
  el('task-name').value = form.task_name || '';
  el('operator-name').value = form.operator_name || '';
  el('task-description').value = form.task_description || '';
  el('data-dir').value = ''; // the server computes it from the current inputs
  // Camera slots never come from the saved form: they are restored from the
  // server's identification record instead (blank after a GUI restart).
  SLOTS.forEach((s) => { state.saved[s.id] = isCameraSlot(s) ? '' : (form[s.id] || ''); });
}

// ─── devices ───────────────────────────────────────────────────────────────
function deviceOptions(slot) {
  const devices = (state.detection && state.detection[slot.kind]) || [];
  const opts = [`<option value="">${slot.optional ? '— none: record without this camera —' : '— not selected —'}</option>`];
  devices.forEach((d) => {
    opts.push(`<option value="${esc(d.name || '')}"${d.name ? '' : ' disabled'}>${esc(d.label)}</option>`);
  });
  return opts.join('');
}

// Calibration fixes each glove's hand; fill its side from the connected calibrated
// board.

function exoBoard(slot) {
  const boards = ((state.detection && state.detection.exo) || []).filter((d) => d.name && d.side === slot.side);
  return boards.length === 1 ? boards[0] : null;
}

// Require device assignment only for real hardware slots declared by the base config.

function activeSlots() {
  const active = state.defaults.active_slots;
  return Array.isArray(active) ? SLOTS.filter((s) => active.includes(s.id)) : SLOTS;
}

function renderDeviceSelects() {
  const holder = el('device-slots');
  const previous = {};
  const slots = activeSlots();
  const inactive = SLOTS.filter((s) => !slots.includes(s));
  el('device-inactive').textContent = inactive.length
    ? `No device needed for: ${inactive.map((s) => s.label).join(', ')} — the base config uses mock sources there.`
    : '';
  if (!slots.length) {
    holder.innerHTML = '';
    el('step-devices').classList.add('complete');
    return;
  }
  SLOTS.forEach((s) => { const sel = el(`slot-${s.id}`); previous[s.id] = sel ? sel.value : state.saved[s.id] || ''; });
  const exoField = (s) => {
    const board = exoBoard(s);
    const glove = board ? esc(board.label)
      : `No calibrated ${s.side.toLowerCase()} glove connected: plug it in, or calibrate it in <a href="#panel-calibrate">Step 01</a>.`;
    return `
    <div class="camera-card" id="card-${s.id}">
      <div class="exo-slot"><span class="camera-heading"><span>Exoskeleton</span><span class="camera-requirement">From calibration</span></span><p class="exo-glove${board ? '' : ' missing'}">${glove}</p><input type="hidden" id="slot-${s.id}" value="${esc(board ? board.name : '')}"></div>
      <div class="camera-key mono">${esc(s.key)}</div>
    </div>`;
  };
  const field = (s) => s.kind === 'exo' ? exoField(s) : `
    <div class="camera-card" id="card-${s.id}">
      <label><span class="camera-heading"><span>${s.kind === 'iphone' ? 'Tracking camera' : s.optional ? 'Head camera' : 'Wrist camera'}</span><span class="camera-requirement">${s.optional ? 'Optional' : 'Required'}</span></span><select id="slot-${s.id}" aria-label="${esc(s.label)}">${deviceOptions(s)}</select></label>
      <div class="camera-key mono">${esc(s.key)}</div>
    </div>`;
  const hands = ['left', 'right'].map((side) => {
    const handSlots = ['iphone', 'oak', 'exo'].map((kind) => slots.find((s) => s.id === `${side}_${kind}`)).filter(Boolean);
    if (!handSlots.length) return '';
    const title = side === 'left' ? 'Left hand' : 'Right hand';
    return `<section class="hand-card" aria-labelledby="${side}-hand-title"><h3 id="${side}-hand-title">${title}</h3><div class="hand-content"><img class="hand-diagram ${side}" src="/static/hand-devices.svg" alt="Simplified DexUMI finger linkages with a separate iPhone cradle and single-lens OAK camera; schematic connections, not mount positions."><div class="hand-fields">${handSlots.map(field).join('')}</div></div></section>`;
  }).join('');
  holder.innerHTML = hands + slots.filter((s) => s.optional).map((s) => `<div class="optional-device">${field(s)}<p class="field-hint">Leave unselected to record without a head camera.</p></div>`).join('');
  slots.forEach((s) => {
    const sel = el(`slot-${s.id}`);
    if (s.kind !== 'exo') {
      // Cameras are never guessed: which OAK is the head vs a wrist is the operator's call.
      const wanted = previous[s.id];
      sel.value = wanted && Array.from(sel.options).some((o) => o.value === wanted && !o.disabled) ? wanted : '';
      sel.onchange = () => { state.saved[s.id] = sel.value; el(`card-${s.id}`).classList.toggle('assigned', !!sel.value); updateDeviceStatus(); scheduleCompose(); };
    }
    el(`card-${s.id}`).classList.toggle('assigned', !!sel.value);
  });
  updateDeviceStatus();
}

const identifiedNames = (identified) => IDENTIFIED_SLOTS.map((id) => (identified && identified[id]) || '').join('|');

// Refresh identification when a launch consumes/refuses it or another tab scans.
// Preserve selected slots and bind only while all IDs match. force also refreshes
// preflight after recorder exit.





function syncIdentification(body, force = false) {
  if (!body.identified || typeof body.identified !== 'object') return;
  state.scanEveryLaunch = !!body.scan_every_launch;
  if (!force && identifiedNames(body.identified) === identifiedNames(state.identified)) return;
  state.identified = { ...body.identified };
  scheduleCompose(0);
}

// With both wrists identified and exactly three available OAKs, suggest the remaining
// OAK for an empty head slot; retain manual override.

function inferHeadOak() {
  const left = state.saved.left_oak;
  const right = state.saved.right_oak;
  const available = ((state.detection && state.detection.oak) || []).filter((camera) => !camera.in_use);
  if (!left || !right || left === right || available.length !== 3 || new Set(available.map((camera) => camera.name)).size !== 3) return '';
  const remaining = available.filter((camera) => camera.name !== left && camera.name !== right);
  return remaining.length === 1 ? remaining[0].name : '';
}

function updateDeviceStatus() {
  const slots = activeSlots();
  const required = slots.filter((s) => !s.optional);
  const value = (s) => (state.detection && el(`slot-${s.id}`) ? el(`slot-${s.id}`).value : '');
  const found = required.filter((s) => value(s)).length;
  el('step-devices').classList.toggle('complete', found === required.length);
}

async function loadDevices() {
  if (state.detecting || state.identifying || state.live || state.busy) return;
  state.detecting = true;
  updateControls();
  el('device-status').textContent = 'Enumerating iPhones, OAK cameras and exoskeleton boards… (up to 20 s)';
  try {
    const { body } = await api('/api/devices');
    state.detection = body;
    el('device-status').textContent = '';
    el('device-errors').textContent = (body.errors || []).join('\n');
    renderDeviceSelects();
    scheduleCompose(0);
  } catch (e) {
    el('device-status').textContent = '';
    showToast(`Device detection failed: ${e.message}`, 'error');
  } finally {
    state.detecting = false;
    updateControls();
  }
}

// ─── camera identification ─────────────────────────────────────────────────
const PREVIEW_POLL_MS = 100;
// A preview nobody has polled for this long belongs to a page that was reloaded or closed.
const PREVIEW_ORPHAN_S = 1.5;
const PREVIEW_MIN_MARKERS = 2; // mirrors camera_pairing.MIN_MARKERS

function identificationSides() {
  const slots = activeSlots().map((slot) => slot.id);
  return ['left', 'right'].filter((side) => ['iphone', 'oak'].every((kind) => slots.includes(`${side}_${kind}`)));
}

function canIdentifyCameras() {
  return !state.live && !state.busy && !state.detecting && !state.identifying &&
    !!state.detection && identificationSides().length > 0;
}

function emptyPreview() {
  return { active: false, cameras: [], scanning: null, uncertain: false, timer: null, seqs: {}, tileKey: '', error: '', polling: false };
}

function previewSettled() {
  return state.preview.active && state.preview.cameras.every((camera) => camera.state !== 'starting');
}

// Require live iPhone and OAK streams. A failed camera can hide an ambiguous match and
// must block assignment until restarted.


function previewCanScan() {
  const live = state.preview.cameras.filter((camera) => camera.state === 'live');
  return previewSettled() && live.some((camera) => camera.kind === 'iphone') && live.some((camera) => camera.kind === 'oak');
}

function scanInFlight() {
  return state.identifyRequest || !!state.preview.scanning || state.preview.uncertain;
}

function canScan() {
  return previewCanScan() && !scanInFlight() && !state.live && !state.busy;
}

async function openCameraIdentification() {
  if (!canIdentifyCameras()) return;
  state.pairing = { side: null, loaded: false, completed: {}, results: [], message: '', closeRequested: false, cancelling: false };
  setCameraIdentificationSide(identificationSides()[0]);
  el('camera-identify-dialog').showModal();
  await startCameraPreview();
}

// Keep disposable camera workers streaming previews across both side scans.

async function startCameraPreview() {
  clearInterval(state.preview.timer);
  state.preview = { ...emptyPreview(), active: true, uncertain: true };
  state.identifyEpoch += 1;
  state.identifyUncertain = true;
  updateControls();
  try {
    const { body } = await post('/api/cameras/preview/start');
    if (body.detection) {
      state.detection = body.detection;
      renderDeviceSelects();
    }
    if (!state.preview.active) return; // closed while starting
    applyPreviewSnapshot(body.session);
    state.preview.uncertain = false;
    state.preview.timer = setInterval(pollPreview, PREVIEW_POLL_MS);
  } catch (error) {
    clearInterval(state.preview.timer);
    state.preview = { ...emptyPreview(), error: error.message };
    if (state.pairing) state.pairing.message = `The cameras could not be started: ${error.message}. Close, fix the cause and reopen Identify cameras, or pick the devices from the dropdowns.`;
  } finally {
    state.identifyEpoch += 1;
    state.identifyUncertain = true;
    updateControls();
    await pollProcs();
  }
}

function applyPreviewSnapshot(snapshot) {
  if (!snapshot || !state.preview.active) return;
  state.preview.cameras = Array.isArray(snapshot.cameras) ? snapshot.cameras : [];
  state.preview.scanning = snapshot.scanning || null;
}

async function pollPreview() {
  if (!state.preview.active || state.preview.polling) return;
  state.preview.polling = true;
  try {
    const { body } = await api('/api/cameras/preview');
    if (!state.preview.active) return;
    if (!body.active) {
      clearInterval(state.preview.timer);
      state.preview = { ...emptyPreview(), error: 'closed by the server' };
      if (state.pairing && !state.pairing.closeRequested) state.pairing.message = 'The camera preview was closed by the server (idle timeout or shutdown). Close and reopen Identify cameras to restart the cameras.';
    } else {
      applyPreviewSnapshot(body);
      state.preview.uncertain = false;
    }
  } catch (error) {
    // A lost poll does not prove the server is idle: keep Scan locked until one succeeds.
    state.preview.uncertain = true;
  } finally {
    state.preview.polling = false;
  }
  renderCameraIdentification();
}

// Release orphaned previews on reload instead of waiting for the 60 s watchdog.

async function releaseOrphanedPreview(body) {
  if (!body.camera_preview || state.pairing || !(Number(body.camera_preview_idle_s) > PREVIEW_ORPHAN_S)) return;
  try {
    await post('/api/cameras/preview/stop');
    showToast('Released the cameras still held by the previous page load');
  } catch (error) {
    showToast(`Cameras from the previous page load are still held: ${error.message}`, 'error');
  }
}

async function stopCameraPreview() {
  clearInterval(state.preview.timer);
  const wasActive = state.preview.active;
  state.preview = emptyPreview();
  if (wasActive) await post('/api/cameras/preview/stop');
}

async function restartCameraPreview() {
  const pairing = state.pairing;
  if (!pairing || scanInFlight() || pairing.cancelling) return;
  pairing.message = 'Restarting the cameras…';
  renderCameraIdentification();
  try {
    await stopCameraPreview();
  } catch (error) {
    pairing.message = `The cameras could not be released: ${error.message}`;
    renderCameraIdentification();
    return;
  }
  await startCameraPreview();
}

function setCameraIdentificationSide(side) {
  const pairing = state.pairing;
  if (!pairing || scanInFlight() || !identificationSides().includes(side)) return;
  pairing.side = side;
  pairing.loaded = false;
  pairing.results = [];
  pairing.message = `Point the ${side.toUpperCase()} exoskeleton's iPhone and wrist OAK at this board together; when only those two tiles show "sees ${side.toUpperCase()} board", press Scan ${side}.`;
  const board = el('identify-board');
  board.onload = () => { pairing.loaded = true; renderCameraIdentification(); };
  board.onerror = () => { pairing.loaded = false; pairing.message = 'The board could not load. Close and reopen Identify cameras to retry.'; renderCameraIdentification(); };
  board.alt = `ChArUco board for identifying the ${side} camera pair`;
  board.src = `/api/cameras/board/${side}.png`;
  renderCameraIdentification();
}

function tileId(camera) {
  return `identify-camera-${camera.kind}-${String(camera.device_id || camera.name || '').replace(/[^A-Za-z0-9_-]/g, '_')}`;
}

function tileHtml(camera) {
  const id = tileId(camera);
  const name = esc(camera.name || camera.device_id || camera.kind);
  return `<li id="${id}" class="identify-camera ${esc(camera.state)}"><img id="${id}-img" alt="Live view of ${name}"><div class="identify-camera-title"><strong>${name}</strong><span id="${id}-badge" class="badge-sees badge-none"></span></div><span class="identify-camera-kind">${esc(camera.kind)}</span><span id="${id}-status" class="identify-camera-status"></span></li>`;
}

// Update existing image tiles in place to avoid blanking previews on every poll.

function renderCameraTiles(side) {
  const holder = el('identify-cameras');
  const cameras = state.preview.cameras;
  const key = cameras.map((camera) => `${camera.kind}:${camera.device_id}`).join('|');
  if (state.preview.tileKey !== key) {
    holder.innerHTML = cameras.map(tileHtml).join('');
    state.preview.tileKey = key;
    state.preview.seqs = {};
  }
  const other = side === 'left' ? 'right' : 'left';
  cameras.forEach((camera) => {
    const id = tileId(camera);
    const sees = camera.sees || {};
    const current = Number(sees[side]) || 0;
    const otherCount = Number(sees[other]) || 0;
    const badge = el(`${id}-badge`);
    let badgeClass = 'badge-none';
    if (camera.state === 'error') { badge.textContent = 'error'; badgeClass = 'badge-error'; }
    else if (camera.state === 'starting') badge.textContent = 'starting…';
    else if (current >= PREVIEW_MIN_MARKERS) { badge.textContent = `sees ${side.toUpperCase()} board`; badgeClass = 'badge-yes'; }
    else if (otherCount >= PREVIEW_MIN_MARKERS) { badge.textContent = `sees ${other.toUpperCase()} board`; badgeClass = 'badge-other'; }
    else badge.textContent = 'no board';
    badge.className = `badge-sees ${badgeClass}`;
    const frames = Number(camera.frames) || 0;
    el(`${id}-status`).textContent = camera.state === 'error' ? (camera.error || 'failed')
      : camera.state === 'starting' ? 'Starting camera…'
        : `${camera.state === 'scanning' ? 'Scanning' : 'Live'} · ${frames} frames${camera.stalled ? ` · no fresh frame for ${camera.stalled} s` : ''}`;
    el(id).className = `identify-camera ${camera.state}${current >= PREVIEW_MIN_MARKERS ? ' sees' : ''}`;
    if (camera.preview_seq && state.preview.seqs[id] !== camera.preview_seq) {
      state.preview.seqs[id] = camera.preview_seq;
      el(`${id}-img`).src = `/api/cameras/preview/${encodeURIComponent(camera.kind)}/${encodeURIComponent(camera.device_id)}.jpg?seq=${camera.preview_seq}`;
    }
  });
}

function previewStatusText(side) {
  const pairing = state.pairing;
  const preview = state.preview;
  const starting = preview.cameras.filter((camera) => camera.state === 'starting').length;
  const failed = preview.cameras.filter((camera) => camera.state === 'error').length;
  if (pairing.cancelling) return state.identifyRequest ? 'Cancelling — waiting for the scan to finish and release the cameras…' : 'Releasing the cameras…';
  if (state.identifyRequest || preview.scanning) return `Scanning ${side} cameras… keep both lenses on the board (about 10 s).`;
  if (!preview.active) return preview.error ? `Cameras are not running (${preview.error}).` : 'Starting the cameras…';
  if (preview.uncertain) return 'Waiting for the camera status…';
  if (starting) return `Starting cameras: ${preview.cameras.length - starting} of ${preview.cameras.length} live (an OAK boot can take up to 45 s)…`;
  if (failed) return `${failed} camera${failed > 1 ? 's' : ''} failed to start. A failed camera blocks assignment: fix or unplug it, then press Restart cameras.`;
  if (previewCanScan()) return `Live. Aim the ${side.toUpperCase()} iPhone and wrist OAK at the board until only those two show "sees ${side.toUpperCase()} board", then press Scan ${side}.`;
  return 'Live, but identification needs at least one iPhone and one OAK streaming.';
}

function renderCameraIdentification() {
  const pairing = state.pairing;
  if (!pairing) return;
  const side = pairing.side;
  const title = side[0].toUpperCase() + side.slice(1);
  el('identify-title').textContent = `Identify ${side} cameras`;
  el('identify-step').textContent = identificationSides().length === 2 ? `Step ${side === 'left' ? 1 : 2} of 2` : 'Camera pair';
  el('identify-instructions').textContent = `Hold the ${side} glove so its iPhone and wrist OAK both see the full board. Point all other cameras away. When both previews show green board badges, click Scan ${side} and hold still. ${state.scanEveryLaunch ? 'Repeat before each recorder launch.' : 'Rescan after moving cameras or restarting the GUI.'}`;
  el('identify-status').textContent = pairing.message;
  el('identify-lock-status').textContent = previewStatusText(side);
  renderCameraTiles(side);
  el('identify-results').innerHTML = pairing.results.map((camera) => `<li><strong>${esc(camera.name || camera.device_id || camera.kind)}</strong><span>${esc(camera.kind)} · ${Number(camera.hits) || 0} board detections / ${Number(camera.frames) || 0} frames${camera.error ? ` · ${esc(camera.error)}` : ''}</span></li>`).join('');
  el('btn-identify-scan').textContent = `${pairing.completed[side] ? 'Scan again:' : 'Scan'} ${side}`;
  el('btn-identify-scan').disabled = !pairing.loaded || !canScan() || pairing.closeRequested;
  el('btn-identify-next').hidden = side !== 'left' || !identificationSides().includes('right');
  el('btn-identify-next').disabled = scanInFlight() || pairing.cancelling || !pairing.completed.left;
  el('btn-identify-back').hidden = side !== 'right' || !identificationSides().includes('left');
  el('btn-identify-back').disabled = scanInFlight() || pairing.cancelling;
  const failed = state.preview.cameras.some((camera) => camera.state === 'error');
  el('btn-identify-restart').hidden = !failed || scanInFlight() || pairing.cancelling;
  el('btn-identify-restart').disabled = scanInFlight() || pairing.cancelling;
  el('btn-identify-close').textContent = state.identifyRequest ? 'Cancel scan' : pairing.completed[side] ? 'Done' : 'Close';
  el('btn-identify-close').disabled = pairing.cancelling;
  el('identify-board-label').textContent = `${title} camera identification board`;
}

function setCameraSelection(id, value) {
  const select = el(`slot-${id}`);
  if (!select) return;
  select.value = value;
  state.saved[id] = value;
  el(`card-${id}`).classList.toggle('assigned', !!value);
  updateDeviceStatus();
}

function applyCameraAssignments(side, assignments) {
  const pairing = state.pairing;
  const ids = [`${side}_iphone`, `${side}_oak`];
  const valid = ids.every((id) => {
    const select = el(`slot-${id}`);
    return assignments[id] && select && Array.from(select.options).some((option) => option.value === assignments[id] && !option.disabled);
  });
  if (!valid) return false;
  const other = side === 'left' ? 'right' : 'left';
  const cleared = [];
  ids.forEach((id) => {
    const otherId = id.replace(side, other);
    const select = el(`slot-${otherId}`);
    if (select && select.value === assignments[id]) {
      setCameraSelection(otherId, '');
      delete state.identified[otherId];
      cleared.push(otherId.endsWith('iphone') ? 'iPhone' : 'wrist OAK');
      delete pairing.completed[other];
    }
    setCameraSelection(id, assignments[id]);
    state.identified[id] = assignments[id];
  });
  pairing.completed[side] = true;
  const notes = [`Identified ${side}: ${assignments[ids[0]]} + ${assignments[ids[1]]}.`];
  if (cleared.length) notes.push(`Cleared ${other} ${cleared.join(' and ')} because those cameras were detected on the ${side} rig. Scan ${other} again before recording.`);
  const head = activeSlots().some((slot) => slot.id === 'head_oak') && el('slot-head_oak');
  const leftOak = el('slot-left_oak');
  const rightOak = el('slot-right_oak');
  const available = (state.detection.oak || []).filter((camera) => !camera.in_use);
  if (pairing.completed.left && pairing.completed.right && head && !head.value &&
      available.length === 3 && new Set(available.map((camera) => camera.name)).size === 3 &&
      leftOak && rightOak && leftOak.value && rightOak.value && leftOak.value !== rightOak.value) {
    const remaining = available.filter((camera) => camera.name !== leftOak.value && camera.name !== rightOak.value);
    if (remaining.length === 1) {
      setCameraSelection('head_oak', remaining[0].name);
      notes.push(`Head OAK filled with the remaining camera: ${remaining[0].name}.`);
    }
  }
  if (head && head.value === assignments[`${side}_oak`]) notes.push('The head selection uses this wrist camera too. Choose the head OAK manually after closing this dialog.');
  pairing.message = notes.join(' ');
  scheduleCompose(0);
  return true;
}

async function scanCameraIdentification() {
  const pairing = state.pairing;
  if (!pairing || !pairing.loaded || !canScan() || pairing.closeRequested) return;
  const side = pairing.side;
  state.identifyRequest = true;
  state.identifyUncertain = true;
  state.preview.uncertain = true;
  state.identifyEpoch += 1;
  pairing.results = [];
  pairing.message = `Keep both ${side} cameras facing the board until the scan finishes.`;
  updateControls();
  try {
    const { body } = await post('/api/cameras/identify', { side });
    pairing.results = body.cameras || [];
    if (body.cancelled || pairing.closeRequested) pairing.message = 'Scan cancelled. Existing selections were kept.';
    else {
      if (body.detection) {
        state.detection = body.detection;
        renderDeviceSelects();
      }
      if (body.side !== side || !applyCameraAssignments(side, body.assignments || {})) {
        pairing.message = (body.errors || []).join(' ') || 'No unambiguous iPhone + OAK pair was found. Aim both cameras at the whole board and retry.';
      }
    }
  } catch (error) {
    pairing.message = `Scan failed: ${error.message}. Waiting for camera status before retrying.`;
  } finally {
    state.identifyRequest = false;
    state.identifyEpoch += 1;
    // A lost HTTP response does not prove the server stopped using cameras.
    // Only fresh /api/cameras/preview and /api/procs responses unlock things.
    state.identifyUncertain = true;
    updateControls();
    await pollPreview();
    await pollProcs();
  }
}

function closeCameraIdentificationIfReady() {
  if (state.pairing && state.pairing.closeRequested && !state.identifying && !state.preview.active) {
    el('camera-identify-dialog').close();
    state.pairing = null;
  }
}

// Close the dialog only after the server confirms scan cancellation and camera release.


async function cancelCameraIdentification() {
  const pairing = state.pairing;
  if (!pairing || pairing.cancelling) return;
  pairing.closeRequested = true;
  pairing.cancelling = true;
  renderCameraIdentification();
  if (state.identifyRequest) {
    try {
      await post('/api/cameras/identify/cancel');
      clearInterval(state.preview.timer);
      state.preview = emptyPreview();
      await pollProcs();
    } catch (error) {
      pairing.cancelling = false;
      pairing.message = `Cancel request failed: ${error.message}. Retry Cancel scan; camera controls remain locked.`;
      renderCameraIdentification();
    }
    return;
  }
  try {
    await stopCameraPreview();
  } catch (error) {
    pairing.cancelling = false;
    pairing.closeRequested = false;
    pairing.message = `The cameras could not be released: ${error.message}. Retry Close.`;
    renderCameraIdentification();
    return;
  }
  state.identifyEpoch += 1;
  state.identifyUncertain = true;
  updateControls();
  await pollProcs();
  closeCameraIdentificationIfReady();
}

// ─── compose / plan ────────────────────────────────────────────────────────
// Only blockers are shown: warnings and info never stop a recording.
function renderFindings(findings) {
  const blockers = (findings || []).filter((f) => f.level === 'error');
  el('findings').innerHTML = blockers.map((f) => `<li class="error"><span class="finding-level">Blocker</span><span><span class="finding-code">${esc(f.code)}</span>${esc(f.message)}</span></li>`).join('');
}

function renderPreview(body) {
  state.blocked = !!body.blocked;
  renderFindings(body.findings);
  const dir = body.data_dir || '';
  el('data-dir').value = dir;
  el('summary-dir').textContent = dir ? dir.split('/').pop() : '—';
  el('summary-task').textContent = (body.config && body.config.meta && body.config.meta.task_name) || 'Not set';
  const form = collectForm();
  const sessionReady = form.task_name;
  el('step-session').classList.toggle('complete', !!sessionReady);
  updateControls();
}

function scheduleCompose(delay = 350) {
  if (state.live) return;
  clearTimeout(state.composeTimer);
  state.composeSeq += 1;
  state.plan = null;
  state.blocked = true;
  state.streamDiagnostics = [];
  renderStreamDiagnostics();
  updateControls();
  state.composeTimer = setTimeout(compose, delay);
}

async function compose() {
  if (state.live || state.busy) return;
  const seq = ++state.composeSeq;
  try {
    const { body } = await post('/api/compose', collectForm());
    if (seq !== state.composeSeq) return;
    state.plan = null;
    renderPreview(body);
  } catch (e) { showToast(e.message, 'error'); }
}

// ─── recorder ──────────────────────────────────────────────────────────────
function renderStreamDiagnostics() {
  el('camera-stream-check').hidden = !state.streamDiagnostics.length;
  el('camera-stream-results').innerHTML = state.streamDiagnostics.map((camera) =>
    `<li class="${camera.error ? 'error' : ''}"><strong>${esc(camera.name || camera.device_id || camera.kind)}</strong><span>${esc(camera.kind)} · ${Number(camera.frames) || 0} frames${camera.error ? ` · ${esc(camera.error)}` : ''}</span></li>`
  ).join('');
  el('camera-stream-recovery').textContent = 'Check the camera connections and selections. For an iPhone with no frames, restart its Record3D USB stream. For an OAK, reconnect USB and close other apps using it. Refresh devices if a camera is missing, then Retry Start to check the streams and launch.';
}

function renderProcCard() {
  el('proc-cards').className = 'proc-grid single';
  el('proc-cards').innerHTML = `
    <div class="proc" id="proc-${ROLE}">
      <header><div class="proc-title"><span class="proc-symbol"><svg class="icon"><use href="#icon-monitor"/></svg></span><div><strong>Recorder</strong><div class="proc-role">record_data.py</div></div></div><span class="pill" id="pill-${ROLE}">stopped</span></header>
      <div class="proc-meta" id="meta-${ROLE}">Waiting to start</div>
      <details class="proc-logs" open><summary>Recorder log</summary><div class="tail mono" id="tail-${ROLE}" tabindex="0" aria-label="Recorder logs"></div><div class="proc-log-path" id="log-${ROLE}"></div></details>
    </div>`;
}

function updateControls() {
  state.identifying = state.identifyRequest || state.identifyRemote || state.identifyUncertain;
  el('btn-start').disabled = state.live || state.converting || state.blocked || state.busy || state.identifying || state.detecting || calLive();
  el('btn-stop').disabled = !state.live || state.busy;
  el('btn-devices').disabled = state.live || state.detecting || state.busy || state.identifying;
  el('btn-identify').disabled = !canIdentifyCameras();
  const startingCameras = state.startRequest && activeSlots().some((slot) => slot.kind !== 'exo');
  el('btn-cancel-check').hidden = !startingCameras && state.cameraOperation !== 'stream check';
  el('btn-cancel-check').disabled = state.cameraOperation !== 'stream check' || state.cancellingStream;
  el('btn-cancel-check').textContent = state.cancellingStream ? 'Cancelling camera check…' : 'Cancel camera check';
  el('btn-retry-start').disabled = state.live || state.busy || state.identifying || state.detecting;
  const frozen = state.live || state.busy || state.identifying || state.detecting;
  ['task-name', 'operator-name', 'task-description'].forEach((id) => {
    const field = el(id);
    if (field) field.disabled = frozen;
  });
  SLOTS.forEach((slot) => {
    const field = el(`slot-${slot.id}`);
    if (field) field.disabled = frozen;
  });
  const recorderState = state.procs[ROLE]?.state;
  el('gate').textContent = state.live && recorderState === 'failed' ? 'Recorder failed but its process is still running. Stop it to release the devices, then check the log.'
    : state.live && recorderState === 'stopping' ? 'Recorder stopping: discarding any unsaved take and releasing the devices.'
    : state.live && recorderState === 'starting' ? 'Recorder starting: wait for the instructions banner before recording.'
    : state.live ? 'Recorder running: use its window (s / w / a / t / q). Stop here to end the session.'
    : state.cameraOperation === 'stream check' ? state.cancellingStream
      ? 'Cancelling the camera check: waiting for camera workers to release the devices.'
      : 'Checking camera streams before launch. Each selected camera must produce fresh frames.'
    : state.startRequest ? 'Preparing the recorder launch. Camera checks run before recording starts.'
    : state.converting ? 'A LeRobot conversion is running (step 05): wait for it to finish or stop it before recording.'
    : calLive() ? 'A glove calibration is running: finish or cancel it before recording.'
    : state.identifying ? 'Camera identification is using the cameras. Wait for it to finish before recording or refreshing devices.'
    : state.blocked ? 'Resolve the blockers below before starting.' : 'Ready: Start regenerates the yaml with the current folder index and launches the recorder.';
  el('gate').classList.toggle('open', !state.live && !state.blocked && !state.identifying);
  renderCameraIdentification();
  closeCameraIdentificationIfReady();
}

async function startRecorder() {
  if (state.busy || state.identifying || state.live || state.detecting) return;
  clearTimeout(state.composeTimer);
  state.composeSeq += 1;
  state.procsSeq += 1;
  state.busy = true;
  state.startRequest = true;
  state.streamDiagnostics = [];
  state.identifyEpoch += 1;
  state.identifyUncertain = activeSlots().some((slot) => slot.kind !== 'exo');
  renderStreamDiagnostics();
  el('process-operation').textContent = state.identifyUncertain
    ? 'Checking selected camera streams before launching the recorder…'
    : 'Validating the configuration and launching the recorder…';
  updateControls();
  try {
    // Launch what the page shows (the same form /api/compose validated), never a stale plan.
    const { status, body } = await post(`/api/procs/${ROLE}/start`, collectForm());
    if (status === 422) {
      state.plan = null;
      state.streamDiagnostics = body.camera_diagnostics || [];
      renderPreview(body);
      renderStreamDiagnostics();
      el('process-operation').textContent = body.cancelled
        ? 'Camera check cancelled. The recorder was not started.'
        : state.streamDiagnostics.length ? 'Camera stream check failed. Review the results below before retrying.'
          : 'Start refused. Fix the blockers listed below.';
      showToast(body.cancelled ? 'Camera check cancelled' : 'Start refused: the plan has blockers', body.cancelled ? undefined : 'error');
      return;
    }
    state.live = true;
    if (body.plan) { state.plan = body.plan; renderPreview(body.plan); }
    state.logSeq = 0;
    el(`tail-${ROLE}`).textContent = '';
    showToast('Recorder starting — its window opens on the workstation desktop');
    el('process-operation').textContent = 'Recorder starting — wait for the instructions banner in its window.';
    await restoreSession(false);
  } catch (e) {
    el('process-operation').textContent = `Start did not complete: ${e.message}`;
    showToast(e.message, 'error');
  } finally {
    state.busy = false;
    state.startRequest = false;
    state.cancellingStream = false;
    state.identifyEpoch += 1;
    await pollProcs();
  }
}

async function cancelStreamCheck() {
  if (state.cameraOperation !== 'stream check' || state.cancellingStream) return;
  state.cancellingStream = true;
  updateControls();
  try {
    await post('/api/cameras/identify/cancel');
    await pollProcs();
  } catch (error) {
    state.cancellingStream = false;
    showToast(`Camera-check cancellation failed: ${error.message}. Retry Cancel camera check.`, 'error');
    updateControls();
  }
}

const STOP_WARNING = 'Stop sends Ctrl-C to the recorder.\n\n' +
  'An episode that is still being recorded is DISCARDED, not saved. ' +
  'Alignments still running are interrupted; they resume the next time this folder is started (q in the recorder window waits for them instead).\n\n' +
  'To keep the last episode: press w in the recorder window to save it, then press q there.\n\nStop anyway?';

async function stopRecorder() {
  if (state.busy) return;
  if (!window.confirm(STOP_WARNING)) return;
  state.busy = true;
  updateControls();
  el('process-operation').textContent = 'Stopping: Ctrl-C sent; waiting for the recorder to exit…';
  try {
    await post(`/api/procs/${ROLE}/stop`);
    el('process-operation').textContent = 'Recorder stopped.';
  } catch (e) { showToast(e.message, 'error'); } finally { state.busy = false; await pollProcs(); }
}

async function pollProcs() {
  const seq = ++state.procsSeq;
  const identifyEpoch = state.identifyEpoch;
  try {
    const { body } = await api('/api/procs');
    if (seq !== state.procsSeq) return;
    const s = body.procs[ROLE] || { state: 'stopped' };
    const wasLive = state.live;
    state.live = !!body.live;
    state.procs = body.procs;
    updateConversion(body);
    if (wasLive && !state.live) {
      state.identificationResync = true;
      loadConvertSessions(); // the session just recorded has new episodes
    }
    if (identifyEpoch === state.identifyEpoch) {
      state.identifyRemote = !!body.identifying;
      state.identifyUncertain = false;
      state.cameraOperation = body.camera_operation || null;
      if (!state.cameraOperation && !state.startRequest) state.cancellingStream = false;
      // Retain launched camera selections while recording. On return to idle, refresh
      // identification and preflight.


      if (!state.live && !body.identifying && !state.pairing && !state.startRequest && !state.busy) {
        const force = !!state.identificationResync;
        state.identificationResync = false;
        syncIdentification(body, force);
      }
    }
    el('connection-dot').classList.add('online');
    el('connection-label').textContent = 'Connected to workstation';
    const pill = el(`pill-${ROLE}`);
    pill.textContent = s.restart_required ? 'restart required' : s.state;
    pill.className = 'pill ' + (s.restart_required ? 'starting' : s.state);
    const uptime = s.uptime_s !== null && s.uptime_s !== undefined ? `${Math.round(s.uptime_s)} s` : '';
    const runtime = uptime || (state.live && s.state === 'failed' ? 'Process cleanup pending'
      : ['stopped', 'done', 'exited', 'failed'].includes(s.state) ? 'Not running' : 'Waiting for the instructions banner');
    const lastError = s.last_error ? ` | <span class="badge bad" title="${esc(s.last_error)}">error</span> ${esc(s.last_error)}` : '';
    el(`meta-${ROLE}`).innerHTML = `<span class="dot ${s.state === 'ready' || s.state === 'adopted' ? 'ok' : ''}"></span>${runtime}${s.pid ? ' | pid ' + s.pid : ''}${s.exit_code !== null && s.exit_code !== undefined ? ' | exit ' + s.exit_code : ''}${s.error ? ' | ' + esc(s.error) : ''}${lastError}`;
    el(`log-${ROLE}`).innerHTML = s.log_path ? `log: <a href="/api/procs/${ROLE}/logfile" target="_blank">${esc(s.log_path)}</a>` : '';
    el('process-count').textContent = s.state;
    el('step-record').classList.toggle('complete', state.live && ['ready', 'adopted'].includes(s.state));
    el('step-record').classList.toggle('blocked', s.state === 'failed');
    if (wasLive && !state.live) el('process-operation').textContent = s.state === 'failed' ? `Recorder exited with an error (exit ${s.exit_code}). See the log.` : 'Recorder finished.';
    updateControls();
    await pollLogs();
  } catch (e) { el('connection-dot').classList.remove('online'); el('connection-label').textContent = 'Reconnecting to workstation'; }
}

async function pollLogs() {
  const { body } = await api(`/api/procs/${ROLE}/logs?since=${state.logSeq}&limit=500`);
  if (body.next_seq < state.logSeq) {
    // Reset log position when a new or adopted recorder restarts its sequence counter.

    state.logSeq = 0;
    el(`tail-${ROLE}`).textContent = '';
    return pollLogs();
  }
  if (body.lines.length) {
    const tail = el(`tail-${ROLE}`);
    tail.textContent = (tail.textContent + '\n' + body.lines.join('\n')).split('\n').slice(-300).join('\n');
    tail.scrollTop = tail.scrollHeight;
  }
  state.logSeq = body.next_seq;
}

// ─── convert for training (step 05) ────────────────────────────────────────
// The conversion runs as the supervisor's `lerobot` role: same status, log tail
// and Stop as the recorder, one session at a time, never while it records.
const CONVERT_ROLE = 'lerobot';
const convert = { sessions: [], logSeq: 0, busy: false, wasConverting: false, loaded: false };

function renderConvertCard() {
  el('convert-proc').innerHTML = `
    <div class="proc" id="proc-${CONVERT_ROLE}">
      <header><div class="proc-title"><span class="proc-symbol"><svg class="icon"><use href="#icon-cube"/></svg></span><div><strong>LeRobot conversion</strong><div class="proc-role">dexumi_lerobot_conversion.py</div></div></div><div class="device-actions"><span class="pill" id="pill-${CONVERT_ROLE}">stopped</span><button id="btn-convert-stop" class="danger-secondary compact" disabled>Stop conversion</button></div></header>
      <div class="proc-meta" id="meta-${CONVERT_ROLE}">No conversion running</div>
      <details class="proc-logs"><summary>Conversion log</summary><div class="tail mono" id="tail-${CONVERT_ROLE}" tabindex="0" aria-label="Conversion logs"></div><div class="proc-log-path" id="log-${CONVERT_ROLE}"></div></details>
    </div>`;
  el('btn-convert-stop').onclick = stopConversion;
}

function datasetText(s) {
  if (s.converted === null) return 'not converted';
  if (s.up_to_date) return `LeRobot dataset up to date (${s.converted} episodes)`;
  return `LeRobot dataset out of date (${s.converted} of ${s.aligned} episodes)`;
}

function renderConvertSessions() {
  const box = el('convert-sessions');
  if (!convert.loaded) { box.innerHTML = '<p class="muted empty">Loading sessions…</p>'; return; }
  if (!convert.sessions.length) {
    box.innerHTML = '<p class="muted empty">No recorded sessions in the data folder yet.</p>';
    return;
  }
  const locked = state.live || state.converting || convert.busy;
  box.innerHTML = convert.sessions.map((s, i) => {
    const kind = s.blocker ? 'missing' : s.up_to_date ? 'calibrated' : '';
    const label = s.converted === null ? 'Convert to LeRobot' : 'Convert again';
    const reason = s.blocker ? `${s.blocker}: start the recorder on this session again to align it, or run ./launch.sh align`
      : state.live ? 'Quit the recorder first' : '';
    return `<div class="glove-card ${kind}"><h3><span>${esc(s.name)}</span></h3>
      <p>${s.recorded} recorded · ${s.aligned} aligned · ${esc(datasetText(s))}${reason ? ` · ${esc(reason)}` : ''}</p>
      <button class="${s.up_to_date ? 'secondary' : 'primary'} compact" data-convert="${i}"${s.blocker || locked ? ' disabled' : ''}>${label}</button></div>`;
  }).join('');
  box.querySelectorAll('button[data-convert]').forEach((button) => {
    button.onclick = () => startConversion(convert.sessions[Number(button.dataset.convert)]);
  });
}

async function loadConvertSessions() {
  try {
    const { body } = await api('/api/lerobot');
    convert.sessions = body.sessions || [];
    convert.loaded = true;
  } catch (e) { showToast(`Cannot list sessions: ${e.message}`, 'error'); }
  renderConvertSessions();
}

async function startConversion(session) {
  if (!session || convert.busy) return;
  if (session.converted !== null && !window.confirm(
    `Replace ${session.name}_lerobot?\n\nThe dataset is rebuilt from the ${session.aligned} aligned episodes; the current one is kept until the new one is complete.`)) return;
  convert.busy = true;
  renderConvertSessions();
  try {
    await post('/api/lerobot/start', { session: session.name });
    convert.logSeq = 0;
    el(`tail-${CONVERT_ROLE}`).textContent = '';
    el(`proc-${CONVERT_ROLE}`).querySelector('details').open = true;
    showToast(`Converting ${session.name}…`);
  } catch (e) { showToast(e.detail || e.message, 'error'); } finally { convert.busy = false; await pollProcs(); }
}

async function stopConversion() {
  if (!window.confirm('Stop the conversion? The existing LeRobot dataset is kept; convert again later.')) return;
  try { await post(`/api/procs/${CONVERT_ROLE}/stop`); } catch (e) { showToast(e.message, 'error'); }
  await pollProcs();
}

function updateConversion(body) {
  const s = (body.procs || {})[CONVERT_ROLE] || { state: 'stopped' };
  state.converting = !!body.converting;
  const pill = el(`pill-${CONVERT_ROLE}`);
  if (!pill) return;
  const succeeded = s.state === 'exited' && s.exit_code === 0;
  pill.textContent = state.converting ? 'converting' : succeeded ? 'done' : s.state;
  pill.className = 'pill ' + (state.converting ? 'ready' : s.state);
  const uptime = s.uptime_s !== null && s.uptime_s !== undefined ? `running ${Math.round(s.uptime_s)} s` : '';
  el(`meta-${CONVERT_ROLE}`).textContent = state.converting ? uptime
    : succeeded ? 'Last conversion finished.'
    : s.state === 'failed' ? `Last conversion failed${s.exit_code !== null && s.exit_code !== undefined ? ` (exit ${s.exit_code})` : ''}: see the log.`
    : s.state === 'stopped' && s.pid ? 'Last conversion was stopped.' : 'No conversion running';
  el(`log-${CONVERT_ROLE}`).innerHTML = s.log_path ? `log: <a href="/api/procs/${CONVERT_ROLE}/logfile" target="_blank">${esc(s.log_path)}</a>` : '';
  el('btn-convert-stop').disabled = !state.converting;
  if (convert.wasConverting && !state.converting) {
    showToast(succeeded ? 'LeRobot conversion finished.' : 'LeRobot conversion ended without a new dataset: see the conversion log.', succeeded ? undefined : 'error');
    loadConvertSessions();
  }
  if (state.converting || convert.wasConverting) pollConversionLogs();
  convert.wasConverting = state.converting;
  renderConvertSessions();
}

async function pollConversionLogs() {
  try {
    const { body } = await api(`/api/procs/${CONVERT_ROLE}/logs?since=${convert.logSeq}&limit=500`);
    if (body.next_seq < convert.logSeq) { convert.logSeq = 0; el(`tail-${CONVERT_ROLE}`).textContent = ''; return; }
    if (body.lines.length) {
      const tail = el(`tail-${CONVERT_ROLE}`);
      tail.textContent = (tail.textContent + '\n' + body.lines.join('\n')).split('\n').slice(-300).join('\n');
      tail.scrollTop = tail.scrollHeight;
    }
    convert.logSeq = body.next_seq;
  } catch (e) { /* the next poll retries */ }
}

// ─── restore ───────────────────────────────────────────────────────────────
async function restoreSession(full) {
  const { body } = await api('/api/state');
  state.defaults = body.defaults || {};
  el('summary-date').textContent = state.defaults.today || '—';
  const s = body.session || {};
  state.scanEveryLaunch = !!body.scan_every_launch;
  if (full) {
    fillForm(s.form);
    state.identified = { ...(body.identified || {}) };
    // A running recorder displays its launch selections even after consuming the scan.

    const launched = body.live ? (s.form || {}) : {};
    IDENTIFIED_SLOTS.forEach((id) => { state.saved[id] = state.identified[id] || launched[id] || ''; });
    if (body.detection) state.detection = body.detection;
    state.saved.head_oak = body.live ? (launched.head_oak || '') : inferHeadOak();
    if (body.detection) renderDeviceSelects();
  }
  if (s.data_dir) { el('summary-dir').textContent = s.data_dir.split('/').pop(); }
  if (body.plan) { state.plan = body.plan; renderPreview(body.plan); }
  if (full) await releaseOrphanedPreview(body);
  return body;
}

function wire() {
  el('btn-devices').onclick = () => loadDevices();
  el('btn-identify').onclick = openCameraIdentification;
  el('btn-identify-scan').onclick = scanCameraIdentification;
  el('btn-identify-next').onclick = () => setCameraIdentificationSide('right');
  el('btn-identify-back').onclick = () => setCameraIdentificationSide('left');
  el('btn-identify-close').onclick = cancelCameraIdentification;
  el('btn-identify-restart').onclick = restartCameraPreview;
  el('camera-identify-dialog').oncancel = (event) => { event.preventDefault(); cancelCameraIdentification(); };
  // Native dialog cancel events can be non-cancelable (e.g. repeated Escape).
  // Intercept the key before the browser closes a dialog whose scan is active.
  el('camera-identify-dialog').onkeydown = (event) => {
    if (event.key === 'Escape') { event.preventDefault(); cancelCameraIdentification(); }
  };
  el('btn-start').onclick = startRecorder;
  el('btn-stop').onclick = stopRecorder;
  el('btn-convert-refresh').onclick = loadConvertSessions;
  el('btn-cancel-check').onclick = cancelStreamCheck;
  el('btn-retry-start').onclick = startRecorder;
  el('btn-cal-left').onclick = () => startCalibration('Left');
  el('btn-cal-right').onclick = () => startCalibration('Right');
  el('btn-cal-recal').onclick = () => cal.target && startCalibration(cal.target.side, true);
  el('btn-cal-back').onclick = () => { cal.target = null; renderCalibration(); };
  el('btn-cal-record').onclick = () => calibrationAction('/api/calibration/confirm');
  el('btn-cal-save').onclick = () => calibrationAction('/api/calibration/save', { save: true });
  el('btn-cal-discard').onclick = () => calibrationAction('/api/calibration/save', { save: false });
  el('btn-cal-cancel').onclick = () => calibrationAction('/api/calibration/cancel');
  ['task-name', 'operator-name', 'task-description'].forEach((id) => {
    el(id).addEventListener('input', () => scheduleCompose());
    el(id).addEventListener('change', () => scheduleCompose(0));
  });
}

function observeWorkflow() {
  if (typeof IntersectionObserver === 'undefined') return;
  const links = [...document.querySelectorAll('.step-link')];
  const activate = (id) => links.forEach((link) => {
    const active = link.getAttribute('href') === `#${id}`;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'step');
    else link.removeAttribute('aria-current');
  });
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting);
    if (visible.length) activate(visible[0].target.id);
  }, { rootMargin: '-10% 0px -65% 0px' });
  document.querySelectorAll('.panel').forEach((panel) => observer.observe(panel));
  links.forEach((link) => link.addEventListener('click', () => activate(link.hash.slice(1))));
}

// ─── glove calibration (step 01) ───────────────────────────────────────────
// Run glove calibration as a supervised child; poll its job and answer its two prompts
// through the UI.


const CAL_LIVE = ['starting', 'waiting', 'recording', 'review', 'saving'];
const CAL_TERMINAL = ['done', 'failed', 'cancelled'];
const CAL_POLL_MS = { live: 700, idle: 2500 };
const CAL_STATUS_TEXT = {
  starting: 'Starting… reading the glove board.',
  waiting: 'Ready. Keep the glove seated in the calibration tool and still, then press Record pose.',
  recording: 'Recording the pose… keep the glove still.',
  review: 'Pose recorded. Press Save calibration to write it, or Discard to try again.',
  saving: 'Saving…',
};
const cal = { data: null, seq: 0, log: [], timer: null, prevState: 'idle', busy: false, keepOpen: false, target: null };
const calLive = () => !!(cal.data && CAL_LIVE.includes(cal.data.job.state));
const shortSerial = (v) => (v && v.length > 12 ? `${v.slice(0, 4)}…${v.slice(-6)}` : v || '');

async function loadCalibration() {
  clearTimeout(cal.timer);
  try {
    const { body } = await api(`/api/calibration?since=${cal.seq}`);
    applyCalibration(body);
  } catch (e) { /* Retry on the next poll if the server is unreachable. */ }
  scheduleCalibrationPoll();
}

function scheduleCalibrationPoll() {
  clearTimeout(cal.timer);
  cal.timer = setTimeout(loadCalibration, calLive() ? CAL_POLL_MS.live : CAL_POLL_MS.idle);
}

function applyCalibration(body) {
  const job = body.job;
  // A new run (or a restarted server) numbers its output from 1 again: start the log over.
  if (job.seq < cal.seq) { cal.log = []; cal.seq = 0; }
  job.lines.forEach((line) => cal.log.push(line.text));
  cal.log = cal.log.slice(-400);
  cal.seq = job.seq;
  if (CAL_LIVE.includes(job.state)) cal.keepOpen = true;
  const wasLive = CAL_LIVE.includes(cal.prevState);
  cal.data = body;
  // A glove the library knows had its saved calibration loaded: list the boards again.
  if ((body.loaded || []).length) {
    showToast(`Loaded the saved calibration of ${body.loaded.map((g) => `${g.side} glove ${shortSerial(g.serial)}`).join(', ')}`);
    loadDevices();
  }
  renderCalibration();
  if (wasLive && !CAL_LIVE.includes(job.state)) onCalibrationFinished(job);
  cal.prevState = job.state;
  updateControls();
}

function onCalibrationFinished(job) {
  cal.keepOpen = true;
  if (job.state === 'done') {
    showToast(job.message);
    // Refresh device assignments after calibration gives the board a name.
    loadDevices();
  } else if (job.state === 'failed') showToast(`Calibration failed: ${job.message}`, 'error');
}

function renderCalibration() {
  const d = cal.data;
  if (!d) return;
  const job = d.job;
  const live = CAL_LIVE.includes(job.state);
  // A connected glove is calibrated only if a saved calibration matches its board ID.

  const fresh = d.boards.filter((b) => !b.calibrated);
  const cards = d.boards.map((b) => {
    const title = b.side ? `${esc(b.side)} glove` : 'New glove';
    const detail = b.calibrated ? `Board ${esc(shortSerial(b.serial))} · calibration ${esc(b.recorded_by)}`
      : b.saved_sides.length ? `Board ${esc(shortSerial(b.serial))} · a saved calibration exists (${esc(b.saved_sides.join(', '))}) but could not be loaded: plug in only this glove and refresh`
        : `Board ${esc(shortSerial(b.serial))} · no calibration for this glove yet`;
    const redo = b.calibrated ? `<button class="secondary compact" data-recal="${esc(b.side)}"${live || d.recorder_live || cal.busy ? ' disabled' : ''}>Recalibrate</button>` : '';
    return `<div class="glove-card ${b.calibrated ? 'calibrated' : 'missing'}"><h3>${title}<span class="glove-state">${b.calibrated ? '✓ Calibrated' : 'Not calibrated'}</span></h3><p>${detail}</p>${redo}</div>`;
  });
  el('glove-cards').innerHTML = cards.length ? cards.join('')
    : '<p class="field-hint">No glove is connected. Plug in your gloves to check their calibration.</p>';
  el('glove-cards').querySelectorAll('[data-recal]').forEach((button) => {
    button.onclick = () => { cal.target = { side: button.dataset.recal }; cal.keepOpen = true; renderCalibration(); el('calibrate-guide').scrollIntoView({ block: 'nearest' }); };
  });
  // The guide serves one glove: the one being recalibrated, or a new one.
  const redoing = live ? !!job.recalibrate : !!cal.target;
  const redoSide = live ? job.side : cal.target && cal.target.side;
  // With every plugged-in glove calibrated and nothing being redone, the guide only shows a finished run's result.
  const resultOnly = !live && !cal.target && fresh.length === 0;
  el('calibrate-guide-title').textContent = redoing ? `Recalibrate the ${redoSide} glove` : resultOnly ? 'Calibration result' : 'Calibrate a new glove';
  document.querySelector('.calibrate-steps').hidden = resultOnly;
  el('cal-step-plug').hidden = redoing;
  el('cal-step-hand').hidden = redoing;
  el('glove-saved').textContent = d.saved.length
    ? `Saved calibrations of gloves that are not plugged in (they load when the glove is): ${d.saved.map((g) => `${g.side} ${shortSerial(g.serial)}`).join(', ')}.`
    : '';
  // Done only when a glove is plugged in and every plugged-in glove is calibrated.
  const settled = d.boards.length > 0 && fresh.length === 0;
  el('calibrate-tag').textContent = d.boards.length
    ? `${d.boards.length - fresh.length} of ${d.boards.length} connected glove${d.boards.length > 1 ? 's' : ''} calibrated`
    : 'No gloves connected';
  el('step-calibrate').classList.toggle('complete', settled);
  el('step-calibrate').classList.toggle('blocked', d.boards.length > 0 && !settled);

  const showGuide = fresh.length > 0 || live || cal.keepOpen || !!cal.target;
  el('calibrate-guide').hidden = !showGuide;
  if (!showGuide) return;

  const single = d.boards.length === 1 ? d.boards[0] : null;
  let boards = '';
  if (!live) {
    if (cal.target) {
      const g = d.boards.find((b) => b.side === cal.target.side);
      boards = g ? `Recalibrating the ${cal.target.side} glove (board ${shortSerial(g.serial)}). Seat it in the ${d.tool_version} tool with every finger in its slot, then press Start recalibration. Other gloves can stay plugged in. The current calibration is kept as a backup and put back if you cancel.` : `The ${cal.target.side} glove is not plugged in. Plug it in, or go back.`;
    } else if (!d.boards.length) boards = 'No glove board is connected. Plug in the glove you are calibrating.';
    else if (d.boards.length > 1) boards = `${d.boards.length} glove boards are connected. Unplug every glove except the one you are calibrating.`;
    else if (single.recorded_by) boards = `The connected glove is already calibrated (${single.recorded_by}). Plug in the glove that has never been calibrated.`;
    else boards = `New glove detected (board ${shortSerial(single.serial)}). Choose its hand to start.${d.gloves.some((g) => g.calibrated) ? ' The calibration now on that hand stays saved and loads again when its glove is plugged in.' : ''}`;
  }
  el('calibrate-boards').textContent = resultOnly ? '' : boards;

  const canStart = !live && !d.recorder_live && !cal.busy && !!single && !single.recorded_by;
  [['Left', 'btn-cal-left'], ['Right', 'btn-cal-right']].forEach(([side, id]) => {
    const g = d.gloves.find((x) => x.side === side);
    const button = el(id);
    button.hidden = live || !!cal.target || resultOnly;
    button.disabled = !canStart;
    button.textContent = `Calibrate as ${side} glove`;
  });
  const redoReady = !live && !!cal.target && d.boards.some((b) => b.side === cal.target.side);
  el('btn-cal-recal').hidden = live || !cal.target;
  el('btn-cal-recal').disabled = !redoReady || d.recorder_live || cal.busy;
  el('btn-cal-back').hidden = live || !cal.target;
  el('btn-cal-record').hidden = job.state !== 'waiting';
  el('btn-cal-save').hidden = job.state !== 'review';
  el('btn-cal-discard').hidden = job.state !== 'review';
  el('btn-cal-cancel').hidden = !live;
  el('btn-cal-cancel').disabled = job.state === 'saving';
  if (d.recorder_live && !live) el('calibrate-boards').textContent = 'The recorder is running. Stop it before calibrating a glove.';

  let status = '';
  let kind = '';
  if (live) status = CAL_STATUS_TEXT[job.state] || '';
  else if (job.state === 'failed') { status = `Calibration failed: ${job.message}`; kind = 'error'; }
  else if (job.state === 'cancelled') status = job.message;
  const box = el('calibrate-status');
  box.hidden = !status;
  box.textContent = status;
  box.className = 'gate' + (kind ? ` ${kind}` : '');

  const log = el('calibrate-log');
  const text = cal.log.join('\n') + (job.pending ? `\n${job.pending}` : '');
  if (log.textContent !== text) {
    const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
    log.textContent = text;
    if (atEnd) log.scrollTop = log.scrollHeight;
  }
}

async function startCalibration(side, recalibrate = false) {
  if (cal.busy) return;
  cal.busy = true;
  renderCalibration();
  try {
    await post('/api/calibration/start', { side, recalibrate });
    cal.target = null;
    cal.log = [];
    cal.seq = 0;
    cal.keepOpen = true;
  } catch (e) {
    showToast(e.detail || e.message, 'error');
  } finally {
    cal.busy = false;
  }
  await loadCalibration();
}

async function calibrationAction(path, payload) {
  try { await post(path, payload); } catch (e) { showToast(e.detail || e.message, 'error'); }
  await loadCalibration();
}

async function init() {
  observeWorkflow();
  renderProcCard();
  renderConvertCard();
  renderConvertSessions();
  wire();
  try { await restoreSession(true); } catch (e) { showToast(`Cannot reach the GUI server: ${e.message}`, 'error'); }
  await loadCalibration();
  await pollProcs();
  await loadConvertSessions();
  if (!activeSlots().length) renderDeviceSelects();
  else if (!state.live && !state.identifying) await loadDevices();
  else { el('device-status').textContent = 'Cameras in use: device detection paused until they are released.'; renderDeviceSelects(); }
  if (!state.live && (state.detection || el('task-name').value)) scheduleCompose(0);
  updateControls();
  setInterval(pollProcs, 2000);
}

if (typeof document !== 'undefined') init();
