/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
(function () {
  'use strict';
  const dialog = document.getElementById('camera-identify-dialog');
  if (!dialog) return;
  const el = id => document.getElementById(id);
  let session = null, sides = [], index = 0, busy = false, closing = false, timer = null;
  let completed = [], latest = null, fields = {}, opening = false, closeRequested = false;
  let refreshing = false, generation = 0;
  async function request(path, data) {
    const response = await fetch(path, data === undefined ? {cache: 'no-store'} : {method: 'POST', body: data, credentials: 'same-origin'});
    const body = await response.json();
    if (!response.ok) {
      if (body.id) session = body.id;
      throw new Error(body.detail || body.error || (body.errors || []).join(' ') || 'The camera operation failed.');
    }
    return body;
  }
  function message(text) {
    el('identify-status').textContent = text;
    el('identify-status').hidden = !text;
  }
  function complete() { return sides.length > 0 && sides.every(side => completed.includes(side)); }
  function sideUnlocked(position) { return sides.slice(0, position).every(side => completed.includes(side)); }
  function progress(working) {
    const list = el('identify-progress');
    if (list.children.length !== sides.length || sides.some((side, i) => list.children[i].firstChild.dataset.identifySide !== side)) {
      list.replaceChildren();
      sides.forEach(side => {
        const item = document.createElement('li');
        const button = document.createElement('button'); button.type = 'button';
        button.className = 'identify-side'; button.dataset.identifySide = side;
        const title = document.createElement('strong'); title.textContent = side[0].toUpperCase() + side.slice(1) + ' cameras';
        button.append(title, document.createElement('span')); item.append(button); list.append(item);
      });
    }
    sides.forEach((side, i) => {
      const button = list.children[i].firstChild;
      const identified = completed.includes(side), unlocked = sideUnlocked(i);
      const status = busy && i === index ? (identified ? '✓ Identified · scanning again…' : '◐ Scanning…') : identified ? '✓ Identified' : unlocked ? '○ Not identified' : '○ Identify ' + sides[i - 1] + ' first';
      if (button.lastChild.textContent !== status) button.lastChild.textContent = status;
      button.classList.toggle('identified', identified);
      if (i === index) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
      button.disabled = working || closing || !unlocked;
    });
  }
  function controls() {
    const side = sides[index];
    const working = busy || opening || refreshing;
    el('identify-step').textContent = 'STEP ' + (index + 1) + ' OF ' + sides.length;
    el('identify-title').textContent = side ? 'Identify ' + side + ' cameras' : 'Identify Cameras';
    el('identify-marker-label').textContent = side ? 'Aim the ' + side + ' cameras at this marker' : '';
    el('identify-instructions').textContent = 'Wait for the cameras to open, then aim this gripper’s cameras at the marker. Point the other gripper’s cameras away. Once their previews show the marker, click Scan ' + (side || 'left') + '.';
    el('identify-scan').textContent = busy ? 'Scanning…' : (completed.includes(side) ? 'Scan again: ' : 'Scan ') + (side || 'cameras');
    el('identify-scan').disabled = working || closing || !side || !sideUnlocked(index) || !latest || !latest.cameras.length || latest.cameras.some(c => !c.ready || c.error);
    el('identify-back').disabled = working || closing || index === 0;
    el('identify-next').hidden = index + 1 >= sides.length;
    el('identify-next').textContent = 'Next: ' + (sides[index + 1] || 'right') + ' →';
    el('identify-next').disabled = working || closing || !completed.includes(side) || !sideUnlocked(index + 1);
    el('identify-refresh').disabled = working || closing;
    el('identify-refresh').textContent = refreshing ? 'Refreshing devices…' : 'Refresh Devices';
    dialog.querySelectorAll('[data-identify-close]').forEach(button => {
      button.disabled = closing;
      button.textContent = closing ? 'Releasing cameras…' : busy ? 'Cancel scan' : complete() ? 'Done' : 'Close';
      button.classList.toggle('btn--primary', complete() && !busy && !closing);
    });
    progress(working);
  }
  function showSide() {
    el('identify-marker').src = '/config/identify/marker/' + sides[index] + '.svg';
    controls();
  }
  function selectSide(position) {
    if (busy || opening || refreshing || closing || position < 0 || position >= sides.length || !sideUnlocked(position)) return;
    index = position; showSide();
    message('');
  }
  function renderCameras(body) {
    latest = body;
    completed = body.completed || [];
    const list = el('identify-cameras');
    const present = new Set();
    body.cameras.forEach(camera => {
      const key = camera.kind + ':' + camera.device_id;
      present.add(key);
      let item = Array.from(list.children).find(row => row.dataset.camera === key);
      if (!item) {
        item = document.createElement('li'); item.dataset.camera = key;
        item.append(document.createElement('img'), document.createElement('strong'), document.createElement('span'), document.createElement('span'));
        list.append(item);
      }
      const sees = camera.sees.includes(sides[index]);
      item.className = 'identify-camera' + (sees ? ' sees' : '') + (camera.error ? ' error' : '');
      const img = item.children[0]; img.alt = 'Live view from ' + camera.name;
      if (camera.ready && !closing) img.src = camera.frame_url + '?t=' + Date.now();
      item.children[1].textContent = camera.name;
      item.children[2].textContent = camera.kind + ' · ' + camera.device_id;
      item.children[3].textContent = camera.error ? '✕ ' + camera.error : !camera.ready ? '◐ Opening camera…' : sees ? '✓ Sees ' + sides[index] + ' marker' : '○ Live — aim at the marker';
    });
    Array.from(list.children).forEach(item => { if (!present.has(item.dataset.camera)) item.remove(); });
    if (body.released) { finish(); return; }
    if (body.closing) { closing = true; message('Releasing cameras…'); }
    controls();
  }
  async function poll() {
    timer = null;
    if (!session) return;
    const current = session, version = generation;
    try {
      const body = await request('/config/identify/' + current + '/status');
      if (version !== generation || current !== session) return;
      renderCameras(body);
    } catch (error) { message(error.message); }
    if (session && version === generation) timer = setTimeout(poll, 450);
  }
  async function connect() {
    const form = document.querySelector('.config-form');
    if (!form) throw new Error('Choose a setup before identifying cameras.');
    const body = await request('/config/identify/start', new FormData(form));
    session = body.id; sides = body.sides; index = 0; closing = false;
    renderCameras(body); showSide();
    message('');
    timer = setTimeout(poll, 0);
  }
  async function open() {
    if (session || dialog.open) return;
    sides = []; index = 0; busy = false; closing = false; completed = []; latest = null; fields = {}; opening = true; closeRequested = false; generation++;
    el('identify-cameras').replaceChildren(); el('identify-marker').removeAttribute('src');
    dialog.showModal(); message(''); controls();
    try {
      await connect();
    } catch (error) { message(error.message); controls(); }
    finally { opening = false; controls(); if (closeRequested) close(); }
  }
  async function refresh() {
    if (busy || opening || closing || refreshing) return;
    refreshing = true; generation++; clearTimeout(timer); timer = null;
    latest = null; completed = [];
    controls(); message('Releasing cameras, then rescanning connected devices…');
    try {
      const form = document.querySelector('.config-form');
      if (!form) throw new Error('Choose a setup before refreshing devices.');
      const scan = await request('/config/identify/refresh', new FormData(form));
      session = null; latest = null; completed = []; sides = []; index = 0;
      el('identify-cameras').replaceChildren(); controls();
      while (!closeRequested) {
        const inventory = await request('/config/identify/inventory');
        if (inventory.scan_id !== scan.scan_id) throw new Error('The device scan changed. Refresh devices again.');
        if (!inventory.running) {
          if (window.htmx) htmx.ajax('GET', '/config/devices/status', {target: '#config-scan', swap: 'outerHTML'});
          if (inventory.error) throw new Error(inventory.error);
          if (!inventory.count) throw new Error('No devices detected. Check the connections and refresh again.');
          if (!closeRequested) await connect();
          break;
        }
        message('Scanning connected devices… Keep this window open.');
        await new Promise(resolve => setTimeout(resolve, 450));
      }
    } catch (error) { message(error.message); }
    finally {
      refreshing = false; controls();
      if (closeRequested) close();
    }
  }
  function fill(assignments) {
    const inputs = [];
    Object.entries(assignments).forEach(([name, value]) => {
      const input = document.getElementById(name);
      if (input) {
        input.value = value;
        inputs.push(input);
      }
    });
    // Set all identified fields before the gripper change posts the complete form.

    inputs.forEach(input => {
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    });
  }
  async function scan() {
    if (busy || !session) return;
    busy = true; controls(); message('Scanning ' + sides[index] + ' cameras. Hold the marker in view…');
    const data = new FormData(); data.set('side', sides[index]);
    try {
      const body = await request('/config/identify/' + session + '/scan', data);
      Object.assign(fields, body.assignments); fill(body.assignments);
      completed = body.completed; if (latest) latest.completed = completed;
      message(body.message);
    } catch (error) { message(error.message); }
    finally { busy = false; controls(); }
  }
  function finish() {
    generation++; clearTimeout(timer); timer = null; session = null; closing = false; busy = false;
    dialog.close();
    if (Object.keys(fields).length) fill(fields);
    if (window.refreshCalibrationLayout) window.refreshCalibrationLayout();
    controls();
  }
  async function close() {
    if (opening || refreshing) { closeRequested = true; message('Waiting for the camera operation to close…'); return; }
    if (!session) { finish(); return; }
    if (closing) return;
    closing = true; controls(); message('Releasing cameras…');
    try {
      const body = await request('/config/identify/' + session + '/stop', new FormData());
      renderCameras(body);
      if (session && !timer) poll();
    } catch (error) { closing = false; message(error.message); controls(); }
  }
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-identify-open]');
    if (button && !button.disabled) open();
  });
  dialog.querySelectorAll('[data-identify-close]').forEach(button => button.addEventListener('click', close));
  dialog.addEventListener('cancel', event => { event.preventDefault(); close(); });
  el('identify-scan').addEventListener('click', scan);
  el('identify-refresh').addEventListener('click', refresh);
  el('identify-back').addEventListener('click', () => selectSide(index - 1));
  el('identify-next').addEventListener('click', () => selectSide(index + 1));
  el('identify-progress').addEventListener('click', event => {
    const button = event.target.closest('[data-identify-side]');
    if (button && !button.disabled) selectSide(sides.indexOf(button.dataset.identifySide));
  });
  window.addEventListener('pagehide', () => { if (session) fetch('/config/identify/' + session + '/stop', {method: 'POST', keepalive: true}); });
})();
