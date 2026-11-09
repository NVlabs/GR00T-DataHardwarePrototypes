/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 */
(function () {
  'use strict';
  const anchors = { '/': 'setup', '/config': 'configuration', '/calibration': 'calibration', '/record': 'record', '/audit': 'audit' };
  const scrollKey = 'umi.workflow.position';
  let resume = null;
  try {
    resume = JSON.parse(sessionStorage.getItem(scrollKey));
    sessionStorage.removeItem(scrollKey);
    if (!resume || Date.now() - resume.at > 300000) resume = null;
  } catch (_) { /* Storage can be disabled in a station browser. */ }
  function position(source) {
    const sections = [...document.querySelectorAll('.workflow-section')];
    const section = source?.closest('.workflow-section') || sections.find(section => section.getBoundingClientRect().bottom > 0);
    return {x: scrollX, y: scrollY, section: section?.id, offset: section ? -section.getBoundingClientRect().top : 0, at: Date.now()};
  }
  function restorePosition(saved) {
    const section = saved.section && document.getElementById(saved.section);
    const top = section ? scrollY + section.getBoundingClientRect().top + saved.offset : saved.y;
    window.scrollTo({left: saved.x, top, behavior: 'instant'});
  }
  // Restore the current section's viewport offset after a normal form redirect.

  document.addEventListener('submit', function (event) {
    if (event.defaultPrevented || event.target.method.toLowerCase() !== 'post') return;
    const action = event.submitter?.getAttribute('formaction') || event.target.action;
    if (new URL(action, location.href).origin !== location.origin || event.target.target === '_blank') return;
    try { sessionStorage.setItem(scrollKey, JSON.stringify(position(event.submitter || event.target))); } catch (_) {}
  });
  const swapping = new WeakMap();
  document.body.addEventListener('htmx:beforeSwap', function (event) {
    const detail = event.detail;
    if (detail?.xhr && detail.shouldSwap && detail.xhr.status < 400) swapping.set(detail.xhr, position());
  });
  function restoreSwap(event) {
    const saved = event.detail?.xhr && swapping.get(event.detail.xhr);
    if (saved) restorePosition(saved);
  }
  document.body.addEventListener('htmx:afterSwap', restoreSwap);
  document.body.addEventListener('htmx:afterSettle', restoreSwap);
  document.querySelectorAll('a[href]').forEach(function (link) {
    const url = new URL(link.href, location.href);
    if (url.origin === location.origin && anchors[url.pathname] && !url.hash && !url.search) {
      link.href = '#' + anchors[url.pathname];
    }
  });
  if (!resume && !location.hash && location.pathname.startsWith('/audit/jobs/')) location.hash = 'audit';
  if (!resume && !location.hash && anchors[location.pathname] && location.pathname !== '/') {
    location.hash = anchors[location.pathname];
  }
  function links() { return document.querySelectorAll('[data-workflow-link]'); }
  function active(id) {
    links().forEach(function (link) {
      if (link.dataset.workflowLink === id) link.setAttribute('aria-current', 'step');
      else link.removeAttribute('aria-current');
    });
  }
  if (window.IntersectionObserver) {
    const observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) { if (entry.isIntersecting) active(entry.target.id); });
    }, { rootMargin: '-5% 0px -70% 0px' });
    document.querySelectorAll('.workflow-section').forEach(function (section) { observer.observe(section); });
  }
  active(resume?.section || location.hash.slice(1) || 'setup');
  function progress(state) {
    links().forEach(function (link) {
      const route = Object.keys(anchors).find(path => anchors[path] === link.dataset.workflowLink);
      let check = link.querySelector('.nav__done');
      if (state.done.includes(route) && !check) {
        check = document.createElement('span'); check.className = 'nav__done';
        check.setAttribute('aria-label', 'complete'); check.textContent = '✓'; link.append(check);
      } else if (!state.done.includes(route) && check) check.remove();
    });
    const banner = document.getElementById('workflow-recording-lock');
    if (banner) {
      banner.hidden = !state.locked;
      if (state.locked && !banner.textContent.trim()) {
        const text = document.createElement('p');
        text.append('A recording is in progress, so setup, configuration, and calibration are read-only. ');
        const link = document.createElement('a'); link.href = '#record'; link.textContent = 'Go to Record';
        text.append(link, ' to stop it.'); banner.append(text);
      }
    }
    document.querySelectorAll('#setup button, #configuration button, #configuration input[data-recording-check], #calibration button').forEach(button => {
      if ((state.locked || !state.owner) && !button.disabled) {
        button.dataset.workflowLocked = 'true'; button.disabled = true;
      } else if (!state.locked && state.owner && button.dataset.workflowLocked) {
        delete button.dataset.workflowLocked; button.disabled = false;
      }
    });
  }
  document.body.addEventListener('htmx:afterSwap', function (event) {
    const xhr = event.detail && event.detail.xhr;
    const value = xhr && xhr.getResponseHeader('X-Workflow-Progress');
    if (value) { try { progress(JSON.parse(value)); } catch (error) { console.error(error); } }
    // Refresh recording readiness when calibration or config fragments change it.

    const target = event.detail?.target || event.target;
    const updated = (target?.id && document.getElementById(target.id)) || target;
    if (updated instanceof Element && (updated.closest('#configuration') || ['cal-steps', 'check-panel'].includes(updated.id) || updated.id.startsWith('vive-panel'))) {
      scheduleWorkflowStatus();
    }
  });
  const initial = document.getElementById('workflow-progress');
  if (initial) progress(JSON.parse(initial.textContent));
  function updateRecordingChecks() {
    const inputs = [...document.querySelectorAll('[data-recording-check]')];
    const count = document.getElementById('recording-check-count');
    if (!count) return;
    const enforced = inputs.filter(input => input.checked).length;
    const relaxed = inputs.length - enforced;
    count.textContent = relaxed ? enforced + ' enforced · ' + relaxed + ' warn only' : 'all ' + inputs.length + ' enforced';
  }
  // Build Start confirmation from the current submitted check-enforcement settings.

  function syncStartConfirm() {
    const warn = [...document.querySelectorAll('[data-recording-check]')]
      .filter(input => !input.checked)
      .map(input => input.closest('label')?.querySelector('strong')?.textContent.trim())
      .filter(Boolean);
    document.querySelectorAll('[data-confirm-base]').forEach(element => {
      const base = element.dataset.confirmBase;
      element.setAttribute('hx-confirm', warn.length
        ? base + '\n\nWarn only (will not stop recording): ' + warn.join(', ') + '.'
        : base);
    });
  }
  document.addEventListener('change', function (event) {
    if (event.target.matches('[data-recording-check]')) { updateRecordingChecks(); syncStartConfirm(); }
  });
  document.body.addEventListener('htmx:afterSwap', syncStartConfirm);
  // Refresh sections after Start saves the page, but retain its recording response and
  // any refusal.


  document.body.addEventListener('workflow-refresh', function () {
    if (window.refreshCalibrationLayout) window.refreshCalibrationLayout();
  });
  syncStartConfirm();
  document.addEventListener('click', function (event) {
    const button = event.target.closest('[data-recording-checks-all]');
    if (!button || button.disabled) return;
    document.querySelectorAll('[data-recording-check]:not([disabled])').forEach(input => {
      input.checked = button.dataset.recordingChecksAll === 'true';
    });
    updateRecordingChecks();
    syncStartConfirm();
  });
  function showConfigurationError() {
    resume = null;
    const section = document.getElementById('configuration');
    if (section) {
      location.hash = 'configuration';
      section.scrollIntoView({block: 'start'});
      const invalid = section.querySelector('[aria-invalid="true"]');
      if (invalid) invalid.focus({preventScroll: true});
    }
  }
  document.body.addEventListener('configuration-error', showConfigurationError);
  if (document.querySelector('#configuration .banner--error')) showConfigurationError();
  if (resume) {
    const restore = () => { if (resume) restorePosition(resume); };
    restore(); requestAnimationFrame(restore);
    window.addEventListener('load', restore, {once: true});
  }
  window.refreshCalibrationLayout = function () {
    if (!window.htmx || !document.getElementById('cal-steps')) return;
    htmx.ajax('GET', '/calibration', {target: '#cal-steps', select: '#cal-steps', swap: 'outerHTML'});
  };
  // Refresh live status without discarding the operator's unsaved configuration.
  window.refreshWorkflowStatus = function () {
    if (!window.htmx) return;
    htmx.ajax('GET', '/record', {target: '#record-status', select: '#record-status', swap: 'outerHTML'});
  };
  let statusTimer = null;
  function scheduleWorkflowStatus() {
    clearTimeout(statusTimer);
    statusTimer = setTimeout(() => window.refreshWorkflowStatus(), 80);
  }
})();
