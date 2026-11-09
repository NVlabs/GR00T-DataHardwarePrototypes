/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Open the selected camera's existing server-managed JPEG preview and poll its
   status. Closing or navigating away releases it promptly; server disconnect and
   timeout cleanup are additional safeguards. */
(function () {
  var POLL_MS = 1000;
  var dialog, nameEl, statusEl, image;

  var current = null;
  var poll = null;
  // Advance the request generation so slow responses cannot replace a newer preview.

  var attempt = 0;

  function setStatus(text, isError) {
    statusEl.textContent = text || "";
    statusEl.hidden = !text;
    statusEl.classList.toggle("field__warn", !!isError);
    statusEl.classList.toggle("muted", !isError);
  }

  function stopRemote(preview) {
    if (navigator.sendBeacon) {
      navigator.sendBeacon(preview.stop_url);
    } else {
      fetch(preview.stop_url, {
        method: "POST",
        credentials: "same-origin",
        keepalive: true,
      });
    }
  }

  function release() {
    if (poll !== null) {
      clearInterval(poll);
      poll = null;
    }
    // Dropping the stream releases the server preview; explicitly stop it for prompt
    // cleanup.

    image.removeAttribute("src");
    image.hidden = true;
    if (current) {
      stopRemote(current);
      current = null;
    }
  }

  function check(preview) {
    fetch(preview.status_url, { credentials: "same-origin", cache: "no-store" })
      .then(function (response) {
        return response.json();
      })
      .then(function (status) {
        if (current !== preview) {
          return;
        }
        if (status.running) {
          if (status.streaming) {
            image.hidden = false;
            setStatus("");
          }
          return;
        }

        current = null;
        release();
        setStatus(status.message || "The preview ended.", true);
      })
      .catch(function () {});
  }

  function open(kind, device) {
    release();
    var mine = ++attempt;
    nameEl.textContent = device;
    image.alt = "Live picture from " + device;
    setStatus("Starting the camera… this takes a few seconds.", false);
    if (!dialog.open) {
      dialog.showModal();
    }
    var body = new FormData();
    body.append("kind", kind);
    body.append("device", device);
    fetch("/config/preview/start", {
      method: "POST",
      body: body,
      credentials: "same-origin",
    })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, data: data };
        });
      })
      .then(function (result) {
        if (mine !== attempt || !dialog.open) {
          // Release a preview if its response arrives after the dialog closes or
          // another device is selected.

          if (result.ok) {
            stopRemote(result.data);
          }
          return;
        }
        if (!result.ok) {
          setStatus(result.data.message || "The preview could not be started.", true);
          return;
        }
        current = result.data;
        image.src = current.stream_url;
        var preview = current;
        poll = setInterval(function () {
          check(preview);
        }, POLL_MS);
      })
      .catch(function () {
        if (mine === attempt) {
          setStatus("The preview could not be started. Is the console still running?", true);
        }
      });
  }

  function init() {
    dialog = document.getElementById("device-preview");
    if (!dialog || typeof dialog.showModal !== "function") {
      return;
    }
    nameEl = dialog.querySelector("[data-preview-name]");
    statusEl = dialog.querySelector("[data-preview-status]");
    image = dialog.querySelector("[data-preview-image]");


    image.addEventListener("load", function () {
      if (current) {
        image.hidden = false;
        setStatus("");
      }
    });
    dialog.addEventListener("close", function () {
      attempt++;
      release();
    });
    dialog.querySelector("[data-preview-close]").addEventListener("click", function () {
      dialog.close();
    });
    // Delegate events because discovery replaces the complete table.
    document.addEventListener("click", function (event) {
      var button = event.target.closest && event.target.closest("[data-preview-device]");
      if (button) {
        event.preventDefault();
        open(button.getAttribute("data-preview-kind"), button.getAttribute("data-preview-device"));
      }
    });
    window.addEventListener("pagehide", release);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
