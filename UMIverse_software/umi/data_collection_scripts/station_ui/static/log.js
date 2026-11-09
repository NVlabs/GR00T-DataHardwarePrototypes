/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Stream the persistent station log without job-completion reloads. Report
   disconnection rather than leaving a false Live badge. Keep server-rendered text
   and Refresh as a fallback. */
(function () {
  var log = document.getElementById("station-log");
  var state = document.getElementById("log-stream-state");
  if (!log || !window.EventSource) return;
  if (log.getAttribute("data-state") !== "live") return;

  var url = log.getAttribute("data-stream");
  if (!url) return;

  var source = new EventSource(url);
  var replaced = false;

  source.onmessage = function (event) {
    // Clear the static tail once because the stream replays it.

    if (!replaced) {
      log.textContent = "";
      replaced = true;
    }
    window.appendLogLine(log, event.data);
  };

  source.onerror = function () {
    source.close();
    log.setAttribute("data-state", "stopped");
    if (state) state.textContent = "Not live";
    log.textContent += "[log] stream stopped — press Refresh\n";
  };
})();
