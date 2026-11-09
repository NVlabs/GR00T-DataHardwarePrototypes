/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Stream audit logs with EventSource and retain server-rendered output plus Refresh
   as a fallback. */
(function () {
  var log = document.getElementById("job-log");
  if (!log || !window.EventSource) return;
  if (log.getAttribute("data-state") !== "live") return;

  var url = log.getAttribute("data-stream");
  if (!url) return;

  var source = new EventSource(url);
  var replaced = false;

  source.onmessage = function (event) {
    // Clear the server-rendered copy once because the stream replays its lines.

    if (!replaced) {
      log.textContent = "";
      replaced = true;
    }
    window.appendLogLine(log, event.data);
    log.scrollTop = log.scrollHeight;
  };

  source.addEventListener("end", function () {
    source.close();
    // Reload on completion so the server remains responsible for final job status.


    if (document.body.classList.contains("page--workflow") && window.htmx) {
      htmx.ajax("GET", window.location.pathname, {target: "#audit", select: "#audit", swap: "outerHTML"});
    } else { window.location.reload(); }
  });

  source.onerror = function () {
    // A dropped stream must not look like a finished job: leave what arrived
    // on screen and let the operator use Refresh.
    source.close();
  };
})();
