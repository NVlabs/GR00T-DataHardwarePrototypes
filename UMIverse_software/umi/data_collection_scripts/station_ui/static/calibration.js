/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Calibration only: show the latest frame from the already-running check.
 * Panel swaps and navigation retire their images; no extra camera is opened.
 */
(function () {
  "use strict";
  function refresh() {
    if (document.hidden) { return; }
    document.querySelectorAll("[data-calibration-camera-src]").forEach(function (image) {
      if (image.dataset.loading === "true") { return; }
      var status = image.parentElement.querySelector("[data-calibration-camera-status]");
      image.dataset.loading = "true";
      image.onload = function () {
        image.dataset.loading = "false";
        image.hidden = false;
        if (status) { status.hidden = true; }
      };
      image.onerror = function () {
        image.dataset.loading = "false";
        image.hidden = true;
        if (status) {
          status.hidden = false;
          var message = "Waiting for the live preview. If it does not appear, check the camera output below.";
          if (status.textContent !== message) { status.textContent = message; }
        }
      };
      image.src = image.dataset.calibrationCameraSrc + "?t=" + Date.now();
    });
  }
  refresh();
  var timer = window.setInterval(refresh, 300);
  window.addEventListener("pagehide", function () { window.clearInterval(timer); });
})();
