/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Preserve each details panel's user-selected open state across polling swaps.
   Untouched panels retain the server default. */
(function () {

  var chosen = Object.create(null);

  // Listen for toggle during capture because it does not bubble.

  document.addEventListener(
    "toggle",
    function (event) {
      var el = event.target;
      if (
        el.tagName === "DETAILS" &&
        el.classList &&
        el.classList.contains("job-details") &&
        el.id
      ) {
        chosen[el.id] = el.open;
      }
    },
    true
  );

  function restore(root) {
    var scope = root && root.querySelectorAll ? root : document;
    var panels = scope.querySelectorAll("details.job-details[id]");
    for (var i = 0; i < panels.length; i++) {
      var el = panels[i];
      // Restore only explicit user choices; keep server-opened failure logs visible.

      if (el.id in chosen) {
        el.open = chosen[el.id];
      }
    }
  }

  // Restore panel choices after HTMX replaces a fragment.

  document.addEventListener("htmx:afterSwap", function () {
    restore(document);
  });
  document.addEventListener("DOMContentLoaded", function () {
    restore(document);
  });
})();
