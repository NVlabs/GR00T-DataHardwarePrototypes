/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Display failed HTMX requests in the page. Native form navigation remains the
   fallback when this script is unavailable. */
(function () {
  /* Display server crash responses in the banner rather than swapping a complete
     page into a control. */
  var slot = document.getElementById("crash-slot");
  if (slot) {
    document.body.addEventListener("htmx:responseError", function (evt) {
      var xhr = evt.detail && evt.detail.xhr;
      if (!xhr || xhr.status !== 500) return;
      slot.innerHTML = xhr.responseText;
      slot.focus();
      slot.scrollIntoView({ block: "nearest" });
    });
  }

  /* Swap expected-error responses only when the server marks them as fragments. Mark
     rendered refusals handled; complete error documents must not replace cards. */
  document.body.addEventListener("htmx:beforeSwap", function (evt) {
    var xhr = evt.detail && evt.detail.xhr;
    if (!xhr || xhr.status < 400 || xhr.status >= 500) return;
    if (!xhr.getResponseHeader("X-Station-Fragment")) return;
    evt.detail.shouldSwap = true;
    evt.detail.isError = false;
  });
})();
