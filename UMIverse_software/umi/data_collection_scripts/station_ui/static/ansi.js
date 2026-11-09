/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Render streamed ANSI SGR colors consistently with highlight.ansi_to_html. Discard
   other terminal escapes. Callers fall back to textContent if this enhancement is
   unavailable. */
(function () {
  var SGR = /\x1b\[([0-9;]*)m/g;
  var OTHER = /\x1b\[[0-9;?]*[A-Za-z]|\x1b[()][0-9A-B]|\x1b[=>]/g;
  var CLASS = {
    1: "b",
    30: "k", 31: "r", 32: "g", 33: "y", 34: "bl", 35: "m", 36: "c", 37: "w",
    90: "k", 91: "r", 92: "g", 93: "y", 94: "bl", 95: "m", 96: "c", 97: "w"
  };

  function escape(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  /* Spans are opened and closed per run rather than nested, so a child killed
     mid-colour cannot leak its styling into the rest of the page. */
  window.ansiHtml = function (text) {
    if (!text) return "";
    var out = "";
    var classes = [];
    var at = 0;
    var match;

    SGR.lastIndex = 0;
    while ((match = SGR.exec(text)) !== null) {
      out += wrap(text.slice(at, match.index), classes);
      at = SGR.lastIndex;
      var codes = match[1] ? match[1].split(";") : ["0"];
      for (var i = 0; i < codes.length; i++) {
        var code = parseInt(codes[i] || "0", 10);
        if (code === 0) classes = [];
        else if (CLASS[code]) {
          var name = "ansi-" + CLASS[code];
          if (classes.indexOf(name) === -1) classes.push(name);
        }
      }
    }
    out += wrap(text.slice(at), classes);
    return out.replace(OTHER, "");
  };

  function wrap(chunk, classes) {
    if (!chunk) return "";
    var safe = escape(chunk);
    return classes.length
      ? '<span class="' + classes.join(" ") + '">' + safe + "</span>"
      : safe;
  }

  /* Share streamed-line rendering and its plain-text fallback across log views. */
  window.appendLogLine = function (node, text) {
    if (window.ansiHtml) {
      node.insertAdjacentHTML("beforeend", window.ansiHtml(text) + "\n");
    } else {
      node.textContent += text + "\n";
    }
    node.scrollTop = node.scrollHeight;
  };
})();
