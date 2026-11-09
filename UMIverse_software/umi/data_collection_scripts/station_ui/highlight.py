# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Colour for a child process's output, the one thing on this console that is
read rather than filled in.

The station's own tools already colour their output
(`umi/common/utility/print_utils.py` writes `\\033[91m` for `print_red` and
friends), and a browser renders that as literal escape gibberish. This turns
the colour codes into styled spans.

Escaping happens here, not in the template. `ansi_to_html` returns `Markup`,
which Jinja renders unescaped, so every path into it has to escape first, and
there is exactly one path in. A helper that returned a plain `str` would be
escaped by the template and show its own tags.
"""

from __future__ import annotations

import html
import re

from markupsafe import Markup

__all__ = ["ansi_to_html"]

# ---------------------------------------------------------------- ANSI

# Render SGR color sequences only; discard cursor and other terminal control
# escapes that have no meaning in a browser log.
_SGR = re.compile(r"\x1b\[([0-9;]*)m")
_OTHER_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[()][0-9A-B]|\x1b[=>]")

# Support the eight standard/bright colors and bold emitted by print_utils.
_CLASS = {
    1: "b",
    30: "k",
    31: "r",
    32: "g",
    33: "y",
    34: "bl",
    35: "m",
    36: "c",
    37: "w",
    90: "k",
    91: "r",
    92: "g",
    93: "y",
    94: "bl",
    95: "m",
    96: "c",
    97: "w",
}


def ansi_to_html(text: str) -> Markup:
    """One log line, with its ANSI colours as spans. Escapes as it goes.

    Spans are opened and closed per run rather than nested, so a line that sets
    a colour and never resets (which a killed child process leaves behind)
    cannot leak its styling into the rest of the page. Every run this emits is
    balanced by construction.
    """
    if not text:
        return Markup("")

    out = []
    classes: list[str] = []
    position = 0

    def emit(chunk: str) -> None:
        if not chunk:
            return
        safe = html.escape(chunk)
        if classes:
            out.append(f'<span class="{" ".join(classes)}">{safe}</span>')
        else:
            out.append(safe)

    for match in _SGR.finditer(text):
        emit(text[position : match.start()])
        position = match.end()
        # An empty parameter list is `ESC[m`, which means reset.
        codes = (
            [int(c or 0) for c in match.group(1).split(";")] if match.group(1) else [0]
        )
        for code in codes:
            if code == 0:
                classes = []
            elif code in _CLASS:
                name = f"ansi-{_CLASS[code]}"
                if name not in classes:
                    classes.append(name)
    emit(text[position:])

    return Markup(_OTHER_ESCAPE.sub("", "".join(out)))
