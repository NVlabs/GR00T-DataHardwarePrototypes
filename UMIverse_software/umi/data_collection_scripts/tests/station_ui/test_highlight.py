# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Colour for a child process's output, which is read rather than filled in.

The station's own tools already colour their output --
`umi/common/utility/print_utils.py` writes `\\033[91m` for `print_red` -- and a
browser renders that as literal `[91m` unless it is translated.

What these tests actually protect is the escaping. `ansi_to_html` returns
`Markup`, so Jinja renders it unescaped and the escaping is its own to get
right; a regression here is a log line that closes a `<pre>` and rewrites the
page, from a filename a child process printed.
"""

from station_ui import highlight

RED = "\033[91m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ------------------------------------------------------------------- the log


def test_a_colour_code_becomes_a_span_and_the_escape_does_not_survive():
    out = str(highlight.ansi_to_html(f"{RED}camera would not open{RESET}"))

    assert out == '<span class="ansi-r">camera would not open</span>'
    assert "\033" not in out, "the raw escape reached the page"
    assert "[91m" not in out


def test_text_outside_any_colour_is_left_alone():
    out = str(highlight.ansi_to_html(f"opening {GREEN}ok{RESET} done"))

    assert out == 'opening <span class="ansi-g">ok</span> done'


def test_markup_in_the_childs_output_is_escaped_not_rendered():
    """The whole reason this returns Markup is that it escapes first.

    A child process printing a path or a repr can print anything at all, and
    `<script>` reaching a page unescaped from a subprocess's stdout is the one
    way this console could execute someone else's code.
    """
    out = str(highlight.ansi_to_html(f"{RED}<script>alert(1)</script>{RESET}"))

    assert "<script>" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out


def test_an_unclosed_colour_cannot_leak_past_its_own_line():
    """A killed child leaves a colour set and never resets it.

    Spans are emitted per run rather than nested, so the markup is balanced by
    construction and the next line starts clean.
    """
    out = str(highlight.ansi_to_html(f"{RED}died mid-print"))

    assert out.count("<span") == out.count("</span>") == 1
    assert out.endswith("</span>")


def test_bold_and_a_colour_are_both_carried():
    """`print_utils._format_message` bolds the `[device]` prefix inside the
    colour, so the two arrive nested and both have to survive."""
    out = str(highlight.ansi_to_html(f"{RED}{BOLD}[oak-06]{RESET}{RED} gone{RESET}"))

    assert 'class="ansi-r ansi-b"' in out
    assert "[oak-06]" in out
    assert "gone" in out


def test_a_cursor_escape_is_dropped_rather_than_printed():
    """A `<pre>` has no cursor. Rendering `[2J` literally is worse than no
    colour at all, and tqdm emits them by the hundred."""
    out = str(highlight.ansi_to_html("\033[2Kprogress 40%"))

    assert out == "progress 40%"


def test_an_empty_line_stays_empty():
    assert str(highlight.ansi_to_html("")) == ""
