# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The design contract of the shell: measured contrast, the layout, and the
accessibility structure every later page inherits from base.html.

Contrast is computed from the tokens actually declared in app.css, not from a
copy of them, so editing a token re-runs the audit.

The layout half pins the page measure, the preset grid, the window chrome and
the wizard progression, which are easy to regress without a test: full-bleed at
1440, for example, would put the four preset cards in a thin 4-across strip.
"""

import pathlib
import re
from html.parser import HTMLParser

from fastapi.testclient import TestClient

from station_ui.app import create_app
from station_ui.core import setup_presets as sp
from station_ui.settings import Settings

REPO = pathlib.Path(__file__).resolve().parents[2]
CSS = (REPO / "station_ui" / "static" / "app.css").read_text()

# WCAG 2.1 SC 1.4.3 — normal-size text. Nothing in this shell is "large text"
# (>=18.66px bold or >=24px), so 3:1 never applies to a foreground here.
AA_TEXT = 4.5
# SC 1.4.11 — non-text contrast, used for the focus ring.
AA_NON_TEXT = 3.0

# Measure foreground/background pairs that the stylesheet actually renders.
PAIRS = [
    ("--color-text", "--color-bg", "body copy, and the active tab on the tabbar"),
    ("--color-text", "--color-surface", ".card / .btn / .preset-card / .banner"),
    ("--color-text", "--color-surface-alt", ".input / .tabbar / .btn:hover"),
    ("--color-text-2", "--color-bg", ".muted in the main region"),
    ("--color-text-2", "--color-surface", ".field__label"),
    ("--color-text-2", "--color-surface-alt", ".badge, .pill, .job-log, .nav__link"),
    ("--color-text-3", "--color-bg", "tertiary metadata, and .appbar__identity"),
    ("--color-text-3", "--color-surface", "tertiary metadata inside a card"),
    ("--color-text-3", "--color-surface-alt", ".nav__step, .nav__link--disabled"),
    ("--color-primary", "--color-header", "brand mark in the dark utility bar"),
    ("--color-header-text", "--color-header", "utility bar and station identity"),
    ("--color-header-error", "--color-header", "error link in the utility bar"),
    ("--color-bg", "--color-header", "white text on a dark accent"),
    ("--color-bg", "--status-ok", "check mark in the pairing status circle"),
    ("--color-text", "--color-primary", ".btn--primary label"),
    ("--status-ok", "--color-bg", "status text on the page"),
    ("--status-ok", "--color-surface", "status text inside a card"),
    ("--status-ok", "--color-surface-alt", ".badge--ok"),
    ("--status-warn", "--color-bg", "status text on the page"),
    ("--status-warn", "--color-surface", "status text inside a card"),
    ("--status-warn", "--color-surface-alt", ".badge--warn"),
    ("--status-error", "--color-bg", "status text on the page"),
    ("--status-error", "--color-surface", ".btn--danger label, .field__error"),
    ("--status-error", "--color-surface-alt", ".badge--error under an input"),
    # Machine-readable text. A separate hue from the brand, deliberately:
    # painting filenames and pills brand green would leave the single primary
    # action indistinguishable from every code span on the page.
    ("--color-code", "--color-bg", "<code>, .mono, .field__streams on the page"),
    ("--color-code", "--color-surface", ".btn--accent label, code inside a card"),
    ("--color-code", "--color-surface-alt", ".path, .pill, code in an input row"),
    # Status tints. A status drawn as one coloured line among twenty grey ones
    # is a status nobody reads, so every status surface is tinted as well as
    # iconed and worded — which composes three more backgrounds.
    ("--color-text", "--tint-warn", ".banner--warn body copy"),
    ("--color-text", "--tint-error", ".banner--error body copy"),
    ("--color-text", "--tint-info", ".banner--info iPhone-generation notice"),
    ("--color-text-2", "--tint-warn", ".muted inside a warning banner"),
    ("--color-text-2", "--tint-error", ".muted inside an error banner"),
    ("--status-ok", "--tint-ok", ".badge--ok"),
    ("--status-warn", "--tint-warn", ".badge--warn, .banner--warn__icon"),
    (
        "--status-error",
        "--tint-error",
        ".badge--error, .banner--error__icon, .field__error",
    ),
    ("--color-code", "--tint-warn", "<code> inside a warning banner"),
    ("--color-code", "--tint-error", "<code> inside an error banner"),
]

# Measure text-free value indicators against their backgrounds at the
# 3:1 non-text contrast threshold, separately from text pairs.
NON_TEXT_FILLS = [
    ("--color-code", "--color-surface", ".preset-card selected radio"),
    ("--status-warn", "--color-surface-alt", ".disp-bar__fill on its own track"),
    ("--status-ok", "--color-surface-alt", ".disp-bar__fill--ok on its own track"),
]

SURFACES = ("--color-bg", "--color-surface", "--color-surface-alt")

# The website uses fine sage dividers on white. These layout boundaries are
# intentionally subtle; keyboard focus uses a separate dark-green token, measured
# at 3:1 on every surface below. The boundary token is never used for body text.
EXEMPT_BOUNDARY_TOKEN = "--color-border"


def _tokens() -> dict[str, str]:
    """Hex custom properties declared in app.css's :root block."""
    root = re.search(r":root\s*\{(.*?)\n\}", CSS, re.S)
    assert root, "app.css has no :root token block"
    return dict(re.findall(r"(--[\w-]+):\s*(#[0-9A-Fa-f]{6})\b", root.group(1)))


def _relative_luminance(hex_colour: str) -> float:
    raw = hex_colour.lstrip("#")
    channels = [int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    la, lb = _relative_luminance(a), _relative_luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def test_the_contrast_helper_agrees_with_a_known_value():
    """Guard the guard: a broken ratio function would pass everything."""
    tokens = _tokens()
    assert round(_contrast("#FFFFFF", "#000000"), 2) == 21.0
    assert round(_contrast("#1A1A1A", "#1A1A1A"), 2) == 1.0
    assert tokens["--color-primary"].upper() == "#8BD62A"
    assert tokens["--color-bg"].upper() == "#FFFFFF"


def test_every_text_pair_meets_wcag_aa():
    tokens = _tokens()
    failures = []
    for fg, bg, where in PAIRS:
        ratio = _contrast(tokens[fg], tokens[bg])
        if ratio < AA_TEXT:
            failures.append(f"{fg} on {bg} = {ratio:.2f}:1 (< {AA_TEXT}) — {where}")
    assert not failures, "WCAG AA text failures:\n  " + "\n  ".join(failures)


def test_every_fill_that_carries_no_text_still_meets_non_text_contrast():
    """A bar an operator glances at across a bench is only a status if it is
    visible against its own track. Same 3:1 the focus ring is held to, and the
    reason NON_TEXT_FILLS exists rather than a PAIRS entry with an invented
    foreground."""
    tokens = _tokens()
    for fill, surface, where in NON_TEXT_FILLS:
        ratio = _contrast(tokens[fill], tokens[surface])
        assert ratio >= AA_NON_TEXT, f"{fill} on {surface} = {ratio:.2f}:1 — {where}"


def test_the_focus_ring_meets_non_text_contrast_on_every_surface():
    tokens = _tokens()
    for surface in SURFACES:
        ratio = _contrast(tokens["--color-focus"], tokens[surface])
        assert ratio >= AA_NON_TEXT, f"focus ring on {surface} = {ratio:.2f}:1"


def test_the_border_shortfall_is_the_known_accepted_one():
    """The exemption, asserted. Keeps it a measured decision rather than an
    omission, and fails the moment the boundary token is used as text."""
    tokens = _tokens()
    for surface in SURFACES:
        ratio = _contrast(tokens[EXEMPT_BOUNDARY_TOKEN], tokens[surface])
        assert ratio < AA_NON_TEXT, (
            f"{EXEMPT_BOUNDARY_TOKEN} now clears 1.4.11 on {surface} ({ratio:.2f}:1) — "
            "delete this exemption and the note in design spec section 5.2"
        )
    assert not re.search(rf"(?<!-)\bcolor:\s*var\({EXEMPT_BOUNDARY_TOKEN}\)", CSS), (
        f"{EXEMPT_BOUNDARY_TOKEN} is a boundary colour, never a text colour"
    )


def _same_rule_pairs() -> list[tuple[str, str, str]]:
    """(foreground, background, selector) for every rule that sets both."""
    found = []
    for chunk in CSS.split("}"):
        if "{" not in chunk:
            continue
        selector, _, body = chunk.rpartition("{")
        fg = re.search(r"(?<!-)\bcolor:\s*var\((--[\w-]+)\)", body)
        bg = re.search(r"background:\s*var\((--[\w-]+)\)", body)
        if fg and bg:
            name = selector.strip().splitlines()[-1].strip()
            found.append((fg.group(1), bg.group(1), name))
    return found


def test_no_colour_used_in_app_css_escapes_the_contrast_audit():
    """The part that binds every page's styles. Two checks, because either
    alone has a hole:

    - every token used as a `color:` or `background:` must appear in PAIRS, so a
      brand-new token cannot ship unmeasured;
    - every rule that sets both must have that exact combination in PAIRS, so
      recombining two already-listed tokens into an unmeasured pair (which the
      token-presence check alone waves through) fails too.

    Cross-element pairs — a colour on one element over an ancestor's background
    — are not statically derivable, which is why PAIRS is curated by hand.
    """
    used_fg = set(re.findall(r"(?<!-)\bcolor:\s*var\((--[\w-]+)\)", CSS))
    used_bg = set(re.findall(r"background:\s*var\((--[\w-]+)\)", CSS))
    covered_fg = {fg for fg, _, _ in PAIRS}
    covered_bg = {bg for _, bg, _ in PAIRS} | {fill for fill, _, _ in NON_TEXT_FILLS}
    assert used_fg <= covered_fg, f"uncovered foreground tokens: {used_fg - covered_fg}"
    assert used_bg <= covered_bg, f"uncovered background tokens: {used_bg - covered_bg}"

    listed = {(fg, bg) for fg, bg, _ in PAIRS}
    unmeasured = [
        f"{fg} on {bg} in `{sel}`"
        for fg, bg, sel in _same_rule_pairs()
        if (fg, bg) not in listed
    ]
    assert not unmeasured, "rules compose unmeasured pairs:\n  " + "\n  ".join(
        unmeasured
    )


def test_app_css_never_fades_text_with_opacity():
    """An `opacity: .55` disabled treatment composites --color-text-3 to
    #575757 = 2.15:1. Opacity is invisible to a token audit, so it is banned
    outright; the sole exception is the @keyframes that pulses an empty
    .skeleton placeholder, which contains no text."""
    without_keyframes = re.sub(r"@keyframes[^{]*\{.*?\}\s*\}", "", CSS, flags=re.S)
    offenders = re.findall(r"^[^\n]*\bopacity\s*:[^\n]*$", without_keyframes, re.M)
    assert not offenders, "opacity outside @keyframes fades text: " + "; ".join(
        offenders
    )


def test_focus_rings_are_never_removed():
    assert ":focus-visible" in CSS
    assert not re.search(r"outline:\s*(none|0)\b", CSS), "a focus ring is being removed"


def test_reduced_motion_disables_every_transition():
    assert "@media (prefers-reduced-motion: reduce)" in CSS
    block = CSS.split("@media (prefers-reduced-motion: reduce)")[1]
    for prop in ("transition-duration", "animation-duration"):
        assert prop in block, f"{prop} is not neutralised for reduced motion"


# Cover container width, preset grid, application chrome and workflow progression.


def _declared(selector: str, prop: str) -> list[str]:
    """Every value app.css declares for `prop` on `selector`, in source order.

    All of them, not just the winning one: an override tucked into a later rule
    or a @media block is exactly how a fixed column count would quietly become a
    viewport-dependent one again.
    """
    values = []
    for chunk in CSS.split("}"):
        if "{" not in chunk:
            continue
        selectors, _, body = chunk.rpartition("{")
        names = [s.strip() for s in selectors.strip().splitlines()[-1].split(",")]
        if selector not in names:
            continue
        match = re.search(rf"(?<![\w-]){re.escape(prop)}\s*:\s*([^;]+);", body)
        if match:
            values.append(" ".join(match.group(1).split()))
    return values


# HTML void elements: they have no end tag, so a depth counter must not treat
# them as opening anything.
_VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


class _ByClass(HTMLParser):
    """Attributes and visible text of every element carrying `classname`.

    Depth-tracked rather than regex-matched: every entry in this shell wraps
    nested spans and an inline <svg>, and a non-greedy regex for the closing tag
    stops at the first of those instead of at the element's own.
    """

    def __init__(self, classname: str):
        super().__init__()
        self.classname = classname
        self.found: list[dict] = []
        self._current: dict | None = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if self._current is not None:
            # HTML void elements have no end tag; do not count them toward nesting
            # depth.
            if tag not in _VOID_ELEMENTS:
                self._depth += 1
            return
        attr = dict(attrs)
        if self.classname in (attr.get("class") or "").split():
            self._current = {**attr, "text": ""}
            self._depth = 1

    def handle_endtag(self, tag):
        if self._current is None:
            return
        self._depth -= 1
        if self._depth == 0:
            self._current["text"] = " ".join(self._current["text"].split())
            self.found.append(self._current)
            self._current = None

    def handle_data(self, data):
        if self._current is not None:
            self._current["text"] += data


def _by_class(html: str, classname: str) -> list[dict]:
    parser = _ByClass(classname)
    parser.feed(html)
    return parser.found


def _nav_entries(html: str) -> list[dict]:
    return _by_class(html, "nav__link")


def _completed(html: str) -> set[str]:
    """The hrefs whose step reports itself finished. The marker is a word, so
    this also proves progress is not signalled by colour alone."""
    routes = {
        "#setup": "/",
        "#configuration": "/config",
        "#calibration": "/calibration",
        "#record": "/record",
        "#audit": "/audit",
    }
    return {
        routes.get(e["href"], e["href"])
        for e in _nav_entries(html)
        if "complete" in e["text"]
    }


def test_the_page_content_sits_in_a_container_with_a_bounded_measure():
    """Full-bleed at 1440 would put the preset cards in a 4-across strip and
    stretch every form line to a 1400px reading measure."""
    values = _declared(".container", "max-width")
    assert values, ".container declares no max-width, so the page is full-bleed"
    assert all(re.fullmatch(r"\d+px", value) for value in values), (
        f"a non-px max-width does not bound anything: {values}"
    )
    widths = [int(value[:-2]) for value in values]
    assert max(widths) <= 1280, (
        f"max-width {max(widths)}px does not bound the content on the 1440px "
        "station monitor"
    )

    # And the container is really wrapped around the page, not just declared.
    html = TestClient(create_app()).get("/").text
    assert re.search(r'<main[^>]*class="[^"]*\bcontainer\b', html), (
        "<main> does not carry the container, so nothing is bounded"
    )


def test_the_workflow_uses_the_documentation_sidebar_and_mobile_row():
    """The same server-controlled links reflow without changing navigation."""
    assert _declared(".shell", "grid-template-columns")[0] == "260px minmax(0, 1fr)"
    assert _declared(".shell", "grid-template-areas") == [
        '"header header" "sidebar main"'
    ]
    assert _declared(".nav", "flex-direction") == ["column", "row"]
    assert _declared(".tabbar", "position") == ["sticky", "static"]
    assert '.nav__link[aria-current="page"]::before' in CSS


def test_the_preset_grid_has_two_desktop_columns_and_stacks_on_mobile():
    """Keep setup choices paired on desktop and readable on narrow screens."""
    values = _declared(".preset-grid", "grid-template-columns")
    assert values, ".preset-grid does not declare a column count"
    for value in values:
        assert "auto-fit" not in value and "auto-fill" not in value, (
            f"{value!r} lets the viewport pick the column count"
        )
        assert value.startswith("repeat(2") or value == "minmax(0, 1fr)", (
            f"{value!r} is neither the desktop pair nor a single mobile column"
        )

    html = TestClient(create_app()).get("/").text
    grid = _by_class(html, "preset-grid")
    assert len(grid) == 1, "the cards are not inside a single preset grid"
    # Every preset is a card on the grid.
    assert html.count('class="preset-card"') == len(sp.SETUP_PRESETS)


def test_there_is_no_admin_mode():
    """Every browser gets the operator's console: no admin entry point in the
    chrome, and no page or unlock endpoint behind it."""
    client = TestClient(create_app())
    assert 'href="/admin"' not in client.get("/").text
    assert client.get("/admin").status_code == 404
    assert client.post("/admin/unlock", data={}).status_code == 404


def test_the_chrome_says_which_station_this_is(monkeypatch):
    """Several stations share a bench. A tab that does not say which rig it
    drives is a mis-recording waiting to happen."""
    monkeypatch.setenv("UMI_STATION_ID", "gear-umi-42")
    app = create_app()
    html = TestClient(app).get("/").text

    identity = _by_class(html, "appbar__identity")
    assert len(identity) == 1, "no station identity in the app chrome"
    text = identity[0]["text"]
    assert "gear-umi-42" in text, f"the strip does not name the station: {text!r}"


def test_the_nav_numbers_every_step_in_workflow_order():
    app = create_app()
    entries = _nav_entries(TestClient(app).get("/").text)
    labels = [item["label"] for item in app.state.templates.env.globals["nav_items"]]
    assert len(entries) == len(labels)

    for position, (entry, label) in enumerate(zip(entries, labels), start=1):
        assert entry["text"].startswith(f"{position:02d} "), (
            f"{entry['text']!r} does not lead with step {position}"
        )
        assert label in entry["text"], entry["text"]


def test_a_step_is_ticked_only_once_the_draft_really_holds_it():
    """Progress comes from the draft, never from a list of steps someone
    remembered to tick."""
    app = create_app()
    client = TestClient(app)

    assert _completed(client.get("/").text) == set(), (
        "a step claims to be complete before anything has been chosen"
    )

    client.post("/setup/preset", data={"preset_id": sp.SETUP_PRESETS[-1].id})
    assert _completed(client.get("/").text) == {"/"}, (
        "choosing a preset did not complete Setup, or completed something else"
    )

    app.state.rig.draft.preset_id = None
    assert _completed(client.get("/").text) == set(), (
        "the tick outlived the state it is supposed to report"
    )


def test_configuration_is_ticked_only_when_its_config_file_exists(tmp_path):
    """The artefact, not the appearance: Configuration is done when the file the
    station records from is on disk."""
    # Two roots: templates are read from the checkout, the generated config is
    # written under the state dir (settings.STATE_DIR_ENV).
    settings = Settings(repo_root=tmp_path, state_dir=tmp_path / "state")
    app = create_app(settings)
    client = TestClient(app)
    client.get("/")
    preset_id = sp.SETUP_PRESETS[-1].id
    # Not followed: /config against an empty tree is not what is under test.
    client.post("/setup/preset", data={"preset_id": preset_id}, follow_redirects=False)

    assert _completed(client.get("/").text) == {"/"}, (
        "Configuration reports itself done with no config written"
    )

    generated = settings.state_dir.joinpath("data_recording", f"{preset_id}.yaml")
    generated.parent.mkdir(parents=True)
    generated.write_text("schema_version: 2\n")

    assert _completed(client.get("/").text) == {"/", "/config"}, (
        "the tick ignores the config file the station actually records from"
    )


class _Shell(HTMLParser):
    """Just enough parsing to assert the landmarks; no bs4 dependency."""

    def __init__(self):
        super().__init__()
        self.headings: list[int] = []
        self.ids: set[str] = set()
        self.main_count = 0
        self.skip_href: str | None = None
        self.html_lang: str | None = None
        self.scripts: list[str] = []
        self.stylesheets: list[str] = []
        self.nav_labels: list[str] = []

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if attr.get("id"):
            self.ids.add(attr["id"])
        if tag == "main":
            self.main_count += 1
        elif tag == "html":
            self.html_lang = attr.get("lang")
        elif tag == "nav":
            self.nav_labels.append(attr.get("aria-label") or "")
        elif tag == "script" and attr.get("src"):
            self.scripts.append(attr["src"])
        elif tag == "link" and attr.get("rel") == "stylesheet":
            self.stylesheets.append(attr.get("href", ""))
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.headings.append(int(tag[1]))
        elif tag == "a" and "skip-link" in (attr.get("class") or ""):
            self.skip_href = attr.get("href")


def _parse_shell() -> _Shell:
    parser = _Shell()
    parser.feed(TestClient(create_app()).get("/").text)
    return parser


def test_the_shell_has_a_skip_link_that_targets_the_main_landmark():
    shell = _parse_shell()
    assert shell.skip_href, "no skip link"
    assert shell.skip_href.startswith("#"), "skip link must target an in-page anchor"
    assert shell.skip_href[1:] in shell.ids, (
        f"skip link points at {shell.skip_href}, which no element declares"
    )


def test_the_shell_has_exactly_one_main_landmark_and_a_labelled_nav():
    shell = _parse_shell()
    assert shell.main_count == 1, f"found {shell.main_count} <main> landmarks"
    assert shell.nav_labels and all(shell.nav_labels), "every <nav> needs an aria-label"
    assert shell.html_lang, "<html> needs a lang attribute"


def test_the_shell_skips_no_heading_level():
    shell = _parse_shell()
    assert shell.headings, "the page has no headings"
    assert shell.headings[0] == 1, f"first heading is h{shell.headings[0]}, not h1"
    for previous, current in zip(shell.headings, shell.headings[1:]):
        assert current <= previous + 1, f"h{previous} is followed by h{current}"


def test_every_nav_item_renders_once():
    app = create_app()
    html = TestClient(app).get("/").text
    nav = re.search(r"<nav[^>]*>(.*?)</nav>", html, re.S)
    assert nav, "no <nav> in the shell"
    entries = re.findall(r'class="nav__link[^"]*"[^>]*', nav.group(1))
    assert len(entries) == len(app.state.templates.env.globals["nav_items"]) == 5


def test_a_disabled_nav_item_is_marked_not_merely_greyed():
    """State is never colour alone, so a disabled entry must carry
    aria-disabled and not rely on the grey alone.

    The disabled entry is injected rather than taken from the real nav: the
    shipped nav may have no disabled item, and asserting on real rows would then
    pass vacuously. Injecting one keeps the property under test permanently,
    and keeps it honest if a future page ships gated.
    """
    app = create_app()
    nav_items = app.state.templates.env.globals["nav_items"]
    injected = list(nav_items) + [
        {"href": "/not-yet", "label": "Not Yet", "enabled": False}
    ]
    app.state.templates.env.globals["nav_items"] = injected
    try:
        html = TestClient(app).get("/log").text
    finally:
        app.state.templates.env.globals["nav_items"] = nav_items

    nav = re.search(r"<nav[^>]*>(.*?)</nav>", html, re.S)
    assert nav, "no <nav> in the shell"
    entries = re.findall(r'class="nav__link[^"]*"[^>]*', nav.group(1))
    disabled = [e for e in entries if "nav__link--disabled" in e]
    assert disabled, "the injected entry did not render as disabled"
    for entry in disabled:
        assert 'aria-disabled="true"' in entry, f"greyed but unmarked: {entry}"


def test_the_shell_loads_the_vendored_script_and_stylesheet():
    """Without this, deleting the <script> tag would leave the suite green."""
    shell = _parse_shell()
    assert any(src.endswith("/static/htmx.min.js") for src in shell.scripts), (
        f"htmx is not loaded; scripts = {shell.scripts}"
    )
    assert any(href.endswith("/static/app.css") for href in shell.stylesheets), (
        f"app.css is not loaded; stylesheets = {shell.stylesheets}"
    )
