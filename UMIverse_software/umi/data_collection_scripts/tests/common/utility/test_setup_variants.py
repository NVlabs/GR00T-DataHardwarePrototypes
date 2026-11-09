# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-episode setup variants: which option of each ``{a|b}`` group an
episode shows (``umi.common.utility.setup_variants``)."""

from collections import Counter
from itertools import product

from umi.common.utility import setup_variants

TOWEL = (
    "Place blue towel folded {once|twice|three times|four times}, tray on the table."
)
OPTIONS = ["once", "twice", "three times", "four times"]


def _option(text):
    return next(o for o in OPTIONS if f"folded {o}," in text)


def test_one_group_rotates_with_equal_shares():
    seq = [_option(setup_variants.resolve(TOWEL, "s1", e)) for e in range(12)]
    assert set(Counter(seq).values()) == {3}
    for a, b in zip(seq, seq[1:]):
        assert OPTIONS.index(b) == (OPTIONS.index(a) + 1) % 4


def test_same_session_and_episode_gives_same_text():
    assert setup_variants.resolve(TOWEL, "s1", 5) == setup_variants.resolve(
        TOWEL, "s1", 5
    )


def test_sessions_start_on_every_option():
    starts = {
        _option(setup_variants.resolve(TOWEL, f"2026-10-06_s{i}", 0)) for i in range(40)
    }
    assert starts == set(OPTIONS)


def test_groups_count_like_an_odometer():
    text = "Place {red|blue} cup {left|right|center}."
    for session in ("a", "b", "c"):
        seen = [
            (
                next(c for c in ("red", "blue") if c in t),
                next(s for s in ("left", "right", "center") if s in t),
            )
            for t in (setup_variants.resolve(text, session, e) for e in range(6))
        ]
        # Any 6 consecutive episodes cover all 2x3 combinations once.
        assert sorted(seen) == sorted(
            product(("red", "blue"), ("left", "right", "center"))
        )
        # The first group steps every episode; the second every 2.
        assert seen[0][1] == seen[1][1] and seen[1][1] != seen[2][1]


def test_text_without_groups_is_unchanged():
    text = "Place cup on the table."
    assert setup_variants.resolve(text, "s", 3) == text
    assert not setup_variants.has_variants(text)


def test_braces_without_a_bar_are_not_a_group():
    text = "Place {object} here."
    assert not setup_variants.has_variants(text)
    assert setup_variants.resolve(text, "s", 1) == text


def test_strip_removes_groups():
    assert setup_variants.strip(TOWEL) == "Place blue towel folded, tray on the table."
