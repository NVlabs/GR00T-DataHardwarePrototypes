# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A rolled session_uid that happens to be all digits must stay a string.

`session_uid` is `uuid4().hex[:12]`, so roughly one in 29,000 contains no
letters. If `_replace_yaml_value` decided whether to quote by looking at the
value it is replacing, such a uid would be written unquoted, read back as an
int, and `meta.get("session_uid")` downstream would hold a number where
everything else holds a string. A failure that rare surfaces as an unrelated
flaky test (`assert 466545581325 == '466545581325'` in test_session_roll);
these tests make it deterministic.
"""

import yaml

from umi.common.recording.session_slot import _replace_yaml_value


def _round_trip(text: str, key: str, value: str):
    return yaml.safe_load(_replace_yaml_value(text, key, value))[key]


def test_an_all_digit_uid_survives_as_a_string():
    out = _round_trip("session_uid: a3f9c2b1d4e5\n", "session_uid", "466545581325")
    assert out == "466545581325"
    assert isinstance(out, str), "YAML read it back as a number"


def test_a_normal_uid_is_still_written_bare():
    # The common case must not grow quotes it never had: the diff a station
    # operator sees should stay empty when nothing meaningful changed.
    text = _replace_yaml_value(
        "session_uid: a3f9c2b1d4e5\n", "session_uid", "b7c1d2e3f4a5"
    )
    assert text.strip() == "session_uid: b7c1d2e3f4a5"


def test_an_already_quoted_value_stays_quoted():
    text = _replace_yaml_value(
        'session_uid: "a3f9c2b1d4e5"\n', "session_uid", "b7c1d2e3f4a5"
    )
    assert text.strip() == 'session_uid: "b7c1d2e3f4a5"'


def test_other_yaml_traps_are_quoted_too():
    # The reason this asks the YAML parser instead of testing for digits: the
    # set of scalars it reinterprets is longer than anyone remembers.
    for value in ("yes", "no", "null", "~", "1e5", "0x1f", "on", "off"):
        out = _round_trip("k: placeholder\n", "k", value)
        assert out == value, f"{value!r} came back as {out!r}"
        assert isinstance(out, str)


def test_a_path_with_digits_is_untouched():
    # Not everything numeric-looking is a trap; a data_dir is still a string.
    out = _round_trip(
        "data_dir: ./data\n", "data_dir", "./data/2026-09-17_x_466545581325"
    )
    assert out == "./data/2026-09-17_x_466545581325"
