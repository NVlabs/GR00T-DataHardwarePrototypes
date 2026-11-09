# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest

from station_ui.core import optional_devices as optional_devices
from station_ui.core.optional_devices import (
    OPTIONAL_DEVICE_TYPES,
    PLACEMENTS_BY_SECTION,
    OptionalDeviceAssignment,
    allocate_logical_name,
    can_add,
    get_optional_type,
    normalize_placement,
    placement_label,
    placement_options,
    placements_for_section,
    types_for_section,
    validate_placement,
)


def _a(type_id, name):
    return OptionalDeviceAssignment(type_id=type_id, logical_name=name)


def test_head_offers_exactly_oak_and_iphone_in_that_order():
    """Order is load-bearing: it is the order the head's Add buttons render in
    and the order its rows come back in. The OAK leads because it is what the
    rigs carry and what a fresh session starts with -- declared second, the
    head would open on "Head iPhone" and read as though the phone were the
    default."""
    assert [t.id for t in types_for_section("head")] == ["head_oak", "head_iphone"]


def test_bottom_offers_exactly_the_oak():
    """The static table camera is one OAK with a fixed name, in a section of
    its own: it is not on the head mount."""
    assert [t.id for t in types_for_section("bottom")] == ["bottom_oak"]


def test_the_optional_sections_are_head_and_bottom():
    """The Configuration page renders one section per section named here, so
    this is the list of optional-camera sections an operator sees."""
    assert {t.section for t in OPTIONAL_DEVICE_TYPES} == {"head", "bottom"}


def test_the_bottom_oak_has_a_fixed_name_and_a_cap_of_one():
    assert allocate_logical_name("bottom_oak") == "bottom_oak_camera"
    assert can_add("bottom_oak", []) is True
    assert can_add("bottom_oak", [_a("bottom_oak", "bottom_oak_camera")]) is False
    # Independent of the head: a rig may carry a head OAK and a bottom OAK.
    assert can_add("bottom_oak", [_a("head_oak", "head_oak_camera")]) is True


def test_a_bottom_camera_takes_only_bottom_placements():
    assert validate_placement("table", "bottom") == "table"
    assert validate_placement("", "bottom") == ""
    with pytest.raises(ValueError, match="bottom"):
        validate_placement("neck", "bottom")
    with pytest.raises(ValueError, match="bottom"):
        validate_placement("left_front", "bottom")


def test_head_names_are_fixed():
    assert allocate_logical_name("head_iphone") == "head_iphone_camera"
    assert allocate_logical_name("head_oak") == "head_oak_camera"


def test_head_allows_one_iphone_and_one_oak_together():
    existing = [_a("head_iphone", "head_iphone_camera")]
    assert can_add("head_oak", existing) is True
    assert can_add("head_iphone", existing) is False
    assert can_add("head_oak", existing + [_a("head_oak", "head_oak_camera")]) is False


def test_placement_is_optional_and_defaults_empty():
    a = OptionalDeviceAssignment(type_id="head_oak", logical_name="head_oak_camera")
    assert a.placement == ""


def test_validate_placement_accepts_the_known_vocabulary_and_empty():
    for section in ("head", "bottom"):
        for value in placements_for_section(section):
            assert validate_placement(value, section) == value


def test_validate_placement_rejects_unknown_values():
    with pytest.raises(ValueError, match="placement"):
        validate_placement("somewhere", "head")


# ------------------------------------------------------- per-section vocabulary


def test_the_head_sits_on_a_body_and_the_bottom_on_the_table():
    # The two sections describe different things, so one shared list could only
    # ever be right for one of them.
    assert placements_for_section("head") == ("", "head", "neck", "static")
    assert placements_for_section("bottom") == ("", "table")


def test_every_section_that_has_devices_has_a_placement_list():
    for descriptor in OPTIONAL_DEVICE_TYPES:
        assert placements_for_section(descriptor.section)


def test_a_section_nothing_declares_is_a_mistake_not_an_empty_list():
    with pytest.raises(ValueError, match="section"):
        placements_for_section("underside")


def test_a_head_camera_cannot_take_a_placement_its_section_does_not_offer():
    with pytest.raises(ValueError, match="head"):
        validate_placement("left_front", "head")


# ------------------------------------------------------------- multi-word values


def test_stored_placements_are_single_tokens():
    # The value travels into the session config and on into the LeRobot
    # conversion, so it is stored the way a key is, never with a space.
    for section in PLACEMENTS_BY_SECTION:
        for value in placements_for_section(section):
            assert " " not in value


def test_a_multi_word_placement_is_shown_as_words():
    assert placement_label("left_front") == "left front"
    assert placement_label("head") == "head"
    assert placement_label("") == ""


def test_the_spelling_the_operator_reads_normalises_to_the_stored_token():
    assert normalize_placement("left front") == "left_front"
    assert normalize_placement("  Left-Front ") == "left_front"
    assert normalize_placement("") == ""


def test_validate_accepts_the_spoken_spelling_and_returns_the_stored_one():
    assert validate_placement(" Neck ", "head") == "neck"


# -------------------------------------------------------------- the picker


def test_placement_options_offer_the_section_vocabulary_in_order():
    options = placement_options("head")
    assert [o.value for o in options] == list(placements_for_section("head"))
    assert [o.label for o in options] == ["", "head", "neck", "static"]


def test_placement_options_mark_the_current_choice_selected():
    options = placement_options("head", "neck")
    selected = [o for o in options if o.selected]
    assert [o.value for o in selected] == ["neck"]
    assert selected[0].label == "neck"


def test_nothing_chosen_selects_the_empty_option():
    options = placement_options("head")
    assert [o.value for o in options if o.selected] == [""]


# --------------------------------------------------------------- migration


def test_a_placement_from_before_the_split_still_validates():
    # Persisted placement aliases remain loadable for existing head-camera configs.
    for value in ("left", "right", "front", "rear", "overhead", "table"):
        assert validate_placement(value, "head") == value


def test_an_older_choice_is_offered_back_rather_than_vanishing():
    options = placement_options("head", "overhead")
    assert [o.value for o in options if o.selected] == ["overhead"]
    # Offered because it is what this camera is set to -- not because the head
    # section offers it.
    assert "overhead" not in placements_for_section("head")


def test_an_older_choice_says_why_it_is_there():
    kept = [o for o in placement_options("head", "table") if o.value == "table"]
    assert len(kept) == 1
    assert "table" in kept[0].label and kept[0].label != "table"


def test_an_older_choice_is_not_offered_twice():
    options = placement_options("head", "overhead")
    assert [o.value for o in options].count("overhead") == 1


def test_a_placement_no_section_ever_offered_is_still_rejected():
    with pytest.raises(ValueError, match="placement"):
        validate_placement("upside_down", "head")


# ------------------------------------------------------------ the rendered row


def _render_row(section, placement):
    """The one optional-camera card, rendered the way the config page renders it."""
    jinja2 = pytest.importorskip("jinja2")
    templates = pathlib.Path(__file__).resolve().parents[2] / "station_ui" / "templates"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(templates)))
    row = {
        "label": "Head OAK",
        "section": section,
        "section_id": f"{section}-section",
        "logical_name": "head_oak_camera",
        "device": "oak-W97-51",
        "placement": placement,
        "placement_field": "optional__head_oak__placement",
        "device_field": "optional__head_oak__device",
        "error": "",
        "error_id": "e",
        "confirm": False,
        "confirm_message": "",
        "device_view": {
            "field": "optional__head_oak__device",
            "kind": "oak",
            "label": "Head OAK",
            "value": "oak-W97-51",
            "list_id": "l",
            "options": [],
            "serves": "",
            "error": "",
            "error_id": "e",
            "message_id": "m",
            "message_state": "",
            "message": "",
        },
    }
    return env.get_template("_partials/device_row.html").render(
        row=row, placements=placement_options, can_edit=True
    )


def _selected(html):
    """The value of the one <option> the browser would submit."""
    import re

    return re.findall(r'<option value="([^"]*)"[^>]*\bselected', html)


def test_the_row_offers_only_its_own_section_vocabulary():
    html = _render_row("head", "")
    assert 'value="neck"' in html
    assert 'value="table"' not in html


def test_a_saved_choice_comes_back_as_the_selection():
    assert _selected(_render_row("head", "neck")) == ["neck"]


def test_an_older_choice_is_still_the_selection_in_the_rendered_row():
    assert _selected(_render_row("head", "overhead")) == ["overhead"]


def test_every_descriptor_has_its_own_fixed_name():
    names = [t.fixed_name for t in OPTIONAL_DEVICE_TYPES]
    assert all(names)
    assert len(set(names)) == len(names)


def test_an_optional_camera_is_recognised_by_its_fixed_name_only():
    """How a config written by an earlier session is read back: a source is an
    optional camera exactly when a descriptor owns its name."""
    head_oak = get_optional_type("head_oak")
    assert optional_devices.matches_logical_name(head_oak, "head_oak_camera")
    assert not optional_devices.matches_logical_name(head_oak, "head_iphone_camera")
    assert not optional_devices.matches_logical_name(head_oak, "left_umi_camera")
    assert optional_devices.type_for_logical_name("bottom_oak_camera").id == (
        "bottom_oak"
    )
    assert optional_devices.type_for_logical_name("right_tracking_camera") is None


def test_unknown_type_id_raises():
    with pytest.raises(KeyError):
        get_optional_type("side_lidar")
