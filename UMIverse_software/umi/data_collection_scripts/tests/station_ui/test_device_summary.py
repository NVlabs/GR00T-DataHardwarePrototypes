# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The detected devices, grouped by gripper, at the terminal and on /config.

A gripper's OAK board and iPhone hang off the gripper's own USB hub. Listing
every device on its own -- two look-alike OAK serials, two UDIDs and two
gripper serials -- would leave the operator to find out which camera rides on
which gripper by unplugging cables. The summary puts one gripper per row with
the camera and phone that share its hub.

`BENCH` is a scan taken on a V3 bench whose grippers carried their devices on
a hub the same way: two grippers on ports 3 and 4 of one external hub, and a
standalone head OAK on port 1. `V4_STATION` is cabled like a V4 station, whose
trigger cameras plug into the computer and not the gripper.

No test here opens hardware: the scan is either a list of records or a fake
child that prints the scan's wire format.
"""

import io
import re
import shutil
import sys
import time
from html.parser import HTMLParser

import pytest
import test_unavailable_device_gate as gate  # noqa: E402
from fastapi.testclient import TestClient

from station_ui import (  # noqa: E402
    device_summary,
    jobs,  # noqa: E402
    scan_worker,
)
from station_ui import devices as devices_core  # noqa: E402
from station_ui.app import create_app  # noqa: E402
from station_ui.core import device_detector  # noqa: E402
from station_ui.routes import config as config_routes  # noqa: E402
from station_ui.settings import Settings  # noqa: E402
from umi.common.hardware_sdk.device_discovery import ConnectedDevice  # noqa: E402
from umi.common.tasks import task_registry_store  # noqa: E402


def record(kind, device_id, location=None, registry_name=None, state="available"):
    found = gate.found_record(kind, device_id, registry_name, state=state)
    if location:
        found["metadata"] = {"usb_location": location}
    return found


BENCH = [
    record("oak", "1944301000000000A9", "0-1.3.3"),
    record("oak", "1944301000000000A8", "0-1.4.3"),
    record("oak", "1944301000000000AE", "0-1.1"),
    record("iphone", "00008130-00000000000000A1", "0-1.3.1"),
    record("iphone", "00008130-00000000000000A2", "0-1.4.1", "iphone15p-28"),
    record("gripper", "534A3035000000B8", "0-1.3.4"),
    record("gripper", "534A3035000000B7", "0-1.4.4"),
]

BENCH_ROWS = [
    {
        "port": "0-1.3",
        "gripper": ["534A3035000000B8"],
        "oak": ["1944301000000000A9"],
        "iphone": ["00008130-00000000000000A1"],
        "uvc": [],
    },
    {
        "port": "0-1.4",
        "gripper": ["534A3035000000B7"],
        "oak": ["1944301000000000A8"],
        "iphone": ["iphone15p-28"],
        "uvc": [],
    },
    {
        "port": "0-1.1",
        "gripper": [],
        "oak": ["1944301000000000AE"],
        "iphone": [],
        "uvc": [],
    },
]

# Each gripper's own hub carries its iPhone (port 1), OAK board (3) and
# controller (4); the trigger cameras, USB cameras under the grippers, take a
# root port of the computer each.
V4_STATION = [
    record("gripper", "534A3035000000B8", "3-1.2.2.4"),
    record("oak", "1944301000000000A9", "3-1.2.2.3"),
    record("iphone", "00008130-00000000000000A1", "3-1.2.2.1"),
    record("gripper", "534A3035000000B7", "3-1.2.3.4"),
    record("oak", "1944301000000000A8", "3-1.2.3.3"),
    record("iphone", "00008130-00000000000000A2", "3-1.2.3.1", "iphone15p-28"),
    record("uvc", "0", "3-7"),
    record("uvc", "2", "3-8"),
]

# A kind the table has no column for. The scan enumerates none today, so it is
# made up.
UNTABLED = record("tracker", "LHR-0B5F2C3A", "3-9")


def texts(found):
    """`device_summary.rows`, each cell as the text the terminal prints for it."""
    return [
        {
            key: [str(cell) for cell in value] if isinstance(value, list) else value
            for key, value in row.items()
        }
        for row in device_summary.rows(found)
    ]


# ------------------------------------------------------------------ grouping


def test_each_gripper_gets_the_camera_and_phone_on_its_own_hub():
    assert texts(BENCH) == BENCH_ROWS


def test_the_order_of_the_scan_does_not_change_the_rows():
    assert texts(list(reversed(BENCH))) == BENCH_ROWS


def test_a_cell_is_what_the_device_box_takes():
    """A registered device by its name, an unregistered one by its raw id."""
    rows = texts(BENCH)

    assert rows[1]["iphone"] == ["iphone15p-28"]
    assert rows[0]["iphone"] == ["00008130-00000000000000A1"]


def test_grippers_wired_straight_into_one_hub_are_not_paired_with_anything():
    """Two grippers on one hub means it is the bench's hub, not a gripper's own.
    Pairing either with the OAK beside them would be a guess."""
    found = [
        record("gripper", "AAAA0001", "1-2.1"),
        record("gripper", "AAAA0002", "1-2.2"),
        record("oak", "1944301000000001", "1-2.3"),
    ]

    rows = device_summary.rows(found)

    assert [row["port"] for row in rows] == ["1-2.1", "1-2.2", "1-2.3"]
    assert all(len(row["gripper"]) + len(row["oak"]) == 1 for row in rows)


def test_a_device_with_no_known_location_gets_a_row_of_its_own_last():
    found = [*BENCH, record("iphone", "00008140-000000000099")]

    rows = texts(found)

    assert rows[-1] == {
        "port": None,
        "gripper": [],
        "oak": [],
        "iphone": ["00008140-000000000099"],
        "uvc": [],
    }


def test_each_usb_camera_gets_a_row_of_its_own():
    """A trigger camera is not on its gripper's hub, so there is nothing to group
    it with: it gets its own row, after the grippers, with its own port."""
    rows = texts(V4_STATION)

    assert [row["gripper"] for row in rows[:2]] == [
        ["534A3035000000B8"],
        ["534A3035000000B7"],
    ]
    assert all(row["uvc"] == [] for row in rows[:2])
    assert rows[2:] == [
        {"port": "3-7", "gripper": [], "oak": [], "iphone": [], "uvc": ["0"]},
        {"port": "3-8", "gripper": [], "oak": [], "iphone": [], "uvc": ["2"]},
    ]


def test_a_usb_camera_on_a_grippers_hub_shares_its_row():
    """The rule is where it is plugged in, not what kind it is."""
    found = [*V4_STATION[:6], record("uvc", "0", "3-1.2.2.2")]

    assert texts(found)[0]["uvc"] == ["0"]


def test_a_device_that_would_not_open_says_so():
    found = [record("oak", "1944301000000001", "0-1.3.3", state="unavailable")]

    cell = device_summary.rows(found)[0]["oak"][0]
    assert str(cell) == "1944301000000001 (unavailable)"
    assert cell.name == "1944301000000001", "the box still takes the bare id"
    assert not cell.available


def test_kinds_without_a_column_are_left_out():
    found = [*BENCH, UNTABLED]

    assert texts(found) == BENCH_ROWS


def test_nothing_found_says_no_devices():
    assert device_summary.table_lines([]) == ["No devices detected."]


def test_only_other_kinds_found_does_not_say_no_devices():
    """The strip beside it counts that device; "no devices" would contradict it."""
    lines = device_summary.table_lines([UNTABLED])

    assert lines == ["No grippers, OAK cameras, iPhones or USB cameras detected."]


def test_a_usb_camera_alone_is_a_table_not_a_no_devices_message():
    assert device_summary.table_lines([record("uvc", "4", "3-7")]) == [
        "USB port | Gripper | OAK camera | iPhone | USB camera",
        "---------+---------+------------+--------+-----------",
        "3-7      | —       | —          | —      | 4",
    ]


def test_the_terminal_table():
    assert device_summary.table_lines(BENCH) == [
        "USB port | Gripper          | OAK camera         | iPhone                    | USB camera",
        "---------+------------------+--------------------+---------------------------+-----------",
        "0-1.3    | 534A3035000000B8 | 1944301000000000A9 | 00008130-00000000000000A1 | —",
        "0-1.4    | 534A3035000000B7 | 1944301000000000A8 | iphone15p-28              | —",
        "0-1.1    | —                | 1944301000000000AE | —                         | —",
    ]


# --------------------------------------------------------- the scan worker


def _device(found):
    return ConnectedDevice(
        kind=found["kind"],
        device_id=found["device_id"],
        registry_name=found["registry_name"],
        state=found["state"],
        metadata=found["metadata"],
    )


@pytest.fixture
def fake_enumerators(monkeypatch):
    """The worker's enumerators, answering with *found* and touching nothing."""

    def _install(found):
        by_kind = {}
        for item in found:
            by_kind.setdefault(item["kind"], []).append(_device(item))
        monkeypatch.setattr(
            scan_worker,
            "_enumerators",
            lambda: tuple(
                (kind, label, lambda kind=kind: by_kind.get(kind, []))
                for kind, label in (
                    ("oak", "OAK cameras"),
                    ("iphone", "iPhones"),
                    ("gripper", "UMI grippers"),
                )
            ),
        )

    return _install


def test_the_summary_comes_after_everything_the_scan_already_printed(
    fake_enumerators,
):
    fake_enumerators(BENCH)
    alone = io.StringIO()
    scan_worker.scan(out=alone)
    out = io.StringIO()

    assert scan_worker.main([], out=out) == 0

    printed = out.getvalue()
    assert printed.startswith(alone.getvalue()), "the existing output changed"
    tail = printed[len(alone.getvalue()) :].splitlines()
    assert tail[2:] == device_summary.table_lines(BENCH)


def test_the_result_line_is_still_found_after_the_summary(fake_enumerators):
    """The console reads the result by its prefix, not by position."""
    fake_enumerators(BENCH)
    out = io.StringIO()

    scan_worker.main([], out=out)

    found = devices_core.parse_result(out.getvalue().splitlines())
    assert [item["device_id"] for item in found] == [
        "1944301000000000A9",
        "1944301000000000A8",
        "1944301000000000AE",
        "00008130-00000000000000A1",
        "00008130-00000000000000A2",
        "534A3035000000B8",
        "534A3035000000B7",
    ]


def test_a_scan_that_finds_nothing_says_so(fake_enumerators):
    fake_enumerators([])
    out = io.StringIO()

    assert scan_worker.main([], out=out) == 0

    assert out.getvalue().rstrip().endswith("No devices detected.")


def test_a_summary_that_cannot_be_built_is_a_message_not_a_failed_scan(
    fake_enumerators, monkeypatch
):
    fake_enumerators(BENCH)

    def _broken(found):
        raise ValueError("unexpected record")

    monkeypatch.setattr(device_summary, "table_lines", _broken)
    out = io.StringIO()

    assert scan_worker.main([], out=out) == 0

    assert "Device information could not be loaded: unexpected record" in out.getvalue()
    assert scan_worker.RESULT_PREFIX in out.getvalue(), "the result was lost"


def test_a_scan_that_breaks_prints_a_message_and_exits_non_zero(monkeypatch):
    def _broken(out=None, probe_devices=False):
        raise RuntimeError("USB subsystem unavailable")

    monkeypatch.setattr(scan_worker, "scan", _broken)
    out = io.StringIO()

    assert scan_worker.main([], out=out) == 1

    assert out.getvalue() == (
        "[scan] Device information could not be loaded: USB subsystem unavailable\n"
    )


# -------------------------------------------------------------- the page


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr(task_registry_store, "get_all_tasks", lambda: dict(gate.TASKS))
    monkeypatch.setattr(
        device_detector, "get_available_oak_cameras", lambda: [gate.OAK_R, gate.OAK_L]
    )
    monkeypatch.setattr(
        device_detector, "get_available_iphones", lambda: [gate.IPHONE_R, gate.IPHONE_L]
    )
    monkeypatch.setattr(
        config_routes,
        "list_umi_gripper_device_names_for_pipeline",
        lambda pipeline: ["UMI-V4-0001", "UMI-V4-0002"],
    )


@pytest.fixture
def client(tmp_path):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    shutil.copytree(
        gate.REPO / "config" / "data_recording",
        root / "config" / "data_recording",
        ignore=shutil.ignore_patterns("generated"),
    )
    shutil.copytree(gate.REPO / "config" / "data_check", root / "config" / "data_check")
    built = TestClient(create_app(Settings(repo_root=root)))
    built.get("/")
    built.post("/setup/preset", data={"preset_id": "bimanual_v4"})
    return built


class _Table(HTMLParser):
    """The summary's rows as lists of cell text, and whether it was out-of-band."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.oob = False
        self._in_summary = 0
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if attr.get("id") == "device-summary":
            self._in_summary = 1
            self.oob = "hx-swap-oob" in attr
            return
        if not self._in_summary:
            return
        if tag == "div":
            self._in_summary += 1
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag):
        if not self._in_summary:
            return
        if tag in ("td", "th") and self._cell is not None:
            self.rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "div":
            self._in_summary -= 1

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def table(html):
    parsed = _Table()
    parsed.feed(html)
    return parsed


def summary_text(html):
    match = re.search(r'<div [^>]*id="device-summary".*?</div>', html, re.S)
    assert match, "the summary is not on the page"
    return " ".join(re.sub(r"<[^>]+>", " ", match.group(0)).split())


def finish(client, monkeypatch, argv):
    monkeypatch.setattr(devices_core, "scan_argv", lambda: argv)
    client.post("/config/devices/scan")
    job = devices_core.current_job(client.app)
    end = time.monotonic() + gate.DEADLINE_S
    while not job.finished and time.monotonic() < end:
        time.sleep(0.02)
    assert job.finished, f"the fake scan never ended: {job.state}"


def test_the_page_groups_each_gripper_with_its_camera_and_phone(client, monkeypatch):
    gate.run_scan(client, monkeypatch, BENCH)

    rows = table(client.get("/config").text).rows

    assert rows == [
        ["USB port", "Gripper", "OAK camera", "iPhone", "USB camera"],
        [
            "0-1.3",
            "534A3035000000B8",
            "1944301000000000A9",
            "00008130-00000000000000A1",
            "—",
        ],
        ["0-1.4", "534A3035000000B7", "1944301000000000A8", "iphone15p-28", "—"],
        ["0-1.1", "—", "1944301000000000AE", "—", "—"],
    ]


def test_the_page_gives_each_usb_camera_a_row_of_its_own(client, monkeypatch):
    gate.run_scan(client, monkeypatch, V4_STATION)

    rows = table(client.get("/config").text).rows

    assert rows[0][-1] == "USB camera"
    assert rows[-2:] == [["3-7", "—", "—", "—", "0"], ["3-8", "—", "—", "—", "2"]]


def test_the_table_sits_in_devices_above_the_boxes(client, monkeypatch):
    gate.run_scan(client, monkeypatch, BENCH)

    page = client.get("/config").text

    controls = page.index("data-identify-open")
    summary = page.index('id="device-summary"')
    boxes = page.index('class="grid arm-grid')
    assert controls < summary < boxes
    assert "devices found" not in page


def test_a_finished_scan_brings_the_table_with_it(client, monkeypatch):
    """The poller swaps the table out-of-band, so it fills in when the scan ends
    without re-rendering a device box."""
    gate.run_scan(client, monkeypatch, BENCH)

    fragment = table(client.get("/config/devices/status").text)

    assert fragment.oob, "the table would land nowhere without hx-swap-oob"
    assert fragment.rows[1][1] == "534A3035000000B8"


def test_before_any_scan_the_page_says_how_to_get_one(client):
    page = client.get("/config").text

    assert "Press Refresh devices to see what is connected." in summary_text(page)


def test_while_scanning_the_page_says_it_is_looking(client, monkeypatch):
    monkeypatch.setattr(
        devices_core,
        "scan_argv",
        lambda: [sys.executable, "-c", "import time\ntime.sleep(60)"],
    )
    client.post("/config/devices/scan")
    job = devices_core.current_job(client.app)
    try:
        page = client.get("/config").text
    finally:
        jobs.cancel(devices_core.registry_for(client.app), job.id)
        end = time.monotonic() + gate.DEADLINE_S
        while not job.finished and time.monotonic() < end:
            time.sleep(0.02)

    assert "Looking for connected devices" in summary_text(page)


def test_a_scan_that_found_nothing_says_no_devices_detected(client, monkeypatch):
    gate.run_scan(client, monkeypatch, [])

    assert "No devices detected." in summary_text(client.get("/config").text)


def test_a_failed_scan_says_the_information_could_not_be_loaded(client, monkeypatch):
    # A child that exits cleanly without ever printing the result line.
    finish(client, monkeypatch, [sys.executable, "-c", "print('[scan] starting')"])

    text = summary_text(client.get("/config").text)

    assert "Device information could not be loaded" in text


def test_a_summary_that_cannot_be_built_leaves_the_page_usable(client, monkeypatch):
    gate.run_scan(client, monkeypatch, BENCH)

    def _broken(found):
        raise ValueError("unexpected record")

    monkeypatch.setattr(device_summary, "rows", _broken)

    response = client.get("/config")

    assert response.status_code == 200
    assert "Device information could not be loaded" in summary_text(response.text)
    assert 'class="grid arm-grid' in response.text, "the device boxes are gone"
