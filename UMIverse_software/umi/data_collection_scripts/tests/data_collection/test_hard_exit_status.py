# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The recorder hard-exits, so it has to report its own status.

48 C++ threads from `libdepthaicore` and `libusb1` survive `device.close()` and
never join, so ordinary interpreter shutdown blocks forever: a recorder that has
finished every bit of its work stays on the process list, in its process group,
holding the console's recording lock. Threads this process does not own cannot
be made daemonic, so `main` ends with `os._exit` and the interpreter never
finishes.

That brings two obligations:

* **The status is whatever is passed.** A hardcoded 0 makes a session that died
  mid-recording indistinguishable from a clean one -- and the console picks
  between a green "Stopped" and a red "Failed" by reading exactly this
  (`station_ui/routes/record.status_of`). The traceback goes too, because
  `os._exit` skips the default excepthook.

* **Nothing after it runs.** Inside `_run_session` that would include the stdio
  tee's own teardown, so the session log would never get its
  `<<< session end` footer or a `close()`.

These run the real interpreter in a subprocess rather than mocking `os._exit`:
the whole point is what the process reports, which a mock cannot observe.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Mirrors main()'s shape: tee around a body that may raise, traceback printed
# from the inner `finally` while the tee is still installed, hard exit outside
# it. Kept as source rather than importing `main`, which would need a rig.
HARNESS = """
import os, sys, traceback
from contextlib import contextmanager

@contextmanager
def tee(path):
    f = open(path, "a", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    class T:
        def __init__(self, a, b): self.a, self.b = a, b
        def write(self, s): self.a.write(s); self.b.write(s)
        def flush(self): self.a.flush(); self.b.flush()
    sys.stdout, sys.stderr = T(old_out, f), T(old_err, f)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        f.write("<<< session end\\n")
        f.close()

def body():
    try:
        {BODY}
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Recording stopped")
        exc = sys.exc_info()[1]
        if exc is not None:
            traceback.print_exception(exc)

status = 0
try:
    with tee({LOG!r}):
        body()
except SystemExit as exc:
    status = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
except BaseException:
    status = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
"""


def _run(tmp_path, body):
    log = tmp_path / "record_data.log"
    script = tmp_path / "harness.py"
    script.write_text(
        HARNESS.replace("{BODY}", body).replace("{LOG!r}", repr(str(log)))
    )
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )
    return proc, log.read_text() if log.exists() else ""


def test_a_clean_session_exits_zero(tmp_path):
    proc, _log = _run(tmp_path, "pass")
    assert proc.returncode == 0
    assert "Recording stopped" in proc.stdout


def test_a_crash_exits_non_zero(tmp_path):
    """A failed session must not exit 0: the console maps 0 to a green
    "Stopped", so a failed recording would look like a good one."""
    proc, _log = _run(tmp_path, "raise RuntimeError('camera fell off the bus')")
    assert proc.returncode != 0, (
        "a crashed session reported success -- the console will badge it "
        "'Stopped' and say nothing went wrong"
    )


def test_a_crash_still_prints_its_traceback(tmp_path):
    """`os._exit` skips the default excepthook, so this has to be explicit or
    the operator gets a dead recorder and no reason for it."""
    proc, _log = _run(tmp_path, "raise RuntimeError('camera fell off the bus')")
    assert "camera fell off the bus" in proc.stdout + proc.stderr
    assert "Traceback" in proc.stdout + proc.stderr


def test_the_traceback_reaches_the_session_log(tmp_path):
    """Printed from the inner `finally`, where the tee is still installed. From
    `main`'s handler it would land on the console only, and the log is what
    anyone reads after the fact."""
    _proc, log = _run(tmp_path, "raise RuntimeError('camera fell off the bus')")
    assert "camera fell off the bus" in log


def test_the_shutdown_runs_even_when_the_body_raises(tmp_path):
    proc, _log = _run(tmp_path, "raise RuntimeError('boom')")
    assert "Recording stopped" in proc.stdout


def test_ctrl_c_is_a_clean_stop(tmp_path):
    proc, _log = _run(tmp_path, "raise KeyboardInterrupt")
    assert proc.returncode == 0
    assert "Interrupted by user" in proc.stdout


def test_an_explicit_status_is_honoured(tmp_path):
    """Flattening `SystemExit(3)` to 1 would lose a caller's own answer."""
    proc, _log = _run(tmp_path, "raise SystemExit(3)")
    assert proc.returncode == 3


def test_the_session_log_is_closed_out(tmp_path):
    """The hard exit sits outside the tee. Inside `_run_session` it would skip
    the context manager's teardown and leave every session log unterminated."""
    _proc, log = _run(tmp_path, "pass")
    assert "<<< session end" in log, (
        "the stdio tee never tore down -- the hard exit is inside it"
    )


@pytest.mark.parametrize("body", ["pass", "raise RuntimeError('boom')"])
def test_output_is_never_swallowed_by_the_hard_exit(tmp_path, body):
    """`os._exit` skips buffer flushing, and the recorder's progress lines go
    through `print`, which buffers when stdout is a pipe rather than a tty --
    which is exactly how the console runs it."""
    proc, _log = _run(tmp_path, body)
    assert "Recording stopped" in proc.stdout


# ------------------------------------------------- the real module, not a copy


def test_record_data_hard_exits_outside_the_tee():
    """The harness above only proves the shape is right. This pins that
    `record_data` has that shape: `os._exit` in `main`, after the `with`, and
    not back inside `_run_session` where it skips the tee's teardown."""
    import inspect

    from real_scripts.data_collection import record_data

    # `.callback`: `main` is a click.Command, and `getsource` on the Command
    # object raises rather than returning the function it wraps.
    main_src = inspect.getsource(record_data.main.callback)
    assert "os._exit(status)" in main_src, "the hard exit left main"
    assert "os._exit(0)" not in main_src, (
        "hardcoded success: every crashed session would report clean and the "
        "console would badge it green"
    )
    assert main_src.index("with tee_stdio_to_file") < main_src.index("os._exit(status)")

    # Comments stripped: this file's own explanation of why the hard exit is
    # not here mentions it by name, and matching that would make the assertion
    # permanently true regardless of the code.
    session_src = "\n".join(
        line
        for line in inspect.getsource(record_data._run_session).splitlines()
        if not line.strip().startswith("#")
    )
    assert "os._exit" not in session_src, (
        "a hard exit inside _run_session skips the stdio tee's teardown, so the "
        "session log is never closed out"
    )
    assert "traceback.print_exception" in session_src, (
        "nothing prints the traceback while the tee is still installed"
    )
