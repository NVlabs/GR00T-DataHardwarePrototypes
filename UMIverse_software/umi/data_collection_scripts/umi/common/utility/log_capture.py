# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tee stdout/stderr to a log file while still printing to the terminal.

Used by ``record_data.py`` to keep a persistent session log without refactoring
existing ``print`` / ``click.echo`` calls.

- Append-only: re-running the script on the same ``data_dir`` extends the log.
- Each session is wrapped in a visible divider so gaps between runs are obvious.
- Every line is prefixed with ``[HH:MM:SS]`` (added only once per line, even
  when a caller writes multiple ``write()`` chunks for a single line).
- Also exposes ``worker_redirect_stdio`` for use as a ``ProcessPoolExecutor``
  initializer so subprocess output (e.g. ``check_dataset``) lands in the same
  file.
"""

import atexit
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import IO, Union

_DIVIDER = "=" * 80


class _TimestampedTee:
    """File-like object that writes to every underlying stream, prefixing
    each new line with a wall-clock timestamp.

    The timestamp is inserted lazily: it is only emitted when the *next* chunk
    begins a fresh line. This keeps partial writes (e.g. progress bars that
    use ``\\r``) from getting a timestamp mid-line.
    """

    def __init__(self, *streams: IO, add_timestamp: bool = True):
        self._streams = streams
        self._add_timestamp = add_timestamp
        self._at_line_start = True

    def write(self, data: Union[str, bytes]) -> int:
        if not data:
            return 0
        orig_len = len(data)
        if isinstance(data, bytes):
            enc = getattr(self._streams[0], "encoding", None) or "utf-8"
            data = data.decode(enc, errors="replace")
        if not self._add_timestamp:
            for s in self._streams:
                s.write(data)
            return orig_len

        # Split while keeping the newline characters so each logical line is
        # tagged exactly once.
        out_parts = []
        for line in data.splitlines(keepends=True):
            if self._at_line_start and line.strip():
                out_parts.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
            else:
                out_parts.append(line)
            self._at_line_start = line.endswith(("\n", "\r"))

        merged = "".join(out_parts)
        for s in self._streams:
            s.write(merged)
        return orig_len

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)

    def fileno(self) -> int:
        # Some libraries ask for the underlying fd; hand back the first stream
        # that has one (the real terminal), not the log file.
        for s in self._streams:
            try:
                return s.fileno()
            except Exception:
                continue
        raise OSError("no underlying fileno")


@contextmanager
def tee_stdio_to_file(log_path: Union[str, Path]):
    """Tee ``sys.stdout`` and ``sys.stderr`` into ``log_path`` (append mode).

    Writes visible dividers at session start/end so restarts are easy to spot.
    Restores the original streams on exit, including on exception.
    """
    log_path = Path(log_path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    started_at = datetime.now()
    header = (
        f"\n{_DIVIDER}\n"
        f">>> session start {started_at.isoformat(timespec='seconds')} "
        f"(pid={os.getpid()}, host={socket.gethostname()})\n"
        f"{_DIVIDER}\n"
    )
    log_file.write(header)
    log_file.flush()

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _TimestampedTee(old_out, log_file)
    sys.stderr = _TimestampedTee(old_err, log_file)

    # Expose the log path to subprocess initializers so worker output joins the same
    # session log.
    os.environ["UMI_SESSION_LOG_PATH"] = str(log_path)

    try:
        yield log_path
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        ended_at = datetime.now()
        footer = (
            f"{_DIVIDER}\n"
            f"<<< session end {ended_at.isoformat(timespec='seconds')}\n"
            f"{_DIVIDER}\n"
        )
        try:
            log_file.write(footer)
            log_file.flush()
        finally:
            log_file.close()
        os.environ.pop("UMI_SESSION_LOG_PATH", None)


def worker_redirect_stdio(log_path: Union[str, Path, None] = None) -> None:
    """ProcessPoolExecutor / multiprocessing worker initializer: tee worker
    stdio into the session log.

    If ``log_path`` is ``None``, reads it from the ``UMI_SESSION_LOG_PATH``
    env var (set by ``tee_stdio_to_file``). Each worker appends a short
    marker so its output is attributable.

    Idempotent: if ``sys.stdout`` is already a ``_TimestampedTee`` (e.g. the
    worker was started with the ``fork`` method and inherited the parent's
    tee), this is a no-op. Without this guard, wrapping an already-teed
    stream would cause each line to be timestamped twice and written to the
    log file twice. Calling this explicitly is still valuable under
    ``spawn`` / ``forkserver`` where inheritance does not apply.
    """
    if isinstance(sys.stdout, _TimestampedTee) and isinstance(
        sys.stderr, _TimestampedTee
    ):
        return

    if log_path is None:
        log_path = os.environ.get("UMI_SESSION_LOG_PATH")
    if not log_path:
        return

    log_file = open(log_path, "a", buffering=1, encoding="utf-8")
    log_file.write(
        f"--- worker start pid={os.getpid()} "
        f"at {datetime.now().isoformat(timespec='seconds')} ---\n"
    )
    log_file.flush()

    sys.stdout = _TimestampedTee(sys.stdout, log_file)
    sys.stderr = _TimestampedTee(sys.stderr, log_file)

    # Keep the worker log open until process shutdown, then flush and close it
    # to preserve buffered output.
    def _close_log_file() -> None:
        try:
            log_file.flush()
        except Exception:
            pass
        try:
            log_file.close()
        except Exception:
            pass

    atexit.register(_close_log_file)
