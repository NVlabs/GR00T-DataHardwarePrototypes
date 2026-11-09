# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections import deque
from threading import Condition, Lock
from typing import Any, List, Optional


class FrameBuffer:
    """Thread-safe frame buffer with blocking support."""

    def __init__(self, maxlen: int):
        self._buffer = deque(maxlen=maxlen)
        self._lock = Lock()
        self._new_frame = Condition(self._lock)
        self._last_frame: Optional[Any] = None  # Always track latest for quick peek

    @property
    def maxlen(self) -> int:
        """Retention capacity (frames kept before the oldest is dropped)."""
        return self._buffer.maxlen

    def put(self, item: Any) -> None:
        """Add item, automatically dropping oldest if full."""
        with self._lock:
            self._buffer.append(item)
            self._last_frame = item
            self._new_frame.notify_all()  # Wake up waiting readers

    def full(self) -> bool:
        with self._lock:
            return len(self._buffer) == self._buffer.maxlen

    def empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0

    def peek_latest(self) -> Optional[Any]:
        """Get newest frame without removing. Never blocks."""
        with self._lock:
            return self._last_frame

    def wait_for_new_frame(
        self,
        last_timestamp: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Any]:
        """
        Wait efficiently for a frame newer than last_timestamp.

        Args:
            last_timestamp: Only return frames with receive_time > this
            timeout: Max seconds to wait (None = forever)

        Returns:
            New frame or None if timeout
        """
        with self._lock:
            deadline = None if timeout is None else time.monotonic() + timeout

            while True:
                if self._last_frame is not None:
                    if last_timestamp is None:
                        return self._last_frame
                    if (
                        hasattr(self._last_frame, "receive_time")
                        and self._last_frame.receive_time is not None
                        and self._last_frame.receive_time > last_timestamp
                    ):
                        return self._last_frame

                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None

                self._new_frame.wait(timeout=remaining)

    def get_last_k(self, k: int, remove: bool = False) -> Optional[List[Any]]:
        """
        Get last k items.

        Args:
            k: Number of items to get
            remove: If True, remove oldest k items (FIFO order)

        Returns:
            List of items or None if empty
        """
        with self._lock:
            if not self._buffer:
                return None

            items = list(self._buffer)[-k:]

            if remove:
                for _ in range(min(k, len(self._buffer))):
                    self._buffer.popleft()

            return items

    def clear(self) -> None:
        """Clear all items from the buffer."""
        with self._lock:
            self._buffer.clear()
            self._last_frame = None
            self._new_frame.notify_all()  # Wake up any waiting threads

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
