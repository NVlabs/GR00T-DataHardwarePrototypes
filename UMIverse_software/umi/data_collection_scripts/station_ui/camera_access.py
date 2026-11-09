# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialize camera starts across awaits, without blocking status or cancellation."""

import asyncio
import functools
import inspect
import threading

_ATTACH = threading.Lock()


def serialize_start(function):
    signature = inspect.signature(function, eval_str=True)

    @functools.wraps(function)
    async def serialized(*args, **kwargs):
        request = signature.bind(*args, **kwargs).arguments["request"]
        with _ATTACH:
            lock = getattr(request.app.state, "camera_start_lock", None)
            if lock is None:
                lock = request.app.state.camera_start_lock = threading.Lock()
        # Nonblocking acquisition also works for callers on separate event loops.
        while not lock.acquire(blocking=False):
            await asyncio.sleep(0.01)
        try:
            return await function(*args, **kwargs)
        finally:
            lock.release()

    serialized.__signature__ = signature
    return serialized
