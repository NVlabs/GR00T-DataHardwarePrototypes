# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Station web UI (the Station Console): a local FastAPI + HTMX app.

Served from the station itself (`python -m station_ui`), because the browser
cannot enumerate USB, open DepthAI, or read `/dev`.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
