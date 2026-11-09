# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DexUMI data-collection GUI: generate the bimanual recording yaml and launch record_data.py.

Everything for the feature lives in this folder: ``core/`` (pure python, no
FastAPI, unit-testable without hardware), ``server.py`` (thin FastAPI routes),
``static/`` (vanilla JS page), ``tests/`` and ``README.md``. Run as modules from
the repo root (``python -m gui.dc_ui.server``); nothing here is installed as a package.
"""
