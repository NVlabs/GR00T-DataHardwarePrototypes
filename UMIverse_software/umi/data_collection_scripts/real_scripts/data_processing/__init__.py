# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data Processing Scripts
=======================

The station-side checks and alignment that run on recorded data. Everything
reads and writes the local output folder (``umi.common.utility.output_layout``):
``output/raw``, ``output/quarantine``, ``output/aligned`` and ``output/reports``.

File Structure
--------------

- ``data_check/``             – Validate data quality (timestamps, gripper,
  tracking, …). Run after every take by ``record_data.py``.
- ``align_data.py``           – Interpolate & align multi-source recordings via
  DataBuffer into ``output/aligned/{task}/{session}``.
- ``session_health_check.py`` – Audit a day's sessions: find episodes with no
  valid aligned copy and re-align them; report to ``output/reports``.

Dependency Graph (simplified)
-----------------------------
::

    session_health_check.py ──▸ align_data.py
"""
