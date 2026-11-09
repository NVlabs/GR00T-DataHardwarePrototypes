# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config assembly, device enumeration and session identity for the station console.

UI-free by construction: nothing here imports a web framework or a widget
toolkit. `real_scripts/` and the recording process must not import it.

This `__init__.py` is load-bearing: it is what makes `station_ui.core` a
regular package rather than a namespace one, so the tree can only ever resolve
through a single path entry. Without it, one file could be imported as two
distinct module objects.
"""
