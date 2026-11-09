# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completion markers for derived groups in aligned DexUMI sessions.

Writers mark groups as ``writing`` before deleting existing data and as
``complete`` after writing all chunks. Readers reject incomplete groups because
Zarr fills missing chunks with zeros. Groups without a marker are treated as
complete for compatibility with existing datasets. These markers detect
interrupted writes; they do not provide locking between concurrent writers.
"""

WRITE_STATE = "write_state"
WRITING = "writing"
COMPLETE = "complete"


def complete(node) -> bool:
    """Return whether the node is complete or predates completion markers."""
    return node.attrs.get(WRITE_STATE) != WRITING


def mark_writing(parent, name: str) -> None:
    """Mark an existing node before its data is replaced.

    Unreadable metadata is left for the writer to replace; readers already
    reject it.
    """
    try:
        node = parent.get(name)
    except ValueError:  # JSONDecodeError, zarr's MetadataValidationError
        return
    if node is not None and complete(node):
        node.attrs[WRITE_STATE] = WRITING
