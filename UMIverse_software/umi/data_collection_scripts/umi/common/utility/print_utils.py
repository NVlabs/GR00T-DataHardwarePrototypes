# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Colored print utilities for terminal output.
"""

import os
import re


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def get_rank_prefix() -> str:
    """Return a '[rank N]' prefix based on the RANK env var, or '' if not distributed."""
    rank = os.environ.get("RANK")
    if rank is not None:
        return f"[rank {rank}] "
    return ""


def _format_message(message: str, color: str, bold: bool = True) -> str:
    """Format message with color and rank prefix, optionally making leading [brackets] bold."""
    prefix = get_rank_prefix()
    match = re.match(r"^(\[[^\]]*\])\s*(.*)", message)
    if match and bold:
        bracket_part, rest = match.groups()
        return f"{color}{prefix}{Colors.BOLD}{bracket_part}{Colors.RESET}{color} {rest}{Colors.RESET}"
    return f"{color}{prefix}{message}{Colors.RESET}"


def print_red(*args, bold: bool = True, **kwargs):
    """Print text in red color."""
    message = " ".join(str(arg) for arg in args)
    print(_format_message(message, Colors.RED, bold), **kwargs)


def print_green(*args, bold: bool = True, **kwargs):
    """Print text in green color."""
    message = " ".join(str(arg) for arg in args)
    print(_format_message(message, Colors.GREEN, bold), **kwargs)


def print_yellow(*args, bold: bool = True, **kwargs):
    """Print text in yellow color."""
    message = " ".join(str(arg) for arg in args)
    print(_format_message(message, Colors.YELLOW, bold), **kwargs)


def print_blue(*args, bold: bool = True, **kwargs):
    """Print text in blue color."""
    message = " ".join(str(arg) for arg in args)
    print(_format_message(message, Colors.BLUE, bold), **kwargs)
