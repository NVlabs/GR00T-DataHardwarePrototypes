#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export the software-only copy of this repository published on GitHub.

GitLab holds everything, including the hardware designs and the documentation
website, which are published on research.nvidia.com. The GitHub repository
(NVlabs/GR00T-DataHardwarePrototypes) carries only the software, its license
records and the README:

    python3 tools/github_export.py --list             # print the files it exports
    python3 tools/github_export.py <github checkout>  # make the checkout match

With a checkout, files are copied in when they differ and tracked files that
are no longer exported are deleted, except the GitHub copy's own
.gitattributes. Review and commit the result there. Only git-tracked files are
exported, so build output and local files never leave this repository.
"""

import filecmp
import fnmatch
import os
import shutil
import subprocess
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Root files and whole folders that are exported.
ROOT_FILES = {
    ".gitignore",
    ".python-version",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "CONTRIBUTORS.md",
    "LICENSE",
    "NOTICE",
    "README.md",
    "pyproject.toml",
    "uv.lock",
}
FOLDERS = (
    ".github/",
    "LICENSES/",
    "UMIverse_software/",
    "tools/data_collection/",
    # Software that lives beside the hardware it runs on.
    "GEAR_UMI/files/firmware/",
    "GEAR_Neck/files/software/neck_control/",
)
FILES = {
    "tools/licenses.py",
    "tools/github_export.py",
    "tools/build_umi_electronics_zips.py",
    "GEAR_DexUMI/files/software/dexumi-hand-monitor-v2.0.zip",
}
# README GIFs are kept; other video files stay on research.nvidia.com.
KEEP_MEDIA = ("media/*.gif",)
VIDEO = (".mp4", ".gif", ".mov", ".webm", ".m4v")
# Files in the GitHub checkout that are maintained there, not exported.
GITHUB_ONLY = {".gitattributes"}


def exported(path):
    if any(fnmatch.fnmatch(path, p) for p in KEEP_MEDIA):
        return True
    if path.lower().endswith(VIDEO):
        return False
    return path in ROOT_FILES or path in FILES or path.startswith(FOLDERS)


def tracked(root):
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout.decode("utf8")
    return sorted(p for p in out.split("\0") if p)


def main(argv):
    files = [p for p in tracked(REPO) if exported(p)]
    pointers = [
        p for p in files
        if open(os.path.join(REPO, p), "rb").read(40).startswith(b"version https://git-lfs")
    ]
    if pointers:
        raise SystemExit(f"Git LFS pointers, not files: {pointers[:3]}. Run `git lfs pull` first.")
    if argv[1:] == ["--list"]:
        print("\n".join(files))
        return
    if len(argv) != 2:
        raise SystemExit(__doc__)
    dest = os.path.abspath(argv[1])
    if not os.path.isdir(os.path.join(dest, ".git")) or os.path.samefile(dest, REPO):
        raise SystemExit(f"{dest} is not a separate git checkout")
    copied = []
    for p in files:
        src, dst = os.path.join(REPO, p), os.path.join(dest, p)
        if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(p)
    keep = set(files) | GITHUB_ONLY
    removed = [p for p in tracked(dest) if p not in keep]
    for p in removed:
        os.remove(os.path.join(dest, p))
    print(f"{len(files)} files exported: {len(copied)} copied, {len(removed)} removed")


if __name__ == "__main__":
    main(sys.argv)
