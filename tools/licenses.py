#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate and check the repository's software license records.

  python3 tools/licenses.py deps    # rewrite LICENSES/python-dependencies.md from uv.lock (reads PyPI)
  python3 tools/licenses.py check   # offline checks; exits non-zero if anything is missing

The software listed in NOTICE (SOFTWARE below) is Apache-2.0 by default; hardware designs and
documentation are CC BY 4.0 and need no headers. `check` verifies that:
  1. program files in SOFTWARE (.py .js .mjs .css .sh .c .h .pio and standalone .html pages)
     carry an SPDX-License-Identifier header. Vendored code (vendor/ folders, htmx.min.js),
     Jinja templates (templates/ folders) and empty files are exempt;
  2. no file outside SOFTWARE carries the NVIDIA Apache-2.0 header;
  3. every repository path named in NOTICE, LICENSE and the LICENSES/ READMEs exists, and
     every relative Markdown link in those files, README.md and CONTRIBUTING.md resolves;
  4. LICENSES/python-dependencies.md lists exactly the third-party packages pinned in uv.lock;
  5. every LICENSE/COPYING/NOTICE/AUTHORS file in SOFTWARE is named in NOTICE;
  6. every downloadable archive that contains source code also contains LICENSE and NOTICE;
  7. both Record3D adapters retain LGPL-2.1-only attribution and modification notices;
  8. tracked paths and ZIP/TAR member names contain no common bundled runtime artifacts.

Run `deps` after changing uv.lock, then `check`.
"""

import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import tomllib
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import github_export  # noqa: E402  (same folder)

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LOCK = os.path.join(REPO, "uv.lock")
DEPS_MD = os.path.join(REPO, "LICENSES", "python-dependencies.md")

# Workspace members in uv.lock, and the label used for the packages each one pulls in.
MEMBERS = {
    "umi": "UMI",
    "dexumi-data-collection": "DexUMI",
    "gear-neck": "Neck",
    "gear-data-collection-commands": "tools",
}
DEV_GROUPS = {"dev"}

# Retained license texts in LICENSES/ whose folder name is not <name>-<version>.
RETAINED = {("record3d", "1.4.1"): "DexUMI/record3d-79e6f446/"}

# Packages whose PyPI release declares no license, or only a copyright line. The license
# named here is the one in the LICENSE file that the package itself ships.
SHIPPED_LICENSE = {
    "cmeel-octomap": "BSD-3-Clause",
    "cmeel-qhull": "Qhull",
    "embreex": "BSD-2-Clause",
    "google-crc32c": "Apache-2.0",
    "matplotlib-inline": "BSD-3-Clause",
    "torchcodec": "BSD-3-Clause",
    "vhacdx": "MIT",
}

# Facts that the publisher's license declaration alone does not show. Keep these short;
# NOTICE has the full explanation.
NOTES = {
    "av": "Linux wheels bundle FFmpeg built with GPL components (x264, x265); see NOTICE.",
    "cmeel-console-bridge": "Metadata says Zlib; the source carries BSD-3-Clause.",
    "glfw": "Wheel bundles the GLFW native library (Zlib).",
    "mujoco": "Wheel bundles third-party native components; see the retained LICENSES_THIRD_PARTY.md.",
    "opencv-python": "Wheel bundles native libraries (FFmpeg, Qt and others) with their own terms; see its LICENSE-3RD-PARTY.txt.",
    "pin": "Metadata says BSD-3-Clause; the shipped LICENSE (Pinocchio) is BSD-2-Clause.",
    "pyopengl": "Complete license includes additional component terms.",
    "python-can": "LGPL-3.0-only; the GEAR Neck software imports it unmodified; see NOTICE.",
    "python-xlib": "LGPL-2.1-or-later.",
    "record3d": "LGPL-2.1; bundled native components keep their own notices.",
    "torch": "Wheel bundles third-party components; release metadata lists Apache-2.0, BSD-2-Clause, BSD-3-Clause, BSL-1.0 and MIT parts.",
    "torchcodec": "Wheel bundles license texts for dav1d, libavif, libjpeg-turbo, libpng, libwebp, libyuv and zlib.",
}

# Licensed software, including third-party exceptions identified in NOTICE.
SOFTWARE = (
    "UMIverse_software/umi/data_collection_scripts/",
    "UMIverse_software/dexumi/data_collection_scripts/",
    "GEAR_Neck/files/software/neck_control/",
    "GEAR_UMI/files/firmware/",
    "GEAR_DexUMI/files/software/dexumi-hand-monitor-v2.0.zip",
    "tools/data_collection/",
    "tools/licenses.py",
    "tools/build_umi_electronics_zips.py",
    "tools/github_export.py",
)
NVIDIA_HEADER = "NVIDIA CORPORATION & AFFILIATES"
HEADER_EXTENSIONS = (".py", ".js", ".mjs", ".css", ".sh", ".c", ".h", ".pio", ".html")
HEADER_EXEMPT_NAMES = {"htmx.min.js"}
LICENSE_FILE = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|AUTHORS)([.-].*)?$", re.I)
CODE_MEMBER = re.compile(r"\.(py|js|mjs|c|h|cpp|hpp|pio|sh)$")
LEGAL_DOCS = ["NOTICE", "LICENSE", "LICENSES/README.md"]
PATH_TOKEN = re.compile(
    r"(?<![\w/.])((?:GEAR_[A-Za-z]+|UMIverse_software|LICENSES|tools)/[^\s,;:()'\"\]`<>]+)"
)
RECORD3D_ADAPTERS = tuple(
    f"UMIverse_software/{pipeline}/data_collection_scripts/"
    "umi/common/hardware_sdk/camera/iphone_camera.py"
    for pipeline in ("umi", "dexumi")
)
RECORD3D_REFERENCE = (
    "https://github.com/marek-simonik/record3d/blob/"
    "79e6f44697150a4f036034af8fec903d11ac6f00/demo-main.py"
)
RECORD3D_LICENSE = "LICENSES/DexUMI/record3d-79e6f446/LICENSE"
RECORD3D_LICENSE_SHA256 = (
    "e237fa56668030e928551ddd60f05df5fe957f75eab874bbd017e085ed722e7c"
)
RUNTIME_DIRECTORIES = {
    ".venv",
    "venv",
    "site-packages",
    "dist-packages",
    "__pycache__",
    "wheelhouse",
    ".uv-cache",
    ".cache",
}
RUNTIME_SUFFIXES = (
    ".whl",
    ".egg",
    ".pyc",
    ".pyo",
    ".so",
    ".pyd",
    ".dll",
    ".dylib",
    ".exe",
    ".appimage",
    ".sif",
)
TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2")


def tracked():
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True
    ).stdout
    return [p for p in out.decode("utf8").split("\0") if p]


def read_text(path, limit=None):
    with open(os.path.join(REPO, path), "rb") as f:
        data = f.read(limit) if limit else f.read()
    return data.decode("utf8", "replace")


# ----------------------------------------------------------------------------- uv.lock


def lock_packages():
    with open(LOCK, "rb") as f:
        lock = tomllib.load(f)
    packages = lock["package"]
    third_party = [
        p
        for p in packages
        if "registry" in p["source"] or "git" in p["source"] or "url" in p["source"]
    ]

    by_name = {}
    for p in packages:
        by_name.setdefault(p["name"], []).append(p)

    def reach(start_names):
        seen, todo = set(), list(start_names)
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            for p in by_name.get(name, []):
                todo += [d["name"] for d in p.get("dependencies", [])]
                for extra in p.get("optional-dependencies", {}).values():
                    todo += [d["name"] for d in extra]
        return seen

    used_by = {}
    for member, label in MEMBERS.items():
        for p in by_name.get(member, []):
            runtime = [d["name"] for d in p.get("dependencies", [])]
            dev = []
            for group, deps in p.get("dev-dependencies", {}).items():
                (dev if group in DEV_GROUPS else runtime).extend(
                    d["name"] for d in deps
                )
            run_set = reach(runtime)
            for name in run_set:
                used_by.setdefault(name, set()).add(label)
            for name in reach(dev) - run_set:
                used_by.setdefault(name, set()).add(label + " dev")
    return third_party, used_by


def lock_sha256():
    with open(LOCK, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ----------------------------------------------------------------------------- deps


def pypi_license(name, version):
    base = version.split("+")[0]
    url = f"https://pypi.org/pypi/{name}/{base}/json"
    with urllib.request.urlopen(url, timeout=60) as r:
        info = json.load(r)["info"]
    expression = (info.get("license_expression") or "").strip()
    if expression:
        return expression
    legacy = (info.get("license") or "").strip()
    classifiers = [
        c.split(" :: ")[-1]
        for c in info.get("classifiers") or []
        if c.startswith("License ::")
    ]
    short = legacy and "\n" not in legacy and len(legacy) <= 60
    if (
        short
        and legacy.upper() not in ("UNKNOWN", "OTHER")
        and not legacy.lower().startswith("copyright")
    ):
        return legacy
    if classifiers:
        return "; ".join(classifiers)
    if name in SHIPPED_LICENSE:
        return f"{SHIPPED_LICENSE[name]} (from its LICENSE file)"
    return "not declared"


def cell(text):
    return text.replace("|", "\\|")


def deps():
    third_party, used_by = lock_packages()
    rows = sorted(third_party, key=lambda p: (p["name"], p["version"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        licenses = list(pool.map(lambda p: pypi_license(p["name"], p["version"]), rows))

    lines = [
        "# Python dependency licenses",
        "",
        "<!-- Generated by `python3 tools/licenses.py deps`. Do not edit by hand. -->",
        "",
        "The third-party Python packages pinned by the repository-root `uv.lock`, which installs the",
        "GEAR UMI and GEAR DexUMI data-collection software and the GEAR Neck control software into",
        "one environment with `uv sync`.",
        "These packages are downloaded at install time; none of them is distributed in this",
        "repository. Each keeps its own license.",
        "",
        "*Declared license* is what the publisher declares on PyPI for that release (license",
        "expression, license field or classifiers). Wheels can bundle native libraries under other",
        "terms; *Notes* points out the ones that matter, and [NOTICE](../NOTICE) explains them.",
        "*Used by* names the project whose dependencies pull the package in (`dev`: only through",
        "that project's lint and test tools). Platform-specific alternatives are included.",
        "",
        f"`uv.lock` SHA-256: `{lock_sha256()}`",
        f"{len(rows)} packages ({len({p['name'] for p in rows})} distribution names).",
        "",
        "| Package | Version | Declared license | Used by | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for p, lic in zip(rows, licenses):
        name, version = p["name"], p["version"]
        note = NOTES.get(name, "")
        retained = RETAINED.get((name, version))
        if retained is None and os.path.isdir(
            os.path.join(REPO, "LICENSES", "DexUMI", f"{name}-{version}")
        ):
            retained = f"DexUMI/{name}-{version}/"
        if retained:
            note = (note + " " if note else "") + f"[Retained texts]({retained})"
        source = p["source"].get("registry", "")
        if source and "pypi.org" not in source:
            note = (note + " " if note else "") + f"From {source}."
        labels = sorted(used_by.get(name, set()), key=lambda s: (s.endswith(" dev"), s))
        lines.append(
            f"| {cell(name)} | {cell(version)} | {cell(lic)} | {', '.join(labels) or '-'} | {cell(note)} |"
        )
    with open(DEPS_MD, "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(DEPS_MD, REPO)}: {len(rows)} packages")


# ----------------------------------------------------------------------------- check


def in_software(path):
    return path.startswith(SOFTWARE)


def check_headers(files):
    missing = []
    for path in files:
        if not path.endswith(HEADER_EXTENSIONS) or not in_software(path):
            continue
        parts = path.split("/")
        if "vendor" in parts or os.path.basename(path) in HEADER_EXEMPT_NAMES:
            continue
        if path.endswith(".html") and "templates" in parts:
            continue
        full = os.path.join(REPO, path)
        if not os.path.isfile(full) or os.path.getsize(full) == 0:
            continue
        head = read_text(path, 6000)
        if head.startswith("version https://git-lfs"):
            continue
        if "SPDX-License-Identifier:" not in head:
            missing.append(path)
    return [f"no SPDX header: {p}" for p in missing]


def check_no_header_outside(files):
    """Only the software is licensed: flag NVIDIA Apache-2.0 headers on anything else."""
    problems = []
    for path in files:
        if (
            in_software(path)
            or path.startswith("LICENSES/")
            or path in ("LICENSE", "NOTICE")
        ):
            continue
        full = os.path.join(REPO, path)
        if not os.path.isfile(full):
            continue
        head = read_text(path, 6000)
        if NVIDIA_HEADER in head and "SPDX-License-Identifier: Apache-2.0" in head:
            problems.append(
                f"NVIDIA Apache-2.0 header outside the licensed software: {path}"
            )
    return problems


def github_copy():
    """True in the software-only GitHub copy, which has no hardware folders."""
    return not os.path.exists(os.path.join(REPO, "GEAR_UMI", "index.html"))


def check_paths(files):
    problems = []
    tracked_set = set(files)
    docs = LEGAL_DOCS + [
        p for p in files if p.startswith("LICENSES/") and p.endswith("README.md")
    ]
    for doc in docs:
        text = read_text(doc)
        for token in PATH_TOKEN.findall(text):
            token = token.rstrip(".")
            if os.path.exists(os.path.join(REPO, token)):
                continue
            # Paths with spaces are cut at the space; accept a tracked path that continues the token.
            if any(t.startswith(token) for t in tracked_set):
                continue
            # The software-only GitHub copy leaves out the hardware and website files
            # (they are on research.nvidia.com), so the legal texts may still name them.
            if github_copy() and not github_export.exported(token):
                continue
            problems.append(f"{doc}: names a missing path: {token}")
    for doc in docs + ["README.md", "CONTRIBUTING.md"]:
        if not doc.endswith(".md"):
            continue
        base = os.path.dirname(doc)
        for target in re.findall(r"\]\(([^)\s]+)\)", read_text(doc)):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            path = os.path.normpath(os.path.join(base, target.split("#")[0]))
            if not os.path.exists(os.path.join(REPO, path)):
                problems.append(f"{doc}: broken link: {target}")
    return problems


def check_deps():
    if not os.path.exists(DEPS_MD):
        return [
            f"missing {os.path.relpath(DEPS_MD, REPO)}; run `python3 tools/licenses.py deps`"
        ]
    listed = set()
    for line in read_text(os.path.relpath(DEPS_MD, REPO)).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if (
            line.startswith("| ")
            and len(cells) == 5
            and cells[0] not in ("Package", "---")
        ):
            listed.add((cells[0], cells[1]))
    third_party, _ = lock_packages()
    pinned = {(p["name"], p["version"]) for p in third_party}
    problems = [
        f"uv.lock pins {n} {v}, not listed in python-dependencies.md"
        for n, v in sorted(pinned - listed)
    ]
    problems += [
        f"python-dependencies.md lists {n} {v}, not in uv.lock"
        for n, v in sorted(listed - pinned)
    ]
    if problems:
        problems.append("run `python3 tools/licenses.py deps`")
    return problems


def check_license_files(files):
    notice = read_text("NOTICE")
    problems = []
    for path in files:
        if not in_software(path):
            continue
        if LICENSE_FILE.match(os.path.basename(path)) and path not in notice:
            problems.append(f"license file not named in NOTICE: {path}")
    return problems


def check_archives(files):
    problems = []
    for path in files:
        if not path.endswith(".zip") or path.startswith("LICENSES/"):
            continue
        full = os.path.join(REPO, path)
        try:
            with zipfile.ZipFile(full) as z:
                names = z.namelist()
        except zipfile.BadZipFile:
            problems.append(f"cannot read {path} (Git LFS pointer? run `git lfs pull`)")
            continue
        if not any(CODE_MEMBER.search(n) for n in names):
            continue
        base = {os.path.basename(n.rstrip("/")) for n in names}
        for required in ("LICENSE", "NOTICE"):
            if required not in base:
                problems.append(f"{path}: contains source code but no {required}")
    return problems


def check_record3d_adapters():
    problems = []
    notice = read_text("NOTICE")
    for path in RECORD3D_ADAPTERS:
        try:
            text = read_text(path)
        except FileNotFoundError:
            problems.append(f"missing Record3D adapter: {path}")
            continue
        licenses = re.findall(r"^# SPDX-License-Identifier: (.+)$", text, re.M)
        if licenses != ["LGPL-2.1-only"]:
            problems.append(f"Record3D adapter must declare LGPL-2.1-only: {path}")
        if RECORD3D_REFERENCE not in text or "Marek Simonik" not in text:
            problems.append(f"Record3D adapter is missing upstream attribution: {path}")
        if not re.search(r"^# \d{4}-\d{2}-\d{2}: .+", text, re.M):
            problems.append(
                f"Record3D adapter is missing a dated modification notice: {path}"
            )
        if path not in notice:
            problems.append(f"Record3D adapter is missing from NOTICE: {path}")
    try:
        with open(os.path.join(REPO, RECORD3D_LICENSE), "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != RECORD3D_LICENSE_SHA256:
            problems.append(
                f"retained Record3D LGPL text differs from upstream: {RECORD3D_LICENSE}"
            )
    except FileNotFoundError:
        problems.append(f"missing Record3D LGPL text: {RECORD3D_LICENSE}")
    return problems


def runtime_artifact(path):
    """Inspect names only; local, ignored station environments are not releases."""
    normalized = path.replace("\\", "/").lower()
    parts = normalized.split("/")
    return bool(
        RUNTIME_DIRECTORIES.intersection(parts)
        or normalized.endswith(RUNTIME_SUFFIXES)
        or re.search(r"\.so\.\d", parts[-1])
    )


def check_source_distribution(files):
    """Reject common runtime payloads in tracked files and ZIP/TAR member names."""
    problems = []
    for path in files:
        if runtime_artifact(path):
            problems.append(
                f"runtime artifact excluded from source distribution: {path}"
            )
            continue
        full = os.path.join(REPO, path)
        try:
            if path.lower().endswith(".zip"):
                with zipfile.ZipFile(full) as archive:
                    members = archive.namelist()
            elif path.lower().endswith(TAR_SUFFIXES):
                with tarfile.open(full, "r:*") as archive:
                    members = archive.getnames()
            else:
                continue
        except (OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
            problems.append(f"cannot inspect source-distribution archive {path}: {exc}")
            continue
        for member in members:
            if runtime_artifact(member):
                problems.append(
                    f"runtime artifact excluded from source distribution: {path}!{member}"
                )
        names = {name.replace("\\", "/").removeprefix("./") for name in members}
        if "oci-layout" in names or (
            "manifest.json" in names
            and (
                "repositories" in names or any(n.endswith("/layer.tar") for n in names)
            )
        ):
            problems.append(
                f"container image excluded from source distribution: {path}"
            )
    return problems


def check():
    files = tracked()
    problems = (
        check_headers(files)
        + check_no_header_outside(files)
        + check_paths(files)
        + check_deps()
        + check_license_files(files)
        + check_archives(files)
        + check_record3d_adapters()
        + check_source_distribution(files)
    )
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)" if problems else "license checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["deps"]:
        deps()
    elif sys.argv[1:] == ["check"]:
        sys.exit(check())
    else:
        sys.exit(__doc__)
