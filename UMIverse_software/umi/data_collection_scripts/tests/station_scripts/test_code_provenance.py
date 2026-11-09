# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`code_provenance` must never be the reason a recording fails.

It shells out to git, so it has several ways to go wrong: the root can be no
checkout at all, git can be missing, the subprocess can hang. Every one of
those has to degrade to "no provenance recorded" rather than propagating into
`_save_meta_data` and taking the rest of the session metadata down with it.
"""

from __future__ import annotations

import subprocess

import pytest

from umi.common.utility.code_provenance import (
    code_provenance,
    head_commit,
    repo_root,
)


def _git(repo, *args):
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_repo_root_is_the_checkout_not_the_cwd(tmp_path, monkeypatch):
    """Derived from __file__, so a launcher that does not cd is still correct."""
    monkeypatch.chdir(tmp_path)
    root = repo_root()
    assert (pytest.importorskip("pathlib").Path(root) / "scripts").is_dir(), root


def test_returns_the_expected_shape():
    info = code_provenance()
    assert info is not None, "provenance unavailable in a full checkout"
    assert set(info) == {"version", "commit", "branch", "upstream", "dirty"}
    assert info["commit"] == head_commit()


def test_clean_checkout(checkout):
    info = code_provenance(str(checkout))
    assert info["branch"] == "main"
    assert info["upstream"] == "(none)"
    assert info["dirty"] is False
    assert info["commit"].startswith(info["version"])


def test_modified_tracked_file_is_dirty(checkout):
    (checkout / "a.txt").write_text("changed\n")
    info = code_provenance(str(checkout))
    assert info["dirty"] is True
    assert info["version"].endswith("-dirty"), info["version"]


def test_untracked_file_is_not_dirty(checkout):
    (checkout / "new.txt").write_text("x\n")
    assert code_provenance(str(checkout))["dirty"] is False


def test_tag_and_detached_head(checkout):
    _git(checkout, "tag", "v1.0")
    _git(checkout, "checkout", "-q", "--detach")
    info = code_provenance(str(checkout))
    assert info["version"] == "v1.0"
    assert info["branch"] == "detached HEAD"


def test_not_a_checkout_returns_none(tmp_path):
    """A directory that is not a git checkout records nothing rather than raising."""
    assert code_provenance(str(tmp_path)) is None


def test_subprocess_failure_returns_none(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert code_provenance(str(tmp_path)) is None


def test_timeout_returns_none(tmp_path, monkeypatch):
    def hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(subprocess, "run", hang)
    assert code_provenance(str(tmp_path)) is None


def test_head_commit_is_a_sha():
    sha = head_commit()
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), sha


def test_head_commit_empty_when_git_unavailable(tmp_path, monkeypatch):
    """Empty, not an exception: `_save_meta_data` still writes operator_name,
    task_name, HW_version and session_uid after this returns."""

    def boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert head_commit(str(tmp_path)) == ""
