# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Release checks must catch changed adapter terms and bundled runtime payloads."""

import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "software_licenses", REPO / "tools/licenses.py"
)
licenses = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(licenses)


class Record3DNoticeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for path in (*licenses.RECORD3D_ADAPTERS, licenses.RECORD3D_LICENSE, "NOTICE"):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPO / path).read_bytes())
        patch = mock.patch.object(licenses, "REPO", str(self.root))
        patch.start()
        self.addCleanup(patch.stop)

    def test_current_notices_and_original_lgpl_text_pass(self):
        self.assertEqual(licenses.check_record3d_adapters(), [])

    def test_apache_header_cannot_replace_or_supplement_lgpl(self):
        for replacement in (
            "Apache-2.0",
            "LGPL-2.1-only OR Apache-2.0",
            "LGPL-2.1-only\n# SPDX-License-Identifier: Apache-2.0",
        ):
            with self.subTest(replacement=replacement):
                path = licenses.RECORD3D_ADAPTERS[0]
                target = self.root / path
                target.write_text(
                    (REPO / path)
                    .read_text()
                    .replace(
                        "# SPDX-License-Identifier: LGPL-2.1-only",
                        f"# SPDX-License-Identifier: {replacement}",
                    )
                )
                self.assertTrue(
                    any("must declare" in p for p in licenses.check_record3d_adapters())
                )

    def test_upstream_attribution_and_dated_notice_are_required(self):
        path = licenses.RECORD3D_ADAPTERS[1]
        target = self.root / path
        target.write_text(
            (REPO / path)
            .read_text()
            .replace(licenses.RECORD3D_REFERENCE, "")
            .replace(
                "# 2026-11-05:",
                "# Notice:",
            )
        )
        problems = licenses.check_record3d_adapters()
        self.assertTrue(any("upstream attribution" in p for p in problems))
        self.assertTrue(any("dated modification" in p for p in problems))

    def test_missing_adapter_and_license_are_reported(self):
        (self.root / licenses.RECORD3D_ADAPTERS[0]).unlink()
        (self.root / licenses.RECORD3D_LICENSE).unlink()
        problems = licenses.check_record3d_adapters()
        self.assertTrue(any("missing Record3D adapter" in p for p in problems))
        self.assertTrue(any("missing Record3D LGPL" in p for p in problems))

    def test_truncated_license_is_rejected(self):
        (self.root / licenses.RECORD3D_LICENSE).write_text("LGPL 2.1\n")
        self.assertTrue(
            any(
                "differs from upstream" in p for p in licenses.check_record3d_adapters()
            )
        )


class SourceDistributionTests(unittest.TestCase):
    def test_bundled_runtime_names_are_rejected(self):
        for path in (
            "vendor/record3d.whl",
            "release/.venv/bin/python",
            "env/site-packages/record3d.py",
            "wheelhouse/index.html",
            ".cache/uv/archive-v0/record3d/__init__.py",
            "camera/__pycache__/iphone_camera.cpython-312.pyc",
            "vendor/record3d.cpython-312-x86_64-linux-gnu.so",
            "vendor/libusb.so.1",
            "release/recorder.exe",
            "release/recorder.AppImage",
            "release/station.sif",
            "release\\venv\\Scripts\\python.exe",
        ):
            with self.subTest(path=path):
                self.assertTrue(licenses.check_source_distribution([path]))

    def test_source_hardware_media_and_lock_metadata_are_allowed(self):
        self.assertEqual(
            licenses.check_source_distribution(
                [
                    "uv.lock",
                    "pyproject.toml",
                    "LICENSES/DexUMI/record3d-79e6f446/LICENSE",
                    "umi/iphone_camera.py",
                    "assets/iphone.glb",
                    "parts/iphone.stl",
                    "assets/demo.mp4",
                ]
            ),
            [],
        )

    def test_zip_and_tar_payloads_are_checked_without_extracting(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(licenses, "REPO", temporary),
        ):
            root = Path(temporary)
            with zipfile.ZipFile(root / "source.zip", "w") as archive:
                archive.writestr("source/vendor/record3d.whl", b"wheel")
            with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
                member = tarfile.TarInfo("source/.venv/lib/record3d.so")
                member.size = 6
                archive.addfile(member, io.BytesIO(b"binary"))
            problems = licenses.check_source_distribution(
                ["source.zip", "source.tar.gz"]
            )
            self.assertEqual(len(problems), 2)
            self.assertTrue(
                any("source.zip!source/vendor/record3d.whl" in p for p in problems)
            )
            self.assertTrue(
                any("source.tar.gz!source/.venv/lib/record3d.so" in p for p in problems)
            )
            self.assertFalse((root / "source").exists())

    def test_container_image_archives_are_rejected(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(licenses, "REPO", temporary),
        ):
            for name, members in (
                ("docker.zip", ("manifest.json", "repositories")),
                ("oci.zip", ("oci-layout", "index.json")),
            ):
                with zipfile.ZipFile(Path(temporary) / name, "w") as archive:
                    for member in members:
                        archive.writestr(member, "{}")
                with self.subTest(name=name):
                    self.assertTrue(
                        any(
                            "container image" in p
                            for p in licenses.check_source_distribution([name])
                        )
                    )

    def test_source_zip_is_allowed_and_unreadable_archives_fail(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(licenses, "REPO", temporary),
        ):
            with zipfile.ZipFile(Path(temporary) / "source.zip", "w") as archive:
                archive.writestr(
                    "source/iphone_camera.py", "from record3d import Record3DStream\n"
                )
            self.assertEqual(licenses.check_source_distribution(["source.zip"]), [])
            (Path(temporary) / "broken.tar.gz").write_text("not an archive")
            self.assertTrue(licenses.check_source_distribution(["broken.tar.gz"]))


if __name__ == "__main__":
    unittest.main()
