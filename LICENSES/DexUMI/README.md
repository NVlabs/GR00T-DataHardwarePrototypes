# DexUMI license and notice texts

The DexUMI station software in
[UMIverse_software/dexumi/data_collection_scripts](../../UMIverse_software/dexumi/data_collection_scripts/DEXUMI_USAGE_GUIDE.md)
and the hand monitor use Apache-2.0 by default (see the root
[LICENSE](../../LICENSE)). The complete Record3D-derived iPhone adapter,
including NVIDIA adaptations, is an LGPL-2.1-only exception; see the
[Record3D attribution and replacement note](../Record3D/README.md).
Original third-party terms remain applicable; see [NOTICE](../../NOTICE).

## Hand monitor components

[current-components.json](current-components.json) records exact versions, paths,
publisher hashes and distribution scope for:

| Component | Distribution / terms | License and notices |
| --- | --- | --- |
| pySerial 3.5 | Bundled with the downloadable hand-monitor source ZIP; BSD-3-Clause | [Exact publisher text](pyserial-3.5/LICENSE.txt); also retained inside the ZIP |
| windows-curses | Unpinned, separately installed by Windows users; publisher metadata identifies PSF2 and its PDCurses basis | [Publisher metadata](https://pypi.org/project/windows-curses/); no wheel/native binary is bundled here |

The [standalone monitor ZIP](../../GEAR_DexUMI/files/software/dexumi-hand-monitor-v2.0.zip)
includes its project LICENSE, component NOTICE and pySerial license.

## X Hand model

The two X Hand URDFs in
[umi/assets/urdf/xhand/](../../UMIverse_software/dexumi/data_collection_scripts/umi/assets/urdf/xhand/)
are licensed by their vendor, Beijing Xingdongjiyuan Co. Ltd., under the
[MIT License](XHAND-URDF-MIT.txt). The same text is kept beside the files as
their `LICENSE`, so it travels with them, including in a built DexUMI wheel.
They are not NVIDIA material and are not covered by the root LICENSE.

## Python dependency notices

The DexUMI software installs together with the UMI software from the
repository-root `uv.lock`. [python-dependencies.md](../python-dependencies.md)
lists every pinned package, its declared license and which project uses it.

This directory retains the license and notice files of 19 dependency
versions that the DexUMI software has used, including retired dependencies.
They were copied from publisher artifacts or versioned upstream sources;
[dependencies.json](dependencies.json) records their source URLs, archive
members and SHA-256 hashes. When the two lock files were merged into the root
workspace, DexUMI moved to the versions UMI already used for av, fsspec and
idna, and gTTS was dropped; the texts for av 12.3.0, fsspec 2026.3.0,
gtts 2.5.4 and idna 3.13 were removed accordingly. Removing the optional Viser
browser hand viewer also removed Viser and its seven exclusive Python
dependencies from the lock; their retained texts and browser notice inventory
were removed accordingly. The native MuJoCo hand simulation has also been
removed, along with absl-py, etils, glfw, mujoco, PyOpenGL and python-xlib from
the workspace lock. Their retained license and source records remain here
for reference; [python-dependencies.md](../python-dependencies.md) lists the
current dependencies. Pinocchio and its eleven exclusive dependencies were
removed with the offline force/torque calculations and admittance relabeling,
and are back in the workspace lock at the same versions for the tactile
robot-action conversion (the DexUMI `ik` group). The camera/tactile replay
preview remains. Package source and license
declarations describe the inspected releases; they do not certify a particular
installed image.

| Dependency | Recorded terms | License and notice files |
| --- | --- | --- |
| absl-py 2.4.0 | Apache-2.0 | [Files](absl-py-2.4.0/) |
| cmeel 0.59.0 | BSD-2-Clause | [Files](cmeel-0.59.0/) |
| cmeel-assimp 5.4.3.1 | BSD-3-Clause; component-specific notices retained | [Files](cmeel-assimp-5.4.3.1/) |
| cmeel-boost 1.87.0.1 | BSL-1.0 (Boost 1.87.0) | [Files](cmeel-boost-1.87.0.1/) |
| cmeel-console-bridge 1.0.2.3 | BSD-3-Clause (upstream source; metadata says Zlib) | [Files](cmeel-console-bridge-1.0.2.3/) |
| cmeel-octomap 1.10.0 | BSD-3-Clause (OctoMap library) | [Files](cmeel-octomap-1.10.0/) |
| cmeel-qhull 8.0.2.1 | Qhull | [Files](cmeel-qhull-8.0.2.1/) |
| cmeel-tinyxml2 10.0.0 | Zlib | [Files](cmeel-tinyxml2-10.0.0/) |
| cmeel-urdfdom 4.0.1 | BSD-3-Clause | [Files](cmeel-urdfdom-4.0.1/) |
| cmeel-zlib 1.3.1 | Zlib | [Files](cmeel-zlib-1.3.1/) |
| coal 3.0.1 | BSD-3-Clause | [Files](coal-3.0.1/) |
| eigenpy 3.10.3 | BSD-2-Clause | [Files](eigenpy-3.10.3/) |
| etils 1.14.0 | Apache-2.0 | [Files](etils-1.14.0/) |
| glfw 2.10.0 | MIT (Python wrapper); Zlib (GLFW 3.4 native library) | [Files](glfw-2.10.0/) |
| mujoco 3.3.5 | Apache-2.0; packaged third-party notices retained | [Files](mujoco-3.3.5/) |
| pin 3.7.0 | BSD-2-Clause (Pinocchio) | [Files](pin-3.7.0/) |
| pyopengl 3.1.0 | BSD-style main terms and additional component licenses; see complete license | [Files](pyopengl-3.1.0/) |
| python-xlib 0.33 | LGPL-2.1-or-later | [Files](python-xlib-0.33/) |
| record3d 1.4.1 | LGPL-2.1; bundled native components and pybind11 retain their own terms | [Files](record3d-79e6f446/) |

## Important distinctions

console_bridge declares Zlib in package metadata but supplies
BSD 3-Clause source notices. Resolve these declaration discrepancies with
the publishers if a definitive single label is needed.

Record3D's texts were taken from its source tree at commit `79e6f446`. The
root lock installs the PyPI 1.4.1 wheel instead; that wheel's LICENSE is
byte-identical to the retained one. Record3D and Python-Xlib retain their
LGPL terms; supplying notices alone does not complete any binary
source/relinking requirements. PyAV's GPL-enabled FFmpeg build is described
in [NOTICE](../../NOTICE).

MuJoCo's packaged `LICENSES_THIRD_PARTY.md`, GLFW's native Zlib license,
the Qhull notice, and the actual native-library source notices are included.
Assimp contributes independently licensed code, so its relevant source notices
are retained without claiming every optional subcomponent was linked. Test
models, other language ports and the unbuilt Octovis GUI are not covered by
this selected-library notice set.

The earlier-source/release [UMI MIT attribution](../GEAR_UMI/UMI-MIT.txt) and
[Three.js 0.164.1 notice](../GEAR_UMI/Three-0.164.1-MIT.txt) are retained for
earlier distributions. The calibration viewer has been removed from the
current software. See [NOTICE](../../NOTICE) for the current components and
their terms.
