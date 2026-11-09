# Record3D-derived iPhone adapters

The complete iPhone adapter files in both pipelines, including NVIDIA's
adaptations, are distributed under **LGPL-2.1-only**:

- [UMI adapter](../../UMIverse_software/umi/data_collection_scripts/umi/common/hardware_sdk/camera/iphone_camera.py)
- [DexUMI adapter](../../UMIverse_software/dexumi/data_collection_scripts/umi/common/hardware_sdk/camera/iphone_camera.py)

These files are exceptions to the repository's Apache-2.0 default. The
complete LGPL 2.1 text is retained in
[record3d-79e6f446/LICENSE](../DexUMI/record3d-79e6f446/LICENSE).
[NOTICE](../../NOTICE) records the component scope and modifications.

## Upstream material and modifications

The adapters contain material matching
[Record3D's demo-main.py](https://github.com/marek-simonik/record3d/blob/79e6f44697150a4f036034af8fec903d11ac6f00/demo-main.py),
by Marek Simonik. This revision identifies the matching material; the original
copy revision is unknown. The example has no per-file copyright notice;
the upstream package identifies its author and declares `lgpl-2.1` in
[setup.py](https://github.com/marek-simonik/record3d/blob/79e6f44697150a4f036034af8fec903d11ac6f00/setup.py).
This release uses LGPL-2.1-only without assuming permission for later versions.

NVIDIA adaptations include the Camera/RGBDCameraFrame integration, UDID
selection, capture timestamps, pose conversion, frame backlog handling,
depth conversion, resolution/calibration handling, and diagnostics. The
DexUMI adapter also reports device identity. NVIDIA's copyright notice
remains in each file; the LGPL treatment includes those adaptations.

On 2026-11-05 the notices were corrected and the existing adaptations
documented without changing executable camera code. This date records the
notice correction, not the dates of the earlier adaptations. Future changes
must identify what changed and the dates in the affected source files.

## Using a modified Record3D package

Record3D is installed separately and can be replaced with a modified,
compatible package. When using a replacement, launch with
`uv run --no-sync umi_dc` or `uv run --no-sync dexumi_dc` to avoid automatic
synchronization. See [uv's sync behavior](https://docs.astral.sh/uv/concepts/projects/sync/).

## Source release boundary

This submission distributes the adapter source and its required notices and
license text. Third-party wheels, installed environments, dependency caches,
container images and frozen executables are excluded. Users download
dependencies separately during installation. Keep the adapters editable and
Record3D replaceable; do not distribute only compiled or frozen copies of
these adapters under this source-only submission.

When exporting either pipeline separately, include its adapter source,
the relevant NOTICE entry, this attribution guide, the LGPL text, and the
other license files required by its components. Preserve the relative
license links or update them for the exported layout. Notices alone do not
fulfill the obligations of a future binary/environment distribution.

`python3 tools/licenses.py check` verifies the adapter notices, retained LGPL
text and common bundled runtime artifacts in tracked paths and ZIP/TAR member
names. It does not audit arbitrary external release artifacts or establish
the legal boundary of other code that uses Record3D.
