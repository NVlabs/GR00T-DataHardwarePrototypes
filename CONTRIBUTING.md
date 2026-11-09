# Contributing to GR00T Hardware Designs

External contributions to all GEAR projects in this repository are welcome.
Report issues and submit pull requests at
[https://github.com/NVlabs/GR00T-DataHardwarePrototypes](https://github.com/NVlabs/GR00T-DataHardwarePrototypes).
The software listed in [NOTICE](NOTICE) under "Software license scope" (the UMI and DexUMI data-collection
software, the UMI gripper firmware and the DexUMI hand monitor) uses
Apache-2.0 by default. The two Record3D-derived iPhone adapters, including
their NVIDIA adaptations, are exceptions licensed under LGPL-2.1-only;
see [their attribution and terms](LICENSES/Record3D/README.md). Contributions
to those adapters must retain that license and identify the modifications
and their dates in the affected files. Contributions to other third-party
portions retain their applicable terms. Every contribution must follow the
Developer Certificate of Origin (DCO) sign-off
process below. A Contributor License Agreement (CLA) is not required.

Preserve original third-party copyright, license and attribution notices.
Contributions to third-party portions must respect their applicable terms.
Identify the source and redistribution permissions for incorporated code,
models or other external material.

See the [licensing section](README.md#license) and [NOTICE](NOTICE).
Contact: gear-hardware@nvidia.com.

This submission distributes software source only. Do not add third-party
wheels, installed environments, dependency caches, container images or
frozen executables to source releases. Keep Record3D separately installable
and replaceable. Run `python3 tools/licenses.py check` before submitting;
the checker also examines tracked files and ZIP/TAR member names for common
bundled runtime artifacts. Any future binary distribution needs separate
artifact-specific licensing work.

## Signing Off Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```

Link to DCO: <https://developercertificate.org/>
