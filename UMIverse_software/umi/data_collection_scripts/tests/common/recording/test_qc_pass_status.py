# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `qc_pass_status_from_attrs` (episode_utils.py): each episode's
QC status, with the legacy `discarded_episode_indices` folded in."""


class TestQcPassStatusFromAttrs:
    def test_legacy_discarded_fills_none_slots(self):
        from umi.common.recording.episode_utils import qc_pass_status_from_attrs

        attrs = {"discarded_episode_indices": [1], "num_episodes": 3}
        assert qc_pass_status_from_attrs(attrs) == [None, 0, None]

    def test_explicit_pass_overrides_legacy_absence(self):
        from umi.common.recording.episode_utils import qc_pass_status_from_attrs

        attrs = {
            "discarded_episode_indices": [1],
            "qc_pass_status": [None, 1, None],
            "num_episodes": 3,
        }
        assert qc_pass_status_from_attrs(attrs) == [None, 1, None]
