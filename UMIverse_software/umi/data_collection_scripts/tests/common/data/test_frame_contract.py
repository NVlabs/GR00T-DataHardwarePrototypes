# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the splittable + transform contract on `Frame`."""

from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Type

import numpy as np

from umi.common.data.frame import (
    FRAME_CLASS_MAP,
    EndEffectorFrame,
    Frame,
    GripperFrame,
    GripperObsFrame,
    NumericFrame,
    RGBCameraFrame,
    register_frame,
)

# --- default split/splittable_keys/sub_frame_classes --------------------------


def test_default_split_returns_self_under_empty_key():
    f = Frame(receive_time=1.0)
    out = f.split()
    assert out == {"": f}
    assert out[""] is f


def test_default_splittable_keys_is_empty_string_singleton():
    assert Frame.splittable_keys() == [""]
    # Inherited unchanged on built-ins
    for cls in (
        GripperFrame,
        RGBCameraFrame,
        NumericFrame,
        EndEffectorFrame,
    ):
        assert cls.splittable_keys() == [""], f"{cls.__name__} lost the default"


def test_default_sub_frame_classes_maps_empty_key_to_self():
    assert Frame.sub_frame_classes() == {"": Frame}
    assert GripperFrame.sub_frame_classes() == {"": GripperFrame}


# --- default transform identity -----------------------------------------------


def test_default_transform_is_identity_on_base_frame():
    f = Frame(receive_time=1.0, frame_source="x")
    assert f.transform() is f


def test_end_effector_frame_transform_is_identity():
    e = EndEffectorFrame(joint_positions=np.zeros(16), receive_time=1.0)
    assert e.transform() is e


def test_builtin_frames_inherit_identity_transform_except_gripper():
    # GripperFrame is the only built-in that overrides; everyone else is identity.
    overrides = {GripperFrame}
    for cls in (
        Frame,
        RGBCameraFrame,
        NumericFrame,
        EndEffectorFrame,
    ):
        assert cls not in overrides
        # Verify the class did not override transform
        assert cls.transform is Frame.transform, (
            f"{cls.__name__} overrode transform unexpectedly"
        )


# --- GripperFrame.transform shape preservation --------------------------------


def test_gripper_transform_returns_gripper_obs_frame_with_shape_1_distance():
    g = GripperFrame(
        joint_positions=np.array([10.0]),
        distance=42.5,
        capture_time=0.5,
        receive_time=1.0,
        frame_source="grp",
    )
    derived = g.transform()
    assert isinstance(derived, GripperObsFrame)
    # distance is shape (1,) np.float32 - so per-field interpolation yields (T, 1)
    assert isinstance(derived.distance, np.ndarray)
    assert derived.distance.shape == (1,)
    assert derived.distance.dtype == np.float32
    assert derived.distance[0] == np.float32(42.5)

    # timestamps preserved
    assert derived.capture_time == 0.5
    assert derived.receive_time == 1.0
    assert derived.frame_source == "grp"

    # Only non-metadata field on the derived class is `distance`
    nonmeta = [
        f.name
        for f in fields(GripperObsFrame)
        if f.name not in {"capture_time", "receive_time", "frame_source"}
    ]
    assert nonmeta == ["distance"]


def test_gripper_transform_with_none_distance_returns_self():
    g = GripperFrame(distance=None, joint_positions=np.array([5.0]), receive_time=1.0)
    assert g.transform() is g


def test_gripper_obs_frame_registered():
    assert "GripperObsFrame" in FRAME_CLASS_MAP
    assert FRAME_CLASS_MAP["GripperObsFrame"] is GripperObsFrame


# --- MockTransformableFrame -- transform returns a derived Frame -------------


@register_frame
@dataclass
class _MockPackedFrame(NumericFrame):
    """Derived frame produced by _MockTransformableFrame.transform()."""

    packed: Optional[np.ndarray] = field(default=None, metadata={"interpolate": True})


@register_frame
@dataclass
class _MockTransformableFrame(NumericFrame):
    """Non-splittable frame that defines a packing transform."""

    a: Optional[float] = field(default=None, metadata={"interpolate": True})
    b: Optional[float] = field(default=None, metadata={"interpolate": True})
    c: Optional[float] = field(default=None, metadata={"interpolate": True})

    def transform(self) -> Frame:
        return _MockPackedFrame(
            packed=np.asarray([self.a, self.b, self.c], dtype=np.float32),
            capture_time=self.capture_time,
            receive_time=self.receive_time,
            frame_source=self.frame_source,
        )


def test_mock_transformable_frame_packs_in_declared_order():
    f = _MockTransformableFrame(a=1.0, b=2.0, c=3.0, receive_time=0.5)
    derived = f.transform()
    assert isinstance(derived, _MockPackedFrame)
    np.testing.assert_array_equal(
        derived.packed, np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    )
    assert derived.receive_time == 0.5


def test_mock_packed_frame_registered_by_name():
    # Saved sessions resolve frame classes by name lookup in FRAME_CLASS_MAP.
    assert "_MockPackedFrame" in FRAME_CLASS_MAP
    assert FRAME_CLASS_MAP["_MockPackedFrame"] is _MockPackedFrame


# --- Splittable mock for downstream recorder tests ---------------------------


@register_frame
@dataclass
class _MockSubA(NumericFrame):
    """Sub-stream A of a splittable mock frame."""

    a_value: Optional[float] = field(default=None, metadata={"interpolate": True})


@register_frame
@dataclass
class _MockSubB(NumericFrame):
    """Sub-stream B of a splittable mock frame."""

    b_value: Optional[np.ndarray] = field(default=None, metadata={"interpolate": True})


@register_frame
@dataclass
class _MockSplittableFrame(NumericFrame):
    """Mock combined frame with two sub-streams."""

    a_value: Optional[float] = field(default=None, metadata={"interpolate": True})
    b_value: Optional[np.ndarray] = field(default=None, metadata={"interpolate": True})

    def split(self) -> Dict[str, Frame]:
        return {
            "a": _MockSubA(
                a_value=self.a_value,
                capture_time=self.capture_time,
                receive_time=self.receive_time,
                frame_source=self.frame_source,
            ),
            "b": _MockSubB(
                b_value=self.b_value,
                capture_time=self.capture_time,
                receive_time=self.receive_time,
                frame_source=self.frame_source,
            ),
        }

    @classmethod
    def splittable_keys(cls) -> List[str]:
        return ["a", "b"]

    @classmethod
    def sub_frame_classes(cls) -> Dict[str, Type[Frame]]:
        return {"a": _MockSubA, "b": _MockSubB}


def test_mock_splittable_frame_split_returns_two_sub_streams():
    f = _MockSplittableFrame(
        a_value=7.0,
        b_value=np.array([1.0, 2.0]),
        capture_time=0.25,
        receive_time=0.5,
        frame_source="mock",
    )
    out = f.split()
    assert set(out) == {"a", "b"}
    assert isinstance(out["a"], _MockSubA)
    assert out["a"].a_value == 7.0
    assert isinstance(out["b"], _MockSubB)
    np.testing.assert_array_equal(out["b"].b_value, np.array([1.0, 2.0]))
    # Timestamps copy through
    for sub in out.values():
        assert sub.capture_time == 0.25
        assert sub.receive_time == 0.5
        assert sub.frame_source == "mock"


def test_mock_splittable_keys_and_sub_classes():
    assert _MockSplittableFrame.splittable_keys() == ["a", "b"]
    assert _MockSplittableFrame.sub_frame_classes() == {"a": _MockSubA, "b": _MockSubB}
