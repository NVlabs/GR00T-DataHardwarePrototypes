# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass, field, fields, make_dataclass
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from zarr.codecs import BloscCodec

from umi.common.data.interpolator import (
    BaseInterpolator,
    LinearInterpolator,
    PoseInterpolator,
)

# Registry mapping interpolator class names to classes for serialization
INTERPOLATOR_REGISTRY: Dict[str, Type[BaseInterpolator]] = {
    "PoseInterpolator": PoseInterpolator,
    "LinearInterpolator": LinearInterpolator,
}

# Built-in frame classes that don't need serialization
BUILTIN_FRAME_NAMES = {
    "Frame",
    "NumericFrame",
    "EncoderFrame",
    "RGBCameraFrame",
    "RGBDCameraFrame",
}


@dataclass(frozen=True)
class ZarrStorageConfig:
    """Zarr storage configuration for a frame field."""

    compressors: Optional[Any] = None  # e.g., BloscCodec instance
    chunk_frames: int = (
        0  # Frames per chunk along axis 0. 0 = zarr default, 1 = per-frame
    )


DEPTH_STORAGE = ZarrStorageConfig(
    compressors=BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle"),
    chunk_frames=128,
)

# Registry of all frame classes
FRAME_CLASS_MAP: Dict[str, Type["Frame"]] = {}


def register_frame(cls: Type["Frame"]) -> Type["Frame"]:
    """Decorator to register a Frame class in FRAME_CLASS_MAP.

    Also attaches ``cls`` as an attribute of its declared ``__module__``
    when one isn't already there. Dynamic classes built via
    ``make_dataclass`` have ``__module__`` pointing at their construction
    site (e.g. this module when built via ``create_frame``) but aren't
    actually accessible there by attribute lookup, which breaks
    ``pickle.dumps`` of their instances. Setting the attribute fixes the
    lookup without changing class identity.

    Usage:
        @register_frame
        @dataclass
        class MyCustomFrame(Frame):
            my_field: Optional[np.ndarray] = None
    """
    FRAME_CLASS_MAP[cls.__name__] = cls
    module = sys.modules.get(cls.__module__)
    if module is not None and not hasattr(module, cls.__name__):
        setattr(module, cls.__name__, cls)
    return cls


def is_interpolatable(frame_class: Type["Frame"], field_name: str) -> bool:
    """Check if a field should be interpolated.

    Args:
        frame_class: The Frame class to check.
        field_name: Name of the field.

    Returns:
        True if the field should be interpolated, False otherwise.
    """
    for f in fields(frame_class):
        if f.name == field_name:
            if f.metadata.get("interpolate") is False:
                return False
            return True
    return False


def get_field_interpolator(
    frame_class: Type["Frame"], field_name: str
) -> Optional[Type["BaseInterpolator"]]:
    """Get interpolator class for a field from its metadata.

    Args:
        frame_class: The Frame class to check.
        field_name: Name of the field.

    Returns:
        Interpolator class if specified in metadata, None otherwise.
    """
    for f in fields(frame_class):
        if f.name == field_name:
            return f.metadata.get("interpolator", None)
    return None


def get_zarr_storage_config(
    frame_class: Type["Frame"], field_name: str
) -> Optional[ZarrStorageConfig]:
    """Get Zarr storage configuration for a field from its metadata.

    Args:
        frame_class: The Frame class to check.
        field_name: Name of the field.

    Returns:
        ZarrStorageConfig if specified in metadata, None otherwise.
    """
    for f in fields(frame_class):
        if f.name == field_name:
            return f.metadata.get("zarr_storage", None)
    return None


_BLUE = "\033[94m"
_CYAN = "\033[96m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _format_value(value: any) -> str:
    """Format a value for pretty printing."""
    if isinstance(value, np.ndarray):
        if value.size > 50:
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        return np.array2string(value, precision=3, separator=", ")
    elif isinstance(value, dict):
        if not value:
            return "{}"
        items = []
        for k, v in value.items():
            items.append(f"{k}: {_format_value(v)}")
        return "{" + ", ".join(items) + "}"
    elif isinstance(value, float):
        return f"{value:.4f}"
    return repr(value)


@register_frame
@dataclass
class Frame:
    # Timestamp fields - used for alignment, not interpolated as data
    capture_time: Optional[float] = field(default=None, metadata={"interpolate": False})
    receive_time: Optional[float] = field(default=None, metadata={"interpolate": False})
    # Metadata fields - strings can't be interpolated
    frame_source: Optional[str] = field(default=None, metadata={"interpolate": False})

    def __str__(self) -> str:
        """Pretty print frame with blue color."""
        lines = [f"{_BOLD}{_BLUE}{self.__class__.__name__}{_RESET}"]

        for f in fields(self):
            if f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if value is not None:
                formatted = _format_value(value)
                lines.append(f"  {_CYAN}{f.name}{_RESET}: {formatted}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Splittable contract (recorder fan-out)
    # ------------------------------------------------------------------
    def split(self) -> Dict[str, "Frame"]:
        """Return sub-frames keyed by sub-stream name. Default: not splittable.

        Subclasses that carry multiple logical streams (e.g. joint + tactile)
        override this to fan the frame out for separate recording.
        """
        return {"": self}

    @classmethod
    def splittable_keys(cls) -> List[str]:
        """Sub-stream names this frame class fans out into. Default: [""]."""
        return [""]

    @classmethod
    def sub_frame_classes(cls) -> Dict[str, Type["Frame"]]:
        """Sub-stream name -> Frame class. Default: self under the empty key."""
        return {"": cls}


@register_frame
@dataclass
class RGBCameraFrame(Frame):
    # Video frames stored as mp4, not interpolated
    rgb: Optional[np.ndarray] = field(default=None, metadata={"interpolate": False})
    # 4x4 homogeneous matrix - uses Slerp interpolation
    pose: Optional[np.ndarray] = field(
        default=None, metadata={"interpolate": True, "interpolator": PoseInterpolator}
    )
    # Camera intrinsics are constant, not interpolated
    intrinsics: Optional[np.ndarray] = field(
        default=None, metadata={"interpolate": False}
    )


@register_frame
@dataclass
class RGBDCameraFrame(RGBCameraFrame):
    # Depth frames stored like video, not interpolated
    depth: Optional[np.ndarray] = field(
        default=None, metadata={"interpolate": False, "zarr_storage": DEPTH_STORAGE}
    )


@register_frame
@dataclass
class NumericFrame(Frame):
    pass


@register_frame
@dataclass
class EncoderFrame(Frame):
    """Encoder sensor frame with position data.

    Contains position values from multiple encoders.
    """

    position: Union[List[float], np.ndarray] = field(
        default_factory=list, metadata={"interpolate": True}
    )  # Position values


def create_frame(
    class_name: str,
    entries: Dict[str, Dict[str, Any]],
    base_class: Type[Frame] = NumericFrame,
) -> Type[Frame]:
    """Create a frame dataclass dynamically with the given entries.

    The resulting class works exactly like hand-written frame dataclasses
    (e.g., EncoderFrame) — is_interpolatable(), fields(), Frame.__str__,
    and the recorder all work natively.

    Args:
        class_name: Name for the new frame class.
        entries: Dict mapping entry names to field metadata.
            Metadata keys match those used by hand-written frame classes:
            - "interpolate": bool (default True)
            - "interpolator": interpolator class (e.g., PoseInterpolator)
            - "zarr_storage": ZarrStorageConfig instance
        base_class: Base class to inherit from (default NumericFrame).

    Returns:
        A new dataclass type, registered in FRAME_CLASS_MAP.

    Example:
        TactileFrame = create_frame("TactileFrame", {
            "tactile_0": {"interpolate": False},
            "depth": {"interpolate": False, "zarr_storage": DEPTH_STORAGE},
        })
        frame = TactileFrame(capture_time=1.0, tactile_0=1.5)
        is_interpolatable(TactileFrame, "tactile_0")  # False
    """
    fields_list = [
        (name, Optional[Any], field(default=None, metadata=metadata))
        for name, metadata in entries.items()
    ]
    cls = make_dataclass(class_name, fields_list, bases=(base_class,))
    return register_frame(cls)


def flatten_sample(prefix: str, sample: Any) -> Dict[str, Any]:
    """Recursively flatten a sample value into {field_name: leaf_value} pairs.

    Dicts are recursively flattened with '_' as separator.
    Non-dict values (np.ndarray, numbers, lists) become a single entry.

    Args:
        prefix: Prefix for field names (e.g., sensor name).
        sample: Sample value to flatten.

    Returns:
        Dict mapping flattened field names to leaf values.

    Example:
        >>> flatten_sample("tactile_0", {"force": {"normal": 1.0}, "prox": 2.0})
        {"tactile_0_force_normal": 1.0, "tactile_0_prox": 2.0}
        >>> flatten_sample("encoder", np.array([1.0, 2.0]))
        {"encoder": np.array([1.0, 2.0])}
    """
    if isinstance(sample, dict):
        result = {}
        for key, value in sample.items():
            child_prefix = f"{prefix}_{key}" if prefix else key
            result.update(flatten_sample(child_prefix, value))
        return result
    return {prefix: sample}


def create_frame_from_sample(
    class_name: str,
    samples: Dict[str, Any],
    base_class: Type[Frame] = NumericFrame,
    **metadata: Any,
) -> Type[Frame]:
    """Create a frame class by inspecting sample values.

    Recursively flattens nested dicts into individual fields.
    Non-dict values (np.ndarray, numbers, lists) become single fields.

    Args:
        class_name: Name for the new frame class.
        samples: Dict mapping entry names to sample values.
            Dict values are recursively flattened with '_' separator.
        base_class: Base class (default NumericFrame).
        **metadata: Field metadata applied to all entries
            (e.g., interpolate=False).

    Returns:
        A new dataclass type, registered in FRAME_CLASS_MAP.

    Example:
        decoded = {"raw_channels": np.zeros(13), "normal_force_1": 0.0}
        TactileFrame = create_frame_from_sample(
            "TactileFrame",
            {"tactile_0": decoded, "tactile_1": decoded},
            interpolate=False,
        )
        # Creates fields: tactile_0_raw_channels, tactile_0_normal_force_1,
        #                  tactile_1_raw_channels, tactile_1_normal_force_1
    """
    entries = {}
    for name, sample in samples.items():
        for field_name in flatten_sample(name, sample):
            entries[field_name] = metadata
    return create_frame(class_name, entries, base_class)


def serialize_frame_definition(frame_class: Type["Frame"]) -> Dict[str, Any]:
    """Serialize a dynamic frame class definition for YAML storage.

    Extracts non-inherited fields and their metadata, converting class
    references (e.g., PoseInterpolator) to strings.

    Args:
        frame_class: The frame class to serialize.

    Returns:
        Dict with "base_class" name and "fields" mapping field names to metadata.
    """
    base_class = NumericFrame  # default
    for cls in frame_class.__mro__[1:]:
        if cls in (object,):
            continue
        if cls.__name__ in FRAME_CLASS_MAP:
            base_class = cls
            break

    base_field_names = {f.name for f in fields(base_class)}

    serialized_fields: Dict[str, Dict[str, Any]] = {}
    for f in fields(frame_class):
        if f.name in base_field_names:
            continue
        metadata = dict(f.metadata)
        if "interpolator" in metadata and metadata["interpolator"] is not None:
            metadata["interpolator"] = metadata["interpolator"].__name__
        metadata.pop("zarr_storage", None)
        serialized_fields[f.name] = metadata

    return {
        "base_class": base_class.__name__,
        "fields": serialized_fields,
    }


def deserialize_frame_definition(
    class_name: str, definition: Dict[str, Any]
) -> Type["Frame"]:
    """Recreate a dynamic frame class from a serialized definition.

    Args:
        class_name: Name for the frame class.
        definition: Dict with "base_class" and "fields" as produced by
            serialize_frame_definition().

    Returns:
        The recreated and registered frame class.
    """
    base_class_name = definition.get("base_class", "NumericFrame")
    base_class = FRAME_CLASS_MAP.get(base_class_name, NumericFrame)

    entries: Dict[str, Dict[str, Any]] = {}
    for field_name, metadata in definition.get("fields", {}).items():
        metadata = dict(metadata)  # copy
        if "interpolator" in metadata and isinstance(metadata["interpolator"], str):
            metadata["interpolator"] = INTERPOLATOR_REGISTRY.get(
                metadata["interpolator"]
            )
        entries[field_name] = metadata

    return create_frame(class_name, entries, base_class)


def load_frame_definitions(config: Dict[str, Any]) -> None:
    """Load and register dynamic frame classes from a config dict.

    Reads the "frame_definitions" key and recreates any frame classes
    not already in FRAME_CLASS_MAP.

    Args:
        config: Config dict (typically loaded from config.yaml).
    """
    frame_defs = config.get("frame_definitions", {})
    for class_name, definition in frame_defs.items():
        if class_name not in FRAME_CLASS_MAP:
            deserialize_frame_definition(class_name, definition)
