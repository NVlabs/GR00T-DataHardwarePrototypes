# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic numeric-stream checks.

Seven sub-checks, each optional per (device, array_key) entry in the YAML:

  - `exists`     — zarr key is present and has chunks written
  - `no_nan`     — no NaN values anywhere in the array
  - `in_range`   — all elements within [min, max] (inclusive, element-wise)
  - `jump`       — max abs consecutive-frame diff per axis <= threshold
  - `movement`   — scalar form: at least one axis has (max - min) >= threshold
                   (any-axis pass). Dict form `{threshold, required_axes}`:
                   each listed axis must span >= threshold; unlisted axes are
                   unconstrained. `required_axes` may instead be an
                   `{axis: threshold}` mapping giving each required axis its
                   own threshold (the top-level `threshold` must be omitted
                   then). Axis indices address the flattened non-time dims in
                   `per_axis_range` order.
  - `shape`      — `array.shape[1:]` equals the configured tuple (first axis
                   = number of frames, unconstrained)
  - `start_within_tolerance` — values[0] is within `tolerance` of `target`
                   element-wise (per-axis abs deviation). Useful for pinning
                   the initial pose to a target set by a pre-episode check.
                   Target/tolerance can be literals in the YAML, or pulled
                   from the recording config via `from_recording_config`.

Multi-D arrays are handled element-wise across all non-time axes. Violation
lists report `frame_index` and `axis_index` (None for 1D, int for 2D, tuple for
higher dims) plus the offending value(s), so downstream tooling has enough
information to debug or fix individual frames — matching the depth of
`check_gripper_jumps`.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr

__all__ = [
    "check_numeric_array",
    "check_numeric_values",
    "run_numeric_value_checks",
]

# Cap stored violation lists to keep JSON reports manageable. Counts are
# preserved on the result; the lists themselves are truncated.
_MAX_VIOLATIONS_RETAINED = 100


def _axis_key(idx_tuple: Tuple[int, ...]) -> Optional[Any]:
    """Convert a numpy index tuple (excluding frame axis) into the report's
    `axis_index` field: None for 1D, an int for 2D, a list for higher dims."""
    if len(idx_tuple) == 0:
        return None
    if len(idx_tuple) == 1:
        return int(idx_tuple[0])
    return list(int(x) for x in idx_tuple)


def _no_nan(values: np.ndarray) -> Dict[str, Any]:
    nan_mask = np.isnan(values)
    nan_positions = np.argwhere(nan_mask)
    violations: List[Dict[str, Any]] = []
    for pos in nan_positions[:_MAX_VIOLATIONS_RETAINED]:
        violations.append(
            {
                "frame_index": int(pos[0]),
                "axis_index": _axis_key(tuple(pos[1:])),
            }
        )
    return {
        "passed": bool(nan_positions.size == 0),
        "num_nans": int(nan_positions.shape[0]),
        "nan_positions": violations,
    }


def _in_range(values: np.ndarray, lo: float, hi: float) -> Dict[str, Any]:
    # Suppress NaN warnings here — `no_nan` is the right place to report those.
    with np.errstate(invalid="ignore"):
        below = values < lo
        above = values > hi
    violation_mask = below | above
    positions = np.argwhere(violation_mask)
    violations: List[Dict[str, Any]] = []
    for pos in positions[:_MAX_VIOLATIONS_RETAINED]:
        idx = tuple(int(x) for x in pos)
        violations.append(
            {
                "frame_index": idx[0],
                "axis_index": _axis_key(tuple(pos[1:])),
                "value": float(values[idx]),
            }
        )
    actual_min = float(np.nanmin(values)) if values.size else float("nan")
    actual_max = float(np.nanmax(values)) if values.size else float("nan")
    return {
        "passed": bool(positions.size == 0),
        "min_threshold": float(lo),
        "max_threshold": float(hi),
        "num_out_of_range": int(positions.shape[0]),
        "actual_min": actual_min,
        "actual_max": actual_max,
        "violations": violations,
    }


def _jump(values: np.ndarray, threshold: float) -> Dict[str, Any]:
    if values.shape[0] < 2:
        return {
            "passed": True,
            "threshold": float(threshold),
            "num_jumps": 0,
            "jump_frames": [],
        }
    diffs = np.abs(np.diff(values, axis=0))
    jump_mask = diffs > threshold
    positions = np.argwhere(jump_mask)
    violations: List[Dict[str, Any]] = []
    for pos in positions[:_MAX_VIOLATIONS_RETAINED]:
        # In `diffs`, axis 0 corresponds to the gap between frame i and i+1.
        # Report the frame that jumped (i + 1) to match check_gripper_jumps.
        before_idx = (int(pos[0]),) + tuple(int(x) for x in pos[1:])
        after_idx = (int(pos[0]) + 1,) + tuple(int(x) for x in pos[1:])
        violations.append(
            {
                "frame_index": int(pos[0]) + 1,
                "axis_index": _axis_key(tuple(pos[1:])),
                "before_value": float(values[before_idx]),
                "after_value": float(values[after_idx]),
                "magnitude": float(diffs[tuple(int(x) for x in pos)]),
            }
        )
    return {
        "passed": bool(positions.size == 0),
        "threshold": float(threshold),
        "num_jumps": int(positions.shape[0]),
        "jump_frames": violations,
    }


def _movement(
    values: np.ndarray,
    min_range: Optional[float] = None,
    axis_thresholds: Optional[List[Tuple[int, float]]] = None,
) -> Dict[str, Any]:
    """Span check over the episode: per-axis (max - min) vs a threshold.

    Any-axis mode (`axis_thresholds` is None): at least one axis must span
    `min_range`. Required-axes mode (`axis_thresholds` = ordered
    (axis, threshold) pairs): each listed axis must span its own threshold;
    other axes are unconstrained. Axes index the flattened non-time dims in
    the same order as `per_axis_range` (for a 1-D-per-frame stream, the axis
    index is just the element index, e.g. the joint number).

    `min_range` is reported when given: the uniform-threshold dict form passes
    it alongside `axis_thresholds`; the per-axis mapping form has no single
    threshold, so the result carries `axis_thresholds` instead.
    """
    if values.size == 0:
        out: Dict[str, Any] = {"passed": False}
        if min_range is not None:
            out["min_range"] = float(min_range)
        out["per_axis_range"] = []
        out["max_range"] = 0.0
        if axis_thresholds is not None:
            out["required_axes"] = [a for a, _ in axis_thresholds]
            out["axis_thresholds"] = {a: float(t) for a, t in axis_thresholds}
            out["failing_axes"] = []
            out["out_of_bounds_axes"] = [a for a, _ in axis_thresholds]
        return out
    # Reduce over the frame axis; ranges has shape = values.shape[1:]
    with np.errstate(invalid="ignore"):
        per_axis = (np.nanmax(values, axis=0) - np.nanmin(values, axis=0)).astype(float)
    flat = per_axis.ravel().tolist()
    max_range = float(per_axis.max()) if per_axis.size else 0.0
    out = {"passed": True}
    if min_range is not None:
        out["min_range"] = float(min_range)
    out["per_axis_range"] = [float(x) for x in flat]
    out["max_range"] = max_range
    if axis_thresholds is None:
        out["passed"] = bool(max_range >= min_range)
    else:
        out_of_bounds = [a for a, _ in axis_thresholds if a >= len(flat)]
        failing = [
            {"axis_index": a, "range": float(flat[a]), "threshold": float(t)}
            for a, t in axis_thresholds
            if a < len(flat) and not flat[a] >= t
        ]
        out["required_axes"] = [a for a, _ in axis_thresholds]
        out["axis_thresholds"] = {a: float(t) for a, t in axis_thresholds}
        out["failing_axes"] = failing
        out["out_of_bounds_axes"] = out_of_bounds
        out["passed"] = bool(not failing and not out_of_bounds)
    return out


def _shape(values: np.ndarray, expected: List[int]) -> Dict[str, Any]:
    """`array.shape[1:]` must equal `expected` (first axis = frames, unconstrained)."""
    actual = list(values.shape[1:])
    return {
        "passed": actual == list(int(x) for x in expected),
        "expected": list(int(x) for x in expected),
        "actual": [int(x) for x in actual],
    }


def _start_within_tolerance(
    values: np.ndarray, target: np.ndarray, tolerance: float
) -> Dict[str, Any]:
    """Element-wise check that `|values[0] - target| <= tolerance`.

    Lets the data check verify, post-recording, that a start-pose constraint
    the operator was asked to satisfy (e.g. by a pre-episode check) actually
    held in frame 0.
    """
    if values.shape[0] == 0:
        return {
            "passed": False,
            "error": "no frames",
            "tolerance": float(tolerance),
        }
    first = values[0]
    target_arr = np.asarray(target, dtype=np.float64)
    if first.shape != target_arr.shape:
        return {
            "passed": False,
            "error": (
                f"shape mismatch: first frame {tuple(first.shape)} vs "
                f"target {tuple(target_arr.shape)}"
            ),
            "tolerance": float(tolerance),
        }
    diffs = first.astype(np.float64) - target_arr
    abs_diffs = np.abs(diffs)
    if not np.all(np.isfinite(abs_diffs)):
        # `ravel()` so this works for both 1-D and 0-D (scalar) results.
        bad = np.where(~np.isfinite(abs_diffs).ravel())[0].tolist()
        return {
            "passed": False,
            "error": f"non-finite deviation at axes {bad}",
            "tolerance": float(tolerance),
        }
    in_tol_mask = abs_diffs <= tolerance
    # Flatten scalar or vector results so argmax and subsequent indexing use the same
    # axis.
    flat_diffs = diffs.ravel()
    flat_abs = abs_diffs.ravel()
    worst_idx = int(np.argmax(flat_abs))
    return {
        "passed": bool(in_tol_mask.all()),
        "target": [float(x) for x in target_arr.ravel().tolist()],
        "tolerance": float(tolerance),
        "actual_first_frame": [float(x) for x in np.asarray(first).ravel().tolist()],
        "abs_deviations": [float(x) for x in flat_abs.tolist()],
        "n_in_tolerance": int(in_tol_mask.sum()),
        "total_axes": int(in_tol_mask.size),
        "worst_axis_index": worst_idx,
        "worst_deviation_signed": float(flat_diffs[worst_idx]),
    }


def _filter_matches(item: Dict[str, Any], segment: Dict[str, Any]) -> bool:
    """True when every (key, value) in a list-filter segment matches `item`.
    A dotted key (`params.source_name`) walks nested dicts; a missing key at
    any depth is a non-match, never an error."""
    for key, expected in segment.items():
        node: Any = item
        for part in str(key).split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        if node != expected:
            return False
    return True


def _walk_recording_config_path(
    recording_config: Dict[str, Any], path: List[Any]
) -> Any:
    """Resolve a path inside the recording config.

    Each segment is one of:
      - str: dict key lookup
      - int: list index lookup
      - dict: list-of-dicts filter; returns the first list item where every
              (k, v) in the segment dict matches `item[k] == v`. This is how
              `pre_episode_checks` (a list keyed by `name`) is addressed.
              A filter key may be dotted (`params.source_name`) to match a
              nested value — needed when several checks share a `name` and
              differ only inside `params` (one per hand on a bimanual rig).

    Example: to read the target a pre-episode check was given for the right
    side of a bimanual rig,
      [pre_episode_checks,
       {name: <check_name>, params.source_name: <right_device>},
       params,
       <target_key>]
    """
    if not isinstance(path, list):
        raise ValueError(
            f"recording-config path must be a list, got {type(path).__name__}"
        )
    node: Any = recording_config
    for i, segment in enumerate(path):
        if isinstance(segment, str):
            if not isinstance(node, dict):
                raise ValueError(
                    f"path {path[: i + 1]}: expected dict at this depth, got {type(node).__name__}"
                )
            if segment not in node:
                raise ValueError(f"path {path[: i + 1]}: key {segment!r} not found")
            node = node[segment]
        elif isinstance(segment, bool):
            # bool is a subclass of int — reject it before the int branch.
            raise ValueError(f"path {path[: i + 1]}: bool segments are not supported")
        elif isinstance(segment, int):
            if not isinstance(node, list):
                raise ValueError(
                    f"path {path[: i + 1]}: expected list at this depth, got {type(node).__name__}"
                )
            node = node[segment]
        elif isinstance(segment, dict):
            if not isinstance(node, list):
                raise ValueError(
                    f"path {path[: i + 1]}: filter {segment} requires a list, got {type(node).__name__}"
                )
            found = None
            for item in node:
                if isinstance(item, dict) and _filter_matches(item, segment):
                    found = item
                    break
            if found is None:
                raise ValueError(
                    f"path {path[: i + 1]}: no list item matched filter {segment}"
                )
            node = found
        else:
            raise ValueError(
                f"path {path[: i + 1]}: unsupported segment type {type(segment).__name__}"
            )
    return node


def _resolve_value(value: Any, recording_config: Optional[Dict[str, Any]]) -> Any:
    """If `value` is `{from_recording_config: <path>}`, resolve it from the
    recording config. Otherwise return `value` unchanged.

    This is the single extension point for "the data-check YAML wants to
    reference a value that lives in the recording config." Future fields
    (other targets, expected frame counts, etc.) can use the same syntax.
    """
    if isinstance(value, dict) and "from_recording_config" in value:
        if recording_config is None:
            raise ValueError(
                "from_recording_config requires the recording config to be "
                "available; check that the data-check pipeline is plumbed "
                "to pass `config` through to run_numeric_value_checks."
            )
        return _walk_recording_config_path(
            recording_config, value["from_recording_config"]
        )
    return value


def check_numeric_array(values: np.ndarray, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run every sub-check the spec requests against an already-loaded array.

    Excludes the `exists` sub-check (which is handled by the caller, since it
    requires zarr access to a missing key). Each sub-check is optional — absent
    keys map to `None` in the result.

    For sub-checks whose parameters can be sourced from the recording config
    (e.g. `start_within_tolerance.target`), the caller must resolve those
    `from_recording_config` references into literal values before reaching
    this function.
    """
    out: Dict[str, Any] = {
        "no_nan": None,
        "in_range": None,
        "jump": None,
        "movement": None,
        "shape": None,
        "start_within_tolerance": None,
    }

    if spec.get("no_nan"):
        out["no_nan"] = _no_nan(values)

    in_range_spec = spec.get("in_range")
    if in_range_spec is not None:
        if not isinstance(in_range_spec, (list, tuple)) or len(in_range_spec) != 2:
            raise ValueError(
                f"`in_range` must be a [min, max] pair; got {in_range_spec!r}"
            )
        out["in_range"] = _in_range(values, in_range_spec[0], in_range_spec[1])

    jump_spec = spec.get("jump")
    if jump_spec is not None:
        out["jump"] = _jump(values, float(jump_spec))

    movement_spec = spec.get("movement")
    if movement_spec is not None:
        if isinstance(movement_spec, dict):
            required_axes = movement_spec.get("required_axes")
            if isinstance(required_axes, dict):
                # Per-axis thresholds accept integer keys from YAML or digit-string keys
                # from JSON task overrides.
                if "threshold" in movement_spec:
                    raise ValueError(
                        "dict-form `movement` cannot combine `threshold` with "
                        "a per-axis `required_axes` mapping; put each axis's "
                        f"threshold in the mapping instead. Got {movement_spec!r}"
                    )

                def _valid_axis_key(a: Any) -> bool:
                    if isinstance(a, bool):
                        return False
                    if isinstance(a, int):
                        return a >= 0
                    return isinstance(a, str) and a.isdigit()

                if (
                    len(required_axes) == 0
                    or any(not _valid_axis_key(a) for a in required_axes)
                    or any(
                        isinstance(t, bool) or not isinstance(t, (int, float))
                        for t in required_axes.values()
                    )
                ):
                    raise ValueError(
                        "per-axis `movement.required_axes` must be a non-empty "
                        "{axis: threshold} mapping of non-negative int axes "
                        "(digit strings accepted for JSON-delivered specs) to "
                        f"numeric thresholds; got {required_axes!r}"
                    )
                out["movement"] = _movement(
                    values,
                    axis_thresholds=[
                        (int(a), float(t)) for a, t in required_axes.items()
                    ],
                )
            else:
                if "threshold" not in movement_spec:
                    raise ValueError(
                        f"dict-form `movement` requires `threshold`; got {movement_spec!r}"
                    )
                if (
                    not isinstance(required_axes, (list, tuple))
                    or len(required_axes) == 0
                    or any(
                        isinstance(a, bool) or not isinstance(a, int) or a < 0
                        for a in required_axes
                    )
                ):
                    raise ValueError(
                        "dict-form `movement` requires `required_axes` as a non-empty "
                        f"list of non-negative ints; got {required_axes!r}"
                    )
                threshold = float(movement_spec["threshold"])
                out["movement"] = _movement(
                    values,
                    threshold,
                    axis_thresholds=[(int(a), threshold) for a in required_axes],
                )
        else:
            out["movement"] = _movement(values, float(movement_spec))

    shape_spec = spec.get("shape")
    if shape_spec is not None:
        out["shape"] = _shape(values, shape_spec)

    swt_spec = spec.get("start_within_tolerance")
    if swt_spec is not None:
        target = np.asarray(swt_spec["target"], dtype=np.float64)
        tolerance = float(swt_spec["tolerance"])
        out["start_within_tolerance"] = _start_within_tolerance(
            values, target, tolerance
        )

    return out


def check_numeric_values(
    episode_data: zarr.Group,
    device: str,
    array_key: str,
    spec: Dict[str, Any],
    recording_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all configured sub-checks for one (device, array_key) pair.

    `recording_config` (the session's `config.yaml`) is consulted when the
    spec uses `from_recording_config` references (e.g. to pull the
    `start_within_tolerance.target` from the same pre-episode-check params
    the operator was held to at recording time).

    Returns a dict ready to merge into the per-episode report.
    """
    result: Dict[str, Any] = {
        "passed": True,
        "device": device,
        "array_key": array_key,
        "exists": None,
        "no_nan": None,
        "in_range": None,
        "jump": None,
        "movement": None,
        "shape": None,
        "start_within_tolerance": None,
        "issues": [],
    }

    qualified = f"{device}.{array_key}"

    # Require both array metadata and stored chunks: reading an unwritten Zarr
    # array returns fill values and can conceal missing data. Writers must
    # materialize all-zero chunks for valid zero-valued readings to pass.
    # If exists fails, skip checks that require actual samples.
    device_present = device in episode_data
    array_present = device_present and array_key in episode_data[device]
    if spec.get("exists"):
        if not array_present:
            result["exists"] = {
                "passed": False,
                "key_present": False,
                "chunks_initialized": None,
                "total_chunks": None,
            }
            result["passed"] = False
            result["issues"].append(f"{qualified}: missing")
            return result
        arr = episode_data[device][array_key]
        ninit = int(arr.nchunks_initialized)
        ntotal = int(arr.nchunks)
        result["exists"] = {
            "passed": ninit > 0,
            "key_present": True,
            "chunks_initialized": ninit,
            "total_chunks": ntotal,
        }
        if ninit == 0:
            result["passed"] = False
            result["issues"].append(
                f"{qualified}: registered but no chunks written ({ninit}/{ntotal})"
            )
            return result
    elif not array_present:
        # No `exists` requirement and the array isn't there — nothing to check.
        return result

    values = np.asarray(episode_data[device][array_key][:])

    # Resolve from_recording_config references before passing rules to the NumPy checks.
    resolved_spec = spec
    swt_spec = spec.get("start_within_tolerance")
    if isinstance(swt_spec, dict):
        try:
            target = _resolve_value(swt_spec.get("target"), recording_config)
            tolerance = _resolve_value(swt_spec.get("tolerance"), recording_config)
        except ValueError as e:
            result["passed"] = False
            result["issues"].append(
                f"{qualified}: start_within_tolerance config error ({e})"
            )
            return result
        if target is None or tolerance is None:
            result["passed"] = False
            result["issues"].append(
                f"{qualified}: start_within_tolerance requires both `target` "
                "and `tolerance`"
            )
            return result
        resolved_spec = dict(spec)
        resolved_spec["start_within_tolerance"] = {
            "target": target,
            "tolerance": tolerance,
        }

    sub_results = check_numeric_array(values, resolved_spec)
    result.update(sub_results)

    no_nan_r = result.get("no_nan")
    if no_nan_r is not None and not no_nan_r["passed"]:
        result["passed"] = False
        result["issues"].append(f"{qualified}: {no_nan_r['num_nans']} NaN values")

    in_range_r = result.get("in_range")
    if in_range_r is not None and not in_range_r["passed"]:
        result["passed"] = False
        result["issues"].append(
            f"{qualified}: out of range "
            f"[{in_range_r['min_threshold']}, {in_range_r['max_threshold']}] "
            f"({in_range_r['num_out_of_range']} elements)"
        )

    jump_r = result.get("jump")
    if jump_r is not None and not jump_r["passed"]:
        result["passed"] = False
        result["issues"].append(
            f"{qualified}: {jump_r['num_jumps']} jumps detected "
            f"(threshold={jump_r['threshold']})"
        )

    movement_r = result.get("movement")
    if movement_r is not None and not movement_r["passed"]:
        result["passed"] = False
        if movement_r.get("required_axes") is not None:
            parts = [
                f"axis {f['axis_index']}: range {f['range']:.2f} < {f['threshold']}"
                for f in movement_r.get("failing_axes", [])
            ]
            oob = movement_r.get("out_of_bounds_axes") or []
            if oob:
                n_axes = len(movement_r.get("per_axis_range", []))
                parts.append(
                    f"required_axes {oob} out of bounds for {n_axes}-axis stream"
                )
            result["issues"].append(
                f"{qualified}: required axes did not move ({'; '.join(parts)})"
            )
        else:
            result["issues"].append(
                f"{qualified}: no movement "
                f"(max range {movement_r['max_range']} < {movement_r['min_range']})"
            )

    shape_r = result.get("shape")
    if shape_r is not None and not shape_r["passed"]:
        result["passed"] = False
        result["issues"].append(
            f"{qualified}: shape mismatch "
            f"(expected {shape_r['expected']}, got {shape_r['actual']})"
        )

    swt_r = result.get("start_within_tolerance")
    if swt_r is not None and not swt_r["passed"]:
        result["passed"] = False
        if "error" in swt_r:
            result["issues"].append(
                f"{qualified}: start_within_tolerance: {swt_r['error']}"
            )
        else:
            n_ok = swt_r.get("n_in_tolerance", 0)
            total = swt_r.get("total_axes", 0)
            worst_axis = swt_r.get("worst_axis_index")
            worst_dev = swt_r.get("worst_deviation_signed", 0.0)
            tol = swt_r.get("tolerance", 0.0)
            result["issues"].append(
                f"{qualified}: start frame out of tolerance "
                f"({n_ok}/{total} axes in tol; worst axis {worst_axis}: "
                f"{worst_dev:+.2f} > {tol:.2f})"
            )

    return result


def run_numeric_value_checks(
    episode_data: zarr.Group,
    numeric_values_config: Dict[str, Any],
    recording_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Iterate {device: {array_key: spec}} in YAML order.

    `recording_config` is forwarded so sub-checks can resolve
    `from_recording_config` references.

    Returns (results_by_device_and_key, issues). Iteration order is deterministic
    because dict iteration in Python 3.7+ preserves insertion order, and PyYAML
    inherits that.
    """
    results: Dict[str, Dict[str, Any]] = {}
    issues: List[str] = []
    for device, per_key in numeric_values_config.items():
        if device == "enabled" or not isinstance(per_key, dict):
            continue
        results[device] = {}
        for array_key, spec in per_key.items():
            if not isinstance(spec, dict):
                continue
            r = check_numeric_values(
                episode_data, device, array_key, spec, recording_config
            )
            results[device][array_key] = r
            issues.extend(r.get("issues", []))
    return results, issues
