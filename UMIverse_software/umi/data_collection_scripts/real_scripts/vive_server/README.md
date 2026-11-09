# Vive tracker pose bridge

`vive_server.py` streams Vive tracker poses over ZMQ for UMI data collection.
It runs on the **tracker host** (the machine with the OpenXR runtime and the
base stations), not on a recording station.

## Why it lives here

The wire format is a contract with `umi/common/hardware_sdk/vive/client.py`.
Keeping the server in the same repository versions both halves together, so a
change to the reply (such as the `tracked` flag below, without which a puck that
lost tracking would keep streaming a stale pose) reaches both ends at once.

It is under `real_scripts/` rather than beside the client because it needs
`pyopenxr` (`import xr`), which is deliberately **not** a UMI dependency —
stations never run this. Nothing in the repo imports from `real_scripts/`, so
the undeclared dependency cannot leak into a station environment.

## Running it

On the tracker host, in an environment with `pyopenxr` and a working OpenXR
runtime:

```bash
python vive_server.py --port 5555
```

`--frequency` is accepted for compatibility with existing launch scripts but has
no effect: the loop is paced by `xr.wait_frame`, measured at about 31 Hz.

## Protocol

One command, `get_vive_data`, over `zmq.REP`. The command string is ignored —
any request gets the same reply:

```json
{"left_knee": {"position":    {"x": 0.0, "y": 0.0, "z": 0.0},
               "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
               "tracked":     true,
               "t":           1755130000.123}}
```

| field | meaning |
|---|---|
| `position` / `orientation` | last pose while `tracked` was true; not refreshed while untracked |
| `tracked` | both `POSITION_VALID_BIT` and `POSITION_TRACKED_BIT` set, this frame |
| `t` | wall clock of the frame that produced this entry |

A puck appears only after it has been tracked at least once, so an empty reply
means nothing has been seen yet — not that the server is unhealthy.

**`tracked` is required.** `ViveTracker.start_streaming()` refuses a server that
omits it, because OpenXR keeps `POSITION_VALID` set on a puck that has lost
tracking and clears `POSITION_TRACKED` instead. A server testing VALID alone
republishes the last known pose indefinitely, which records as a flat,
plausible-looking trajectory indistinguishable from a motionless gripper.

## Roles are not fixed

Tracker keys are OpenXR role paths (`left_knee`, `right_shoulder`, …) assigned in
SteamVR, and they change when pucks are re-bound, which can happen more than
once a day. The station config maps them explicitly:

```yaml
- type: vive_tracker
  name: left_tracking_pose      # UMI's name, drives the zarr group
  params:
    tracker_key: left_knee      # the server's name, verify before a session
```

Check what is being served before recording:

```bash
uv run python -m umi.common.hardware_sdk.vive.tracker \
    --endpoint tcp://<host>:5555 --tracker-key left_knee --seconds 3
```

A wrong key fails at startup and lists what is available.

## Known limits

- **~31 Hz ceiling.** `xr.wait_frame` paces the loop and poses update once per
  frame, so a client that polls faster reads the same pose more than once. The
  Vive sources in `bimanual_v4_vive.yaml` stream and record at 30 Hz, just under
  that ceiling.
- **Serving is on its own thread**, so requests are answered in ~0.02 ms and do
  not wait for a frame. It answers `{}` before the XR session comes up, which is
  what makes "server dead" distinguishable from "session not running".
