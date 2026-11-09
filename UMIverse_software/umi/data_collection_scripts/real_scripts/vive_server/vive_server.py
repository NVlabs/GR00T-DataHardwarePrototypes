# SPDX-FileCopyrightText: Copyright 2021 Christopher Bruns
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Original pyopenxr portions: Apache-2.0; UMI and NVIDIA modifications: Apache-2.0.
# Adapted from the Vive tracker example; see NOTICE and LICENSES/GEAR_UMI/pyopenxr.txt.

"""Vive tracker pose bridge: OpenXR -> ZMQ, for UMI data collection.

Runs on the tracker host, not on a station. It needs an OpenXR runtime and
`pyopenxr` (`import xr`), neither of which is a UMI dependency, which is why it
lives under real_scripts/ (standalone programs, nothing imports from here) and
not under umi/common/hardware_sdk/vive/, which the recorder imports at runtime.

It is kept in this repository because its wire format is a contract with
umi/common/hardware_sdk/vive/client.py. See README.md.

Serves one command, `get_vive_data`, replying with JSON keyed by tracker role::

    {"left_knee": {"position":    {"x":..,"y":..,"z":..},
                   "orientation": {"x":..,"y":..,"z":..,"w":..},
                   "tracked":     true,
                   "t":           1755130000.123}}

`tracked` is required by the client: OpenXR leaves POSITION_VALID set on a puck
that has lost tracking and signals the loss by clearing POSITION_TRACKED, so a
server testing VALID alone would republish a stale but plausible pose forever.
The client rejects a server that omits it.

Adapted from the pyopenxr vive_tracker example. Linux setup notes (the udev
symlink trick is the crucial part):
https://gist.github.com/DanielArnett/c9a56c9c7cc0def20648480bca1f6772
"""

import ctypes
import json
import sys
import threading
import time
from collections import deque
from ctypes import POINTER, byref, cast

import click
import xr
import zmq

# The headless-SteamVR clock fallback in main() reads the Windows performance
# counter, so it is only enabled on Windows.
_WINDOWS = sys.platform == "win32"


def update_vive_data(latest_vive_data, pose, role, tracked, now):
    """
    Update latest_vive_data.

    `tracked` matters most. OpenXR keeps returning a pose with
    POSITION_VALID_BIT set after tracking is lost: it is the last known pose,
    and the runtime signals the loss by clearing POSITION_TRACKED_BIT. Gating
    the write on VALID alone would rewrite the same stale pose every frame, so
    a client could not tell a frozen puck from a motionless one.

    The pose is only overwritten while it is actually tracked, but `tracked`
    and `t` are always published so a client can tell the difference.
    """
    if tracked:
        latest_vive_data[role] = {
            "position": {
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z,
            },
            "orientation": {
                "w": pose.orientation.w,
                "x": pose.orientation.x,
                "y": pose.orientation.y,
                "z": pose.orientation.z,
            },
        }
    entry = latest_vive_data.get(role)
    if entry is not None:
        entry["tracked"] = tracked
        entry["t"] = now


class SnapshotPublisher:
    """Hands the newest payload to the ZMQ thread without a lock.

    The frame loop serialises once per frame and rebinds `_message`; rebinding a
    name is atomic under the GIL, and the string handed over is immutable, so the
    server thread can never observe a half-written payload. Serialising in the
    frame loop rather than per request also keeps json.dumps off the request path.
    """

    def __init__(self):
        self._message = "{}"

    def publish(self, message: str) -> None:
        self._message = message

    def latest(self) -> str:
        return self._message


def serve_requests(port, publisher, stop_event):
    """Answer `get_vive_data` on its own thread.

    Served from the frame loop (behind a 1 ms `socket.poll`, once per frame), a
    request that missed that window would wait a whole extra frame, so
    client-visible latency would depend on phase and could double. On its own
    thread, a request is answered as soon as it arrives.

    The socket is created here because ZMQ sockets are not thread-safe: it must
    be owned by the one thread that uses it.
    """
    context_zmq = zmq.Context()
    socket = context_zmq.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://*:{port}")
    try:
        while not stop_event.is_set():
            # Handle transient ZMQ errors per request so a single failure cannot leave
            # a live pose loop with a dead publisher thread.
            try:
                if socket.poll(100):
                    socket.recv_string()
                    socket.send_string(publisher.latest())
            except zmq.ContextTerminated:
                break
            except zmq.ZMQError as exc:
                # REP is strict recv->send; if a send ever fails the socket is
                # left mid-transaction and the only recovery is a new one.
                print(f"[vive-server] socket error ({exc}); rebuilding.")
                socket.close()
                socket = context_zmq.socket(zmq.REP)
                socket.setsockopt(zmq.LINGER, 0)
                socket.bind(f"tcp://*:{port}")
    finally:
        socket.close()
        context_zmq.term()


class ContextObject(object):
    def __init__(
        self,
        instance_create_info: xr.InstanceCreateInfo = xr.InstanceCreateInfo(),
        session_create_info: xr.SessionCreateInfo = xr.SessionCreateInfo(),
        reference_space_create_info: xr.ReferenceSpaceCreateInfo = xr.ReferenceSpaceCreateInfo(),
        view_configuration_type: xr.ViewConfigurationType = xr.ViewConfigurationType.PRIMARY_STEREO,
        environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
        form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY,
    ):
        self._instance_create_info = instance_create_info
        self.instance = None
        self._session_create_info = session_create_info
        self.session = None
        self.session_state = xr.SessionState.IDLE
        self._reference_space_create_info = reference_space_create_info
        self.view_configuration_type = view_configuration_type
        self.environment_blend_mode = environment_blend_mode
        self.form_factor = form_factor
        self.graphics = None
        self.graphics_binding_pointer = None
        self.action_sets = []
        self.render_layers = []
        self.exit_render_loop = False
        self.request_restart = False  # restart requests are not handled
        self.session_is_running = False

    def __enter__(self):
        self.instance = xr.create_instance(
            create_info=self._instance_create_info,
        )
        self.system_id = xr.get_system(
            instance=self.instance,
            get_info=xr.SystemGetInfo(
                form_factor=self.form_factor,
            ),
        )

        if self._session_create_info.next is not None:
            self.graphics_binding_pointer = self._session_create_info.next

        self._session_create_info.system_id = self.system_id
        self.session = xr.create_session(
            instance=self.instance,
            create_info=self._session_create_info,
        )
        self.space = xr.create_reference_space(
            session=self.session, create_info=self._reference_space_create_info
        )
        self.default_action_set = xr.create_action_set(
            instance=self.instance,
            create_info=xr.ActionSetCreateInfo(
                action_set_name="default_action_set",
                localized_action_set_name="Default Action Set",
                priority=0,
            ),
        )
        self.action_sets.append(self.default_action_set)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.default_action_set is not None:
            xr.destroy_action_set(self.default_action_set)
            self.default_action_set = None
        if self.space is not None:
            xr.destroy_space(self.space)
            self.space = None
        if self.session is not None:
            xr.destroy_session(self.session)
            self.session = None
        if self.graphics is not None:
            self.graphics.destroy()
            self.graphics = None
        if self.instance is not None:
            xr.destroy_instance(self.instance)
            self.instance = None

    def frame_loop(self):
        xr.attach_session_action_sets(
            session=self.session,
            attach_info=xr.SessionActionSetsAttachInfo(
                count_action_sets=len(self.action_sets),
                action_sets=(xr.ActionSet * len(self.action_sets))(*self.action_sets),
            ),
        )
        while True:
            self.exit_render_loop = False
            self.poll_xr_events()
            if self.exit_render_loop:
                break
            if self.session_is_running:
                if self.session_state in (
                    xr.SessionState.READY,
                    xr.SessionState.SYNCHRONIZED,
                    xr.SessionState.VISIBLE,
                    xr.SessionState.FOCUSED,
                ):
                    frame_state = xr.wait_frame(self.session)
                    xr.begin_frame(self.session)
                    self.render_layers = []

                    yield frame_state

                    xr.end_frame(
                        self.session,
                        frame_end_info=xr.FrameEndInfo(
                            display_time=frame_state.predicted_display_time,
                            environment_blend_mode=self.environment_blend_mode,
                            layers=self.render_layers,
                        ),
                    )
            else:
                # Throttle loop since xrWaitFrame won't be called.
                time.sleep(0.250)

    def poll_xr_events(self):
        self.exit_render_loop = False
        self.request_restart = False
        while True:
            try:
                event_buffer = xr.poll_event(self.instance)
                event_type = xr.StructureType(event_buffer.type)
                if event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                    self.exit_render_loop = True
                    self.request_restart = True
                elif (
                    event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED
                    and self.session is not None
                ):
                    event = cast(
                        byref(event_buffer), POINTER(xr.EventDataSessionStateChanged)
                    ).contents
                    self.session_state = xr.SessionState(event.state)
                    if self.session_state == xr.SessionState.READY:
                        xr.begin_session(
                            session=self.session,
                            begin_info=xr.SessionBeginInfo(
                                self.view_configuration_type,
                            ),
                        )
                        self.session_is_running = True
                    elif self.session_state == xr.SessionState.STOPPING:
                        self.session_is_running = False
                        xr.end_session(self.session)
                    elif self.session_state == xr.SessionState.EXITING:
                        self.exit_render_loop = True
                        self.request_restart = False
                    elif self.session_state == xr.SessionState.LOSS_PENDING:
                        self.exit_render_loop = True
                        self.request_restart = True
                elif (
                    event_type
                    == xr.StructureType.EVENT_DATA_VIVE_TRACKER_CONNECTED_HTCX
                ):
                    vive_tracker_connected = cast(
                        byref(event_buffer),
                        POINTER(xr.EventDataViveTrackerConnectedHTCX),
                    ).contents
                    paths = vive_tracker_connected.paths.contents
                    persistent_path_str = xr.path_to_string(
                        self.instance, paths.persistent_path
                    )
                    if paths.role_path != xr.NULL_PATH:
                        role_path_str = xr.path_to_string(
                            self.instance, paths.role_path
                        )
                        print(
                            f"[vive-server] tracker connected: "
                            f"{persistent_path_str} -> role {role_path_str}"
                        )
                    else:
                        # No role bound yet: it will not appear in the payload
                        # until SteamVR assigns one.
                        print(
                            f"[vive-server] tracker connected with NO role: "
                            f"{persistent_path_str}"
                        )
                elif (
                    event_type
                    == xr.StructureType.EVENT_DATA_INTERACTION_PROFILE_CHANGED
                ):
                    pass
            except xr.EventUnavailable:
                break


@click.command()
@click.option("--port", default=5555, help="port number")
# Accept this option for launch-script compatibility. Sampling is paced by
# xr.wait_frame rather than the requested rate.
@click.option(
    "--frequency",
    default=60,
    help="UNUSED: the loop is paced by xr.wait_frame, not by this.",
)
def main(port, frequency):
    with ContextObject(
        instance_create_info=xr.InstanceCreateInfo(
            enabled_extension_names=[
                "XR_MND_headless",
                "XR_HTCX_vive_tracker_interaction",
                *(
                    [xr.KHR_WIN32_CONVERT_PERFORMANCE_COUNTER_TIME_EXTENSION_NAME]
                    if _WINDOWS
                    else []
                ),
            ],
        ),
    ) as context:
        instance = context.instance
        session = context.session

        enumerateViveTrackerPathsHTCX = cast(
            xr.get_instance_proc_addr(
                instance,
                "xrEnumerateViveTrackerPathsHTCX",
            ),
            xr.PFN_xrEnumerateViveTrackerPathsHTCX,
        )

        # SteamVR hands a headless session predicted_display_time == 0, which
        # xr.locate_space rejects with TimeInvalidError. Fall back to the
        # current XrTime, converted from the Windows performance counter.
        if _WINDOWS:
            convertWin32PerformanceCounterToTimeKHR = cast(
                xr.get_instance_proc_addr(
                    instance,
                    "xrConvertWin32PerformanceCounterToTimeKHR",
                ),
                xr.PFN_xrConvertWin32PerformanceCounterToTimeKHR,
            )

            def get_current_xr_time():
                pc_time = ctypes.c_longlong()
                ctypes.windll.kernel32.QueryPerformanceCounter(byref(pc_time))
                xr_time = xr.Time()
                result = convertWin32PerformanceCounterToTimeKHR(
                    instance, byref(pc_time), byref(xr_time)
                )
                if xr.check_result(result).is_exception():
                    raise result
                return xr_time.value

        # Create the action with subaction path
        # Role strings from
        # https://www.khronos.org/registry/OpenXR/specs/1.0/html/xrspec.html#XR_HTCX_vive_tracker_interaction
        role_strings = [
            "handheld_object",
            "left_foot",
            "right_foot",
            "left_wrist",
            "right_wrist",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "waist",
            "chest",
            "camera",
            "keyboard",
        ]
        role_path_strings = [
            f"/user/vive_tracker_htcx/role/{role}" for role in role_strings
        ]
        role_paths = (xr.Path * len(role_path_strings))(
            *[
                xr.string_to_path(instance, role_string)
                for role_string in role_path_strings
            ],
        )
        pose_action = xr.create_action(
            action_set=context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="tracker_pose",
                localized_action_name="Tracker Pose",
                count_subaction_paths=len(role_paths),
                subaction_paths=role_paths,
            ),
        )
        suggested_binding_paths = (xr.ActionSuggestedBinding * len(role_path_strings))(
            *[
                xr.ActionSuggestedBinding(
                    pose_action,
                    xr.string_to_path(instance, f"{role_path_string}/input/grip/pose"),
                )
                for role_path_string in role_path_strings
            ],
        )
        xr.suggest_interaction_profile_bindings(
            instance=instance,
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(
                    instance, "/interaction_profiles/htc/vive_tracker_htcx"
                ),
                count_suggested_bindings=len(suggested_binding_paths),
                suggested_bindings=suggested_binding_paths,
            ),
        )
        tracker_action_spaces = (xr.Space * len(role_paths))(
            *[
                xr.create_action_space(
                    session=session,
                    create_info=xr.ActionSpaceCreateInfo(
                        action=pose_action,
                        subaction_path=role_path,
                    ),
                )
                for role_path in role_paths
            ],
        )

        n_paths = ctypes.c_uint32(0)
        result = enumerateViveTrackerPathsHTCX(instance, 0, byref(n_paths), None)
        if xr.check_result(result).is_exception():
            raise result
        vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(
            *([xr.ViveTrackerPathsHTCX()] * n_paths.value)
        )
        result = enumerateViveTrackerPathsHTCX(
            instance, n_paths, byref(n_paths), vive_tracker_paths
        )
        if xr.check_result(result).is_exception():
            raise result

        session_was_focused = False  # Check for a common problem

        latest_vive_data = {}

        publisher = SnapshotPublisher()
        stop_event = threading.Event()
        server_thread = threading.Thread(
            target=serve_requests,
            args=(port, publisher, stop_event),
            daemon=True,
        )
        server_thread.start()
        print(f"Serving get_vive_data on tcp://*:{port}")

        dq = deque(maxlen=300)
        last_report_t = 0.0
        for frame_index, frame_state in enumerate(context.frame_loop()):
            if context.session_state == xr.SessionState.FOCUSED:
                session_was_focused = True
                active_action_set = xr.ActiveActionSet(
                    action_set=context.default_action_set,
                    subaction_path=xr.NULL_PATH,
                )
                xr.sync_actions(
                    session=session,
                    sync_info=xr.ActionsSyncInfo(
                        count_active_action_sets=1,
                        active_action_sets=ctypes.pointer(active_action_set),
                    ),
                )

                found_tracker_count = 0
                now = time.time()
                locate_time = frame_state.predicted_display_time
                if locate_time <= 0 and _WINDOWS:
                    locate_time = get_current_xr_time()
                for index, space in enumerate(tracker_action_spaces):
                    space_location = xr.locate_space(
                        space=space,
                        base_space=context.space,
                        time=locate_time,
                    )
                    flags = space_location.location_flags
                    # Both bits: VALID stays set on a puck that has lost
                    # tracking (the runtime hands back its last known pose), so
                    # testing VALID alone would make lost trackers look alive.
                    tracked = bool(
                        flags & xr.SPACE_LOCATION_POSITION_VALID_BIT
                        and flags & xr.SPACE_LOCATION_POSITION_TRACKED_BIT
                    )
                    update_vive_data(
                        latest_vive_data,
                        space_location.pose,
                        role_strings[index],
                        tracked,
                        now,
                    )
                    if tracked:
                        found_tracker_count += 1

                # Publish one immutable serialized snapshot per frame so requests cannot
                # observe partially updated poses.
                publisher.publish(json.dumps(latest_vive_data))

                dq.append(now)
                fps = (len(dq) - 1) / max(dq[-1] - dq[0], 1e-8)
                # Throttle terminal progress to avoid blocking pose sampling on output
                # I/O.
                if now - last_report_t >= 1.0:
                    last_report_t = now
                    print(
                        f"{found_tracker_count} trackers are found. Running fps = {fps:.2f}"
                    )

        stop_event.set()
        server_thread.join(timeout=2.0)

        if not session_was_focused:
            print(
                "This OpenXR session never entered the FOCUSED state. Did you wear the headset?"
            )


if __name__ == "__main__":
    main()
