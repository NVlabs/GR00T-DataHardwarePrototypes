# UMI Camera Calibration

UMI uses two kinds of camera calibration:

- **Intrinsics and lens distortion** for each OAK camera, solved once per camera
  with `calibrate_intrinsics.py` in this folder.
- **Relative calibration between cameras**, solved by the recorder itself during
  each session, through the `camera.relative_calibration` pre-episode check.
  There is no separate tool to run for it.

## Intrinsics: `calibrate_intrinsics.py`

Capture board views from a live OAK stream. The script keeps only sharp views
that differ from the ones it already has:

```bash
uv run real_scripts/camera_calibration/calibrate_intrinsics.py capture \
    --oak-device-name <oak-device-name> --socket CAM_B --resolution 640 400 \
    --output-dir /tmp/umi_intrinsics/right_umi_camera
```

Solve from any directory of images, so the calibration can be re-tuned without
capturing again:

```bash
uv run real_scripts/camera_calibration/calibrate_intrinsics.py calibrate \
    --image-dir /tmp/umi_intrinsics/right_umi_camera \
    --model fisheye --undistort-preview
```

The board defaults match the ChArUco board in
`config/data_recording/production/bimanual_v4.yaml` (14x9 squares, 40 mm squares,
30 mm markers, `DICT_5X5_1000`). `real_scripts/data_collection/generate_charuco_board.py`
prints that board.

## Relative calibration during recording

When a calibration is due, the first `s` in `record_data.py` starts the
CALIBRATION stage: hold the board where every calibration camera can see it
until the overlay shows DONE, then press `s` again to continue. The result is
saved on every episode as its `calibration` attribute, which the LeRobot
converter reads to relate the cameras' world frames. The math lives in
`umi/common/geometry/pose_calibration.py`; the check is
`real_scripts/data_collection/pre_episode_checks/camera_calibration.py`.

The check is configured in the recording config:

```yaml
pre_episode_checks:
  - name: camera.relative_calibration
    params:
      reference_camera: right_tracking_camera
      cameras: [right_tracking_camera, left_tracking_camera]
      run_every_n_episodes: 1
      marker:
        mode: charuco
        squares_x: 14
        squares_y: 9
        square_length: 0.04       # 40mm
        marker_length: 0.03       # 30mm
        aruco_dict: "DICT_5X5_1000"
        min_markers: 24
      align_t_mm: 3.0
      align_r_deg: 0.5
      align_stable_required: 4
      loop_timeout_sec: 30.0
```

| Key                     | Meaning                                                                                  |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `reference_camera`      | Camera whose world frame the calibration is expressed in. Must be listed in `cameras`.  |
| `cameras`               | Cameras to calibrate.                                                                    |
| `run_every_n_episodes`  | Episodes between calibrations; the result is reused in between.                         |
| `align_t_mm`            | Largest translation change (mm) between updates that still counts as stable.            |
| `align_r_deg`           | Largest rotation change (deg) between updates that still counts as stable.              |
| `align_stable_required` | Consecutive stable updates each camera needs before the calibration converges.         |
| `loop_timeout_sec`      | Seconds before an unconverged calibration gives up; press `s` to retry.                 |

Marker settings for `mode: charuco`:

| Key             | Meaning                                                                            |
| --------------- | ---------------------------------------------------------------------------------- |
| `squares_x`     | Number of board squares in X.                                                      |
| `squares_y`     | Number of board squares in Y.                                                      |
| `square_length` | Physical square side length in meters.                                             |
| `marker_length` | Physical ArUco marker side length in meters. Must be smaller than `square_length`. |
| `aruco_dict`    | OpenCV dictionary name, for example `DICT_5X5_1000`.                               |
| `min_markers`   | Minimum detected markers and corners needed before pose estimation.                |

`mode: aruco` uses a single marker (`marker_size` in meters, `aruco_dict`) and
needs depth as well as RGB, so ChArUco is the better choice for iPhone cameras.

## Frame naming

The code uses this convention everywhere:

```text
T_A_B = pose of frame B expressed in frame A
x_A = T_A_B @ x_B
```

| Frame     | Meaning                                                                                                             |
| --------- | ------------------------------------------------------------------------------------------------------------------- |
| `marker`  | ArUco or ChArUco board coordinate frame.                                                                            |
| `optical` | OpenCV optical frame. +X right, +Y down, +Z forward. Marker pose estimation happens here.                           |
| `cam`     | Camera body frame reported by self-tracking systems. For iPhone, this is the ARKit camera convention.               |
| `w`       | Persistent camera world frame. For iPhone, this is the ARKit tracking world. For static cameras, `w == cam == optical`. |
| `w[ref]`  | The reference camera's world frame. Saved calibrations are expressed here.                                          |

For an iPhone, `frame.pose` is `T_w_cam`, and the fixed transform from the ARKit
camera frame to the OpenCV optical frame is `T_cam_optical = diag(1, -1, -1, 1)`,
so `T_w_optical = T_w_cam @ T_cam_optical`. `cam` and `optical` share an origin;
the transform is a 180 degree rotation around X.

## Common issues

- **Marker IDs found but no ChArUco corners.** The board settings do not match
  the printed board. Check `aruco_dict`, `squares_x`, `squares_y`,
  `square_length` and `marker_length`.
- **No progress while one camera sees the board.** An update needs the reference
  camera to detect the board too; make sure it sees enough markers.
- **Wrong poses after changing a camera's resolution.** The camera base class
  scales intrinsics when RGB is resized. A camera driver that does not use
  `_apply_resolution_override()`, or that returns stale intrinsics, gives wrong
  PnP results.
- **Noisy convergence.** Calibration uses the latest frame from each camera, not
  frames matched by timestamp, so keep the board and cameras still while it runs.
