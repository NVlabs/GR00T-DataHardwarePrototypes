# UMI Software Guide

How to set up the station and record demonstrations with the GEAR_UMI Station Console. This guide covers the two supported gripper setups: **Bimanual (iPhone)** and **Bimanual (Vive)**.

**Contents**

1. [Environment set up](#environment-set-up)
2. [Tracking set up](#tracking-set-up)
   - [iPhones](#iphones) · [Vive trackers](#vive-trackers)
3. [Data collection](#data-collection)
   - [Choose the setup](#choose-the-setup) · [Device assignment](#device-assignment) · [Session details](#session-details) · [Calibrate](#calibrate) · [Record data](#record-data) · [Data check](#data-check)

---

## Environment set up

Use an **Ubuntu 24.04 x86_64** station with a local desktop session, **Python 3.12**, and [uv](https://docs.astral.sh/uv/) installed. The recorder opens a desktop preview window, and gripper calibration opens a terminal. Connect two assembled grippers with their wrist OAK cameras, USB trigger cameras, and the tracking devices for your setup.

> **Tested platforms:** the UMI data-collection software has been tested on **Ubuntu 24.04 x86_64** and **Ubuntu 22.04.5 LTS**.

### 1. Set up the station once

Install the system packages:

```bash
sudo apt update
sudo apt install -y git git-lfs curl ffmpeg v4l-utils gnome-terminal
```

From your checkout, run:

```bash
cd <repository>
git lfs install
git lfs pull
bash UMIverse_software/umi/data_collection_scripts/scripts/setup_station.sh
```

Run the setup script as yourself, **without `sudo`**. It installs the shared UMI/DexUMI Python environment, offers to configure gripper serial access, and installs command shortcuts. Accept the serial-access steps; they request `sudo` when needed. **If you were added to `dialout`, log out and back in.** Opening a new terminal is not enough.

The OAK cameras also need a USB permission rule, which the script does not install:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Replug the cameras if they still report a permission error.

### 2. Launch the UMI console

Run from the repository root:

```bash
cd <repository>
uv sync
uv run umi_dc
```

> **Note:** run `uv sync` at the repository root, not inside the UMI folder, so the shared environment keeps both pipelines' dependencies.

The console opens at [http://127.0.0.1:8770/](http://127.0.0.1:8770/). Keep the server terminal running. `uv run umi_dc --no-open` skips opening the browser; `uv run umi_dc --help` lists the options. After station setup, `umi_dc` is also available through the shortcut in `~/.local/bin`.

---

## Tracking set up

Follow the subsection for your rig before assigning devices in the console.

### iPhones

Each hand's iPhone streams its camera to the station through Record3D over USB. Complete these steps on **every phone**.

The Python package is installed separately by `uv sync`. The iPhone adapter
is LGPL-2.1-only; see the [attribution and replacement note](../../../LICENSES/Record3D/README.md).

#### 1. Install Record3D

Download **Record3D** from the App Store and open it.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_DexUMI/assets/software/phone/record3d-app-store.png" alt="Record3D listing in the iPhone App Store" width="300">
</p>

#### 2. Enable Guided Access

1. With Record3D open, triple-click the side button to enter Guided Access.
2. Enter the Guided Access passcode if prompted (the same passcode you use to unlock the phone).
3. Keep Record3D in the foreground while collecting data.

#### 3. Configure Record3D

1. Open the Record3D settings and set **Live RGBD Video Streaming** to **USB**.
2. Open **Export options** and enable **Export files with camera tracking data**, **Filter exported meshes**, **Per-vertex UVs in PLY meshes** and **Camera autofocus**.

<table align="center">
  <tr>
    <td align="center"><img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_DexUMI/assets/software/phone/record3d-settings.png" alt="Record3D settings screen with Live RGBD Video Streaming set to USB" width="220"><br><sub>Settings: USB streaming</sub></td>
    <td align="center"><img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_DexUMI/assets/software/phone/record3d-export-options.png" alt="Record3D export options screen with the four required options enabled" width="220"><br><sub>Export options</sub></td>
  </tr>
</table>

#### 4. Start streaming and connect

1. Select **Record** in Record3D's bottom menu and press the red circular control to start streaming.
2. Mount the phone on its gripper and connect it to the station with a USB-C cable.

<table align="center">
  <tr>
    <td align="center"><img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_DexUMI/assets/software/phone/record3d-streaming-waiting.png" alt="Record3D record screen waiting for the red streaming button to be pressed" width="220"><br><sub>Waiting for the stream</sub></td>
    <td align="center"><img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_DexUMI/assets/software/phone/record3d-streaming-started.png" alt="Record3D record screen with USB streaming started" width="220"><br><sub>Streaming started</sub></td>
  </tr>
</table>

### Vive trackers

The trackers stream poses from a separate **Windows tracker host**. The grippers and cameras connect to the Ubuntu recording station.

1. Complete the [Vive server setup](real_scripts/vive_server/VIVE_SERVER_SETUP.md) once on the Windows host.
2. Start SteamVR, set it as the active OpenXR runtime, and turn on both trackers. Confirm that both are tracking and visible to the base stations.
3. In **Manage Vive Trackers**, give each tracker a distinct role, such as `left_knee` and `right_knee`. Trackers without a role are not reported.
4. Start the Vive server on the Windows host with `vive_server.py`, run from the folder that contains it (see the setup guide above). The script is provided in this repository at `UMIverse_software/umi/data_collection_scripts/real_scripts/vive_server/vive_server.py`.
5. Note the host's IPv4 address (`ipconfig` in another terminal). Keep SteamVR and the server running during collection.

The SteamVR role is a tracker key; verify which physical gripper it belongs to during calibration.

---

## Data collection

The console has **Setup**, **Configuration**, **Calibration**, **Record**, and **Data Audit** on one scrolling page. Use the sidebar to move between sections.

### Choose the setup

In **Setup**, choose **Bimanual (iPhone)** or **Bimanual (Vive)** to match the rig, then continue to **Configuration**.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/setup/choose-setup.png" alt="Setup section of the console with the Bimanual (iPhone) and Bimanual (Vive) cards; Bimanual (iPhone) is selected" width="700"><br>
  <sub>Setup: choose Bimanual (iPhone) or Bimanual (Vive)</sub>
</p>

### Device assignment

In **Configuration**, assign the connected hardware to the correct side.

1. Press **Refresh devices** and wait for detection to finish.
2. Select each side's **Gripper**. The detected-device table groups cameras under their gripper when their USB connections can be matched.
3. Press **Identify Cameras** and wait for live previews.
4. Aim the **left gripper's** wrist OAK camera, USB trigger camera, and iPhone (if used) at the on-screen marker. Point the other gripper's cameras away. Once the marker is visible in the previews, press **Scan left**.
5. When the left cameras show **Identified**, press **Next: right** and repeat for the right gripper.
6. Press **Done** to close the dialog and release the cameras. Review every assignment before continuing.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/configuration/identify-cameras.gif" alt="Screen recording of the Identify Cameras dialog: live previews of the left gripper's cameras aimed at the on-screen marker, then the left side marked Identified" width="700"><br>
  <sub>Identify Cameras: aim one gripper's cameras at the marker, scan, then repeat for the other side</sub>
</p>

Successful scans fill the camera fields. They also fill the gripper field when USB connections identify its board unambiguously; otherwise, select the gripper manually.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/configuration/cameras-assigned.png" alt="Configuration section after identification: the detected-devices table above Left gripper and Right gripper cards with the gripper, phone, top camera and trigger camera fields filled in and Pairing verified shown" width="700"><br>
  <sub>Cameras assigned: both grippers show Pairing verified</sub>
</p>

You can assign cameras manually instead. Click a camera ID in the detected-device table to inspect its preview, then select it in the correct field. Close previews before starting another camera action.

#### Optional head and bottom cameras

New sessions start without a head or bottom camera. Add one only if it is connected and needed for this session, using **Add Head OAK**, **Add Head iPhone**, or **Add Bottom OAK**, then select its device.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/configuration/optional-cameras.png" alt="Configuration section showing the optional Head cameras and Bottom camera panels, each empty with its Add button, and the Reference camera panel set to Automatic" width="700"><br>
  <sub>Optional cameras and the reference camera</sub>
</p>

#### If a device is not found

Read the reported error, check its cable and USB permissions, and replug it if needed. For an iPhone, restart Record3D and start USB streaming again. Press **Refresh devices**, or **Refresh Devices** inside the identify dialog. Refreshing the dialog clears identification results, so scan both sides again.

#### Vive server and tracker assignment

For a Vive rig, also complete **Vive trackers** in Configuration:

1. Enter the Windows host's IP address in **Server**, for example `192.168.1.20`. For another port, use a full endpoint such as `tcp://192.168.1.20:6000`.
2. Press **Test connection** and confirm both intended tracker keys are reported as tracked.
3. Assign a different key to each gripper, using the live server list.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/configuration/vive-trackers-found.png" alt="Vive trackers panel: the server address replied in 3 ms, 3 trackers found, with left_shoulder assigned as the left tracker and right_knee as the right tracker, both marked tracked" width="700"><br>
  <sub>Vive trackers: server reachable and one tracker key assigned to each gripper</sub>
</p>


**Tip:** before testing, confirm that the recording station can reach the Windows host by running `ping <host-ip>` in a terminal on the station.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/configuration/vive-server-ping.png" alt="Terminal output of ping to the Vive server host with ten replies, each taking under 1 ms" width="700"><br>
  <sub>Ping check: the Windows host replies from the recording station</sub>
</p>

#### Reference camera

Under **Reference camera → Anchor the world frame on**, leave **Automatic** unless your collection setup specifies another camera. Review **Automatic anchors on**. This selects the camera used as the world-frame reference for board calibration and alignment.

### Session details

In **Configuration → Session**:

- **Task:** select an existing task, or type a new title and choose **Add new task**. Entering a description is optional; a blank description is recorded as `Unknown`.
- **Operator** (optional): recorded as `Unknown` when empty.
- **Session path:** check that this is the intended recording destination. Use **Start a new session** when beginning a separate session, then check the path again.

Configuration is validated and saved when you start a calibration, camera check, or recording. Editing session details or starting a new session with the same hardware keeps completed hardware checks valid; changing devices requires the relevant checks again.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/configuration/create-task.gif" alt="Screen recording of the Session panel: choosing a task, the optional operator and task description fields, and the session path" width="700"><br>
  <sub>Session details: select or add a task</sub>
</p>

#### Recording checks

Keep all checks enforced for data collection. In **Recording checks**, unchecking a check makes it **warn only**: it still reports problems, but no longer blocks recording or rejects the episode. **Enforce all** and **Warn only for all** change every check at once.

> **Warning:** Disabling recording checks can allow missing or invalid data that prevents LeRobot conversion, even when recording and alignment succeed. Keep all checks enforced for recordings intended for LeRobot.

- **Before recording:** check gripper connections and camera calibration.
- **After saving:** check frame timing, gripper readings, pose tracking, left/right pairing, video resolution, and camera/sensor motion.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/configuration/recording-checks.png" alt="Recording checks panel with all 8 checks enforced, the warning about LeRobot conversion, the Enforce all and Warn only for all buttons, and the checks grouped into Before recording starts and After saving" width="700"><br>
  <sub>Recording checks: all 8 checks enforced</sub>
</p>

### Calibrate

#### 1. Check camera assignments

In **Calibration**, complete **Camera check** if it is shown. Cameras verified through Identify Cameras are already covered; this step checks remaining manual assignments and optional head cameras.

1. Press **Start camera check** and wait for the live preview.
2. Aim the requested side's cameras at its on-screen code. Keep the other side pointed away.
3. Wait for confirmation, then follow the prompt for the next side. Show any configured head cameras their head marker too.
4. Resolve missing-camera or wrong-side errors before continuing.

This checks camera identity. The printed-board calibration happens later in the recorder.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/calibration/camera-check.png" alt="Calibration section, Camera check panel: left and right tracking, wrist and trigger cameras marked Verified, the head camera Waiting, and the Start camera check button" width="700"><br>
  <sub>Camera check: cameras verified in Identify Cameras are already marked Verified</sub>
</p>

#### 2. Check Vive trackers, if used

1. Press **Check left tracker**. Move only the physical left gripper, keeping the right still and both trackers visible to the base stations.
2. Watch the browser's movement plot until the left tracker is confirmed.
3. Press **Check right tracker** and repeat with only the right gripper moving.

If the keys appear swapped, correct them in Configuration and repeat the checks.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/calibration/vive-tracker-check.png" alt="Vive tracker check results: the left tracker confirmed after moving 123 mm, and the right tracker's movement-over-time plot with the right tracker rising past the 50 mm threshold while the left stays still, then confirmed at 224 mm" width="700"><br>
  <sub>Vive tracker check: each tracker is confirmed while only its gripper moves</sub>
</p>

#### 3. Calibrate the gripper ranges

For each gripper, press **Calibrate** under **Calibrate grippers**. A terminal opens on the station:

1. Fully close the gripper and press **Enter** at each of the three closed-position prompts.
2. Fully open it and press **Enter** when requested.
3. Open and close it repeatedly during the short recording pass (five seconds).
4. Wait for the result in the console. Complete both grippers and resolve any failures before recording.

If a measurement fails, follow the terminal's retry prompt. If the mechanism remains faulty, use its hardware-fault option and have the gripper checked.

Calibrations are saved automatically as `config/umi_gripper/left_umi_gripper_config.yaml` and `right_umi_gripper_config.yaml`, relative to this guide's folder. Use **Calibrate again** to repeat a calibration.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/calibration/calibrate-grippers.gif" alt="Screen recording of gripper calibration: a terminal on the station recording the gripper opening and closing for five seconds while the console shows the left gripper Running and the right gripper Done with its range plot" width="700"><br>
  <sub>Calibrate grippers: open and close each gripper during the five-second recording pass</sub>
</p>

#### 4. Prepare the printed board

Generate the ChArUco board once, from the UMI software folder:

```bash
cd <repository>/UMIverse_software/umi/data_collection_scripts
uv run python -m real_scripts.data_collection.generate_charuco_board --output charuco_board.png
```

Print at **100% scale**, with no fit-to-page resizing: **560 × 360 mm at 300 DPI**. Check with a ruler that each square is **40 mm**. The default board has **14 × 9 squares**, **30 mm markers**, and dictionary `DICT_5X5_1000`. Use a board that matches the recording configuration.

### Record data

In **Record**, resolve any items under **Before you start**, then press **Start recording** and confirm. This opens the recorder's live preview on the station desktop. Click that window to give it keyboard focus.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/record/start-recording.gif" alt="Screen recording of the Record section: pressing Start recording opens the recorder window, which shows the left and right wrist, trigger and tracking camera views while the board calibration converges" width="700"><br>
  <sub>Start recording: the console opens the recorder's live preview</sub>
</p>

#### Record an episode

1. Check that all camera views are live and show the correct sides. Move the grippers to confirm their readings change. Follow any displayed task instructions and placement grid.
2. Point the required calibration cameras at the printed ChArUco board and press `s` to begin board calibration. Hold the board steady and visible to all cameras requested by the calibration overlay until it shows **DONE**.
3. Press `s` to continue to the **starting pose**. Arrange the scene and take the starting pose for the task.
4. Press `s` again. The pre-recording checks run; once they pass and recording starts, perform the demonstration.
5. Press `w` to stop and save, or `a` to discard the current episode.
6. After saving, wait for **Data check PASSED** and the episode to be finalized. If it fails, read the findings, correct the problem, and press `r` to acknowledge a recoverable error.
7. Repeat for the next episode. Both shipped setups require board calibration before **every episode**.

| Key | Action |
| --- | --- |
| `s` | Advance through board calibration and the starting pose; the final press starts recording |
| `w` | Stop and save the current episode, then run the data check |
| `a` | Discard the current episode, or reset preparation |
| `g` | Change the placement-grid layout when available and not recording |
| `r` | Acknowledge a recoverable error after addressing it |
| `q` | Quit the recorder |

> **Before quitting:** press `w` and wait for the data-check verdict. Pressing `q` while recording discards that episode; pressing it while the check is pending can discard the pending episode too.

After the final verdict, press `q` and let shutdown finish; queued alignments may take time. Use `w` for episode saves; the console's **Stop recording** control ends the recorder process.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/record/record-episode.gif" alt="Screen recording of recording episodes: the recorder window with the camera views while the operator steps through board calibration, the starting pose, the demonstration, and saving the episode" width="700"><br>
  <sub>Record an episode: board calibration, starting pose, demonstration, then save</sub>
</p>

### Data check

Each saved episode is checked automatically. Passing episodes are finalized; episodes that fail an enforced check are moved to quarantine for review. Warnings alone do not reject an episode.

Open **Data Audit** to review the session's episodes, findings, and alignment status. The most recently recorded session is listed first; if it is missing, press **Refresh**. Use **Check** to inspect completeness and **Re-align** to regenerate missing or incomplete aligned copies. Episodes are aligned automatically in the background.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/data-audit/session-recorded.png" alt="Data Audit section listing the latest session with 3 episodes, Recording and All aligned badges, and the Check, Finish and close session and Convert to LeRobot buttons" width="700"><br>
  <sub>Data Audit: the latest session, 3 episodes recorded and all aligned</sub>
</p>

Closing the recorder does not close the session. If you launch recording again, new episodes are added to the same session. When the session is complete, press **Finish and close session** and confirm. A session must be closed before it can be converted to LeRobot format.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/data-audit/close-session-confirm.png" alt="Data Audit confirmation banner asking to confirm closing the session, with a Finish and close session button; the episodes stay where they are but no more can be recorded into the session" width="700"><br>
  <sub>Finish and close session: confirm to stop adding episodes</sub>
</p>

The session now shows a **Closed** badge, and the console asks you to apply a configuration before recording again.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/data-audit/session-closed.png" alt="Data Audit banner confirming the session is closed, with the session listed as All aligned and Closed and the Convert to LeRobot button now available" width="700"><br>
  <sub>Session closed: Convert to LeRobot is now available</sub>
</p>

Finally, press **Convert to LeRobot**. The status shows **Succeeded**, and the session gets a **LeRobot format** badge.

<p align="center">
  <img src="https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_UMI/assets/software/data-collection/data-audit/lerobot-converted.png" alt="Data Audit conversion status showing Succeeded with the converter output listing the three converted episodes, and the session marked All aligned, Closed and LeRobot format" width="700"><br>
  <sub>LeRobot conversion: all three episodes converted</sub>
</p>

#### Where the data is saved

The default destination is the UMI software folder's `output/` directory. The console shows the current session's exact path in Configuration and Record.

```text
UMIverse_software/umi/data_collection_scripts/output/
├── raw/<task>/<session>/          accepted episodes and session settings
├── aligned/<task>/<session>/      time-aligned copies of accepted episodes
└── quarantine/<task>/<session>/   episodes that failed enforced checks
```

Episodes are named `episode_0/`, `episode_1/`, and so on. A saved episode temporarily remains in an `episode_N.pending/` folder while its check runs. Session names include the date, grippers, tracking setup, task, and session ID. Background alignment is enabled in both shipped setups; wait for it to finish before reviewing the aligned data.

For LeRobot export, exit the recorder, use **Finish and close session** in Data Audit and confirm, then **Convert to LeRobot** on the closed session. All accepted episodes need completed alignment first.


