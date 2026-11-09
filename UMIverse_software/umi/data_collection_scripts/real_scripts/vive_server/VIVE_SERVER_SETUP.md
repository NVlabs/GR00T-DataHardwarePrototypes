# Vive server setup (Windows + SteamVR)

How to install everything `vive_server.py` needs on the tracker host, using either a
conda environment or a `uv` virtual environment.

## What the script needs

| Requirement | Version known to work | Why |
|---|---|---|
| Windows 10/11 | — | The SteamVR time fix uses the Windows performance counter |
| SteamVR, set as the active OpenXR runtime | current Steam build | Provides the trackers through OpenXR |
| Python | 3.10.18 | pyopenxr needs 3.9 or newer |
| `pyopenxr` (imported as `xr`) | 1.1.5001 | OpenXR bindings |
| `pyzmq` (imported as `zmq`) | 27.1.0 | Serves pose data to clients |
| `click` | 8.1.8 | Command-line options |
| `numpy`, `glfw`, `PyOpenGL` | 2.2.6, 2.10.0, 3.1.10 | Installed automatically with pyopenxr |

Nothing else is needed. If you reuse an environment set up for other tools, the server
ignores its extra packages (open3d, pyrealsense2, opencv-python, redis, ...).

The commands below use two placeholders; replace them with the folders on your tracker
host:

- `<server-dir>`: the folder that holds `vive_server.py` (`real_scripts\vive_server` in a
  clone of this repository, or wherever you copied the script).
- `<pyopenxr-dir>`: your local clone of pyopenxr, installed as an **editable install**.
  Installing `pyopenxr==1.1.5001` from PyPI also works if you don't have a clone.

## 1. SteamVR (both options)

1. Install SteamVR from Steam and start it once.
2. SteamVR → Settings → OpenXR → **Set SteamVR as OpenXR Runtime**.
3. Pair the trackers, then SteamVR → Manage Vive Trackers → give **every tracker a role**
   (waist, left_foot, ...). Trackers without a role are not reported.

Check the active runtime:

```powershell
Get-ItemProperty "HKLM:\SOFTWARE\Khronos\OpenXR\1" | Select-Object ActiveRuntime
# Expect: ...\steamapps\common\SteamVR\steamxr_win64.json
```

## 2a. Conda environment

```powershell
conda create -n vive_server python=3.10 -y
conda activate vive_server

# pyopenxr from the local clone (editable), plus the server's own dependencies
pip install -e <pyopenxr-dir>
pip install pyzmq==27.1.0 click==8.1.8

# No local clone? Use PyPI instead of the editable install:
# pip install pyopenxr==1.1.5001
```

## 2b. uv virtual environment

Install uv once (pick one):

```powershell
winget install --id=astral-sh.uv -e
# or
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal so `uv` is on PATH, then create the environment next to the script:

```powershell
cd <server-dir>

# uv downloads Python 3.10 itself if it isn't installed
uv venv --python 3.10 .venv

uv pip install --python .venv -e <pyopenxr-dir>
uv pip install --python .venv pyzmq==27.1.0 click==8.1.8

# No local clone? Use PyPI instead:
# uv pip install --python .venv pyopenxr==1.1.5001
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activation script, allow local scripts once with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or skip activation and call the
interpreter directly: `.\.venv\Scripts\python.exe vive_server.py`.

## 3. Verify the install

With the environment active and SteamVR running:

```powershell
python -c "import xr, zmq, click; print('pyopenxr', xr.__version__, '| pyzmq', zmq.__version__)"

python -c "import xr; e = xr.enumerate_instance_extension_properties(); [print(n, n in e) for n in ('XR_MND_headless', 'XR_HTCX_vive_tracker_interaction', 'XR_KHR_win32_convert_performance_counter_time')]"
```

All three extensions must print `True`.

Don't run these from inside `<pyopenxr-dir>\src\xr`. That folder has its own
`platform` package, which hides Python's built-in `platform` module and breaks the import
with `partially initialized module 'platform'`.

## 4. Run the server

```powershell
cd <server-dir>
python vive_server.py              # listens on tcp://*:5555
python vive_server.py --port 6000  # different port
```

Expected output, about once a second: `N trackers are found. Running fps = ...`

## Troubleshooting

| Symptom | Fix |
|---|---|
| `TimeInvalidError: The provided basetype:XrTime was zero` | The script lacks the SteamVR time fix (the `get_current_xr_time()` fallback). Run the `vive_server.py` from this repository, which includes it. |
| `0 trackers are found` | Assign a role to each tracker in SteamVR → Manage Vive Trackers; make sure they're tracking (solid green). |
| `ExtensionNotPresentError` on startup | SteamVR isn't the active OpenXR runtime (step 1), or SteamVR is outdated. |
| `ModuleNotFoundError: No module named 'xr'` | pyopenxr isn't installed in the environment you're running; activate the right env. |
| `This OpenXR session never entered the FOCUSED state` | SteamVR isn't running or another OpenXR app has focus. |
