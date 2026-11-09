# GEAR Neck control software

Bring-up tools and a simple Python interface for the GEAR Neck: three SteadyWin
GIM3505-9 actuators on GDK drive boards, on one CAN bus at 1 Mbit/s (GDK CAN
protocol V2.0, MIT variant). The [Software Guide](https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_Neck/software.html) walks
through bring-up, homing and use step by step.

## Install

From the repository root, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --package gear-neck   # only the neck software; plain `uv sync` installs every project
```

or, without uv, `pip install GEAR_Neck/files/software/neck_control`. It needs
Python 3.10 or newer, python-can and PyYAML. A pip install does not include the
settings file `config/neck.yaml`: pass a copy of it with `--config`.

On Linux with can-utils, bring the Jhoinrch RH-02 USB-CAN adapter up as
`can_neck`, the interface the commands use by default. The script needs
`slcand` from can-utils (`apt install can-utils`, `dnf install can-utils` or
`pacman -S can-utils`); run it again after every replug or reboot:

```bash
sudo GEAR_Neck/files/software/neck_control/scripts/setup_can.sh
```

Without can-utils, root or Linux (macOS, for example), add `--interface slcan`
to every command instead, after the subcommand for `neck_assign_id`
(`neck_assign_id scan --interface slcan`), and open the neck with
`Neck(interface="slcan")` in Python. It talks to the adapter's serial port
directly, found by its USB id. On Linux the serial port needs the `dialout`
group (`uucp` on Arch): `sudo usermod -aG dialout $USER`, then log in again.
`can.interface` in `config/neck.yaml` sets the interface the commands and
`Neck()` use by default.

## Settings

Everything specific to one neck is in [`config/neck.yaml`](config/neck.yaml):
each joint's ids, mounting direction, gains and limits, the top speed and the CAN
link. The file explains each setting. Every command and `Neck()` read it; a
setting that is missing, misspelled or out of range stops them with an error
that names it. For another neck, copy the file and pass the copy with
`--config my_neck.yaml` (`Neck(config="my_neck.yaml")` in Python). All three
joints use kp 2 Nm/rad and kd 0.05 Nm·s/rad.

## Commands

| Command | What it does |
| --- | --- |
| `neck_assign_id scan` | List the drives on the bus, with voltage, faults and encoder calibration |
| `neck_assign_id assign --joint yaw` | Give the one drive on the bus the yaw (or roll, pitch) ids |
| `neck_zero` | Make the pose the neck holds (Homing Fixture seated) its home position |
| `neck_angles` | Print yaw, pitch and roll in degrees, with torque off |
| `neck_move --yaw 20 --pitch 10` | Turn the neck there, then back home on Enter |

Run them from the repository root with `uv run --package gear-neck`, for example
`uv run --package gear-neck neck_angles`, or directly once `.venv` is activated.
Plain `uv run` would first install every project in the repository.

## Python

```python
from gear_neck import Neck

with Neck() as neck:                         # finds the drives; torque stays off
    print(neck.angles())                     # yaw, pitch and roll, in degrees
    neck.enable()                            # torque on, holding where it is
    neck.move_to(yaw=20, pitch=10, roll=0)   # blocks until the neck is there
    neck.set_target(yaw=-20)                 # returns at once, for streaming targets
# leaving the block switches torque off
```

Yaw turns the head right, roll tilts it right (right ear down) and pitch nods
it down. Torque goes off when the neck is closed, at program exit, and when a
drive faults or stops answering: the drives have no watchdog and otherwise hold
their last command.

## Layout

| File | Contents |
| --- | --- |
| `config/neck.yaml` | Everything specific to one neck: ids, mounting directions, gains, limits, speed, CAN link |
| `gear_neck/load_config.py` | Load and check the settings file |
| `gear_neck/gdk_protocol.py` | Pack and unpack the GDK CAN protocol, with no bus |
| `gear_neck/gdk_link.py` | Request/reply exchanges over python-can |
| `gear_neck/neck.py` | The `Neck` class and its control loop |
| `gear_neck/assign_id.py`, `zeroing.py`, `cli.py` | The commands |
| `scripts/setup_can.sh` | Bring the USB-CAN adapter up as `can_neck` |

The drives' registers and the vendor manual are in [`../../firmware/`](https://research.nvidia.com/labs/gear/hardware-prototypes/GEAR_Neck/files/firmware/README.md).

## Tests

```bash
uv run --package gear-neck pytest GEAR_Neck/files/software/neck_control
```

The tests run against a simulated drive and bus, with no hardware.
