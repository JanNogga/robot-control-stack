# RCS ElRobot Extension

Support for [ElRobot](https://github.com/norma-core/norma-core/tree/main/hardware/elrobot), NormaCore's open
source, fully 3D printed 7 DoF arm, in RCS. The servos are driven through
[`st3215-bus`](https://pypi.org/project/st3215-bus/), Python bindings built on NormaCore's ST3215 protocol code.

This extension depends on [`rcs-core`](https://pypi.org/project/rcs-core/).

## Installation

Install from PyPI, with the `camera` extra for the wrist camera:

```shell
pip install rcs-elrobot
pip install "rcs-elrobot[camera]"
```

Warning: plain `pip install rcs-elrobot` will install the published `rcs-core` dependency from PyPI.

If you want this extension to use your local RCS checkout instead of the published `rcs-core` package, first
install the main package from the repository root:

```shell
pip install -ve . --no-build-isolation
pip install -ve "extensions/rcs_elrobot[camera]"
```

## Servo calibration

RCS maps encoder counts to joint angles with the per motor ranges of NormaCore's calibration. Calibrate the arm
in NormaCore's station as usual and save the calibration there. Then, with the station running with `--tcp`,
export it with the `st3215-export-calibration` tool that ships with `st3215-bus`:

```shell
st3215-export-calibration --list                                  # buses known to the station
st3215-export-calibration --out calibration --name <SERIAL>=elrobot
```

The exporter only reads from the station. It writes one file per bus, here `calibration/elrobot.json`. RCS
loads `<calibration_dir>/<id>.json`, so this file matches `calibration_dir="calibration"` and `id="elrobot"`.
Stop the station before starting RCS, as both need the serial port.

## Serial port

Each arm has its own USB serial adapter. Your user needs access to it, on most distributions through the
`dialout` group:

```shell
sudo usermod -aG dialout $USER     # log in again afterwards
ls -l /dev/serial/by-id/           # stable device names
```

`/dev/ttyACM*` numbers can change between boots, prefer the `/dev/serial/by-id/` path or a udev rule that
gives the adapter a fixed name.

## Usage

```python
from rcs.envs.base import ControlMode
from rcs_elrobot.configs import DefaultElRobotHardwareEnv

env_creator = DefaultElRobotHardwareEnv()
env_creator.port = "/dev/serial/by-id/usb-..."
env_creator.calibration_dir = "calibration"
env_creator.id = "elrobot"

cfg = env_creator.config()
cfg.control_mode = ControlMode.JOINTS

env = env_creator.create_env(cfg)
obs, info = env.reset()
...
env.close()
```

The robot and the gripper share one servo bus, motors 1 to 7 are the arm joints and motor 8 is the gripper.
`create_env` builds the `ElRobotGripper` from the same `ElRobot` instance, and closing the env closes the bus
once.

Without the env wrappers:

```python
from rcs import common
from rcs_elrobot.hw import ElRobot, ElRobotConfig, ElRobotGripper

cfg = ElRobotConfig(port="/dev/ttyACM0", calibration_dir="calibration", id="elrobot")
ik = common.Pin(cfg.kinematic_model_path, cfg.attachment_site)
robot = ElRobot(cfg, ik)
gripper = ElRobotGripper(common.GripperConfig(gripper_type=common.GripperType("ElRobot")), robot)

robot.move_home()
gripper.grasp()
robot.close()
```

See `src/rcs_elrobot/scripts/test_robot.py` for a step by step bring-up script.

## Wrist camera

Set `DefaultElRobotHardwareEnv.camera` to the video device of the wrist camera, e.g. a path from
`/dev/v4l/by-id/`. It is read through `rcs_usb_cam` at 1280x720 and 30 fps, the same mode as the simulated
wrist camera, and shows up as `wrist` in the observations.

This extension ships no camera intrinsics. Without your own calibration the camera reports all zero intrinsics.
To use yours, pass the camera matrix and distortion coefficients of a calibration with OpenCV's
`cv2.calibrateCamera` and its default five coefficient model (k1, k2, p1, p2, k3), together with the
resolution of the calibration images:

```python
import numpy as np
from rcs_elrobot.camera import CameraCalibration

env_creator.camera = "/dev/v4l/by-id/usb-...-video-index0"
env_creator.camera_calibration = CameraCalibration(
    width=1920,
    height=1080,
    camera_matrix=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
    dist_coeffs=np.array([k1, k2, p1, p2, k3]),
)
```

A calibration carries over to a capture resolution of the same aspect ratio by scaling fx, fy, cx and cy,
which holds for cameras that scale the full sensor image, like the 16:9 modes of the ElRobot's U20CAM. A
resolution of another aspect ratio raises an error, calibrate at the resolution you capture with in that case.

### Extrinsics

The extrinsics of the wrist camera are filled in automatically, as the transform from world (the robot base) to
camera, the same as in simulation. They are computed for every frame from the latest joint reading of the arm and
the camera mount in the kinematic model, so no measurement is needed. To override them, pass your own
`rcs.camera.hw.CalibrationStrategy` for the wrist camera:

```python
cfg = env_creator.config()
cfg.camera_cfgs["wrist"].kwargs["calibration_strategy"] = {"wrist": my_strategy}
env = env_creator.create_env(cfg)
```

Without the env wrappers, `rcs_elrobot.camera.WristCameraCalibration(robot, camera_in_flange=pose)` takes the
pose of a measured mount relative to the robot's `attachment_site`.

## Leader arm teleoperation

`ElRobotLeaderOperator` drives an ElRobot from an ElRobot leader arm, a second ElRobot with a handle and a
trigger in place of the gripper. Motors 1 to 7 of the leader give the joint targets of the follower, motor 8, the
trigger, gives its gripper width. The leader's servos are only read.

Each leader has its own serial port and its own calibration, exported in the same way as the follower's, one file
per bus, e.g. `--name <LEADER_SERIAL>=leader` for `calibration/leader.json`. Two settings of
`ElRobotLeaderArmConfig` adapt the mapping:

- `wrist_offset` (rad, default `0.0`) is added to the leader's joint 7, for a handle mounted at an angle to the
  follower's gripper. The result is clipped to the joint limits.
- `trigger_full_open` (default `1.0`) is how far the trigger has to travel for the follower's gripper to be fully
  open, e.g. `0.55` opens it fully at 55% of the trigger travel.

`TeleopLoop` addresses robots by name, so the arm's env is wrapped in a `MultiRobotWrapper`, with the same key as
the leader in `arms`. The leader sends absolute joint targets, so the env runs with `RelativeTo.NONE`:

```python
from rcs.envs.base import CoverWrapper, MultiRobotWrapper, RelativeTo
from rcs.operator.interface import TeleopLoop
from rcs_elrobot.configs import DefaultElRobotHardwareEnv
from rcs_elrobot.operator import ElRobotLeaderArmConfig, ElRobotLeaderConfig, ElRobotLeaderOperator

env_creator = DefaultElRobotHardwareEnv()
env_creator.port = "/dev/serial/by-id/usb-...follower..."
env_creator.calibration_dir = "calibration"
cfg = env_creator.config()
cfg.relative_to = RelativeTo.NONE
cfg.wrapper_cfg.binary_gripper = False
env = CoverWrapper(MultiRobotWrapper({"right": env_creator.create_env(cfg)}))

operator = ElRobotLeaderOperator(
    ElRobotLeaderConfig(
        arms={
            "right": ElRobotLeaderArmConfig(
                port="/dev/serial/by-id/usb-...leader...", id="leader", calibration_dir="calibration"
            )
        },
        simulation=False,
    )
)
tele = TeleopLoop(env, operator, env_frequency=30)
with env, tele:
    tele.environment_step_loop()
```

Keys, with `pynput` installed: `s` moves the follower to the leader and starts following, `r` marks a
failure and resets the env, after which `s` syncs again.

Several leaders work the same way, one entry in `arms` per robot. The simulated `rcs/elrobot` env is already a
`MultiRobotWrapper` with the robot `right`.

## Safety notes

- Creating the robot only reads from the servos. Torque is enabled by the first command, after the goal of
  every motor has been set to its present position, so the arm holds where it is and does not snap to an old
  goal.
- `close()` releases the arm gently: over `release_duration` seconds the torque limit of the arm motors is
  ramped down while their present position is commanded, so the arm sinks to rest instead of dropping. A
  process that is killed leaves the servos holding their last goal.
- `max_joint_step` (default 0.05 rad) bounds how far a joint may move per `set_joint_position` call, measured
  from the last commanded goal. Larger steps are clamped and a warning is logged, as the arm then lags behind
  the commands. Stream commands in smaller steps, or set it to `None` to disable the limit.
- A motor that misses a read keeps its last position for up to `max_missed_reads` reads, after that the robot
  raises an error.
- The extension writes no servo settings and nothing to the servos' EEPROM, the arm runs with the settings
  from NormaCore.
- The gripper is position controlled. `grasp()` and `shut()` both close the jaws, `is_grasped()` reports a
  grasp while closing jaws stop short of closed.

## Simulation

The matching simulated env is registered as `rcs/elrobot` in core RCS. It uses the same model,
`assets/robots/elrobot/elrobot.xml`, with the same joint limits, home pose and a 1280x720 wrist camera, so
joint and cartesian targets carry over between simulation and hardware.
