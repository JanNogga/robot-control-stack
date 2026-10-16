"""Hardware abstraction layer for the ElRobot arm.

ElRobot is NormaCore's open source 7 DoF arm, driven by Feetech ST3215 servos on one serial bus: motors 1-7
are the arm joints `rev_motor_01` to `rev_motor_07` and motor 8 is the gripper. `ElRobot` owns the bus through
the `st3215_bus` bindings, `ElRobotGripper` commands motor 8 through the same handle.

Lifecycle on hardware:

- Construction only reads. It loads the calibration file, opens the port and requires one complete read of
  all 8 motors, nothing is written to the servos.
- The first command (`set_joint_position`, `set_cartesian_position`, `move_home` or a gripper command) engages
  the arm, see `ElRobot.engage`. With the default env wrappers that is the `move_home` of `env.reset()`.
- `close()` releases the arm gently, see `ElRobot.close`.
"""

import logging
import time
import typing
from pathlib import Path

import numpy as np
import st3215_bus
from rcs.common_typing import RobotConfigKwargs
from st3215_bus.export_calibration import load_calibration

import rcs
from rcs import common

logger = logging.getLogger(__name__)

ARM_MOTORS = (1, 2, 3, 4, 5, 6, 7)
GRIPPER_MOTOR = 8
ALL_MOTORS = (*ARM_MOTORS, GRIPPER_MOTOR)

# Range of the gripper joint rev_motor_08 in assets/robots/elrobot/elrobot.xml, 0 is closed.
GRIPPER_LIMITS = (0.0, 2.2028)

# Role name of the single bus in `st3215_bus.Buses`.
BUS = "arm"

# Complete reads tried at construction and when engaging before giving up.
READ_ATTEMPTS = 5

# A bus read is reused for this long in seconds, so that one env step, which asks for the joints, the
# cartesian pose and the gripper, costs a single read.
READ_MAX_AGE = 0.005

# Rate in Hz at which `move_home` streams its ramp.
HOME_RATE = 50.0

RELEASE_STEPS = 30
# Full scale of the TorqueLimit register, restored for motors whose value could not be read before a release.
FULL_TORQUE_LIMIT = 1000

# Minimum time in seconds between two warnings about clamped joint steps.
CLAMP_WARNING_INTERVAL = 1.0

# A closing gripper whose measured width stays strictly inside this range holds an object.
GRASP_WIDTH_RANGE = (0.01, 0.99)


class ElRobotCalibration:
    """Maps encoder counts to joint angles with the per motor ranges of one arm.

    The ranges come from NormaCore's calibration, exported by `st3215-export-calibration` from the
    `st3215-bus` package. Each motor's measured travel `[range_min, range_max]` maps linearly onto the joint
    limits `[lower, upper]` of the model:

        angle = lower + (count - range_min) / (range_max - range_min) * (upper - lower)

    The gripper motor maps onto the range of `rev_motor_08` in the same way, `range_min` is the jaws shut,
    and is reported as a normalized width between 0 (closed) and 1 (open).
    """

    def __init__(self, ranges: dict[int, tuple[int, int]], joint_limits: np.ndarray, source: str = "calibration"):
        missing = [motor for motor in ALL_MOTORS if motor not in ranges]
        if missing:
            msg = f"{source}: no range for motors {missing}, an ElRobot needs motors {list(ALL_MOTORS)}."
            raise ValueError(msg)
        collapsed = [motor for motor in ALL_MOTORS if ranges[motor][0] >= ranges[motor][1]]
        if collapsed:
            msg = f"{source}: motors {collapsed} have range_min >= range_max, redo their calibration."
            raise ValueError(msg)
        joint_limits = np.asarray(joint_limits, dtype=np.float64)
        if joint_limits.shape != (2, len(ARM_MOTORS)):
            msg = f"Expected joint limits of shape (2, {len(ARM_MOTORS)}), got {joint_limits.shape}."
            raise ValueError(msg)
        self.ranges = {motor: ranges[motor] for motor in ALL_MOTORS}
        self._lower, self._upper = joint_limits
        self._count_min = np.array([ranges[motor][0] for motor in ARM_MOTORS], dtype=np.float64)
        self._count_max = np.array([ranges[motor][1] for motor in ARM_MOTORS], dtype=np.float64)

    @classmethod
    def from_file(cls, path: str | Path, joint_limits: np.ndarray) -> "ElRobotCalibration":
        path = Path(path)
        if not path.is_file():
            msg = (
                f"Calibration file {path} not found. Calibrate the arm in NormaCore and export it with "
                "`st3215-export-calibration`, see the rcs_elrobot README."
            )
            raise FileNotFoundError(msg)
        doc = load_calibration(path)
        ranges = {
            int(motor): (int(values["range_min"]), int(values["range_max"])) for motor, values in doc["motors"].items()
        }
        return cls(ranges, joint_limits, source=str(path))

    def arm_angles(self, counts: dict[int, int]) -> np.ndarray:
        raw = np.array([counts[motor] for motor in ARM_MOTORS], dtype=np.float64)
        return self._lower + (raw - self._count_min) / (self._count_max - self._count_min) * (self._upper - self._lower)

    def arm_counts(self, q: np.ndarray) -> dict[int, int]:
        """Inverse of `arm_angles`, clamped to the calibrated travel of each motor."""
        fraction = (np.asarray(q, dtype=np.float64) - self._lower) / (self._upper - self._lower)
        raw = np.clip(
            self._count_min + fraction * (self._count_max - self._count_min), self._count_min, self._count_max
        )
        return {motor: int(round(count)) for motor, count in zip(ARM_MOTORS, raw, strict=True)}

    def gripper_width(self, count: int) -> float:
        range_min, range_max = self.ranges[GRIPPER_MOTOR]
        lower, upper = GRIPPER_LIMITS
        angle = lower + (count - range_min) / (range_max - range_min) * (upper - lower)
        return float(np.clip((angle - lower) / (upper - lower), 0.0, 1.0))

    def gripper_count(self, width: float) -> int:
        range_min, range_max = self.ranges[GRIPPER_MOTOR]
        return int(round(range_min + float(np.clip(width, 0.0, 1.0)) * (range_max - range_min)))


class ElRobotConfig(common.RobotConfig):
    """Configuration of a single ElRobot arm on one serial port.

    The kinematic model, joint limits and home pose default to `rcs.ROBOTS[RobotType("ElRobot")]`.
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        id: str = "elrobot",
        calibration_dir: str = ".",
        max_joint_step: float | None = 0.05,
        home_velocity: float = 0.5,
        release_duration: float = 3.0,
        timeout_ms: int = 20,
        max_missed_reads: int = 10,
        **kwargs: typing.Unpack[RobotConfigKwargs],
    ):
        meta = rcs.ROBOTS[common.RobotType("ElRobot")]
        kwargs.setdefault("kinematic_model_path", meta.mjcf_model_path)
        kwargs.setdefault("attachment_site", meta.attachment_site)
        kwargs.setdefault("dof", meta.dof)
        kwargs.setdefault("joint_limits", meta.joint_limits)
        kwargs.setdefault("q_home", meta.q_home)
        super().__init__(**kwargs)
        self.robot_platform = common.RobotPlatform.HARDWARE
        self.robot_type = common.RobotType("ElRobot")
        self.port = port
        # The calibration is read from <calibration_dir>/<id>.json.
        self.id = id
        self.calibration_dir = calibration_dir
        # Largest change in rad of any joint per `set_joint_position` call, relative to the last commanded goal.
        # Larger steps are clamped with a warning, None disables the limit.
        self.max_joint_step = max_joint_step
        # Velocity in rad/s of the fastest joint during `move_home`.
        self.home_velocity = home_velocity
        # Duration in seconds of the torque ramp in `close`, 0 cuts the torque at once.
        self.release_duration = release_duration
        # Timeout in milliseconds of a single servo request.
        self.timeout_ms = timeout_ms
        # A motor that misses more consecutive reads than this raises an error, until then its last value is held.
        self.max_missed_reads = max_missed_reads


class ElRobot(common.Robot):
    """RCS robot backed by the ST3215 servo bus of one ElRobot arm."""

    def __init__(self, cfg: ElRobotConfig, ik: common.Kinematics):
        super().__init__()
        self._closed = True
        self._config = cfg
        self.ik = ik
        if cfg.dof != len(ARM_MOTORS):
            msg = f"ElRobot has {len(ARM_MOTORS)} arm joints, got dof={cfg.dof}."
            raise ValueError(msg)
        self._calibration = ElRobotCalibration.from_file(
            Path(cfg.calibration_dir) / f"{cfg.id}.json", np.asarray(cfg.joint_limits)
        )
        self._counts: dict[int, int] = {}
        self._missed = dict.fromkeys(ALL_MOTORS, 0)
        self._read_at = -np.inf
        self._engaged = False
        self._last_goal: np.ndarray | None = None
        self._clamped_since_warning = 0
        self._last_clamp_warning = -np.inf

        self._bus = st3215_bus.Buses({BUS: cfg.port}, timeout_ms=cfg.timeout_ms)
        self._closed = False
        try:
            self._read_complete()
        except Exception:
            self.close()
            raise
        logger.info("ElRobot connected on %s.", cfg.port)

    def __del__(self):
        self.close()

    def get_config(self) -> ElRobotConfig:
        return self._config

    def set_config(self, robot_cfg: ElRobotConfig) -> None:
        self._config = robot_cfg

    def get_state(self) -> common.RobotState:
        return common.RobotState()

    def get_ik(self) -> common.Kinematics | None:
        return self.ik

    def get_base_pose_in_world_coordinates(self) -> common.Pose:
        """A real arm has no world frame of its own, so its base is the origin."""
        return common.Pose()

    def reset(self) -> None:
        """Nothing to reset, the arm engages on its first command."""

    def automatic_error_recovery(self) -> None:
        """Called by `RobotWrapper.reset` when homing raises. The servos have no recovery routine."""

    def get_joint_position(self) -> np.ndarray[tuple[typing.Any], np.dtype[np.float64]]:
        return self._calibration.arm_angles(self._read())

    def get_gripper_width(self) -> float:
        """Normalized gripper width, 0 is closed and 1 is open."""
        return self._calibration.gripper_width(self._read()[GRIPPER_MOTOR])

    def get_cartesian_position(self) -> common.Pose:
        # `Kinematics.forward` applies the inverse of the offset it is handed, so the TCP is composed here
        # instead, to match the pose `SimRobot::get_cartesian_position` reports in simulation.
        return self.get_cartesian_flange_position() * self._config.tcp_offset

    def get_cartesian_flange_position(self) -> common.Pose:
        return self.ik.forward(self.get_joint_position(), common.Pose())

    def set_cartesian_position(self, pose: common.Pose) -> None:
        q = self.ik.inverse(pose, self.get_joint_position(), self._config.tcp_offset)
        if q is None:
            logger.warning("IK failed for %s, the command is dropped.", pose)
            return
        # The kinematic model carries the gripper and jaw joints, only the arm joints are commanded.
        self.set_joint_position(np.asarray(q, dtype=np.float64)[: len(ARM_MOTORS)])

    def set_joint_position(self, q: np.ndarray[tuple[typing.Any], np.dtype[np.float64]]) -> None:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (len(ARM_MOTORS),):
            msg = f"Expected {len(ARM_MOTORS)} joint positions, got shape {q.shape}."
            raise ValueError(msg)
        self.engage()
        assert self._last_goal is not None
        max_step = self._config.max_joint_step
        if max_step is not None:
            step = q - self._last_goal
            clamped = np.clip(step, -max_step, max_step)
            if not np.array_equal(step, clamped):
                self._warn_clamp(float(np.max(np.abs(step))))
            q = self._last_goal + clamped
        self._write_arm(q)

    def set_gripper_width(self, width: float) -> None:
        """Command the normalized gripper width, 0 is closed and 1 is open."""
        self.engage()
        self._bus.write_goals({BUS: {GRIPPER_MOTOR: self._calibration.gripper_count(width)}})

    def move_home(self) -> None:
        """Ramp all joints to `q_home` at `home_velocity` (rad/s of the fastest joint).

        Blocks until the last goal is sent.
        """
        if self._config.q_home is None:
            msg = "No home position configured."
            raise ValueError(msg)
        home = np.asarray(self._config.q_home, dtype=np.float64)
        low, high = self._config.joint_limits
        if home.shape != (len(ARM_MOTORS),) or np.any((home < low) | (home > high)):
            msg = f"Home position {home} is out of joint limits."
            raise ValueError(msg)
        if self._config.home_velocity <= 0:
            msg = f"home_velocity must be positive, got {self._config.home_velocity}."
            raise ValueError(msg)
        self.engage()
        self._read_at = -np.inf
        start = self.get_joint_position()
        distance = float(np.max(np.abs(home - start)))
        steps = max(1, int(np.ceil(distance / self._config.home_velocity * HOME_RATE)))
        logger.info("Moving to home position %s in %.1f s.", home, steps / HOME_RATE)
        for i in range(1, steps + 1):
            self._write_arm(start + (home - start) * i / steps)
            time.sleep(1.0 / HOME_RATE)

    def engage(self) -> None:
        """Enable torque, holding the present position. Called by every command, no-op once engaged.

        The goal of every motor is set to its freshly read position before torque is enabled, so the servos
        never snap to a stale goal from an earlier session.
        """
        self._assert_open()
        if self._engaged:
            return
        counts = self._read_complete()
        self._bus.write_goals({BUS: counts})
        self._bus.set_torque(True, list(ALL_MOTORS))
        self._engaged = True
        self._last_goal = self._calibration.arm_angles(counts)
        logger.info("ElRobot on %s engaged, holding the present position.", self._config.port)

    def close(self) -> None:
        """Release the arm and the serial port.

        If engaged, the arm is released gently over `release_duration`: the TorqueLimit of the arm motors is
        ramped down while their present position is re-commanded as the goal each step, so the arm sinks to
        where it rests instead of dropping. Then torque is disabled and TorqueLimit restored. The gripper is
        never ramped, it carries no load.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._engaged:
                self._release()
        finally:
            self._bus.close()

    def _release(self) -> None:
        arm = list(ARM_MOTORS)
        limits = dict.fromkeys(arm, FULL_TORQUE_LIMIT)
        steps = RELEASE_STEPS if self._config.release_duration > 0 else 0
        try:
            limits.update(self._bus.read_register("TorqueLimit", arm).get(BUS, {}))
            for i in range(steps):
                present = self._bus.read_positions(list(ALL_MOTORS)).get(BUS, {})
                self._bus.write_goals({BUS: present})
                fraction = 1.0 - (i + 1) / steps
                self._bus.write_register("TorqueLimit", {BUS: {m: int(limits[m] * fraction) for m in arm}})
                time.sleep(self._config.release_duration / steps)
        finally:
            self._bus.set_torque(False, list(ALL_MOTORS))
            self._bus.write_register("TorqueLimit", {BUS: limits})
            self._engaged = False
            self._last_goal = None
            logger.info("ElRobot on %s released.", self._config.port)

    def _assert_open(self) -> None:
        if self._closed:
            msg = f"ElRobot on {self._config.port} is closed."
            raise RuntimeError(msg)

    def _write_arm(self, q: np.ndarray) -> None:
        low, high = self._config.joint_limits
        q = np.clip(q, low, high)
        self._bus.write_goals({BUS: self._calibration.arm_counts(q)})
        self._last_goal = q

    def _warn_clamp(self, step: float) -> None:
        self._clamped_since_warning += 1
        now = time.monotonic()
        if now - self._last_clamp_warning < CLAMP_WARNING_INTERVAL:
            return
        logger.warning(
            "Joint step of %.3f rad exceeds max_joint_step=%.3f rad and was clamped (%d clamped commands since "
            "the last warning). The arm lags behind the commands, send smaller steps or raise max_joint_step.",
            step,
            self._config.max_joint_step,
            self._clamped_since_warning,
        )
        self._clamped_since_warning = 0
        self._last_clamp_warning = now

    def _read(self) -> dict[int, int]:
        """Present positions in counts. A motor that misses a read keeps its last value for a few reads."""
        self._assert_open()
        if time.monotonic() - self._read_at < READ_MAX_AGE:
            return self._counts
        got = self._bus.read_positions(list(ALL_MOTORS)).get(BUS, {})
        for motor in ALL_MOTORS:
            if motor in got:
                self._counts[motor] = got[motor]
                self._missed[motor] = 0
            else:
                self._missed[motor] += 1
        silent = [motor for motor, missed in self._missed.items() if missed > self._config.max_missed_reads]
        if silent:
            msg = (
                f"Motors {silent} on {self._config.port} did not answer {self._config.max_missed_reads} reads in a "
                "row. Check their cables and the servo power supply."
            )
            raise RuntimeError(msg)
        self._read_at = time.monotonic()
        return self._counts

    def _read_complete(self) -> dict[int, int]:
        """One read in which every motor answered."""
        got: dict[int, int] = {}
        for _ in range(READ_ATTEMPTS):
            got = self._bus.read_positions(list(ALL_MOTORS)).get(BUS, {})
            if all(motor in got for motor in ALL_MOTORS):
                self._counts = dict(got)
                self._missed = dict.fromkeys(ALL_MOTORS, 0)
                self._read_at = time.monotonic()
                return dict(got)
        missing = [motor for motor in ALL_MOTORS if motor not in got]
        msg = (
            f"Motors {missing} on {self._config.port} did not answer in {READ_ATTEMPTS} attempts. "
            "Is the servo power supply on?"
        )
        raise RuntimeError(msg)


class ElRobotGripper(common.Gripper):
    """RCS gripper sharing the servo bus of an `ElRobot` instance.

    Motor 8 is position controlled, it closes until the jaws reach the commanded width or stall on an object.
    """

    def __init__(self, cfg: common.GripperConfig, robot: ElRobot):
        super().__init__()
        self._cfg = cfg
        self._robot = robot
        self._state = common.GripperState()
        self._commanded_width: float | None = None

    def get_config(self) -> common.GripperConfig:
        return self._cfg

    def get_state(self) -> common.GripperState:
        return self._state

    def get_normalized_width(self) -> float:
        return self._robot.get_gripper_width()

    def set_normalized_width(self, width: float, force: float = 0) -> None:
        """Move the jaws to `width`, 0 is closed and 1 is open. `force` is ignored, the servo is position controlled."""
        if not (0 <= width <= 1):
            msg = f"Width must be between 0 and 1, got {width}."
            raise ValueError(msg)
        self._robot.set_gripper_width(width)
        self._commanded_width = width

    def is_grasped(self) -> bool:
        """Heuristic: a close was commanded and the jaws stopped short of closed, but not fully open.

        Mirrors the heuristic of the simulated gripper wrapper. It cannot tell an object from jaws still on
        their way, so read it once the gripper had time to close.
        """
        if self._commanded_width is None or self._commanded_width > GRASP_WIDTH_RANGE[0]:
            return False
        return GRASP_WIDTH_RANGE[0] < self.get_normalized_width() < GRASP_WIDTH_RANGE[1]

    def grasp(self) -> None:
        self.set_normalized_width(0.0)

    def open(self) -> None:
        self.set_normalized_width(1.0)

    def shut(self) -> None:
        """Same as `grasp`, the servo has no separate force mode."""
        self.set_normalized_width(0.0)

    def reset(self) -> None:
        self.open()

    def close(self) -> None:
        """The servo bus belongs to the robot, which closes it."""
