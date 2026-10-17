"""Teleoperation of ElRobot arms with ElRobot leader arms.

A leader is an ElRobot built as an input device, with a handle and a trigger in place of the gripper. Its servos
are only read, nothing is written to them: motors 1-7 give the joint targets of the follower and motor 8, the
trigger, gives its gripper width. Each leader is mapped through its own NormaCore calibration onto the joint limits
of the model, like the follower, so both arms agree on the joint angles.
"""

import copy
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import st3215_bus
from rcs.envs.base import ControlMode, RelativeTo
from rcs.operator.interface import BaseOperator, BaseOperatorConfig, TeleopCommands
from rcs.sim.sim import Sim
from rcs.utils import SimpleFrameRate
from rcs_elrobot.hw import ALL_MOTORS, GRIPPER_MOTOR, READ_ATTEMPTS, ElRobotCalibration

import rcs
from rcs import common

try:
    from pynput import keyboard

    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

logger = logging.getLogger(__name__)

# Role name of the leader's bus in `st3215_bus.Buses`.
BUS = "leader"


@dataclass
class ElRobotLeaderArmConfig:
    port: str = "/dev/ttyACM1"
    # The calibration is read from <calibration_dir>/<id>.json.
    id: str = "elrobot_leader"
    calibration_dir: str = "."
    # Added to the leader's joint 7 in rad before it is sent to the follower.
    wrist_offset: float = 0.0
    # Trigger travel (0 to 1) at which the follower's gripper is fully open, 1.0 maps the trigger 1:1.
    trigger_full_open: float = 1.0
    # Timeout in milliseconds of a single servo request.
    timeout_ms: int = 20
    # A motor that misses more consecutive reads than this raises an error, until then its last value is held.
    max_missed_reads: int = 10


class ElRobotLeader:
    """Reads the servo bus of one leader arm, never writes to it."""

    def __init__(self, config: ElRobotLeaderArmConfig):
        if not config.trigger_full_open > 0:
            msg = f"trigger_full_open must be greater than 0, got {config.trigger_full_open}"
            raise ValueError(msg)
        self._config = config
        self._joint_limits = np.asarray(rcs.ROBOTS[common.RobotType("ElRobot")].joint_limits, dtype=np.float64)
        self._calibration = ElRobotCalibration.from_file(
            Path(config.calibration_dir) / f"{config.id}.json", self._joint_limits
        )
        self._missed = dict.fromkeys(ALL_MOTORS, 0)
        self._bus = st3215_bus.Buses({BUS: config.port}, timeout_ms=config.timeout_ms)
        try:
            self._counts = self._read_complete()
        except Exception:
            self._bus.close()
            raise

    def get_joint_and_gripper_positions(self) -> tuple[np.ndarray, float]:
        """Follower joint targets in rad and the normalized follower gripper width."""
        counts = self._read()
        joints = self._calibration.arm_angles(counts)
        joints[-1] += self._config.wrist_offset
        joints = np.clip(joints, self._joint_limits[0], self._joint_limits[1])
        trigger = self._calibration.gripper_width(counts[GRIPPER_MOTOR])
        gripper = float(np.clip(trigger / self._config.trigger_full_open, 0.0, 1.0))
        return joints, gripper

    def close(self):
        self._bus.close()

    def _read(self) -> dict[int, int]:
        """Present positions in counts. A motor that misses a read keeps its last value for a few reads."""
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
                f"Motors {silent} of the leader on {self._config.port} did not answer {self._config.max_missed_reads} "
                "reads in a row. Check their cables and the servo power supply."
            )
            raise RuntimeError(msg)
        return dict(self._counts)

    def _read_complete(self) -> dict[int, int]:
        """One read in which every motor answered."""
        got: dict[int, int] = {}
        for _ in range(READ_ATTEMPTS):
            got = self._bus.read_positions(list(ALL_MOTORS)).get(BUS, {})
            if all(motor in got for motor in ALL_MOTORS):
                return dict(got)
        missing = [motor for motor in ALL_MOTORS if motor not in got]
        msg = (
            f"Motors {missing} of the leader on {self._config.port} did not answer in {READ_ATTEMPTS} attempts. "
            "Is the servo power supply on?"
        )
        raise RuntimeError(msg)


class ElRobotLeaderOperator(BaseOperator):
    control_mode = (ControlMode.JOINTS, RelativeTo.NONE)

    def __init__(self, config: "ElRobotLeaderConfig", sim: Sim | None = None):
        super().__init__(config, sim)
        self.config: ElRobotLeaderConfig
        self._resource_lock = threading.Lock()
        self._cmd_lock = threading.Lock()

        self._exit_requested = False
        self._commands = TeleopCommands()

        self.controller_names = list(self.config.arms.keys())

        self._last_joints: dict[str, np.ndarray | None] = {name: None for name in self.controller_names}
        self._last_gripper = {name: 1.0 for name in self.controller_names}
        self._leaders: dict[str, ElRobotLeader] = {}
        try:
            for name, arm_cfg in self.config.arms.items():
                self._leaders[name] = ElRobotLeader(arm_cfg)
        except Exception:
            for leader in self._leaders.values():
                leader.close()
            raise

        if HAS_PYNPUT:
            self._listener = keyboard.Listener(on_press=self._on_press)
            self._listener.start()
        else:
            logger.warning("pynput not found. Keyboard triggers disabled.")

    def _on_press(self, key):
        try:
            if hasattr(key, "char"):
                if key.char == "s":
                    with self._cmd_lock:
                        self._commands.sync_position = True
                elif key.char == "r":
                    with self._cmd_lock:
                        self._commands.failure = True
        except AttributeError:
            pass

    def consume_commands(self) -> TeleopCommands:
        with self._cmd_lock:
            cmds = copy.copy(self._commands)
            self._commands = TeleopCommands()
            return cmds

    def reset_operator_state(self):
        # the leader is absolute, there is no internal state to reset
        pass

    def consume_action(self) -> dict[str, Any]:
        actions: dict[str, Any] = {}
        with self._resource_lock:
            for name in self.controller_names:
                joints = self._last_joints[name]
                if joints is not None:
                    actions[name] = {
                        "joints": joints.copy(),
                        "gripper": np.array([self._last_gripper[name]]),
                    }
        return actions

    def run(self):
        rate_limiter = SimpleFrameRate(self.config.read_frequency, "elrobot leader readout")

        while not self._exit_requested:
            for name, leader in self._leaders.items():
                try:
                    joints, gripper = leader.get_joint_and_gripper_positions()
                    with self._resource_lock:
                        self._last_joints[name] = joints
                        self._last_gripper[name] = gripper
                except Exception as e:
                    logger.warning("Error reading ElRobot leader %s: %s", name, e)

            rate_limiter()

    def close(self):
        self._exit_requested = True
        if HAS_PYNPUT and hasattr(self, "_listener"):
            self._listener.stop()
        if self.is_alive() and threading.current_thread() != self:
            self.join(timeout=1.0)
        for leader in self._leaders.values():
            leader.close()


@dataclass(kw_only=True)
class ElRobotLeaderConfig(BaseOperatorConfig):
    operator_class: type[BaseOperator] = field(default=ElRobotLeaderOperator)
    # One leader per robot, keyed by the name of the robot it drives: {"right": ElRobotLeaderArmConfig(...)}
    arms: dict[str, ElRobotLeaderArmConfig] = field(default_factory=lambda: {"right": ElRobotLeaderArmConfig()})
