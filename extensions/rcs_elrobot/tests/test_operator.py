import json
import logging
import time
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import st3215_bus
from rcs.envs.base import ControlMode, CoverWrapper, MultiRobotWrapper, RelativeTo
from rcs.envs.configs import EmptyWorldElRobot
from rcs.operator.interface import TeleopCommands, TeleopLoop
from rcs_elrobot import hw, operator
from rcs_elrobot.configs import DefaultElRobotHardwareEnv
from rcs_elrobot.hw import ALL_MOTORS, ARM_MOTORS, GRIPPER_MOTOR, ElRobotCalibration
from rcs_elrobot.operator import (
    ElRobotLeader,
    ElRobotLeaderArmConfig,
    ElRobotLeaderConfig,
    ElRobotLeaderOperator,
)

import rcs
from rcs import common

# the leader has its own calibration, different from the follower's
LEADER_RANGES = {m: (900 + 20 * m, 3100 - 20 * m) for m in ARM_MOTORS} | {GRIPPER_MOTOR: (1400, 2400)}
FOLLOWER_RANGES = {m: (1000 + 10 * m, 3000 + 10 * m) for m in ARM_MOTORS} | {GRIPPER_MOTOR: (1500, 2600)}
LIMITS = rcs.ROBOTS[common.RobotType("ElRobot")].joint_limits
Q_HOME = np.asarray(rcs.ROBOTS[common.RobotType("ElRobot")].q_home)
LEADER = ElRobotCalibration(LEADER_RANGES, LIMITS)


class FakeLeaderBus:
    """Stands in for the `st3215_bus.Buses` of a leader. Records every call, also of methods it does not have."""

    def __init__(self, ports: dict[str, str], timeout_ms: int = 20):
        self.ports = ports
        self.role = next(iter(ports))
        self.positions = {motor: (low + high) // 2 for motor, (low, high) in LEADER_RANGES.items()}
        self.silent: set[int] = set()
        self.calls: list[str] = []

    def read_positions(self, motors=ALL_MOTORS):
        self.calls.append("read_positions")
        return {self.role: {motor: self.positions[motor] for motor in motors if motor not in self.silent}}

    def close(self):
        self.calls.append("close")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*_args, **_kwargs):
            self.calls.append(name)

        return call


class FakeFollowerBus:
    """Stands in for the `st3215_bus.Buses` of a follower, goals become positions at once."""

    def __init__(self, ports: dict[str, str], timeout_ms: int = 20):
        self.role = next(iter(ports))
        self.positions = {motor: (low + high) // 2 for motor, (low, high) in FOLLOWER_RANGES.items()}

    def read_positions(self, motors=ALL_MOTORS):
        return {self.role: {motor: self.positions[motor] for motor in motors}}

    def write_goals(self, goals):
        self.positions.update(goals[self.role])

    def read_register(self, _register, motors):
        return {self.role: dict.fromkeys(motors, 1000)}

    def write_register(self, register, values):
        pass

    def set_torque(self, enable, motors=ALL_MOTORS):
        pass

    def close(self):
        pass


BUSES: list[typing.Any] = []


def fake_buses(ports: dict[str, str], timeout_ms: int = 20):
    bus = (FakeLeaderBus if operator.BUS in ports else FakeFollowerBus)(ports, timeout_ms)
    BUSES.append(bus)
    return bus


def leader_buses() -> list[FakeLeaderBus]:
    return [bus for bus in BUSES if isinstance(bus, FakeLeaderBus)]


def bus() -> FakeLeaderBus:
    return leader_buses()[-1]


def write_calibration(directory: Path, name: str, ranges) -> Path:
    path = directory / f"{name}.json"
    doc = {
        "format": "st3215-bus-calibration",
        "version": 1,
        "serial": "TEST",
        "motors": {str(m): {"range_min": low, "range_max": high} for m, (low, high) in ranges.items()},
    }
    path.write_text(json.dumps(doc))
    return path


def set_leader(leader_bus: FakeLeaderBus, q: np.ndarray, trigger: float):
    leader_bus.positions.update(LEADER.arm_counts(q))
    low, high = LEADER_RANGES[GRIPPER_MOTOR]
    leader_bus.positions[GRIPPER_MOTOR] = int(round(low + trigger * (high - low)))


def wait_for(predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.002)
    msg = "timed out"
    raise AssertionError(msg)


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    BUSES.clear()
    monkeypatch.setattr(st3215_bus, "Buses", fake_buses)
    # no keyboard listener in the tests, `_on_press` is called directly
    monkeypatch.setattr(operator, "HAS_PYNPUT", False)


@pytest.fixture()
def arm_cfg(tmp_path) -> ElRobotLeaderArmConfig:
    write_calibration(tmp_path, "leader", LEADER_RANGES)
    return ElRobotLeaderArmConfig(port="/dev/fake-leader", id="leader", calibration_dir=str(tmp_path))


@pytest.fixture()
def op(arm_cfg):
    op = ElRobotLeaderOperator(ElRobotLeaderConfig(arms={"right": arm_cfg}))
    yield op
    op.close()


def test_operator_only_reads_the_leader(op):
    assert bus().ports == {operator.BUS: "/dev/fake-leader"}
    op.start()
    wait_for(op.consume_action)
    set_leader(bus(), Q_HOME, 0.5)
    wait_for(lambda: np.allclose(op.consume_action()["right"]["joints"], Q_HOME, atol=2e-3))
    op.consume_commands()
    op.reset_operator_state()
    op.close()
    assert not op.is_alive()
    assert bus().calls.count("read_positions") > 2
    assert set(bus().calls) == {"read_positions", "close"}
    assert bus().calls[-1] == "close"


def test_joints_follow_the_leader_calibration(arm_cfg):
    leader = ElRobotLeader(arm_cfg)
    rng = np.random.default_rng(0)
    for q in [np.zeros(7), Q_HOME] + [rng.uniform(*LIMITS) for _ in range(5)]:
        set_leader(bus(), q, 0.0)
        joints, _ = leader.get_joint_and_gripper_positions()
        assert joints.shape == (7,)
        np.testing.assert_allclose(joints, LEADER.arm_angles(bus().positions))
        np.testing.assert_allclose(joints, q, atol=2e-3)


def read_leader(arm_cfg: ElRobotLeaderArmConfig, q: np.ndarray, trigger: float = 0.0) -> tuple[np.ndarray, float]:
    leader = ElRobotLeader(arm_cfg)
    set_leader(bus(), q, trigger)
    return leader.get_joint_and_gripper_positions()


def test_wrist_offset(arm_cfg):
    assert arm_cfg.wrist_offset == 0.0
    q = Q_HOME.copy()
    q[6] = 0.4
    plain, _ = read_leader(arm_cfg, q)
    np.testing.assert_allclose(plain[6], 0.4, atol=2e-3)

    arm_cfg.wrist_offset = np.pi / 2
    shifted, _ = read_leader(arm_cfg, q)
    np.testing.assert_allclose(shifted[6], plain[6] + np.pi / 2)
    # only joint 7 moves
    np.testing.assert_array_equal(shifted[:6], plain[:6])

    # the result stays inside the follower's joint 7 limits
    q[6] = LIMITS[1][6] - 0.1
    assert read_leader(arm_cfg, q)[0][6] == LIMITS[1][6]
    arm_cfg.wrist_offset = -np.pi / 2
    q[6] = LIMITS[0][6] + 0.1
    assert read_leader(arm_cfg, q)[0][6] == LIMITS[0][6]


@pytest.mark.parametrize(
    ("full_open", "trigger", "expected"),
    [
        (None, 0.0, 0.0),
        (None, 0.4, 0.4),
        (None, 1.0, 1.0),
        (0.5, 0.2, 0.4),
        (0.5, 0.5, 1.0),
        (0.5, 0.8, 1.0),
        (2.0, 0.6, 0.3),
        (2.0, 1.0, 0.5),
    ],
)
def test_trigger_full_open(arm_cfg, full_open, trigger, expected):
    if full_open is None:
        assert arm_cfg.trigger_full_open == 1.0
    else:
        arm_cfg.trigger_full_open = full_open
    _, gripper = read_leader(arm_cfg, Q_HOME, trigger)
    assert gripper == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize("full_open", [0.0, -0.5])
def test_trigger_full_open_must_be_positive(arm_cfg, full_open):
    arm_cfg.trigger_full_open = full_open
    with pytest.raises(ValueError, match="trigger_full_open must be greater than 0"):
        ElRobotLeader(arm_cfg)


def test_missed_reads_hold_last_value_then_raise(arm_cfg):
    leader = ElRobotLeader(arm_cfg)
    held, _ = leader.get_joint_and_gripper_positions()
    bus().silent = {3}
    bus().positions[2] += 100
    for _ in range(arm_cfg.max_missed_reads):
        joints, _ = leader.get_joint_and_gripper_positions()
        assert joints[2] == held[2]
        assert joints[1] != held[1]
    with pytest.raises(RuntimeError, match=r"Motors \[3\] of the leader on /dev/fake-leader did not answer 10 reads"):
        leader.get_joint_and_gripper_positions()

    # a motor that answers again resets its count
    bus().silent = set()
    leader.get_joint_and_gripper_positions()
    bus().silent = {3}
    for _ in range(arm_cfg.max_missed_reads):
        leader.get_joint_and_gripper_positions()


def test_construction_needs_calibration_and_every_motor(monkeypatch, arm_cfg):
    with pytest.raises(FileNotFoundError, match="st3215-export-calibration"):
        ElRobotLeader(ElRobotLeaderArmConfig(id="missing", calibration_dir=arm_cfg.calibration_dir))

    original = FakeLeaderBus.__init__

    def silent_gripper(self, ports, timeout_ms=20):
        original(self, ports, timeout_ms)
        self.silent = {GRIPPER_MOTOR}

    monkeypatch.setattr(FakeLeaderBus, "__init__", silent_gripper)
    with pytest.raises(RuntimeError, match=r"Motors \[8\] of the leader.*power supply"):
        ElRobotLeaderOperator(ElRobotLeaderConfig(arms={"right": arm_cfg}))
    assert bus().calls[-1] == "close"
    assert set(bus().calls) == {"read_positions", "close"}


def test_silent_motor_is_logged_by_the_read_thread(op, caplog):
    op.start()
    wait_for(op.consume_action)
    held = op.consume_action()["right"]["joints"]
    bus().silent = {7}
    bus().positions[1] += 200
    with caplog.at_level(logging.WARNING, logger=operator.__name__):
        wait_for(lambda: any("Motors [7] of the leader" in r.getMessage() for r in caplog.records))
    assert "Error reading ElRobot leader right" in caplog.records[-1].getMessage()
    # the last good reading is kept
    np.testing.assert_array_equal(op.consume_action()["right"]["joints"][1:], held[1:])
    assert op.is_alive()


def test_consume_action_format(tmp_path, arm_cfg):
    write_calibration(tmp_path, "left_leader", LEADER_RANGES)
    left = ElRobotLeaderArmConfig(port="/dev/fake-left", id="left_leader", calibration_dir=str(tmp_path))
    op = ElRobotLeaderOperator(ElRobotLeaderConfig(arms={"right": arm_cfg, "left": left}))
    try:
        assert op.controller_names == ["right", "left"]
        assert op.control_mode == (ControlMode.JOINTS, RelativeTo.NONE)
        # nothing until the leaders were read
        assert op.consume_action() == {}
        right, left_bus = leader_buses()
        set_leader(right, Q_HOME, 0.25)
        set_leader(left_bus, np.zeros(7), 1.0)
        op.start()
        wait_for(lambda: len(op.consume_action()) == 2)
        actions = op.consume_action()
        assert set(actions) == {"right", "left"}
        np.testing.assert_allclose(actions["right"]["joints"], Q_HOME, atol=2e-3)
        for action in actions.values():
            assert set(action) == {"joints", "gripper"}
            assert action["joints"].shape == (7,)
            assert action["gripper"].shape == (1,)
        assert actions["right"]["gripper"][0] == pytest.approx(0.25, abs=2e-3)
        np.testing.assert_allclose(actions["left"]["joints"], np.zeros(7), atol=2e-3)
        assert actions["left"]["gripper"][0] == pytest.approx(1.0)
        # a copy, not the operator's state
        actions["right"]["joints"][:] = 0.0
        assert not np.allclose(op.consume_action()["right"]["joints"], 0.0)
    finally:
        op.close()


def test_keyboard_commands(op):
    assert op.consume_commands() == TeleopCommands()
    op._on_press(SimpleNamespace(char="s"))
    cmds = op.consume_commands()
    assert cmds.sync_position
    assert not cmds.failure
    # edge triggered
    assert op.consume_commands() == TeleopCommands()
    op._on_press(SimpleNamespace(char="r"))
    assert op.consume_commands() == TeleopCommands(failure=True)
    # other keys and special keys without a character do nothing
    op._on_press(SimpleNamespace(char="x"))
    op._on_press(SimpleNamespace(char=None))
    op._on_press(object())
    assert op.consume_commands() == TeleopCommands()


def test_keyboard_commands_with_pynput(op):
    keyboard = pytest.importorskip("pynput.keyboard")
    op._on_press(keyboard.KeyCode.from_char("s"))
    op._on_press(keyboard.Key.space)
    assert op.consume_commands() == TeleopCommands(sync_position=True)


def test_without_pynput_keyboard_is_disabled(arm_cfg, caplog):
    with caplog.at_level(logging.WARNING, logger=operator.__name__):
        op = ElRobotLeaderOperator(ElRobotLeaderConfig(arms={"right": arm_cfg}))
    op.close()
    assert "Keyboard triggers disabled" in caplog.text


def drive(monkeypatch, tele: TeleopLoop, steps: int, on_step: typing.Callable[[int], None]):
    """Run `environment_step_loop`: sync first, then `steps` operator steps, calling `on_step` before each."""
    consume_commands = tele.operator.consume_commands
    count = 0

    def commands() -> TeleopCommands:
        nonlocal count
        count += 1
        cmds = consume_commands()
        if count == 1:
            cmds.sync_position = True
        else:
            on_step(count - 2)
        if count > steps:
            tele._exit_requested = True
        return cmds

    monkeypatch.setattr(tele.operator, "consume_commands", commands)
    tele.environment_step_loop()


def leader_pose(offset: float) -> np.ndarray:
    q = Q_HOME.copy()
    q[[0, 1, 3, 6]] += np.array([0.3, -0.2, 0.25, 0.5]) + offset
    return q


def follow_leader(monkeypatch, env, op: ElRobotLeaderOperator, atol: float):
    """Sync the follower to the leader, move the leader and check that the follower tracks it."""
    leader_bus = leader_buses()[0]
    set_leader(leader_bus, leader_pose(0.0), 0.3)
    seen = []

    def on_step(i):
        seen.append(tele._last_obs["right"])
        if i == 0:
            set_leader(leader_bus, leader_pose(0.04), 0.6)
            wait_for(lambda: np.allclose(op.consume_action()["right"]["joints"], leader_pose(0.04), atol=2e-3))

    tele = TeleopLoop(env, op, env_frequency=10)
    with tele:
        wait_for(op.consume_action)
        drive(monkeypatch, tele, 5, on_step)
    assert not op.is_alive()

    # right after the sync
    np.testing.assert_allclose(seen[0]["joints"], leader_pose(0.0), atol=atol)
    assert seen[0]["gripper"][0] == pytest.approx(0.3, abs=0.05)
    # tracking the leader
    np.testing.assert_allclose(tele._last_obs["right"]["joints"], leader_pose(0.04), atol=atol)
    assert tele._last_obs["right"]["gripper"][0] == pytest.approx(0.6, abs=0.05)
    assert set(leader_bus.calls) == {"read_positions", "close"}


def test_teleop_loop_in_simulation(monkeypatch, arm_cfg):
    creator = EmptyWorldElRobot()
    cfg = creator.config()
    cfg.headless = True
    cfg.sim_cfg.realtime = False
    cfg.camera_cfgs = None
    cfg.control_mode = ElRobotLeaderOperator.control_mode[0]
    cfg.relative_to = ElRobotLeaderOperator.control_mode[1]
    cfg.wrapper_cfg.binary_gripper = False
    env = creator.create_env(cfg)
    op = ElRobotLeaderOperator(ElRobotLeaderConfig(arms={"right": arm_cfg}), env.get_wrapper_attr("sim"))
    with env:
        follow_leader(monkeypatch, env, op, atol=1e-2)


def test_teleop_loop_on_hardware(monkeypatch, tmp_path, arm_cfg):
    write_calibration(tmp_path, "follower", FOLLOWER_RANGES)
    creator = DefaultElRobotHardwareEnv()
    creator.port = "/dev/fake-follower"
    creator.id = "follower"
    creator.calibration_dir = str(tmp_path)
    cfg = creator.config()
    cfg.relative_to = ElRobotLeaderOperator.control_mode[1]
    cfg.wrapper_cfg.binary_gripper = False
    # fast fakes, no need to wait in the test, and every step reads the bus
    cfg.frequency = None
    monkeypatch.setattr(hw, "READ_MAX_AGE", 0.0)
    cfg.robot_cfg.home_velocity = 100.0
    cfg.robot_cfg.release_duration = 0.0
    # TeleopLoop addresses the robots by name, as in the README
    env = CoverWrapper(MultiRobotWrapper({"right": creator.create_env(cfg)}))
    op = ElRobotLeaderOperator(ElRobotLeaderConfig(arms={"right": arm_cfg}, simulation=False))
    with env:
        # quantized to encoder counts on both arms
        follow_leader(monkeypatch, env, op, atol=5e-3)
