import json
import logging
import sys
import threading
import time
import typing
from pathlib import Path

import mujoco
import numpy as np
import pytest
from rcs.camera.hw import DummyCalibrationStrategy
from rcs.camera.interface import CameraFrame, DataFrame, Frame
from rcs.envs.base import ControlMode
from rcs.envs.configs import EmptyWorldElRobot
from rcs_elrobot import creators, hw
from rcs_elrobot.camera import (
    CameraCalibration,
    WristCameraCalibration,
    camera_in_flange,
)
from rcs_elrobot.configs import DefaultElRobotHardwareEnv
from rcs_elrobot.creators import (
    HardwareCameraCreatorConfig,
    _create_usb_camera,
    _with_wrist_extrinsics,
)
from rcs_elrobot.hw import (
    ALL_MOTORS,
    ARM_MOTORS,
    GRIPPER_MOTOR,
    ElRobot,
    ElRobotCalibration,
    ElRobotConfig,
    ElRobotGripper,
)

import rcs
from rcs import common

RANGES = {motor: (1000 + 10 * motor, 3000 + 10 * motor) for motor in ARM_MOTORS} | {GRIPPER_MOTOR: (1500, 2600)}
LIMITS = rcs.ROBOTS[common.RobotType("ElRobot")].joint_limits
# the fixture below patches time.sleep, the fake camera keeps its frame rate with the real one
REAL_SLEEP = time.sleep


class FakeBuses:
    """Stands in for `st3215_bus.Buses`, records every call and never touches a serial port."""

    instances: typing.ClassVar[list["FakeBuses"]] = []

    def __init__(self, ports: dict[str, str], timeout_ms: int = 20):
        self.ports = ports
        self.timeout_ms = timeout_ms
        self.positions = {motor: (low + high) // 2 for motor, (low, high) in RANGES.items()}
        self.torque_limits = dict.fromkeys(ALL_MOTORS, 800)
        self.silent: set[int] = set()
        # motors that miss this many reads first, then answer
        self.miss_first: dict[int, int] = {}
        # goals become positions at once, like an arm that keeps up
        self.follow = True
        self.calls: list[tuple] = []
        self.closed = False
        FakeBuses.instances.append(self)

    def read_positions(self, motors=ALL_MOTORS):
        self.calls.append(("read_positions",))
        answered = {}
        for motor in motors:
            if motor in self.silent:
                continue
            if self.miss_first.get(motor, 0) > 0:
                self.miss_first[motor] -= 1
                continue
            answered[motor] = self.positions[motor]
        return {"arm": answered}

    def write_goals(self, goals):
        self.calls.append(("write_goals", dict(goals["arm"])))
        if self.follow:
            self.positions.update(goals["arm"])

    def read_register(self, register, motors):
        self.calls.append(("read_register", register))
        assert register == "TorqueLimit"
        return {"arm": {motor: self.torque_limits[motor] for motor in motors}}

    def write_register(self, register, values):
        self.calls.append(("write_register", register, dict(values["arm"])))

    def set_torque(self, enable, motors=ALL_MOTORS):
        self.calls.append(("set_torque", enable, list(motors)))

    def close(self):
        self.closed = True

    def writes(self) -> list[tuple]:
        return [call for call in self.calls if call[0] != "read_positions" and call[0] != "read_register"]


def write_calibration(directory: Path, name: str = "elrobot", ranges=RANGES) -> Path:
    path = directory / f"{name}.json"
    doc = {
        "format": "st3215-bus-calibration",
        "version": 1,
        "serial": "TEST",
        "motors": {str(m): {"range_min": low, "range_max": high} for m, (low, high) in ranges.items()},
    }
    path.write_text(json.dumps(doc))
    return path


@pytest.fixture(autouse=True)
def _fake_bus(monkeypatch):
    FakeBuses.instances = []
    monkeypatch.setattr(hw.st3215_bus, "Buses", FakeBuses)
    # every call reads the bus, so the tests see each read
    monkeypatch.setattr(hw, "READ_MAX_AGE", 0.0)
    monkeypatch.setattr(hw.time, "sleep", lambda _: None)


@pytest.fixture()
def robot_cfg(tmp_path) -> ElRobotConfig:
    write_calibration(tmp_path)
    return ElRobotConfig(port="/dev/fake", calibration_dir=str(tmp_path))


@pytest.fixture()
def pin(robot_cfg) -> common.Kinematics:
    return common.Pin(robot_cfg.kinematic_model_path, robot_cfg.attachment_site, urdf=False)


@pytest.fixture()
def robot(robot_cfg, pin):
    robot = ElRobot(robot_cfg, pin)
    yield robot
    robot.close()


def bus() -> FakeBuses:
    return FakeBuses.instances[-1]


def test_calibration_round_trip_and_clamping():
    calibration = ElRobotCalibration(RANGES, LIMITS)
    low_counts = {motor: RANGES[motor][0] for motor in ALL_MOTORS}
    high_counts = {motor: RANGES[motor][1] for motor in ALL_MOTORS}
    np.testing.assert_allclose(calibration.arm_angles(low_counts), LIMITS[0])
    np.testing.assert_allclose(calibration.arm_angles(high_counts), LIMITS[1])

    counts = {motor: RANGES[motor][0] + 123 * motor for motor in ALL_MOTORS}
    assert calibration.arm_counts(calibration.arm_angles(counts)) == {m: counts[m] for m in ARM_MOTORS}

    # targets beyond the joint limits are clamped to the calibrated travel
    assert calibration.arm_counts(LIMITS[1] + 1.0) == {m: high_counts[m] for m in ARM_MOTORS}
    assert calibration.arm_counts(LIMITS[0] - 1.0) == {m: low_counts[m] for m in ARM_MOTORS}


def test_calibration_validation(tmp_path):
    with pytest.raises(FileNotFoundError, match="st3215-export-calibration"):
        ElRobotCalibration.from_file(tmp_path / "missing.json", LIMITS)

    without_gripper = {m: r for m, r in RANGES.items() if m != GRIPPER_MOTOR}
    with pytest.raises(ValueError, match=r"no range for motors \[8\]"):
        ElRobotCalibration.from_file(write_calibration(tmp_path, "a", without_gripper), LIMITS)

    collapsed = RANGES | {3: (2000, 2000)}
    with pytest.raises(ValueError, match=r"motors \[3\] have range_min >= range_max"):
        ElRobotCalibration.from_file(write_calibration(tmp_path, "b", collapsed), LIMITS)

    (tmp_path / "c.json").write_text(json.dumps({"motors": {}}))
    with pytest.raises(ValueError, match="not an st3215-bus-calibration file"):
        ElRobotCalibration.from_file(tmp_path / "c.json", LIMITS)

    calibration = ElRobotCalibration.from_file(write_calibration(tmp_path), LIMITS)
    assert calibration.ranges == RANGES


def test_construction_only_reads(robot_cfg, pin):
    robot = ElRobot(robot_cfg, pin)
    assert bus().ports == {"arm": "/dev/fake"}
    assert bus().writes() == []
    calibration = ElRobotCalibration(RANGES, LIMITS)
    np.testing.assert_allclose(robot.get_joint_position(), calibration.arm_angles(bus().positions))
    robot.close()
    # never engaged, so closing writes nothing either
    assert bus().writes() == []
    assert bus().closed


def test_construction_retries_and_needs_every_motor(monkeypatch, robot_cfg, pin):
    monkeypatch.setattr(FakeBuses, "__init__", _init_with(miss_first={4: 2}))
    ElRobot(robot_cfg, pin).close()

    monkeypatch.setattr(FakeBuses, "__init__", _init_with(silent={8}))
    with pytest.raises(RuntimeError, match=r"Motors \[8\].*power supply"):
        ElRobot(robot_cfg, pin)
    assert bus().closed
    assert bus().writes() == []


def _init_with(silent=frozenset(), miss_first=None):
    original = FakeBuses.__init__

    def init(self, ports, timeout_ms=20):
        original(self, ports, timeout_ms)
        self.silent = set(silent)
        self.miss_first = dict(miss_first or {})

    return init


def test_engage_writes_goal_before_torque(robot):
    present = dict(bus().positions)
    robot.set_joint_position(robot.get_joint_position())
    writes = bus().writes()
    assert writes[0] == ("write_goals", present)
    assert writes[1] == ("set_torque", True, list(ALL_MOTORS))
    assert writes[2][0] == "write_goals"
    # engaging happens once
    robot.set_joint_position(robot.get_joint_position())
    assert [w[0] for w in bus().writes()].count("set_torque") == 1


def test_max_joint_step_clamps_and_warns(robot, caplog):
    q = robot.get_joint_position()
    target = q.copy()
    target[0] += 0.2
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        robot.set_joint_position(target)
        robot.set_joint_position(target)
    warnings = [r for r in caplog.records if "max_joint_step" in r.getMessage()]
    # the warning is rate limited, the clamp is not
    assert len(warnings) == 1
    np.testing.assert_allclose(robot.get_joint_position()[0], q[0] + 0.1, atol=2e-3)
    np.testing.assert_allclose(robot.get_joint_position()[1:], q[1:], atol=2e-3)

    cfg = robot.get_config()
    cfg.max_joint_step = None
    robot.set_config(cfg)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=hw.__name__):
        robot.set_joint_position(target)
    assert not caplog.records
    np.testing.assert_allclose(robot.get_joint_position(), target, atol=2e-3)


def test_hold_last_value_then_error_on_silent_motor(robot):
    held = robot.get_joint_position()
    bus().silent = {3}
    bus().positions[2] += 100
    for _ in range(robot.get_config().max_missed_reads):
        q = robot.get_joint_position()
        assert q[2] == held[2]
        assert q[1] != held[1]
    with pytest.raises(RuntimeError, match=r"Motors \[3\] .* did not answer 10 reads in a row"):
        robot.get_joint_position()

    # a motor that answers again resets its count
    bus().silent = set()
    robot.get_joint_position()
    bus().silent = {3}
    for _ in range(robot.get_config().max_missed_reads):
        robot.get_joint_position()


def test_move_home_ramps_at_bounded_velocity(robot):
    cfg = robot.get_config()
    calibration = ElRobotCalibration(RANGES, LIMITS)
    start = np.asarray(cfg.q_home) + np.array([0.5, -0.3, 0.2, 0.0, 0.4, -0.5, 1.0])
    bus().positions.update(calibration.arm_counts(start))
    robot.move_home()

    goals = [call[1] for call in bus().calls if call[0] == "write_goals"]
    # the first goal is the engage hold at the start, then the ramp to home follows
    assert set(goals[0]) == set(ALL_MOTORS)
    path = np.array([calibration.arm_angles(goal) for goal in goals])
    np.testing.assert_allclose(path[0], start, atol=2e-3)
    np.testing.assert_allclose(path[-1], cfg.q_home, atol=2e-3)
    max_step = cfg.home_velocity / hw.HOME_RATE
    assert np.max(np.abs(np.diff(path, axis=0))) <= max_step + 2e-3
    assert len(path) - 1 == int(np.ceil(np.max(np.abs(cfg.q_home - path[0])) / max_step))

    cfg.q_home = np.array(LIMITS[1]) + 0.1
    robot.set_config(cfg)
    with pytest.raises(ValueError, match="out of joint limits"):
        robot.move_home()


def test_close_releases_gently(robot):
    robot.engage()
    bus().calls.clear()
    robot.close()
    calls = bus().calls
    assert calls[0] == ("read_register", "TorqueLimit")
    ramp = [call for call in calls if call[0] == "write_register"][:-1]
    assert len(ramp) == hw.RELEASE_STEPS
    for i, call in enumerate(ramp):
        assert call[1] == "TorqueLimit"
        # the gripper is never ramped
        assert set(call[2]) == set(ARM_MOTORS)
        assert all(limit == int(800 * (1.0 - (i + 1) / hw.RELEASE_STEPS)) for limit in call[2].values())
    assert ramp[-1][2] == dict.fromkeys(ARM_MOTORS, 0)
    # every step first re-commands the present position
    goals = [i for i, call in enumerate(calls) if call[0] == "write_goals"]
    limits = [i for i, call in enumerate(calls) if call[0] == "write_register"][:-1]
    assert len(goals) == hw.RELEASE_STEPS
    assert all(g < l for g, l in zip(goals, limits, strict=True))
    assert calls[-2] == ("set_torque", False, list(ALL_MOTORS))
    assert calls[-1] == ("write_register", "TorqueLimit", dict.fromkeys(ARM_MOTORS, 800))
    assert bus().closed
    with pytest.raises(RuntimeError, match="closed"):
        robot.set_joint_position(np.zeros(7))


def test_gripper_mapping_and_is_grasped(robot):
    gripper = ElRobotGripper(common.GripperConfig(gripper_type=common.GripperType("ElRobot")), robot)
    low, high = RANGES[GRIPPER_MOTOR]
    bus().follow = False

    bus().positions[GRIPPER_MOTOR] = low
    assert gripper.get_normalized_width() == 0.0
    bus().positions[GRIPPER_MOTOR] = high
    assert gripper.get_normalized_width() == 1.0
    bus().positions[GRIPPER_MOTOR] = (low + high) // 2
    assert gripper.get_normalized_width() == pytest.approx(0.5, abs=1e-3)

    assert not gripper.is_grasped()
    gripper.open()
    assert bus().writes()[-1] == ("write_goals", {GRIPPER_MOTOR: high})
    assert not gripper.is_grasped()
    gripper.grasp()
    assert bus().writes()[-1] == ("write_goals", {GRIPPER_MOTOR: low})
    # stalled half way on an object
    assert gripper.is_grasped()
    bus().positions[GRIPPER_MOTOR] = low
    assert not gripper.is_grasped()
    gripper.set_normalized_width(0.25)
    assert bus().writes()[-1] == ("write_goals", {GRIPPER_MOTOR: low + 275})
    with pytest.raises(ValueError, match="between 0 and 1"):
        gripper.set_normalized_width(1.5)


def test_cartesian_command_slices_ik_to_arm_joints(robot):
    q = robot.get_joint_position()
    pose = robot.get_cartesian_position()
    # pinocchio's model also holds the gripper and jaw joints
    assert robot.ik.inverse(pose, q, common.Pose()).shape[0] > len(ARM_MOTORS)
    robot.set_cartesian_position(pose)
    last = bus().writes()[-1]
    assert last[0] == "write_goals"
    assert set(last[1]) == set(ARM_MOTORS)
    np.testing.assert_allclose(robot.get_joint_position(), q, atol=1e-2)


def test_camera_calibration_scales_with_resolution():
    k = np.array([[1400.0, 0.0, 960.0], [0.0, 1395.0, 540.0], [0.0, 0.0, 1.0]])
    dist = np.array([0.1, -0.2, 0.001, 0.002, 0.05])
    calibration = CameraCalibration(width=1920, height=1080, camera_matrix=k, dist_coeffs=dist)

    for width, height, scale in ((1280, 720, 2 / 3), (640, 360, 1 / 3), (1920, 1080, 1.0)):
        intrinsics = calibration.intrinsics(width, height)
        assert intrinsics.shape == (3, 4)
        np.testing.assert_allclose(intrinsics[:2, :3], k[:2] * scale)
        np.testing.assert_allclose(intrinsics[2], [0.0, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(intrinsics[:, 3], 0.0)
    np.testing.assert_array_equal(calibration.distortion(), dist)

    with pytest.raises(ValueError, match="different aspect ratio"):
        calibration.intrinsics(640, 480)
    with pytest.raises(ValueError, match="five OpenCV coefficients"):
        CameraCalibration(width=1920, height=1080, camera_matrix=k, dist_coeffs=np.zeros(14))


def test_usb_camera_needs_rcs_usb_cam(monkeypatch):
    monkeypatch.setitem(sys.modules, "rcs_usb_cam", None)
    monkeypatch.setitem(sys.modules, "rcs_usb_cam.camera", None)
    cfg = HardwareCameraCreatorConfig(
        camera_type_id="usb",
        camera_cfgs={
            "wrist": common.BaseCameraConfig(
                identifier="/dev/video0", resolution_width=1280, resolution_height=720, frame_rate=30
            )
        },
    )
    with pytest.raises(ImportError, match=r"rcs-elrobot\[camera\]"):
        _create_usb_camera(cfg)


def test_usb_camera_gets_user_calibration():
    pytest.importorskip("rcs_usb_cam")
    k = np.array([[1400.0, 0.0, 960.0], [0.0, 1400.0, 540.0], [0.0, 0.0, 1.0]])
    cfg = HardwareCameraCreatorConfig(
        camera_type_id="usb",
        camera_cfgs={
            "wrist": common.BaseCameraConfig(
                identifier="/dev/video0", resolution_width=1280, resolution_height=720, frame_rate=30
            ),
            "other": common.BaseCameraConfig(
                identifier="/dev/video2", resolution_width=1280, resolution_height=720, frame_rate=30
            ),
        },
        calibrations={"wrist": CameraCalibration(1920, 1080, k, np.ones(5))},
    )
    camera = _create_usb_camera(cfg)
    np.testing.assert_allclose(camera.config("wrist").color_intrinsics[:2, :3], k[:2] * 2 / 3)  # type: ignore
    np.testing.assert_array_equal(camera.config("wrist").distortion_coeffs, np.ones(5))  # type: ignore
    np.testing.assert_array_equal(camera.config("other").color_intrinsics, np.zeros((3, 4)))  # type: ignore


def test_env_lifecycle(tmp_path):
    write_calibration(tmp_path, "left")
    creator = DefaultElRobotHardwareEnv()
    creator.port = "/dev/fake"
    creator.id = "left"
    creator.calibration_dir = str(tmp_path)
    cfg = creator.config()
    assert cfg.control_mode == ControlMode.JOINTS
    assert cfg.camera_cfgs is None
    env = creator.create_env(cfg)
    assert bus().writes() == []

    obs, _ = env.reset()
    assert cfg.robot_cfg.q_home is not None
    np.testing.assert_allclose(obs["joints"], cfg.robot_cfg.q_home, atol=2e-3)
    assert obs["gripper"][0] == pytest.approx(1.0)

    env.close()
    assert bus().writes()[-2] == ("set_torque", False, list(ALL_MOTORS))
    assert bus().closed


def calibrated(strategy: WristCameraCalibration) -> WristCameraCalibration:
    assert strategy.calibrate([], np.zeros((3, 4)), threading.Lock())  # type: ignore[arg-type]
    return strategy


def test_wrist_extrinsics_from_last_reading(robot):
    strategy = WristCameraCalibration(robot)
    assert strategy.get_extrinsics() is None
    calibrated(strategy)

    reads = len(bus().calls)
    extrinsics = strategy.get_extrinsics()
    for _ in range(10):
        strategy.get_extrinsics()
    # the camera thread never touches the bus
    assert len(bus().calls) == reads
    assert extrinsics is not None
    assert extrinsics.shape == (4, 4)
    np.testing.assert_allclose(extrinsics[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(extrinsics[:3, :3] @ extrinsics[:3, :3].T, np.eye(3), atol=1e-12)
    assert np.linalg.det(extrinsics[:3, :3]) == pytest.approx(1.0)

    # the camera is placed from the flange, a TCP offset does not move it
    cfg = robot.get_config()
    cfg.tcp_offset = common.Pose(translation=np.array([0.0, 0.0, 0.1]))  # type: ignore
    robot.set_config(cfg)
    np.testing.assert_array_equal(strategy.get_extrinsics(), extrinsics)

    # it follows the arm once the env thread reads a new position
    bus().positions[1] += 200
    np.testing.assert_array_equal(strategy.get_extrinsics(), extrinsics)
    robot.get_joint_position()
    assert not np.allclose(strategy.get_extrinsics(), extrinsics)  # type: ignore[arg-type]


def test_wrist_extrinsics_with_measured_mount(robot):
    mount = common.Pose(rpy_vector=np.array([0.1, -0.2, 0.3]), translation=np.array([0.01, -0.05, -0.06]))  # type: ignore
    strategy = calibrated(WristCameraCalibration(robot, camera_in_flange=mount))
    flange = robot.ik.forward(robot.get_joint_position(), common.Pose())
    expected = (flange * mount).inverse().pose_matrix()
    np.testing.assert_allclose(strategy.get_extrinsics(), expected, atol=1e-12)  # type: ignore[arg-type]


def test_last_joint_position_is_cached_and_locked(robot):
    q = robot.get_joint_position()
    reads = len(bus().calls)
    bus().positions[2] += 300
    np.testing.assert_array_equal(robot.get_last_joint_position(), q)
    assert len(bus().calls) == reads
    q_new = robot.get_joint_position()
    assert q_new[1] != q[1]
    np.testing.assert_array_equal(robot.get_last_joint_position(), q_new)

    # the camera thread waits while the env thread updates the cache
    result = []
    with robot._read_lock:
        reader = threading.Thread(target=lambda: result.append(robot.get_last_joint_position()))
        reader.start()
        reader.join(timeout=0.1)
        assert reader.is_alive()
    reader.join(timeout=5.0)
    np.testing.assert_array_equal(result[0], q_new)


def test_wrist_extrinsics_match_simulation(robot):
    creator = EmptyWorldElRobot()
    cfg = creator.config()
    cfg.headless = True
    lead = creator.lead_robot_name(cfg)
    prefix = creator.robot_prefix_template.format(robot_name=lead)
    joints = [prefix + joint for joint in cfg.robot_cfgs[lead].joints]
    env = creator.create_env(cfg)
    try:
        sim = env.get_wrapper_attr("sim")
        camera_set = env.get_wrapper_attr("camera_set")
        model, data = sim.model, sim.data
        qpos = [model.jnt_qposadr[model.joint(joint).id] for joint in joints]
        mujoco.mj_forward(model, data)
        # the simulated robot base is the world origin, like the base of the real arm
        base = model.body(prefix + "base_link").id
        np.testing.assert_allclose(data.xpos[base], 0.0, atol=1e-12)
        np.testing.assert_allclose(data.xmat[base].reshape(3, 3), np.eye(3), atol=1e-12)

        strategy = calibrated(WristCameraCalibration(robot))
        calibration = ElRobotCalibration(RANGES, LIMITS)
        rng = np.random.default_rng(0)
        targets = [np.zeros(7), np.asarray(robot.get_config().q_home)] + [rng.uniform(*LIMITS) for _ in range(5)]
        for target in targets:
            bus().positions.update(calibration.arm_counts(target))
            # the angles the arm reports, quantized to encoder counts
            q = robot.get_joint_position()
            data.qpos[qpos] = q
            mujoco.mj_forward(model, data)
            expected = camera_set._extrinsics("wrist")
            np.testing.assert_allclose(strategy.get_extrinsics(), expected, atol=1e-6)  # type: ignore[arg-type]
    finally:
        env.close()


def test_creator_adds_wrist_extrinsics_unless_given(robot):
    cameras = {
        name: common.BaseCameraConfig(identifier=name, resolution_width=1280, resolution_height=720, frame_rate=30)
        for name in ("wrist", "other")
    }
    cfgs = {"usb": HardwareCameraCreatorConfig(camera_type_id="usb", camera_cfgs=cameras)}
    strategies = _with_wrist_extrinsics(cfgs, robot)["usb"].kwargs["calibration_strategy"]  # type: ignore[index]
    assert isinstance(strategies["wrist"], WristCameraCalibration)
    assert isinstance(strategies["other"], DummyCalibrationStrategy)
    # the user's config is not modified
    assert cfgs["usb"].kwargs == {}

    mine = DummyCalibrationStrategy()
    cfgs["usb"].kwargs["calibration_strategy"] = {"wrist": mine, "other": mine}
    assert _with_wrist_extrinsics(cfgs, robot)["usb"].kwargs["calibration_strategy"] == {  # type: ignore[index]
        "wrist": mine,
        "other": mine,
    }
    assert _with_wrist_extrinsics(None, robot) is None


class FakeCamera:
    """Stands in for `USBCameraSet`, uses its calibration strategies the same way."""

    def __init__(self, cfg: HardwareCameraCreatorConfig):
        self.cfg = cfg
        self.strategies = cfg.kwargs.get("calibration_strategy") or {}

    def open(self):
        pass

    def close(self):
        pass

    def config(self, camera_name: str) -> common.BaseCameraConfig:
        return self.cfg.camera_cfgs[camera_name]

    @property
    def camera_names(self) -> list[str]:
        return list(self.cfg.camera_cfgs)

    def poll_frame(self, camera_name: str) -> Frame:
        REAL_SLEEP(0.005)
        color = DataFrame(
            data=np.zeros((2, 2, 3), dtype=np.uint8), extrinsics=self.strategies[camera_name].get_extrinsics()
        )
        return Frame(camera=CameraFrame(color=color), avg_timestamp=time.time())

    def calibrate(self) -> bool:
        return all(strategy.calibrate([], np.zeros((3, 4)), threading.Lock()) for strategy in self.strategies.values())


def test_env_observes_wrist_extrinsics(monkeypatch, tmp_path):
    monkeypatch.setitem(creators.HARDWARE_CAMERA_CREATORS, "usb", FakeCamera)
    write_calibration(tmp_path)
    creator = DefaultElRobotHardwareEnv()
    creator.port = "/dev/fake"
    creator.calibration_dir = str(tmp_path)
    creator.camera = "/dev/video-fake"
    cfg = creator.config()
    env = creator.create_env(cfg)
    try:
        obs, _ = env.reset()
        # present from the first observation
        assert obs["frames"]["wrist"]["rgb"]["extrinsics"].shape == (4, 4)

        robot = env.get_wrapper_attr("robot")
        mount = camera_in_flange(cfg.robot_cfg.kinematic_model_path, cfg.robot_cfg.attachment_site)
        expected = (robot.ik.forward(robot.get_last_joint_position(), common.Pose()) * mount).inverse()
        # the arm rests after the reset, wait for a frame taken there
        REAL_SLEEP(0.1)
        frames = env.get_wrapper_attr("camera_set").get_latest_frames()
        np.testing.assert_allclose(frames.frames["wrist"].camera.color.extrinsics, expected.pose_matrix(), atol=1e-12)
    finally:
        env.close()
