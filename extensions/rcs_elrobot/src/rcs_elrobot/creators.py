import logging
import typing
from dataclasses import dataclass, field

import gymnasium as gym
from rcs._core.common import BaseCameraConfig, GripperConfig
from rcs.camera.hw import HardwareCamera, HardwareCameraSet
from rcs.envs.base import (
    CameraSetWrapper,
    ControlMode,
    CoverWrapper,
    GripperWrapper,
    HardwareEnv,
    RelativeActionSpace,
    RelativeTo,
    RobotWrapper,
)
from rcs.envs.scenes import RCSEnvCreator, WrapperConfig
from rcs_elrobot.camera import CameraCalibration
from rcs_elrobot.hw import ElRobot, ElRobotConfig, ElRobotGripper

import rcs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(kw_only=True)
class HardwareCameraCreatorConfig:
    camera_type_id: str
    camera_cfgs: dict[str, BaseCameraConfig]
    calibrations: dict[str, CameraCalibration] = field(default_factory=dict)
    """Intrinsics per camera name, cameras without an entry report all zero intrinsics."""
    kwargs: dict[str, typing.Any] = field(default_factory=dict)


def _create_usb_camera(cfg: HardwareCameraCreatorConfig) -> HardwareCamera:
    try:
        from rcs_usb_cam.camera import USBCameraConfig, USBCameraSet
    except ImportError as e:
        msg = (
            "USB camera support requires the `rcs_usb_cam` extension to be installed, "
            "e.g. with `pip install rcs-elrobot[camera]`."
        )
        raise ImportError(msg) from e

    cameras: dict[str, USBCameraConfig] = {}
    for name, camera_cfg in cfg.camera_cfgs.items():
        usb_cfg = USBCameraConfig(
            identifier=camera_cfg.identifier,
            resolution_width=camera_cfg.resolution_width,
            resolution_height=camera_cfg.resolution_height,
            frame_rate=camera_cfg.frame_rate,
        )
        calibration = cfg.calibrations.get(name)
        if calibration is not None:
            usb_cfg.color_intrinsics = calibration.intrinsics(  # type: ignore[assignment]
                camera_cfg.resolution_width, camera_cfg.resolution_height
            )
            usb_cfg.distortion_coeffs = calibration.distortion()  # type: ignore[assignment]
        cameras[name] = usb_cfg
    return typing.cast(HardwareCamera, USBCameraSet(cameras=cameras, **cfg.kwargs))


HARDWARE_CAMERA_CREATORS: dict[str, typing.Callable[[HardwareCameraCreatorConfig], HardwareCamera]] = {
    "usb": _create_usb_camera,
}


def _create_hardware_camera_set(
    camera_cfgs: dict[str, HardwareCameraCreatorConfig] | None,
) -> HardwareCameraSet | None:
    if camera_cfgs is None:
        return None
    cameras: list[HardwareCamera] = []
    for cfg in camera_cfgs.values():
        if cfg.camera_type_id not in HARDWARE_CAMERA_CREATORS:
            msg = f"Unknown hardware camera type id: {cfg.camera_type_id}"
            raise ValueError(msg)
        cameras.append(HARDWARE_CAMERA_CREATORS[cfg.camera_type_id](cfg))
    return HardwareCameraSet(cameras) if cameras else None


def _attach_camera_set(
    env: gym.Env, camera_cfgs: dict[str, HardwareCameraCreatorConfig] | None, include_depth: bool
) -> gym.Env:
    camera_set = _create_hardware_camera_set(camera_cfgs)
    if camera_set is None:
        return env
    camera_set.start()
    camera_set.wait_for_frames()
    logger.info("CameraSet started")
    return CameraSetWrapper(env, camera_set, include_depth)


@dataclass(kw_only=True)
class ElRobotHardwareEnvCreatorConfig:
    robot_cfg: ElRobotConfig
    control_mode: ControlMode
    gripper_cfg: GripperConfig | None = None
    camera_cfgs: dict[str, HardwareCameraCreatorConfig] | None = None
    max_relative_movement: float | tuple[float, float] | None = None
    relative_to: RelativeTo = RelativeTo.LAST_STEP
    frequency: float | None = 30.0
    """Control frequency in Hz, rate limits env.step(). None disables rate limiting."""
    wrapper_cfg: WrapperConfig = field(default_factory=WrapperConfig)


class RCSElRobotConfigEnvCreator(RCSEnvCreator[ElRobotHardwareEnvCreatorConfig]):
    def create_env(self, cfg: ElRobotHardwareEnvCreatorConfig) -> gym.Env:
        ik = rcs.common.Pin(
            cfg.robot_cfg.kinematic_model_path,
            cfg.robot_cfg.attachment_site,
            urdf=cfg.robot_cfg.kinematic_model_path.endswith(".urdf"),
        )
        robot = ElRobot(cfg.robot_cfg, ik)
        env: gym.Env = HardwareEnv(frequency=cfg.frequency)
        env = RobotWrapper(env, robot, cfg.control_mode, home_on_reset=cfg.wrapper_cfg.home_on_reset)

        if cfg.gripper_cfg is not None:
            # The gripper motor is on the arm's servo bus, so it shares the robot handle.
            gripper = ElRobotGripper(cfg.gripper_cfg, robot)
            env = GripperWrapper(env, gripper, binary=cfg.wrapper_cfg.binary_gripper)

        env = _attach_camera_set(env, cfg.camera_cfgs, cfg.wrapper_cfg.include_depth)

        if cfg.relative_to != RelativeTo.NONE:
            env = RelativeActionSpace(env, max_mov=cfg.max_relative_movement, relative_to=cfg.relative_to)
        return CoverWrapper(env)

    def config(self) -> ElRobotHardwareEnvCreatorConfig:
        msg = "Implement config() in a subclass or pass `cfg=` explicitly."
        raise NotImplementedError(msg)
