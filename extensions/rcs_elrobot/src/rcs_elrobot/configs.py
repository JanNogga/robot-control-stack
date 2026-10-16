from rcs._core.common import BaseCameraConfig, GripperConfig, GripperType, RobotType
from rcs.envs.base import ControlMode, RelativeTo
from rcs_elrobot.camera import CameraCalibration
from rcs_elrobot.creators import (
    ElRobotHardwareEnvCreatorConfig,
    HardwareCameraCreatorConfig,
    RCSElRobotConfigEnvCreator,
)
from rcs_elrobot.hw import ElRobotConfig

import rcs


class DefaultElRobotHardwareEnv(RCSElRobotConfigEnvCreator):
    port = "/dev/ttyACM0"
    # the calibration is read from <calibration_dir>/<id>.json
    id = "elrobot"
    calibration_dir = "."
    # device of the wrist camera, e.g. "/dev/video0", None runs without camera
    camera: str | None = None
    # your own calibration of the wrist camera, None reports all zero intrinsics
    camera_calibration: CameraCalibration | None = None

    def config(self) -> ElRobotHardwareEnvCreatorConfig:
        robot_type = RobotType("ElRobot")
        robot_cfg = ElRobotConfig(
            port=self.port,
            id=self.id,
            calibration_dir=self.calibration_dir,
            robot_type=robot_type,
            kinematic_model_path=rcs.ROBOTS[robot_type].mjcf_model_path,
            attachment_site=rcs.ROBOTS[robot_type].attachment_site,
            dof=rcs.ROBOTS[robot_type].dof,
            joint_limits=rcs.ROBOTS[robot_type].joint_limits,
            q_home=rcs.ROBOTS[robot_type].q_home,
            # The attachment site of the mjcf is already the grasp point between the jaws.
            tcp_offset=rcs.common.Pose(),
        )

        camera_cfgs = None
        if self.camera is not None:
            camera_cfgs = {
                "wrist": HardwareCameraCreatorConfig(
                    camera_type_id="usb",
                    camera_cfgs={
                        "wrist": BaseCameraConfig(
                            identifier=self.camera,
                            # a native 16:9 mode of the U20CAM, the same as the simulated wrist camera
                            resolution_width=1280,
                            resolution_height=720,
                            frame_rate=30,
                        )
                    },
                    calibrations={"wrist": self.camera_calibration} if self.camera_calibration is not None else {},
                )
            }

        return ElRobotHardwareEnvCreatorConfig(
            control_mode=ControlMode.JOINTS,
            robot_cfg=robot_cfg,
            gripper_cfg=GripperConfig(gripper_type=GripperType("ElRobot")),
            camera_cfgs=camera_cfgs,
            max_relative_movement=0.2,
            relative_to=RelativeTo.LAST_STEP,
            frequency=30.0,
        )
