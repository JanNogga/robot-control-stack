"""Intrinsics and extrinsics of the ElRobot wrist camera.

The extension ships no intrinsics. Without them the wrist camera reports all zero intrinsics, which are obviously
invalid. Calibrate your own camera, e.g. with OpenCV's `calibrateCamera` and its default five coefficient
distortion model, and pass the result as a `CameraCalibration`.

The extrinsics come from the arm: `WristCameraCalibration` places the camera with the joint reading of the arm and
the camera mount in the kinematic model, like the simulated wrist camera.
"""

import threading
import typing
from dataclasses import dataclass

import mujoco
import numpy as np
from rcs.camera.interface import Frame
from rcs_elrobot.hw import ElRobot

from rcs import common

# Name of the wrist camera in the ElRobot mjcf.
WRIST_CAMERA = "wrist_camera"


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole intrinsics and distortion of a camera, measured at `width` x `height` pixels.

    `camera_matrix` is the 3x3 matrix K and `dist_coeffs` the five OpenCV coefficients (k1, k2, p1, p2, k3).
    The calibration carries over to any capture resolution of the same aspect ratio, assuming the camera
    scales the full sensor image to that resolution, as the 16:9 modes of the ElRobot's U20CAM do.
    """

    width: int
    height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            msg = f"Calibration resolution must be positive, got {self.width}x{self.height}."
            raise ValueError(msg)
        if np.asarray(self.camera_matrix).shape != (3, 3):
            msg = f"camera_matrix must be 3x3, got shape {np.asarray(self.camera_matrix).shape}."
            raise ValueError(msg)
        if np.asarray(self.dist_coeffs).reshape(-1).shape != (5,):
            msg = "dist_coeffs must hold the five OpenCV coefficients (k1, k2, p1, p2, k3)."
            raise ValueError(msg)

    def intrinsics(self, width: int, height: int) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """The 3x4 intrinsics matrix [K | 0] at a capture resolution of `width` x `height` pixels."""
        if width * self.height != height * self.width:
            msg = (
                f"Capture resolution {width}x{height} has a different aspect ratio than the calibration at "
                f"{self.width}x{self.height}, calibrate the camera at the resolution you capture with."
            )
            raise ValueError(msg)
        k = np.asarray(self.camera_matrix, dtype=np.float64).copy()
        # fx, fy, cx and cy scale with the image, the bottom row stays.
        k[:2] *= width / self.width
        return np.hstack([k, np.zeros((3, 1))])

    def distortion(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """The distortion coefficients, which are independent of the resolution."""
        return np.asarray(self.dist_coeffs, dtype=np.float64).reshape(5).copy()


def camera_in_flange(model_path: str, site: str, camera: str = WRIST_CAMERA) -> common.Pose:
    """Pose of `camera` relative to `site` in an mjcf, with z along the viewing direction as in `SimCameraSet`.

    Both must be rigidly attached to the same body, then the pose does not depend on the joint positions.
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if site_id < 0 or camera_id < 0:
        msg = f"{model_path} has no site {site!r} or no camera {camera!r}, pass camera_in_flange explicitly."
        raise ValueError(msg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site_pose = common.Pose(rotation=data.site_xmat[site_id].reshape(3, 3), translation=data.site_xpos[site_id])
    camera_pose = common.Pose(rotation=data.cam_xmat[camera_id].reshape(3, 3), translation=data.cam_xpos[camera_id])
    # mujoco cameras look along -z, flip to z in front like `SimCameraSet._extrinsics`
    flip = common.Pose(rpy_vector=np.array([np.pi, 0, 0]), translation=np.zeros(3))  # type: ignore
    return site_pose.inverse() * camera_pose * flip


class WristCameraCalibration:
    """Extrinsics of the wrist camera from the arm pose, a `rcs.camera.hw.CalibrationStrategy`.

    `calibrate` takes the camera pose in the flange frame (the robot's `attachment_site`) from the kinematic
    model, or `camera_in_flange` if given, e.g. for a measured mount. `get_extrinsics` then returns world (robot
    base) to camera as a 4x4 matrix, computed from the arm's last joint reading, the same as in simulation.
    """

    def __init__(self, robot: ElRobot, camera_in_flange: common.Pose | None = None, camera: str = WRIST_CAMERA):
        self._robot = robot
        self._override = camera_in_flange
        self._camera = camera
        self._camera_in_flange: common.Pose | None = None

    def calibrate(
        self,
        samples: list[Frame],
        intrinsics: np.ndarray[tuple[typing.Literal[3], typing.Literal[4]], np.dtype[np.float64]],
        lock: threading.Lock,
    ) -> bool:
        if self._override is not None:
            self._camera_in_flange = self._override
        else:
            cfg = self._robot.get_config()
            self._camera_in_flange = camera_in_flange(cfg.kinematic_model_path, cfg.attachment_site, self._camera)
        return True

    def get_extrinsics(self) -> np.ndarray[tuple[typing.Literal[4], typing.Literal[4]], np.dtype[np.float64]] | None:
        if self._camera_in_flange is None:
            return None
        # Called by the camera's polling thread, so it uses the last reading instead of reading the bus. Calls
        # into the kinematics hold the GIL, so sharing `robot.ik` with the env thread is safe.
        flange = self._robot.ik.forward(self._robot.get_last_joint_position(), common.Pose())
        return (flange * self._camera_in_flange).inverse().pose_matrix()
