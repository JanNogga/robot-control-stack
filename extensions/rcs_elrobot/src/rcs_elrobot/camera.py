"""Intrinsics of the ElRobot wrist camera, supplied by the user.

The extension ships no camera calibration. Without one the wrist camera reports all zero intrinsics, which are
obviously invalid. Calibrate your own camera, e.g. with OpenCV's `calibrateCamera` and its default five
coefficient distortion model, and pass the result as a `CameraCalibration`.
"""

from dataclasses import dataclass

import numpy as np


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
