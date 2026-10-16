"""Script for testing an ElRobot connection, joint control, the gripper and the gentle release.

Set PORT, CALIBRATION_DIR and ID below, see the extension README for the calibration file. Keep the workspace
clear, the arm moves. Every step waits for Enter, Ctrl+C aborts and still releases the arm.
"""

import time

import numpy as np
from rcs_elrobot.hw import ElRobot, ElRobotConfig, ElRobotGripper

import rcs
from rcs import common

PORT = "/dev/ttyACM0"
CALIBRATION_DIR = "."
ID = "elrobot"

# Size in rad of the test moves and the rate at which they are streamed.
TEST_MOVE = 0.1
RATE = 30.0

robot_config = ElRobotConfig(port=PORT, calibration_dir=CALIBRATION_DIR, id=ID)
ik = rcs.common.Pin(robot_config.kinematic_model_path, robot_config.attachment_site, urdf=False)
# Opening the robot only reads, the arm stays limp.
robot = ElRobot(robot_config, ik)
gripper = ElRobotGripper(common.GripperConfig(gripper_type=common.GripperType("ElRobot")), robot)


def ramp_to(target: np.ndarray, duration: float = 1.0) -> None:
    """Stream a straight line to `target` in steps well below `max_joint_step`."""
    start = robot.get_joint_position()
    steps = int(duration * RATE)
    for i in range(1, steps + 1):
        robot.set_joint_position(start + (target - start) * i / steps)
        time.sleep(1.0 / RATE)


try:
    print(f"Joint positions: {robot.get_joint_position()}")
    print(f"Cartesian position: {robot.get_cartesian_position()}")
    print(f"Gripper width: {gripper.get_normalized_width():.3f}")
    input("Move the arm by hand and check that the readings follow. Press Enter to engage torque...")
    robot.engage()
    print("Engaged, the arm should hold its pose without moving.")

    for joint in range(robot_config.dof):
        input(f"Press Enter to move joint {joint + 1} by +-{TEST_MOVE} rad...")
        q = robot.get_joint_position()
        for offset in (TEST_MOVE, -TEST_MOVE, 0.0):
            target = q.copy()
            target[joint] += offset
            ramp_to(target)
        print(f"  joint positions: {robot.get_joint_position()}")

    input("Press Enter to cycle the gripper...")
    for width in (1.0, 0.0, 1.0):
        gripper.set_normalized_width(width)
        time.sleep(1.5)
        print(f"  commanded {width:.1f}, measured {gripper.get_normalized_width():.3f}")

    input("Press Enter to move to the home position...")
    robot.move_home()
    time.sleep(1.0)
    print(f"Joint positions: {robot.get_joint_position()}, home: {robot_config.q_home}")

    input("Press Enter for a small cartesian move down...")
    pose = robot.get_cartesian_position()
    target_pose = common.Pose(
        translation=pose.translation() + np.array([0.0, 0.0, -0.02]), quaternion=pose.rotation_q()
    )
    for _ in range(int(RATE)):
        robot.set_cartesian_position(target_pose)
        time.sleep(1.0 / RATE)
    print(f"Cartesian position now: {robot.get_cartesian_position()}")

    input("Support the arm, then press Enter to release it...")
finally:
    robot.close()
print("done")
