import logging

import gymnasium as gym
import numpy as np
from rcs._core.sim import SimConfig
from rcs.camera.sim import SimCameraSet
from rcs.envs.base import (
    CameraSetWrapper,
    ControlMode,
    CoverWrapper,
    GripperWrapper,
    RelativeActionSpace,
    RelativeTo,
    RobotWrapper,
    SimEnv,
)
from rcs.envs.configs import EmptyWorldElRobot
from rcs.envs.sim import GripperWrapperSim, RobotSimWrapper

import rcs
from rcs import sim

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
This script demonstrates joint control of the simulated ElRobot arm. Actions are relative to the
joint positions of the last step and limited to 5 degrees per joint, the gripper action is binary
(0 closed, 1 open). The observation contains the image of the wrist camera under obs["frames"]["wrist"].
"""


def main():
    scene = EmptyWorldElRobot()
    cfg = scene.prefixed_cfg(scene.config())
    elrobot = scene.lead_robot_name(cfg)

    robot_cfg = cfg.robot_cfgs[elrobot]
    gripper_cfg = cfg.gripper_cfgs[elrobot]  # type: ignore[index]
    camera_cfgs = cfg.camera_cfgs
    sim_cfg = SimConfig(
        realtime=False,
        async_control=False,
    )

    kinematic_model_path, attachment_site = scene.kinematics_cfg(cfg)[elrobot]
    ik = rcs.common.Pin(
        kinematic_model_path,
        attachment_site,
    )
    mjmodel = scene.create_model(cfg)
    simulation = sim.Sim(mjmodel, sim_cfg)

    robot = rcs.sim.SimRobot(simulation, ik, robot_cfg)
    env_rel: gym.Env = SimEnv(simulation)
    env_rel = RobotWrapper(env_rel, robot, ControlMode.JOINTS)

    gripper = sim.SimGripper(simulation, gripper_cfg)
    env_rel = GripperWrapper(env_rel, gripper)

    env_rel = RobotSimWrapper(env_rel)
    env_rel = GripperWrapperSim(env_rel)

    if camera_cfgs is not None:
        camera_set = SimCameraSet(simulation, camera_cfgs, physical_units=True, render_on_demand=True)
        env_rel = CameraSetWrapper(env_rel, camera_set)  # type: ignore[arg-type]

    env_rel = RelativeActionSpace(
        env_rel,
        max_mov=np.deg2rad(5),
        relative_to=RelativeTo.LAST_STEP,
    )
    env_rel = CoverWrapper(env_rel)
    env_rel.get_wrapper_attr("sim").open_gui()

    for _ in range(100):
        obs, info = env_rel.reset()
        for _ in range(10):
            # sample random relative action and execute it
            act = env_rel.action_space.sample()
            obs, reward, terminated, truncated, info = env_rel.step(act)
            if truncated or terminated:
                logger.info("Truncated or terminated!")
                return


if __name__ == "__main__":
    main()
