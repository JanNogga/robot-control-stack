import logging

import gymnasium as gym
import numpy as np
from rcs._core.sim import SimConfig
from rcs.envs.base import (
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
This script demonstrates Cartesian position control of the simulated ElRobot arm in synchronous mode.
Starting from the home pose, the TCP moves 3cm forward along the base x axis and back in 1cm steps,
closing the gripper on the way out and opening it on the way back. Actions are relative to the TCP
pose of the last step and every step goes through inverse kinematics, so the printed TCP positions
tracking the commanded ones show that IK works.
"""

STEP_SIZE = 0.01  # meters per step
STEPS_PER_LEG = 3  # steps forward before reversing
CYCLES = 100


def main():
    scene = EmptyWorldElRobot()
    cfg = scene.prefixed_cfg(scene.config())
    elrobot = scene.lead_robot_name(cfg)

    robot_cfg = cfg.robot_cfgs[elrobot]
    gripper_cfg = cfg.gripper_cfgs[elrobot]  # type: ignore[index]
    # Synchronous mode: the simulation steps until the commanded pose is reached.
    sim_cfg = SimConfig(
        realtime=False,
        async_control=False,
    )

    mjmodel = scene.create_model(cfg)
    simulation = sim.Sim(mjmodel, sim_cfg)

    kinematic_model_path, attachment_site = scene.kinematics_cfg(cfg)[elrobot]
    ik = rcs.common.Pin(
        kinematic_model_path,
        attachment_site,
    )

    robot = rcs.sim.SimRobot(simulation, ik, robot_cfg)
    env_rel: gym.Env = SimEnv(simulation)
    env_rel = RobotWrapper(env_rel, robot, ControlMode.CARTESIAN_TQuat)

    gripper = sim.SimGripper(simulation, gripper_cfg)
    env_rel = GripperWrapper(env_rel, gripper)

    env_rel = RobotSimWrapper(env_rel)
    env_rel = GripperWrapperSim(env_rel)
    env_rel = RelativeActionSpace(
        env_rel,
        max_mov=(0.05, np.deg2rad(5)),
        relative_to=RelativeTo.LAST_STEP,
    )
    env_rel = CoverWrapper(env_rel)
    env_rel.get_wrapper_attr("sim").open_gui()

    obs, info = env_rel.reset()
    print(f"home TCP: {np.round(obs['tquat'][:3], 4)}")

    for _ in range(CYCLES):
        for direction, gripper_action in ((1, 0), (-1, 1)):
            for _ in range(STEPS_PER_LEG):
                act = {"tquat": [direction * STEP_SIZE, 0, 0, 0, 0, 0, 1], "gripper": [gripper_action]}
                obs, reward, terminated, truncated, info = env_rel.step(act)
                if not info["ik_success"]:
                    logger.warning("IK failed")
                if truncated or terminated:
                    logger.info("Truncated or terminated!")
                    return
            print(f"TCP: {np.round(obs['tquat'][:3], 4)}")


if __name__ == "__main__":
    main()
