"""
Thin wrapper around Isaac Sim's RMPflow controller.

RMPflow provides smooth, reactive Cartesian motion. We wrap it so the rest of
the codebase doesn't need to know which IK backend is in use — if we later swap
to Lula or PinocchioIK we only update this file.
"""

import numpy as np

try:
    # Isaac Sim 5.x path
    from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController
except ImportError:
    # Isaac Sim 4.x fallback
    from omni.isaac.franka.controllers import RMPFlowController

from omni.isaac.core.utils.types import ArticulationAction


# Default down-pointing gripper orientation (quaternion w,x,y,z)
DEFAULT_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


class IKController:
    """Single-arm IK controller wrapping RMPflow."""

    def __init__(self, franka, name="rmpflow"):
        self.franka = franka
        self.controller = RMPFlowController(
            name=name,
            robot_articulation=franka,
        )

    def step_to(self, target_pos, target_quat=None):
        """Compute and apply one control step toward the target. Returns the
        end-effector pose after stepping (caller is responsible for world.step)."""
        if target_quat is None:
            target_quat = DEFAULT_DOWN_QUAT
        action = self.controller.forward(
            target_end_effector_position=target_pos,
            target_end_effector_orientation=target_quat,
        )
        self.franka.apply_action(action)

    def reset(self):
        self.controller.reset()

    # ── Gripper helpers ───────────────────────────────────────────────────────
    def set_gripper(self, open: bool):
        width = 0.04 if open else 0.0
        self.franka.gripper.apply_action(
            ArticulationAction(joint_positions=np.array([width, width]))
        )

    # ── Convenience: full move with stepping owned by caller ─────────────────
    def move_to(self, world, target_pos, target_quat=None, steps=120,
                record_callback=None, render=True, stop_tolerance=None,
                max_extra_steps=0, post_step_callback=None):
        """Step toward target_pos for `steps` ticks. If record_callback is given,
        call it each step with the current ee pose so a trajectory can be logged."""
        for _ in range(steps):
            self.step_to(target_pos, target_quat)
            world.step(render=render)
            if post_step_callback is not None:
                post_step_callback()
            if record_callback is not None:
                record_callback()
        if stop_tolerance is None or max_extra_steps <= 0:
            return
        for _ in range(max_extra_steps):
            ee_pos = self.franka.end_effector.get_world_pose()[0]
            if np.linalg.norm(ee_pos - target_pos) <= stop_tolerance:
                break
            self.step_to(target_pos, target_quat)
            world.step(render=render)
            if post_step_callback is not None:
                post_step_callback()
            if record_callback is not None:
                record_callback()
