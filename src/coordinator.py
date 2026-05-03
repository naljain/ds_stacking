"""
Per-arm primitive sequencer.

This is intentionally minimal: it just walks each arm through its block list
in primitive order (reach -> grasp -> lift -> transport -> place) and bumps
the stacking goal Z when 'place' completes. There is NO inter-arm collision
logic here — that is handled smoothly and continuously by the DS modulation
in src/modulation.py, so the closed-loop system remains a pure dynamical
system rather than a hybrid system with discrete holds.

To compute joint-space goals q_goal for each primitive (needed by the joint-space
DS), we use Isaac Sim's IK solver once per primitive transition, given the
Cartesian target from primitives.primitive_target. The DS then drives toward
that q_goal until the next primitive replaces it.
"""

import numpy as np

from .primitives import (
    PRIMITIVE_ORDER,
    DEFAULT_DOWN_QUAT,
    primitive_target,
    grasp_quat_from_block,
    gripper_action_for_primitive,
)


class ArmTaskState:
    """Tracks one arm's progress through its stack."""

    def __init__(self, arm, block_order, goal_xy):
        self.arm         = arm
        self.block_order = list(block_order)
        self.goal_xy     = goal_xy
        self.current_block_idx = 0
        self.current_primitive = "reach"
        self.gripper_open      = True
        self.q_goal = None    # set by the deployment loop after IK
        self.reserved_goal_z = None

    @property
    def current_block(self):
        if self.current_block_idx >= len(self.block_order):
            return None
        return self.block_order[self.current_block_idx]

    def advance_primitive(self):
        idx = PRIMITIVE_ORDER.index(self.current_primitive)
        if idx + 1 < len(PRIMITIVE_ORDER):
            self.current_primitive = PRIMITIVE_ORDER[idx + 1]
        else:
            self.current_block_idx += 1
            self.current_primitive = "reach"

    def is_done(self):
        return self.current_block_idx >= len(self.block_order)


class TaskSequencer:
    """Slim per-arm primitive sequencer. No collision arbitration."""

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

        self.tasks = {}
        for arm in env.arms_active:
            block_order = [b["name"] for b in cfg[f"{arm}_blocks"]]
            goal_xy     = tuple(cfg["goals"][arm])
            self.tasks[arm] = ArmTaskState(arm, block_order, goal_xy)

        block_h = cfg["block"]["size"]
        self.base_z  = cfg["table"]["height"] + block_h / 2
        self.block_h = block_h + 0.002
        # Per-arm block-placement counter — derived goal_z avoids the
        # double-increment race when both arms complete "place" in the same
        # physics step (which the old `self.goal_z += block_h` had).
        self.placed_per_arm = {arm: 0 for arm in env.arms_active}
        self._next_stack_slot = 0

    def cartesian_target(self, arm):
        """Cartesian target for the current primitive on the given arm."""
        task = self.tasks[arm]
        if task.is_done():
            return None
        block_pos = self.env.get_block_positions()[task.current_block]
        goal_z = self._goal_z_for_task(task)
        return primitive_target(
            primitive=task.current_primitive,
            block_pos=block_pos,
            goal_xy=task.goal_xy,
            goal_z=goal_z,
            hover_h=self.cfg["heights"]["hover"],
            lift_h =self.cfg["heights"]["lift"],
            grasp_h=self.cfg["heights"]["grasp"],
        )

    def ee_orientation(self, arm):
        """EE quaternion (w,x,y,z) for the current primitive.

        Reach and grasp align to the block's yaw so fingers don't hit rotated
        faces.  All other primitives use the default straight-down orientation.
        """
        task = self.tasks[arm]
        if not task.is_done() and task.current_primitive in ("reach", "grasp"):
            _, block_quat = self.env.get_block_poses()[task.current_block]
            return grasp_quat_from_block(block_quat)
        return DEFAULT_DOWN_QUAT

    @property
    def goal_z(self):
        n_total = self._next_stack_slot
        return self.base_z + n_total * self.block_h

    def stack_target_position(self, arm):
        """Center position for the active block's reserved stack slot."""
        task = self.tasks[arm]
        z = task.reserved_goal_z if task.reserved_goal_z is not None else self.goal_z
        return np.array([task.goal_xy[0], task.goal_xy[1], z])

    def _goal_z_for_task(self, task):
        """Reserve a unique shared-stack slot before the place primitive.

        Both arms may run transport/place concurrently. If goal height is based
        only on completed placements, two arms can target the same stack layer.
        Reserving at target-generation time keeps the sequencing deterministic
        without adding collision/hold logic to the coordinator.
        """
        if task.current_primitive in ("transport", "place"):
            if task.reserved_goal_z is None:
                task.reserved_goal_z = self.goal_z
                self._next_stack_slot += 1
            return task.reserved_goal_z
        return self.goal_z

    def primitive_complete(self, arm):
        task = self.tasks[arm]
        if task.current_primitive == "place":
            self.placed_per_arm[arm] += 1
            task.reserved_goal_z = None
        task.advance_primitive()

    def gripper_action(self, arm):
        return gripper_action_for_primitive(self.tasks[arm].current_primitive)
