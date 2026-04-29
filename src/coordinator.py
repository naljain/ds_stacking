"""
Per-arm primitive sequencer.

This is intentionally minimal: it just walks each arm through its block list
in primitive order (reach -> grasp -> lift -> transport -> place) and bumps
the stacking goal Z when 'place' completes. There is NO inter-arm collision
logic here — that is handled smoothly and continuously by the DS modulation
in src/modulation.py, so the closed-loop system remains a pure dynamical
system rather than a hybrid system with discrete holds.

To compute joint-space goals q* for each primitive (needed by the joint-space
DS), we use Isaac Sim's IK solver once per primitive transition, given the
Cartesian target from primitives.primitive_target. The DS then drives toward
that q* until the next primitive replaces it.
"""

import numpy as np

from .primitives import (
    PRIMITIVE_ORDER,
    primitive_target,
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
        base_z  = cfg["table"]["height"] + block_h / 2
        self.goal_z = {arm: base_z for arm in env.arms_active}
        self.block_h = block_h + 0.002

    def cartesian_target(self, arm):
        """Cartesian target for the current primitive on the given arm."""
        task = self.tasks[arm]
        if task.is_done():
            return None
        block_pos = self.env.get_block_positions()[task.current_block]
        return primitive_target(
            primitive=task.current_primitive,
            block_pos=block_pos,
            goal_xy=task.goal_xy,
            goal_z=self.goal_z[arm],
            hover_h=self.cfg["heights"]["hover"],
            lift_h =self.cfg["heights"]["lift"],
            grasp_h=self.cfg["heights"]["grasp"],
        )

    def primitive_complete(self, arm):
        task = self.tasks[arm]
        if task.current_primitive == "place":
            self.goal_z[arm] += self.block_h
        task.advance_primitive()

    def gripper_action(self, arm):
        return gripper_action_for_primitive(self.tasks[arm].current_primitive)
