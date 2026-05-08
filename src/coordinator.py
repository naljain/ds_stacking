"""
Per-arm primitive sequencer for shared-goal stacking.

Both arms bring their blocks to a single shared_goal at the centre of the
table and build one unified stack there.

Key changes from the per-arm-goal version:
  - ArmTaskState.goal_xy is the same shared point for both arms.
  - There is a single global goal_z counter, incremented whenever *either*
    arm completes a 'place'. This keeps the stack height correct regardless
    of which arm placed last.
  - can_place(arm) returns False when the other arm's EE is within
    yield_radius (XY) of the stack goal AND that arm is currently in its
    own 'place' or 'transport' primitive. This gives a simple discrete gate:
    only one arm descends to place at a time. The DS modulation still handles
    the smooth spatial avoidance; this gate is the higher-level "take turns"
    rule that prevents simultaneous descents onto the same point.

There is still NO discrete hold/release anywhere in the continuous DS loop
(deploy_dual_arm.py). can_place() is only called at the primitive-completion
check — if it returns False, seq.primitive_complete() is not called, so the
arm stays at 'transport' (hovering above the stack) until the coast is clear.
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
        self.goal_xy     = goal_xy          # shared goal for both arms
        self.current_block_idx = 0
        self.current_primitive = "reach"
        self.gripper_open      = True
        self.q_goal = None    # set by deployment loop after IK

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
    """Slim per-arm primitive sequencer with shared goal and place-yield gate."""

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

        # Both arms target the same shared goal xy
        shared_xy = tuple(cfg["shared_goal"])

        self.tasks = {}
        for arm in env.arms_active:
            block_order = [b["name"] for b in cfg[f"{arm}_blocks"]]
            self.tasks[arm] = ArmTaskState(arm, block_order, shared_xy)

        block_h = cfg["block"]["size"]
        base_z  = cfg["table"]["height"] + block_h / 2

        # Single shared stack height — both arms read/write this
        self.goal_z  = base_z
        self.block_h = block_h + 0.002

        self.yield_radius = cfg["coordination"].get("yield_radius", 0.12)

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
            goal_z=self.goal_z,           # shared height
            hover_h=self.cfg["heights"]["hover"],
            lift_h =self.cfg["heights"]["lift"],
            grasp_h=self.cfg["heights"]["grasp"],
        )

    def can_place(self, arm):
        """Return True if this arm is allowed to proceed through 'place'.

        Blocks the arm if the OTHER arm's EE is within yield_radius (XY) of
        the shared stack goal AND the other arm is in 'transport' or 'place'
        (i.e. it is also heading to or hovering over the stack).

        This prevents two arms from descending onto the same point at once.
        The DS modulation in modulation.py still handles all smooth spatial
        avoidance; this is purely the "take turns descending" gate.
        """
        other = "right" if arm == "left" else "left"
        if other not in self.tasks:
            return True                       # single-arm mode, always go
        other_task = self.tasks[other]
        if other_task.is_done():
            return True
        if other_task.current_primitive not in ("transport", "place"):
            return True

        # Is the other arm's EE close to the stack goal in XY?
        ee_other, _ = self.env.get_ee_pose(other)
        gx, gy = self.tasks[arm].goal_xy
        xy_dist = np.linalg.norm(ee_other[:2] - np.array([gx, gy]))
        return xy_dist > self.yield_radius

    def primitive_complete(self, arm):
        task = self.tasks[arm]
        if task.current_primitive == "place":
            self.goal_z += self.block_h    # shared counter advances once per place
        task.advance_primitive()

    def gripper_action(self, arm):
        return gripper_action_for_primitive(self.tasks[arm].current_primitive)
