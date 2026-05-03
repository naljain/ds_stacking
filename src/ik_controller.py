"""
Cartesian straight-line IK controller for data collection.

Replaces RMPflow with a precise straight-line Cartesian interpolator:
  - Linearly interpolates EE position from current to target (trapezoidal
    velocity profile) and solves Lula IK at every step.
  - Tracks gripper state internally and writes all 9 joints (7 arm + 2
    fingers) in a single apply_action call, avoiding fights between
    franka.apply_action and franka.gripper.apply_action.
  - Rest pose is arm-aware: joint 1 is biased outward so left and right
    arms keep their elbows away from each other.
"""

import numpy as np

from .franka_ik import FrankaIK

# Default down-pointing gripper orientation (quaternion w,x,y,z)
DEFAULT_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

# Gripper finger width (metres) for open/closed
GRIPPER_OPEN   = 0.04
GRIPPER_CLOSED = 0.00

# Per-arm start poses.  Only joints 0 and 1 are specified by you — the rest
# are filled from the remaining values you provided.  The right arm mirrors
# joint 0 and joint 1 so the two arms are symmetric about the table centre.
#
#   Your values: j0=1.93, j1=1.1, j2=-1.8639, j3=-2.6820, j4=1.0833, j5=1.9480, j6=-0.1237
#
#   Left arm  : j0=-1.93 (mirrored), j1=-1.1 (elbow out left), rest unchanged
#   Right arm : j0= 1.93,            j1= 1.1 (elbow out right), rest unchanged

# Left arm: exact reference values.
# Right arm: j0 and j1 mirrored; j2-j6 unchanged until confirmed by testing.
_REST_LEFT  = np.array([ 1.93,  1.10, -1.8639, -2.6820,  1.0833,  1.9480, -0.1237])
_REST_RIGHT = np.array([-1.93, 1.10, 1.8639, -2.6820,  -1.0833,  1.9480, 0.1237])


def _rest_pose(arm: str) -> np.ndarray:
    return _REST_RIGHT.copy() if arm == "right" else _REST_LEFT.copy()


def _articulation_action():
    """Lazy import so Isaac Sim modules are only resolved after SimulationApp starts."""
    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core.utils.types import ArticulationAction
    return ArticulationAction


class IKController:
    """Straight-line Cartesian controller backed by per-step Lula IK.

    Args:
        franka        : Isaac Sim Franka articulation object
        arm           : "left" or "right" — sets elbow-out bias in rest pose
        name          : unused, kept for API compatibility with old RMPflow wrapper
        max_cart_step : max EE displacement per physics step (metres)
        vel_ramp_frac : fraction of steps used for accel/decel ramp
    """

    def __init__(self, franka, arm="left", name="cartesian_ik",
                 max_cart_step=0.005, vel_ramp_frac=0.2, rest_q=None):
        self.franka        = franka
        self.arm           = arm
        self.max_cart_step = max_cart_step
        self.vel_ramp_frac = vel_ramp_frac
        self.ik            = FrankaIK(franka)

        # IK warm-start: prefer caller-supplied rest_q (from config), fall
        # back to the hardcoded per-arm default if not provided.
        self._q_rest       = np.array(rest_q) if rest_q is not None else _rest_pose(arm)
        self._q_last       = self._q_rest.copy()
        self._finger_width = GRIPPER_OPEN              # tracked gripper state

    def reset(self):
        """Re-seed IK warm-start from current joint state (which should be
        the default pose after env.reset_arms() was called)."""""
        q = self.franka.get_joint_positions()
        if q is not None and len(q) >= 7:
            self._q_last = q[:7].copy()
        else:
            self._q_last = self._q_rest.copy()

    # ── Gripper ───────────────────────────────────────────────────────────────
    def set_gripper(self, open: bool):
        """Record desired gripper width; it will be written on the next step_to
        call together with the arm joints in a single apply_action call."""
        self._finger_width = GRIPPER_OPEN if open else GRIPPER_CLOSED

    # ── Core: one IK step to a specific Cartesian waypoint ───────────────────
    def step_to(self, target_pos, target_quat=None):
        """Solve IK for target_pos and apply a single ArticulationAction that
        covers all 9 joints (7 arm + 2 fingers) at once.  Mixing
        franka.apply_action and franka.gripper.apply_action in the same step
        causes the gripper command to be silently dropped, so we own all joints
        here.
        """
        if target_quat is None:
            target_quat = DEFAULT_DOWN_QUAT

        q_goal, ok = self.ik.solve(target_pos, target_quat,
                                   q_seed=self._q_last)
        if ok:
            self._q_last = q_goal.copy()

        # Build full 9-DOF command: arm joints + both finger joints
        full_cmd = np.concatenate([
            self._q_last,                                        # joints 0-6
            np.array([self._finger_width, self._finger_width]),  # joints 7-8
        ])
        ArticulationAction = _articulation_action()
        self.franka.apply_action(ArticulationAction(joint_positions=full_cmd))
        return ok

    # ── Straight-line move with trapezoidal velocity profile ─────────────────
    def move_to(self, world, target_pos, target_quat=None, steps=120,
                record_callback=None, render=True):
        """Move EE in a straight Cartesian line to target_pos over `steps` ticks.

        The path is parameterised by arc-length with a trapezoidal velocity
        profile so the arm accelerates and decelerates smoothly, avoiding
        joint-velocity spikes in the recorded q_dot labels.
        """
        if target_quat is None:
            target_quat = DEFAULT_DOWN_QUAT

        ee_start, _ = self.ik.get_world_pose()
        ee_start = np.array(ee_start).copy()
        ee_end   = np.array(target_pos).copy()

        s_values = _trapezoid_profile(steps, self.vel_ramp_frac)

        for s in s_values:
            waypoint = ee_start + s * (ee_end - ee_start)
            self.step_to(waypoint, target_quat)
            world.step(render=render)
            if record_callback is not None:
                record_callback()


# ── Trapezoidal velocity profile ─────────────────────────────────────────────
def _trapezoid_profile(n_steps, ramp_frac=0.2):
    """Return n_steps values of arc-length parameter s ∈ [0,1] with a
    trapezoidal velocity profile (ramp up, cruise, ramp down)."""
    ramp   = max(1, int(n_steps * ramp_frac))
    cruise = max(0, n_steps - 2 * ramp)

    v = np.zeros(n_steps)
    for i in range(ramp):
        v[i] = (i + 1) / ramp
    for i in range(ramp, ramp + cruise):
        v[i] = 1.0
    for i in range(ramp + cruise, n_steps):
        v[i] = (n_steps - i) / ramp

    v /= v.sum()
    return np.clip(np.cumsum(v), 0.0, 1.0)
