"""
Motion primitive definitions.

Each primitive maps the current world state to a Cartesian target. During data
collection, we use these targets as IK goals; during deployment, the learned
Neural DS replaces the discrete target-following with a continuous velocity
field, but the primitive boundaries (start/end conditions) stay the same.

Primitives:
  reach     — hover above source block
  grasp     — descend to block surface
  lift      — raise block to transport altitude
  transport — move above goal stacking location
  place     — descend onto stack
"""

import numpy as np

PRIMITIVE_ORDER = ["reach", "grasp", "lift", "transport", "place"]

# Only the longer free-space motions are represented by learned DS models.
# Short constrained motions are cleaner and more reliable as scripted IK moves.
DS_PRIMITIVES = ["reach", "transport"]
SCRIPTED_PRIMITIVES = ["grasp", "lift", "place"]

# Gripper pointing straight down, no Z rotation (w,x,y,z Isaac Sim convention)
DEFAULT_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def grasp_quat_from_block(block_quat):
    """Gripper orientation that matches the block's yaw and points straight down.

    Extracts the block's Z-rotation, snaps to the nearest 90° (square block
    symmetry), then composes with DEFAULT_DOWN_QUAT.  Result is always pointing
    straight down — only the in-plane finger alignment changes.
    """
    w, x, y, z = block_quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_snap = np.round(yaw / (np.pi / 2.0)) * (np.pi / 2.0)
    # q_yaw ⊗ q_down with q_down = (0,1,0,0) reduces to (0, cos(θ/2), sin(θ/2), 0)
    return np.array([0.0, np.cos(yaw_snap / 2.0), np.sin(yaw_snap / 2.0), 0.0])


def primitive_target(primitive, block_pos, goal_xy, goal_z,
                     hover_h, lift_h, grasp_h):
    """Compute Cartesian target for a given primitive."""
    bx, by, _ = block_pos
    gx, gy    = goal_xy

    if primitive == "reach":
        return np.array([bx, by, hover_h])
    elif primitive == "grasp":
        return np.array([bx, by, grasp_h])
    elif primitive == "lift":
        return np.array([bx, by, lift_h])
    elif primitive == "transport":
        return np.array([gx, gy, lift_h])
    elif primitive == "place":
        return np.array([gx, gy, goal_z + 0.02])
    else:
        raise ValueError(f"Unknown primitive: {primitive}")


def primitive_goal_state(primitive, block_pos, goal_xy, goal_z,
                          hover_h, lift_h, grasp_h):
    """Same as primitive_target but framed as the DS attractor — the point
    the learned vector field should converge to."""
    return primitive_target(primitive, block_pos, goal_xy, goal_z,
                            hover_h, lift_h, grasp_h)


def primitive_done(primitive, ee_pos, target, tolerance=0.015):
    """Check whether the primitive has reached its goal state."""
    return np.linalg.norm(ee_pos - target) < tolerance


def gripper_action_for_primitive(primitive):
    """Returns 'close', 'open', or None depending on whether gripper should
    actuate at the END of this primitive."""
    if primitive == "grasp":
        return "close"
    if primitive == "place":
        return "open"
    return None
