"""
IK utility for computing target joint configurations from Cartesian targets.

Used in two places:
  1. Data collection: compute q* per transport segment so trajectories are
     labelled with consistent (q, q*, q_dot) tuples.
  2. Deployment: compute q* for the transport target; DS drives q -> q*.

Elbow consistency guarantee
---------------------------
Even with a fixed seed, Lula can return solutions with different elbow
configurations (joint 1 sign) depending on the Cartesian target. This causes
visible elbow flips between demos and discontinuous training data.

solve() now enforces that the returned solution stays in the same elbow
homotopy class as the seed:
  - After a successful solve, check that sign(q[1]) == sign(seed[1]).
  - If the elbow flipped, retry up to `max_elbow_retries` times with a seed
    that is nudged further into the desired elbow half-space.
  - If all retries fail, return the seed (safe fallback, triggers the
    existing [WARN] path in callers).

This means collect_ik.py and deploy_single_arm.py need no changes — the
consistency is enforced transparently inside solve().
"""

import numpy as np

try:
    from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
    LULA_AVAILABLE = True
except ImportError:
    try:
        from omni.isaac.motion_generation.lula import LulaKinematicsSolver
        LULA_AVAILABLE = True
    except ImportError:
        LULA_AVAILABLE = False

DEFAULT_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

# Use Lula's nominal gripper frame for Cartesian DS state and IK targets.
# Isaac's Franka wrapper reports /panda_rightfinger as end_effector, which is
# a different frame; read this frame via Lula FK instead of mixing the two.
DEFAULT_EE_FRAME = "right_gripper"

# How much to nudge joint 1 toward the desired sign on each retry (radians)
_ELBOW_NUDGE = 0.15


class FrankaIK:
    """Wrapper around Lula IK with elbow-consistency enforcement."""

    def __init__(self, franka, robot_description_path=None, urdf_path=None):
        self.franka = franka
        if not LULA_AVAILABLE:
            raise RuntimeError(
                "Lula IK is not available. Install Isaac Sim motion_generation "
                "extension or fall back to numerical IK."
            )
        if robot_description_path is None or urdf_path is None:
            from isaacsim.core.utils.extensions import get_extension_path_from_name
            mg_ext = None
            for ext_name in ("isaacsim.robot_motion.motion_generation",
                             "omni.isaac.motion_generation"):
                try:
                    mg_ext = get_extension_path_from_name(ext_name)
                    if mg_ext:
                        break
                except Exception:
                    continue
            if not mg_ext:
                raise RuntimeError(
                    "Could not locate motion_generation extension. "
                    "Check your Isaac Sim install."
                )
            robot_description_path = robot_description_path or \
                f"{mg_ext}/motion_policy_configs/franka/rmpflow/robot_descriptor.yaml"
            urdf_path = urdf_path or \
                f"{mg_ext}/motion_policy_configs/franka/lula_franka_gen.urdf"

        self.solver = LulaKinematicsSolver(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path,
        )
        self.ee_frame = DEFAULT_EE_FRAME

    def get_world_pose(self, q=None):
        """Return the Lula EE frame pose in world coordinates.

        This is the pose that should be used for Cartesian DS state whenever
        IK targets are solved for self.ee_frame.  It avoids mixing Isaac's
        /panda_rightfinger prim with Lula's synthetic right_gripper frame.
        """
        if q is None:
            q = self.franka.get_joint_positions()[:7].copy()
        base_pos, base_rot = self.franka.get_world_pose()
        self.solver.set_robot_base_pose(base_pos, base_rot)
        pos, rot = self.solver.compute_forward_kinematics(self.ee_frame, q)
        try:
            from scipy.spatial.transform import Rotation
            quat_xyzw = Rotation.from_matrix(rot).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0],
                                  quat_xyzw[1], quat_xyzw[2]])
        except Exception:
            quat_wxyz = None
        return np.asarray(pos), quat_wxyz

    def _solve_once(self, target_pos, target_quat, q_seed):
        """Single Lula IK call. Returns (q, success)."""
        base_pos, base_rot = self.franka.get_world_pose()
        self.solver.set_robot_base_pose(base_pos, base_rot)
        action, success = self.solver.compute_inverse_kinematics(
            frame_name=self.ee_frame,
            target_position=target_pos,
            target_orientation=target_quat,
            warm_start=q_seed,
        )
        if success:
            q = action if isinstance(action, np.ndarray) else action.joint_positions
            return q[:7].copy(), True
        return q_seed.copy(), False

    def solve(self, target_pos, target_quat=None, q_seed=None,
              max_elbow_retries=5):
        """Return (q*, success) with elbow consistency enforced.

        After a successful solve, if the elbow joint (joint 1) has flipped
        sign relative to q_seed, the seed is nudged further into the desired
        half-space and the solve is retried. This keeps all solutions in the
        same homotopy class as the rest/default pose throughout a demo.

        Args:
            target_pos       : (3,) Cartesian target position
            target_quat      : (4,) w,x,y,z orientation (default: down)
            q_seed           : (7,) seed joints (default: current arm pose)
            max_elbow_retries: how many times to retry on elbow flip
        """
        if target_quat is None:
            target_quat = DEFAULT_DOWN_QUAT
        if q_seed is None:
            q_seed = self.franka.get_joint_positions()[:7].copy()

        seed = q_seed.copy()
        desired_elbow_sign = np.sign(seed[1]) if seed[1] != 0 else 1.0

        for attempt in range(max_elbow_retries + 1):
            q, ok = self._solve_once(target_pos, target_quat, seed)
            if not ok:
                return q_seed.copy(), False

            # Check elbow consistency
            if np.sign(q[1]) == desired_elbow_sign or q[1] == 0:
                return q, True

            # Elbow flipped — nudge seed further into desired half-space
            # and retry. Increase nudge each attempt.
            seed = seed.copy()
            seed[1] += desired_elbow_sign * _ELBOW_NUDGE * (attempt + 1)

        # All retries failed to recover correct elbow — return seed as fallback
        return q_seed.copy(), False
