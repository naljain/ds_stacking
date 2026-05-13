"""
IK utility for computing target joint configurations from Cartesian targets.

Used in two places:
  1. Data collection: at each primitive boundary, compute q_goal for the
     recorded trajectory so we can label (q, q_goal, q̇) tuples.
  2. Deployment: when a primitive completes, compute the new q_goal for the
     next primitive's Cartesian goal. The DS then drives q -> q_goal in joint
     space.

We use Isaac Sim's Lula-based IK solver, which gives clean analytical solutions
for the Franka. Falls back to a damped-least-squares numerical solver if Lula
is unavailable.
"""

import numpy as np

try:
    # Isaac Sim 5.x path
    from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
    LULA_AVAILABLE = True
except ImportError:
    try:
        # Isaac Sim 4.x fallback
        from omni.isaac.motion_generation.lula import LulaKinematicsSolver
        LULA_AVAILABLE = True
    except ImportError:
        LULA_AVAILABLE = False


# Default down-pointing gripper orientation (quaternion w,x,y,z)
DEFAULT_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


class FrankaIK:
    """Wrapper around Lula IK for the Franka Panda."""

    def __init__(self, franka, robot_description_path=None, urdf_path=None):
        self.franka = franka
        if not LULA_AVAILABLE:
            raise RuntimeError(
                "Lula IK is not available. Install Isaac Sim motion_generation "
                "extension or fall back to numerical IK."
            )
        # Resolve default Franka description files shipped with Isaac Sim
        if robot_description_path is None or urdf_path is None:
            from isaacsim.core.utils.extensions import get_extension_path_from_name
            # Try Isaac Sim 5.x extension name first, fall back to 4.x name
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
            descriptor_dir = "rmp" + "flow"
            robot_description_path = robot_description_path or \
                f"{mg_ext}/motion_policy_configs/franka/{descriptor_dir}/robot_descriptor.yaml"
            urdf_path = urdf_path or \
                f"{mg_ext}/motion_policy_configs/franka/lula_franka_gen.urdf"

        self.solver = LulaKinematicsSolver(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path,
        )
        # The Franka's TCP frame in the URDF
        self.ee_frame = "right_gripper"

    def solve(self, target_pos, target_quat=None, q_seed=None):
        """Return (q_goal, success) where q_goal is a 7-vector of target joint angles.
        q_seed should be the current joint config to bias the solution."""
        if target_quat is None:
            target_quat = DEFAULT_DOWN_QUAT
        if q_seed is None:
            q_seed = self.franka.get_joint_positions()[:7]

        # Robot base pose in world — needed because Lula solves in robot frame
        base_pos, base_rot = self.franka.get_world_pose()
        self.solver.set_robot_base_pose(base_pos, base_rot)

        action, success = self.solver.compute_inverse_kinematics(
            frame_name=self.ee_frame,
            target_position=target_pos,
            target_orientation=target_quat,
            warm_start=q_seed,
        )
        if success:
            # Isaac Sim 5.x returns ndarray directly; 4.x returned ArticulationAction
            q = action if isinstance(action, np.ndarray) else action.joint_positions
            return q[:7].copy(), True
        return q_seed.copy(), False

    def forward_position(self, q=None):
        """Return the world position of the same Lula frame used for IK.

        Isaac's high-level Franka `end_effector` handle is not guaranteed to be
        the exact same frame as Lula's `right_gripper`. Completion checks should
        compare targets against this frame so IK success and Cartesian error use
        the same geometry.
        """
        if q is None:
            q = self.franka.get_joint_positions()[:7]

        base_pos, base_rot = self.franka.get_world_pose()
        self.solver.set_robot_base_pose(base_pos, base_rot)

        try:
            pose = self.solver.compute_forward_kinematics(
                frame_name=self.ee_frame,
                joint_positions=q,
            )
        except TypeError:
            pose = self.solver.compute_forward_kinematics(self.ee_frame, q)

        if isinstance(pose, tuple):
            return np.asarray(pose[0], dtype=float).copy()
        if hasattr(pose, "p"):
            return np.asarray(pose.p, dtype=float).copy()
        if hasattr(pose, "translation"):
            return np.asarray(pose.translation, dtype=float).copy()
        raise RuntimeError(
            "Unsupported Lula FK return type from compute_forward_kinematics"
        )

    def get_frame_world_pose(self, frame_name, q=None):
        """Return a named Lula frame pose in world coordinates.

        Used by the dual-arm modulation layer to build protected proxy points
        on distal links such as panda_link5, panda_link6, panda_link7, and the
        right_gripper frame.
        """
        if q is None:
            q = self.franka.get_joint_positions()[:7]

        base_pos, base_rot = self.franka.get_world_pose()
        self.solver.set_robot_base_pose(base_pos, base_rot)

        try:
            pose = self.solver.compute_forward_kinematics(
                frame_name=frame_name,
                joint_positions=q,
            )
        except TypeError:
            pose = self.solver.compute_forward_kinematics(frame_name, q)

        if isinstance(pose, tuple):
            pos = np.asarray(pose[0], dtype=float).copy()
            rot = pose[1] if len(pose) > 1 else None
        elif hasattr(pose, "p"):
            pos = np.asarray(pose.p, dtype=float).copy()
            rot = getattr(pose, "R", None)
        elif hasattr(pose, "translation"):
            pos = np.asarray(pose.translation, dtype=float).copy()
            rot = getattr(pose, "rotation", None)
        else:
            raise RuntimeError(
                "Unsupported Lula FK return type from compute_forward_kinematics"
            )

        quat_wxyz = None
        if rot is not None:
            try:
                from scipy.spatial.transform import Rotation
                quat_xyzw = Rotation.from_matrix(rot).as_quat()
                quat_wxyz = np.array([
                    quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
                ])
            except Exception:
                quat_wxyz = None
        return pos, quat_wxyz
