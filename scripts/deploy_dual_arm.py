"""
Dual-arm deployment with joint-space Neural DS, scripted Lula primitives,
end-effector modulation, sampled-link safety holds, and return-home parking.

Reach and transport use learned DS checkpoints. Grasp, lift, and place use a
clamped Lula joint-space controller against the same per-primitive q_goal.
End-effector modulation shapes joint velocities smoothly; the sampled-link hold
is a discrete deployment guard for elbow/forearm clearance.

Usage:
  python scripts/deploy_dual_arm.py
  python scripts/deploy_dual_arm.py --use_safe
  python scripts/deploy_dual_arm.py --no_modulation     # ablation
"""

import os
import sys
import argparse
import signal
import numpy as np
import torch
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["CARB_LOG_LEVEL"] = "error"


# Cartesian EE tolerance for IK-primitive completion (grasp, lift, transport,
# place). Set to the block size: place physically can't descend below this
# because the carried block rests on top of the existing stack, and the other
# IK primitives don't need tighter precision than this either.
IK_CART_DONE_TOL = 0.05

# Joint-space fallback tolerance for IK primitives. The proportional law plus
# Isaac's position controller has a small steady-state Cartesian gap (Lula
# q_goal vs realized FK), so completion accepts EITHER cart_err < 0.05 OR
# joint_err < this — whichever happens first.
IK_JOINT_DONE_TOL = 0.12

# Cartesian EE tolerance that can complete a DS reach primitive even when the
# redundant joint configuration does not match q_goal exactly.
DS_REACH_CART_DONE_TOL = 0.03

# Linear joint-space attraction added to the DS output for every DS primitive:
#     q_dot = ds_scale * f_DS(e) - DS_GOAL_GAIN * (q - q_goal)
# 0.0 = pure neural DS. With uniform state_std normalization at training time,
# the trained DS converges in joint space on its own, so the external linear
# stabilizer is no longer needed. Set to >0 only as a fallback if the trained
# field is divergent in some region (was 1.0 before the uniform-std retrain).
DS_GOAL_GAIN = 0.0


class IsaacProxySphereViz:
    """Small USD sphere markers that follow protected modulation points."""

    def __init__(self, radius=0.025, max_points=32):
        self.radius = float(radius)
        self.max_points = int(max_points)
        self.prims = {}
        self.enabled = False
        try:
            import omni.usd
            from pxr import Gf, UsdGeom
            self.stage = omni.usd.get_context().get_stage()
            self.Gf = Gf
            self.UsdGeom = UsdGeom
            self.enabled = self.stage is not None
        except Exception as exc:
            print(f"[WARN] Proxy sphere visualization unavailable: {exc}")

    def _make(self, arm, idx):
        path = f"/World/ModulationSpheres/{arm}_{idx:02d}"
        sphere = self.UsdGeom.Sphere.Define(self.stage, path)
        sphere.CreateRadiusAttr(self.radius)
        color = (1.0, 0.05, 0.05) if arm == "left" else (0.05, 0.20, 1.0)
        sphere.CreateDisplayColorAttr([self.Gf.Vec3f(*color)])
        sphere.CreateDisplayOpacityAttr([0.45])
        xform = self.UsdGeom.Xformable(sphere.GetPrim())
        xform.ClearXformOpOrder()
        translate = xform.AddTranslateOp()
        self.prims[(arm, idx)] = translate
        return translate

    def update(self, points_by_arm):
        if not self.enabled:
            return
        hidden = self.Gf.Vec3d(0.0, 0.0, -10.0)
        for arm, points in points_by_arm.items():
            points = np.asarray(points, dtype=float).reshape(-1, 3)
            for idx in range(self.max_points):
                op = self.prims.get((arm, idx)) or self._make(arm, idx)
                pos = points[idx] if idx < len(points) else hidden
                op.Set(self.Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))


def _load_deploy_config(path):
    if path is None:
        return {}
    with open(path, "r") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Deploy config must be a mapping: {path}")
    return payload.get("deploy", payload)


def _load_one_ds(ckpt_path, device):
    from src.neural_ds import StableNeuralDS, N_JOINTS
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = StableNeuralDS(
        n_joints    = N_JOINTS,
        hidden_dim  = cfg["hidden_dim"],
        lyap_hidden = cfg["lyapunov_hidden"],
        alpha       = cfg["alpha"],
        stable_skip_gain = cfg.get("stable_skip_gain", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {
        "model":      model,
        "state_mean": ckpt["state_mean"],
        "state_std":  ckpt["state_std"],
        "vel_scale":  ckpt["vel_scale"],
    }


def _quat_wxyz_to_matrix(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        return np.eye(3)
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _protected_points(ee_pos, ee_quat, n_link_spheres, spacing):
    points = [np.asarray(ee_pos, dtype=float)]
    if n_link_spheres <= 0:
        return np.asarray(points)
    R = _quat_wxyz_to_matrix(ee_quat)
    link_axis = R @ np.array([0.0, 0.0, -1.0])
    if np.linalg.norm(link_axis) < 1e-9:
        link_axis = np.array([0.0, 0.0, 1.0])
    link_axis = link_axis / (np.linalg.norm(link_axis) + 1e-12)
    for i in range(1, n_link_spheres + 1):
        points.append(points[0] + i * spacing * link_axis)
    return np.asarray(points)


def _add_gripper_width_points(points, ee_pos, ee_quat, lateral_offsets,
                              body_offsets):
    points = [np.asarray(p, dtype=float) for p in points]
    if not lateral_offsets:
        return np.asarray(points)
    R = _quat_wxyz_to_matrix(ee_quat)
    lateral_axis = R @ np.array([0.0, 1.0, 0.0])
    if np.linalg.norm(lateral_axis) < 1e-9:
        lateral_axis = np.array([1.0, 0.0, 0.0])
    lateral_axis = lateral_axis / (np.linalg.norm(lateral_axis) + 1e-12)
    body_axis = R @ np.array([0.0, 0.0, -1.0])
    if np.linalg.norm(body_axis) < 1e-9:
        body_axis = np.array([0.0, 0.0, 1.0])
    body_axis = body_axis / (np.linalg.norm(body_axis) + 1e-12)
    body_offsets = body_offsets if body_offsets else [0.0]
    ee_pos = np.asarray(ee_pos, dtype=float)
    for body_offset in body_offsets:
        body_center = ee_pos + float(body_offset) * body_axis
        points.append(body_center)
        for offset in lateral_offsets:
            points.append(body_center + float(offset) * lateral_axis)
    return np.asarray(points)


def _protected_points_from_links(ik, q, fallback_ee_pos, fallback_ee_quat,
                                 frames, samples_per_segment,
                                 n_fallback_spheres, fallback_spacing,
                                 gripper_lateral_offsets,
                                 gripper_lateral_body_offsets):
    frame_poses = []
    for frame in frames or []:
        try:
            pos, quat = ik.get_frame_world_pose(frame, q=q)
            frame_poses.append((np.asarray(pos, dtype=float), quat))
        except Exception:
            continue
    frame_points = [p for p, _ in frame_poses]
    if len(frame_points) < 2:
        points = _protected_points(
            fallback_ee_pos, fallback_ee_quat, n_fallback_spheres,
            fallback_spacing
        )
        return _add_gripper_width_points(
            points, fallback_ee_pos, fallback_ee_quat,
            gripper_lateral_offsets, gripper_lateral_body_offsets
        )
    points = [frame_points[0]]
    samples_per_segment = max(0, int(samples_per_segment))
    for a, b in zip(frame_points[:-1], frame_points[1:]):
        for i in range(1, samples_per_segment + 2):
            s = i / (samples_per_segment + 1)
            points.append((1.0 - s) * a + s * b)
    gripper_pos, gripper_quat = frame_poses[-1]
    if gripper_quat is None:
        gripper_quat = fallback_ee_quat
    return _add_gripper_width_points(
        points, gripper_pos, gripper_quat,
        gripper_lateral_offsets, gripper_lateral_body_offsets
    )


def _arm_param(value, arm, default=None):
    if isinstance(value, dict):
        if arm in value:
            return value[arm]
        return value.get("default", default)
    return value


def load_ds_sets(ckpt_dir, ckpt_arm_mode, device, primitives=None):
    """Load DS checkpoints for both arms.

    ckpt_arm_mode:
      'per_arm' -> load left_{primitive}.pt for the left arm and
                   right_{primitive}.pt for the right arm. This is the
                   default after we moved to per-arm training, since pooling
                   left+right with a via-point expert washes out the
                   curvature.
      'left'    -> use left_{primitive}.pt for BOTH arms (ablation).
      'right'   -> use right_{primitive}.pt for BOTH arms (ablation).

    Returns dict keyed by arm: {"left": {primitive: ds_dict}, "right": ...}.
    """
    if primitives is None:
        from src.primitives import DS_PRIMITIVES
        primitives = DS_PRIMITIVES

    if ckpt_arm_mode == "per_arm":
        prefixes = {"left": "left", "right": "right"}
    elif ckpt_arm_mode in ("left", "right"):
        prefixes = {"left": ckpt_arm_mode, "right": ckpt_arm_mode}
    else:
        raise ValueError(f"unknown ckpt_arm_mode: {ckpt_arm_mode!r}")

    return {
        arm: {p: _load_one_ds(ckpt_dir / f"{prefixes[arm]}_{p}.pt", device)
              for p in primitives}
        for arm in ("left", "right")
    }


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--deploy_config", type=str, default=None,
                     help="YAML file with deploy argument defaults.")
    pre_args, _ = pre.parse_known_args()
    deploy_defaults = _load_deploy_config(pre_args.deploy_config)

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt_arm", type=str, default="per_arm",
                        choices=["per_arm", "left", "right"],
                        help="Checkpoint loading mode. 'per_arm' (default) "
                             "loads left_*.pt for the left arm and right_*.pt "
                             "for the right arm. 'left' or 'right' loads one "
                             "arm's checkpoints for both arms (ablation).")
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--use_safe", action="store_true")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override Lyapunov decay rate at deployment "
                             "(higher = more aggressive projection).")
    parser.add_argument("--ds_scale", type=float, default=1.0,
                        help="Scale learned DS velocity. Set to 0 for a pure "
                             "joint-space attractor sanity check (then only "
                             "DS_GOAL_GAIN drives motion).")
    parser.add_argument("--max_joint_vel", type=float, default=None,
                        help="Deployment joint velocity clamp in rad/s. "
                             "Defaults to training.max_joint_vel from config.")
    parser.add_argument("--no_modulation", action="store_true",
                        help="Disable DS modulation (ablation).")
    parser.add_argument("--mod_safe_radius", type=float, default=None,
                        help="Override inter-arm EE modulation safety radius "
                             "in meters. Larger values create earlier, more "
                             "conservative avoidance.")
    parser.add_argument("--mod_reactivity", type=float, default=None,
                        help="Exponent for inter-arm modulation. Lower values "
                             "make modulation act earlier over a wider range.")
    parser.add_argument("--mod_isoline", type=float, default=None,
                        help="Gamma contour to track. Values >1 expand the "
                             "effective modulation boundary.")
    parser.add_argument("--mod_max_pairs", type=int, default=4,
                        help="Number of closest protected sphere pairs to "
                             "modulate. Use 0 for all pairs.")
    parser.add_argument("--no_preserve_mod_speed", action="store_true",
                        help="Do not rescale modulated Cartesian velocity back "
                             "to nominal speed.")
    parser.add_argument("--priority_arm", type=str, default="left",
                        choices=["left", "right"],
                        help="Arm that starts as the lower-modulation priority arm.")
    parser.add_argument("--priority_policy", type=str, default="fixed",
                        choices=["fixed", "closest_to_stack"],
                        help="How to choose the priority arm.")
    parser.add_argument("--priority_hysteresis", type=float, default=0.04,
                        help="Closest-to-stack policy switch margin in metres.")
    parser.add_argument("--priority_mod_weight", type=float, default=0.25,
                        help="Blend weight for modulation on the priority arm.")
    parser.add_argument("--yield_mod_weight", type=float, default=1.0,
                        help="Blend weight for modulation on the yielding arm.")
    parser.add_argument("--yield_mod_speed_scale", type=float, default=1.0,
                        help="Extra velocity scale applied to the non-priority "
                             "arm after modulation. Values below 1 make the "
                             "yielding arm slow down instead of trying to "
                             "route around aggressively.")
    parser.add_argument("--no_lateral_order_modulation", action="store_true",
                        help="Disable virtual separating-plane modulation that "
                             "keeps the left arm on the left side of the right arm.")
    parser.add_argument("--lateral_order_min_separation", type=float, default=None,
                        help="Minimum signed X separation between protected "
                             "left/right points before lateral-order modulation "
                             "pushes an arm back to its own side.")
    parser.add_argument("--lateral_order_reactivity", type=float, default=None,
                        help="Reactivity exponent for lateral-order modulation.")
    parser.add_argument("--lateral_order_mod_weight", type=float, default=None,
                        help="Blend weight for lateral-order modulation.")
    parser.add_argument("--lateral_order_lambda_floor", type=float, default=None,
                        help="Most negative normal eigenvalue allowed inside "
                             "the lateral-order separating margin.")
    parser.add_argument("--link_spheres", type=int, default=None,
                        help="Fallback proxy spheres behind the EE.")
    parser.add_argument("--link_sphere_spacing", type=float, default=None,
                        help="Fallback proxy sphere spacing in metres.")
    parser.add_argument("--link_frames", nargs="+", default=None,
                        help="Lula FK frames used for distal-link proxy points.")
    parser.add_argument("--link_samples_per_segment", type=int, default=None,
                        help="Extra proxy samples between FK frames.")
    parser.add_argument("--gripper_lateral_offsets", nargs="*", type=float,
                        default=None,
                        help="Gripper body lateral proxy offsets in local Y.")
    parser.add_argument("--gripper_lateral_body_offsets", nargs="*", type=float,
                        default=None,
                        help="Offsets back from EE along local -Z for gripper proxies.")
    parser.add_argument("--show_proxy_spheres", action="store_true",
                        help="Show protected modulation proxy spheres in the Isaac viewport.")
    parser.add_argument("--proxy_sphere_radius", type=float, default=0.025,
                        help="Radius (meters) for --show_proxy_spheres markers.")
    parser.add_argument("--yield_radius", type=float, default=None,
                        help="Distance from shared stack where the non-priority "
                             "arm waits before place.")
    parser.add_argument("--stack_keepout_radius", type=float, default=0.0,
                        help="If >0, keep the non-priority arm out of this "
                             "XY radius around the shared stack while the "
                             "priority arm is transporting/placing.")
    parser.add_argument("--stack_keepout_wait_x", type=float, default=0.25,
                        help="Absolute X offset for the non-priority arm's "
                             "side wait pose during stack keepout.")
    parser.add_argument("--link_safety_radius", type=float, default=0.18,
                        help="Minimum sampled link-to-link distance in meters. "
                             "If any sampled links are closer, pause one arm "
                             "so non-EE joints do not collide while EEs avoid.")
    parser.add_argument("--link_safety_hysteresis", type=float, default=0.02,
                        help="Extra clearance above --link_safety_radius "
                             "required before releasing a link-safety hold.")
    parser.add_argument("--link_safety_print_every", type=int, default=60,
                        help="Print repeated link-safety hold messages every "
                             "N simulation steps. Set 0 to print only on "
                             "hold/release transitions.")
    parser.add_argument("--no_link_safety_hold", action="store_true",
                        help="Disable conservative sampled-link safety hold.")
    parser.add_argument("--link_safety_hold", action="store_false",
                        dest="no_link_safety_hold",
                        help="Enable conservative sampled-link safety hold. "
                             "Disabled by default.")
    parser.set_defaults(no_link_safety_hold=True)
    parser.add_argument("--stagger_steps", type=int, default=None,
                        help="Initial right-arm launch delay in physics steps. "
                             "Defaults to coordination.start_stagger_steps "
                             "(0 in the default config).")
    parser.add_argument("--return_home_tol", type=float, default=0.05,
                        help="Joint-space tolerance for parking an arm at its "
                             "initial home pose after its final block is placed.")
    parser.add_argument("--no_return_home_after_done", action="store_true",
                        help="Leave an arm at its final pose after it finishes "
                             "all blocks. Default behavior returns finished "
                             "arms to their initial home pose so they do not "
                             "block the shared stack.")
    parser.add_argument("--kinematic_carry", action="store_true",
                        help="After grasp, attach each active block to its EE "
                             "kinematically until place. Use this to debug the "
                             "DS/task pipeline separately from gripper contact.")
    parser.add_argument("--advance_on_timeout", action="store_true",
                        help="Legacy debug behavior: advance a primitive on "
                             "timeout even when q has not reached q_goal.")
    parser.add_argument("--ik_primitives", type=str, default="grasp,lift,place",
                        help="Comma-separated primitives to execute with the "
                             "Lula joint-space controller instead of the "
                             "learned DS.")
    parser.add_argument("--ik_goal_gain", type=float, default=3.0,
                        help="Joint-space attraction gain for "
                             "Lula-controlled primitives.")
    parser.add_argument("--grasp_height", type=float, default=None,
                        help="Override config heights.grasp in meters.")
    parser.add_argument("--grasp_z_offset", type=float, default=0.0,
                        help="Additive offset to config heights.grasp in meters. "
                             "Negative values make the gripper descend lower.")
    parser.add_argument("--gripper_steps", type=int, default=None,
                        help="Override sim.gripper_steps for close/open dwell.")
    parser.add_argument("--grasp_offset", type=float, nargs=2, default=None,
                        metavar=("DX", "DY"),
                        help="Extra world-frame XY pick offset in meters for "
                             "both arms.")
    parser.add_argument("--debug_grasp", action="store_true",
                        help="Print finger joint positions and block height "
                             "after close/lift for physical grasp debugging.")
    parser.add_argument("--debug_ik", action="store_true",
                        help="Print Cartesian targets, IK success, seeds, and "
                             "q_goal at primitive transitions.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--done_tol", type=float, default=0.05,
                        help="Legacy completion tolerance. Used as the default "
                             "joint-space tolerance unless --joint_done_tol is set.")
    parser.add_argument("--joint_done_tol", type=float, default=None,
                        help="L2 joint-space tolerance for DS primitive completion. "
                             "Defaults to --done_tol.")
    parser.add_argument("--cart_done_tol", type=float, default=IK_CART_DONE_TOL,
                        help="Cartesian EE completion tolerance (meters) for IK primitives "
                             "(grasp/lift/place), except place uses --place_cart_done_tol.")
    parser.add_argument("--place_cart_done_tol", type=float, default=0.03,
                        help="Cartesian EE completion tolerance (meters) for place. "
                             "Use a practical value because Lula/Isaac FK can "
                             "settle a little off the requested Cartesian target.")
    parser.add_argument("--gif_out", type=str, default=None,
                        help="Optional offscreen GIF path. Works with --headless "
                             "when Isaac offscreen rendering is available.")
    parser.add_argument("--video", action="store_true",
                        help="IsaacLab-style convenience flag: record a "
                             "headless-compatible video to --video_dir unless "
                             "--gif_out is set explicitly.")
    parser.add_argument("--video_dir", type=str, default="data/results",
                        help="Output directory used by --video.")
    parser.add_argument("--gif_fps", type=int, default=20)
    parser.add_argument("--gif_stride", type=int, default=4,
                        help="Capture every N sim steps.")
    parser.add_argument("--gif_size", type=str, default="640x480",
                        help="Offscreen GIF size, e.g. 640x480.")
    if deploy_defaults:
        parser.set_defaults(**deploy_defaults)
    args = parser.parse_args()
    if args.video and not args.gif_out:
        args.gif_out = str(Path(args.video_dir) / "dual_neural.gif")
    joint_done_tol = args.done_tol if args.joint_done_tol is None else args.joint_done_tol

    from isaacsim import SimulationApp
    _app_cfg = {"headless": args.headless}
    if args.gif_out:
        try:
            _gif_w, _gif_h = (int(v) for v in args.gif_size.lower().split("x", 1))
        except Exception:
            _gif_w, _gif_h = 640, 480
        _app_cfg.update({
            "width": _gif_w,
            "height": _gif_h,
            "hide_ui": False,
        })
    elif not args.headless:
        _app_cfg.update({"width": 1280, "height": 720})
    simulation_app = SimulationApp(_app_cfg)

    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core.utils.types import ArticulationAction
    from src.env import DualArmEnv
    from src.coordinator import TaskSequencer
    from src.franka_ik import FrankaIK
    from src.modulation import InterArmModulation, jacobian_finite_difference
    from src.offscreen_recorder import OffscreenGifRecorder
    from src.primitives import (
        DS_PRIMITIVES,
        SCRIPTED_PRIMITIVES,
        gripper_action_for_primitive,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DualArmEnv(config_path=args.config, arms=("left", "right"))
    cfg = env.cfg
    if args.grasp_height is not None:
        cfg["heights"]["grasp"] = float(args.grasp_height)
    elif args.grasp_z_offset:
        cfg["heights"]["grasp"] = float(cfg["heights"]["grasp"]) + float(args.grasp_z_offset)
    if args.grasp_height is not None or args.grasp_z_offset:
        print(f"[DEPLOY] Grasp target height -> {cfg['heights']['grasp']:.3f} m")
    if args.gripper_steps is not None:
        cfg["sim"]["gripper_steps"] = int(args.gripper_steps)
        print(f"[DEPLOY] Gripper dwell -> {cfg['sim']['gripper_steps']} steps")
    if args.grasp_offset is not None:
        offsets = cfg.setdefault("block", {}).setdefault("grasp_xy_offset", {})
        for arm in ("left", "right"):
            base = np.asarray(offsets.get(arm, [0.0, 0.0]), dtype=float)
            offsets[arm] = (base + np.asarray(args.grasp_offset, dtype=float)).tolist()
        print(f"[DEPLOY] Grasp XY offsets -> {offsets}")
    try:
        gif_size = tuple(int(v) for v in args.gif_size.lower().split("x", 1))
    except Exception:
        raise ValueError("--gif_size must look like WIDTHxHEIGHT, e.g. 640x480")
    recorder = OffscreenGifRecorder(
        camera_prim_path="/World/Camera",
        out_path=args.gif_out,
        size=gif_size,
        fps=args.gif_fps,
        stride=args.gif_stride,
    )
    render_steps = (not args.headless) or recorder.enabled

    def step_env():
        env.step(render=render_steps)
        recorder.capture()

    def _handle_signal(signum, _frame):
        print(f"[DEPLOY] Received signal {signum}; closing recorder.")
        recorder.close()
        simulation_app.close()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    franka = {"left": env.frankas["left"], "right": env.frankas["right"]}
    ik_kin = {arm: FrankaIK(franka[arm]) for arm in franka}
    ik_primitives = {p.strip() for p in args.ik_primitives.split(",") if p.strip()}
    if not ik_primitives:
        ik_primitives = set(SCRIPTED_PRIMITIVES)
    valid_primitives = {"reach", "grasp", "lift", "transport", "place"}
    bad_primitives = ik_primitives - valid_primitives
    if bad_primitives:
        raise ValueError(f"Unknown --ik_primitives entries: {sorted(bad_primitives)}")

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    if not ckpt_dir.is_absolute() and not ckpt_dir.exists():
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / ckpt_dir
        if candidate.exists():
            ckpt_dir = candidate
    ds_primitives = [p for p in DS_PRIMITIVES if p not in ik_primitives]
    ds_sets = load_ds_sets(ckpt_dir, args.ckpt_arm, device, primitives=ds_primitives)
    print(f"[DEPLOY] DS checkpoint mode: {args.ckpt_arm} "
          f"(left arm <- {'left' if args.ckpt_arm in ('per_arm', 'left') else 'right'}_*.pt, "
          f"right arm <- {'right' if args.ckpt_arm in ('per_arm', 'right') else 'left'}_*.pt)")

    # Override training alpha to drive faster Lyapunov decay at deployment
    if args.alpha is not None:
        for arm_set in ds_sets.values():
            for p in arm_set.values():
                p["model"].alpha = args.alpha
        print(f"[DEPLOY] Overriding alpha -> {args.alpha}")

    seq = TaskSequencer(env, cfg)
    mod_radius = (cfg["coordination"]["ee_safety_radius"]
                  if args.mod_safe_radius is None else args.mod_safe_radius)
    mod_reactivity = args.mod_reactivity
    if mod_reactivity is None:
        mod_reactivity = cfg["coordination"].get("modulation_reactivity", 2.0)
    mod_isoline = args.mod_isoline
    if mod_isoline is None:
        mod_isoline = cfg["coordination"].get("modulation_isoline", 1.0)
    mod = InterArmModulation(
        safe_radius=mod_radius,
        reactivity=mod_reactivity,
        preserve_speed=not args.no_preserve_mod_speed,
        isoline=mod_isoline,
        max_pairs=(None if args.mod_max_pairs == 0 else args.mod_max_pairs),
    )
    lateral_order_min_separation = args.lateral_order_min_separation
    if lateral_order_min_separation is None:
        lateral_order_min_separation = cfg["coordination"].get(
            "lateral_order_min_separation", 0.08
        )
    lateral_order_reactivity = args.lateral_order_reactivity
    if lateral_order_reactivity is None:
        lateral_order_reactivity = cfg["coordination"].get(
            "lateral_order_reactivity", mod_reactivity
        )
    lateral_order_mod_weight = args.lateral_order_mod_weight
    if lateral_order_mod_weight is None:
        lateral_order_mod_weight = cfg["coordination"].get(
            "lateral_order_mod_weight", 1.0
        )
    lateral_order_lambda_floor = args.lateral_order_lambda_floor
    if lateral_order_lambda_floor is None:
        lateral_order_lambda_floor = cfg["coordination"].get(
            "lateral_order_lambda_floor", -0.75
        )
    link_spheres = args.link_spheres
    if link_spheres is None:
        link_spheres = cfg["coordination"].get("link_proxy_spheres", 3)
    link_sphere_spacing = args.link_sphere_spacing
    if link_sphere_spacing is None:
        link_sphere_spacing = cfg["coordination"].get("link_proxy_spacing", 0.055)
    link_frames = args.link_frames
    if link_frames is None:
        link_frames = cfg["coordination"].get(
            "link_proxy_frames",
            ["panda_link5", "panda_link6", "panda_link7", "right_gripper"],
        )
    link_samples_per_segment = args.link_samples_per_segment
    if link_samples_per_segment is None:
        link_samples_per_segment = cfg["coordination"].get(
            "link_proxy_samples_per_segment", 2
        )
    gripper_lateral_offsets = args.gripper_lateral_offsets
    if gripper_lateral_offsets is None:
        gripper_lateral_offsets = cfg["coordination"].get(
            "gripper_proxy_lateral_offsets", [-0.065, 0.065]
        )
    gripper_lateral_body_offsets = args.gripper_lateral_body_offsets
    if gripper_lateral_body_offsets is None:
        gripper_lateral_body_offsets = cfg["coordination"].get(
            "gripper_proxy_body_offsets", [0.075, 0.13, 0.185]
        )
    priority_arm = args.priority_arm
    yield_radius = (
        cfg["coordination"].get("yield_radius", 0.12)
        if args.yield_radius is None else args.yield_radius
    )

    physics_dt = cfg["sim"]["physics_dt"]
    max_joint_vel = (cfg["training"]["max_joint_vel"]
                     if args.max_joint_vel is None else args.max_joint_vel)
    stagger_steps = (cfg["coordination"].get("start_stagger_steps", 0)
                     if args.stagger_steps is None else args.stagger_steps)
    arm_start_step = {"left": 0, "right": max(0, stagger_steps)}
    # Return-home uses the configured trained home pose, not whatever Isaac
    # reports after startup/settling. These are the same joint values used as
    # the collection reset pose.
    home_q = {
        arm: np.asarray(cfg["arms"][f"default_joints_{arm}"], dtype=float).copy()
        for arm in ("left", "right")
    }
    q_cmd_state = {arm: franka[arm].get_joint_positions().copy()
                   for arm in ("left", "right")}
    desired_finger_width = {"left": 0.04, "right": 0.04}
    arm_parked = {arm: False for arm in ("left", "right")}

    proxy_viz = (
        IsaacProxySphereViz(radius=args.proxy_sphere_radius)
        if args.show_proxy_spheres else None
    )

    # Open both grippers
    for arm in franka:
        franka[arm].gripper.apply_action(
            ArticulationAction(joint_positions=np.array([0.04, 0.04]))
        )

    # Let blocks settle before querying their positions
    for _ in range(60):
        step_env()
    q_cmd_state = {arm: franka[arm].get_joint_positions().copy()
                   for arm in ("left", "right")}

    # Initialise q_goals per arm
    ik_failed = {"failed": False}

    def update_q_goal(arm):
        task = seq.tasks[arm]
        cart = seq.cartesian_target(arm)
        if cart is None:
            return None
        # Seed Lula IK from the previous primitive's q_goal when available
        # (matches the collection-time IK seed, which was effectively the
        # previous q_goal because the direct-joint-set expert settled to
        # within joint_done_tol). Seeding from the lagging live pose lets a
        # 0.1–0.2 rad tracking lag flip Lula's IK branch and produce a
        # q_goal 2+ rad away in joint space, which drives the DS OOD.
        prev_q_goal = getattr(task, "q_goal", None)
        if prev_q_goal is not None:
            q_seed = np.asarray(prev_q_goal, dtype=float).copy()
        else:
            q_seed = franka[arm].get_joint_positions()[:7].copy()
        ee_quat = seq.ee_orientation(arm)
        q_goal, ok = ik_kin[arm].solve(cart, target_quat=ee_quat, q_seed=q_seed)
        if args.debug_ik or not ok:
            print(f"[IK] {arm}/{task.current_primitive:9s} ok={ok} "
                  f"cart={cart.round(3)} seed={q_seed.round(3)} "
                  f"q_goal={q_goal.round(3)} "
                  f"||seed-goal||={np.linalg.norm(q_seed - q_goal):.3f}")
        if not ok:
            print(f"[ERROR] {arm}/{task.current_primitive} IK failed. "
                  "Aborting instead of treating the current joint state as "
                  "the primitive goal.")
            ik_failed["failed"] = True
            return None
        seq.tasks[arm].q_goal = q_goal
        if task.current_primitive in ("transport", "place"):
            print(f"[STACK] {arm}/{task.current_primitive} "
                  f"slot={seq.stack_slot_index(arm)} "
                  f"block_center_z={task.reserved_goal_z:.3f} "
                  f"ee_target_z={cart[2]:.3f}")
        return q_goal

    for arm in ("left", "right"):
        update_q_goal(arm)
    if ik_failed["failed"]:
        recorder.close()
        simulation_app.close()
        return

    last_prim = {arm: seq.tasks[arm].current_primitive for arm in ("left", "right")}
    prim_steps = {"left": 0, "right": 0}
    # 30× the collection budget per primitive before we give up.
    prim_timeout = {p: s * 30
                    for p, s in cfg["sim"]["steps_per_primitive"].items()}

    print(f"[DEPLOY] Dual-arm joint-space DS — safe={args.use_safe}, "
          f"modulation={'OFF' if args.no_modulation else 'ON'}, "
          f"ds_scale={args.ds_scale}, ds_goal_gain={DS_GOAL_GAIN}, "
          f"max_joint_vel={max_joint_vel}")
    if not args.no_modulation:
        print(f"[DEPLOY] Modulation radius={mod_radius:.3f}m "
              f"reactivity={mod_reactivity:.2f} isoline={mod_isoline:.2f} "
              f"priority={priority_arm} policy={args.priority_policy} "
              f"priority_w={args.priority_mod_weight} "
              f"yield_w={args.yield_mod_weight}")
        print(f"[DEPLOY] Protected frames={link_frames} "
              f"samples/segment={link_samples_per_segment} "
              f"gripper_offsets={gripper_lateral_offsets}")
        if args.stack_keepout_radius > 0.0:
            print(f"[DEPLOY] Stack keepout radius={args.stack_keepout_radius:.3f}m "
                  f"wait_x={args.stack_keepout_wait_x:.3f}m")
        if not args.no_lateral_order_modulation:
            print(f"[DEPLOY] Lateral order modulation: "
                  f"min_sep={lateral_order_min_separation:.3f}m "
                  f"reactivity={lateral_order_reactivity:.2f} "
                  f"weight={lateral_order_mod_weight:.2f} "
                  f"lambda_floor={lateral_order_lambda_floor:.2f}")
    if ik_primitives:
        print(f"[DEPLOY] IK primitives: {', '.join(sorted(ik_primitives))}")
    if stagger_steps > 0:
        print(f"[DEPLOY] Initial stagger: right arm starts after "
              f"{stagger_steps} steps ({stagger_steps * physics_dt:.2f}s)")

    held_block = {"left": None, "right": None}
    held_offset = {"left": np.zeros(3), "right": np.zeros(3)}
    link_hold_arm = None
    safety_hold_steps = {"left": 0, "right": 0}

    def other_arm(arm):
        return "right" if arm == "left" else "left"

    def goal_xy_for_arm(arm):
        return np.asarray(seq.tasks[arm].goal_xy, dtype=float)

    def avoidance_weight(arm):
        if args.no_modulation:
            return 0.0
        task = seq.tasks[arm]
        if task.is_done():
            return 0.0
        if arm == priority_arm:
            return float(_arm_param(args.priority_mod_weight, arm, 0.25))
        return float(_arm_param(args.yield_mod_weight, arm, 1.0))

    def update_priority():
        nonlocal priority_arm
        if args.priority_policy != "closest_to_stack" or args.no_modulation:
            return
        candidates = []
        for arm in ("left", "right"):
            task = seq.tasks[arm]
            if not task.is_done() and task.current_primitive in ("transport", "place"):
                candidates.append(arm)
        if len(candidates) == 1:
            if priority_arm != candidates[0]:
                priority_arm = candidates[0]
                print(f"[COORD] Priority -> {priority_arm} (sole active arm)")
            return
        if len(candidates) < 2:
            return
        dist = {}
        for arm in candidates:
            ee = ik_frame_position(arm)
            dist[arm] = float(np.linalg.norm(ee[:2] - goal_xy_for_arm(arm)))
        closest = min(candidates, key=lambda arm: dist[arm])
        current = priority_arm if priority_arm in candidates else None
        if current is None or dist[closest] + args.priority_hysteresis < dist[current]:
            priority_arm = closest
            print(f"[COORD] Priority -> {priority_arm} "
                  f"(closer to stack: {dist[closest]:.3f}m)")

    def can_place(arm):
        other = other_arm(arm)
        other_task = seq.tasks[other]
        if other_task.is_done():
            return True
        if other_task.current_primitive == "place":
            return arm == priority_arm
        if other_task.current_primitive not in ("transport", "place"):
            return True
        ee_other = ik_frame_position(other)
        return np.linalg.norm(ee_other[:2] - goal_xy_for_arm(arm)) > yield_radius

    def stack_keepout_velocity(arm, q_now):
        """Move the non-priority arm to a side wait pose near the stack.

        Pure modulation can make the yielding arm arc over the priority arm.
        This small task-level keepout prevents that by reserving the shared
        stack airspace for the priority arm during transport/place.
        """
        if args.stack_keepout_radius <= 0.0 or arm == priority_arm:
            return None
        task = seq.tasks[arm]
        priority_task = seq.tasks[priority_arm]
        if task.is_done() or priority_task.is_done():
            return None
        if task.current_primitive != "transport":
            return None
        if priority_task.current_primitive not in ("transport", "place"):
            return None

        stack_xy = goal_xy_for_arm(priority_arm)
        ee = ik_frame_position(arm)
        dist_to_stack = float(np.linalg.norm(ee[:2] - stack_xy))
        if dist_to_stack > args.stack_keepout_radius:
            return None

        side = -1.0 if arm == "left" else 1.0
        wait_z = max(cfg["heights"]["lift"], seq.stack_clearance_z_for_task(task))
        wait_cart = np.array([
            stack_xy[0] + side * abs(args.stack_keepout_wait_x),
            stack_xy[1],
            wait_z,
        ])
        q_wait, ok = ik_kin[arm].solve(
            wait_cart, target_quat=seq.ee_orientation(arm), q_seed=q_now)
        if not ok:
            return np.zeros(7)
        return np.clip(
            -args.ik_goal_gain * (q_now - q_wait),
            -max_joint_vel,
            max_joint_vel,
        )

    def use_lateral_order_modulation(arm):
        task = seq.tasks[arm]
        if task.is_done():
            return False
        # Both arms eventually need to enter the shared stack column.  The
        # ordering plane prevents transport crossing, but it must not fight the
        # final place primitive once task-level priority has granted access.
        return task.current_primitive != "place"

    def protected_points_for_arm(arm):
        q = franka[arm].get_joint_positions()[:7].copy()
        try:
            ee_pos = ik_kin[arm].forward_position(q)
            _, ee_quat = ik_kin[arm].get_frame_world_pose("right_gripper", q=q)
        except Exception:
            ee_pos, ee_quat = env.get_ee_pose(arm)
        if ee_quat is None:
            ee_quat = seq.ee_orientation(arm)
        return _protected_points_from_links(
            ik_kin[arm], q, ee_pos, ee_quat,
            link_frames, link_samples_per_segment, max(0, link_spheres),
            link_sphere_spacing, gripper_lateral_offsets,
            gripper_lateral_body_offsets,
        )

    def min_link_distance():
        left_pts = env.get_arm_link_positions("left")
        right_pts = env.get_arm_link_positions("right")
        best = float("inf")
        for lp in left_pts:
            for rp in right_pts:
                d = float(np.linalg.norm(lp - rp))
                if d < best:
                    best = d
        return best

    def link_safety_hold_arm():
        """Pick the arm to pause when sampled links are too close.

        A finished arm returning home gets priority because clearing it from
        the stack usually resolves the conflict. Otherwise the arm with fewer
        completed placements moves first; ties use the initial stagger order.
        """
        left_done = seq.tasks["left"].is_done() and not arm_parked["left"]
        right_done = seq.tasks["right"].is_done() and not arm_parked["right"]
        if left_done and not right_done:
            return "right"
        if right_done and not left_done:
            return "left"
        left_count = seq.placed_per_arm["left"]
        right_count = seq.placed_per_arm["right"]
        if left_count < right_count:
            return "right"
        if right_count < left_count:
            return "left"
        return "right"

    def carry_held_blocks():
        for arm in ("left", "right"):
            if held_block[arm] is None:
                continue
            # Use the same frame as IK targets (Lula right_gripper) so the
            # kinematic-carry offset is consistent with reach/grasp/lift/place.
            ee = ik_frame_position(arm).copy()
            obj = env.get_block_obj(held_block[arm])
            obj.set_world_pose(position=ee + held_offset[arm],
                               orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))

    def print_grasp_debug(arm, label, block_name):
        if not args.debug_grasp or block_name is None:
            return
        q_full = franka[arm].get_joint_positions()
        block_pos = env.get_block_positions()[block_name]
        ee_pos = ik_frame_position(arm)
        table_top = cfg["table"]["height"]
        print(f"[GRASP] {arm} {label}: block={block_name} "
              f"block_xy=({block_pos[0]:+.3f},{block_pos[1]:+.3f}) "
              f"ee_xy=({ee_pos[0]:+.3f},{ee_pos[1]:+.3f}) "
              f"xy_err={np.linalg.norm(ee_pos[:2] - block_pos[:2]):.4f} "
              f"z={block_pos[2]:.4f} dz_table={block_pos[2] - table_top:.4f} "
              f"fingers={q_full[7:9].round(4)} "
              f"target_width={desired_finger_width[arm]:.4f}")

    def apply_full_command(arm, q7=None):
        full_cmd = franka[arm].get_joint_positions().copy()
        if q7 is not None:
            full_cmd[:7] = q7[:7]
        full_cmd[7:9] = desired_finger_width[arm]
        franka[arm].apply_action(ArticulationAction(joint_positions=full_cmd))

    def hold_gripper(arm, width, steps):
        desired_finger_width[arm] = float(width)
        for _ in range(steps):
            full_cmd = franka[arm].get_joint_positions().copy()
            full_cmd[7:9] = desired_finger_width[arm]
            franka[arm].apply_action(ArticulationAction(joint_positions=full_cmd))
            step_env()
            carry_held_blocks()
        q_cmd_state[arm] = franka[arm].get_joint_positions().copy()
        q_cmd_state[arm][7:9] = desired_finger_width[arm]

    def snap_held_block_to_stack(arm):
        if held_block[arm] is None:
            return
        obj = env.get_block_obj(held_block[arm])
        obj.set_world_pose(position=seq.stack_target_position(arm),
                           orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))

    def ik_frame_position(arm):
        try:
            return ik_kin[arm].forward_position(
                franka[arm].get_joint_positions()[:7].copy()
            )
        except Exception as exc:
            if args.debug_ik:
                print(f"[WARN] {arm} Lula FK failed, using env EE pose: {exc}")
            return env.get_ee_pose(arm)[0]

    def joint_lula_move_to_cart(arm, target_cart, steps=120, cart_tol=0.03,
                                label="stack clearance"):
        q_seed = franka[arm].get_joint_positions()[:7].copy()
        q_goal, ok = ik_kin[arm].solve(
            target_cart, target_quat=None, q_seed=q_seed)
        if not ok:
            print(f"[WARN] {arm} {label} IK failed for target={target_cart.round(3)}")
            return False
        for _ in range(steps):
            q_now = franka[arm].get_joint_positions()[:7].copy()
            ee_now = ik_frame_position(arm)
            if np.linalg.norm(ee_now - target_cart) < cart_tol:
                break
            q_dot = np.clip(
                -args.ik_goal_gain * (q_now - q_goal),
                -max_joint_vel,
                max_joint_vel,
            )
            q_cmd_state[arm][:7] = q_cmd_state[arm][:7] + q_dot * physics_dt
            apply_full_command(arm, q_cmd_state[arm][:7])
            step_env()
            carry_held_blocks()
        q_cmd_state[arm] = franka[arm].get_joint_positions().copy()
        return np.linalg.norm(ik_frame_position(arm) - target_cart) < cart_tol

    for step in range(args.max_steps):
        if not simulation_app.is_running():
            break
        update_priority()

        # Cache EE positions BEFORE we move so modulation uses consistent state
        ee_pos = {arm: env.get_ee_pose(arm)[0].copy() for arm in ("left", "right")}
        protected_points = {arm: protected_points_for_arm(arm)
                            for arm in ("left", "right")}
        if proxy_viz is not None:
            proxy_viz.update(protected_points)

        # Compute nominal q̇ for each arm (in parallel, before any commits)
        q_dots = {}
        for arm in ("left", "right"):
            task = seq.tasks[arm]
            if step < arm_start_step[arm]:
                q_dots[arm] = None
                q_cmd_state[arm] = franka[arm].get_joint_positions().copy()
                continue
            if task.is_done():
                if args.no_return_home_after_done or arm_parked[arm]:
                    q_dots[arm] = None
                    q_cmd_state[arm] = franka[arm].get_joint_positions().copy()
                    continue
                q = franka[arm].get_joint_positions()[:7].copy()
                x_home = q - home_q[arm]
                if np.linalg.norm(x_home) < args.return_home_tol:
                    q_dots[arm] = None
                    arm_parked[arm] = True
                    q_cmd_state[arm] = franka[arm].get_joint_positions().copy()
                    print(f"[DEPLOY] {arm} arm parked at home.")
                    continue
                q_dots[arm] = np.clip(
                    -args.ik_goal_gain * x_home,
                    -max_joint_vel,
                    max_joint_vel,
                )
                continue

            if task.current_primitive != last_prim[arm]:
                update_q_goal(arm)
                if ik_failed["failed"]:
                    recorder.close()
                    simulation_app.close()
                    return
                last_prim[arm] = task.current_primitive
                prim_steps[arm] = 0
                safety_hold_steps[arm] = 0
                q_cmd_state[arm] = franka[arm].get_joint_positions().copy()

            prim_steps[arm] += 1

            q = franka[arm].get_joint_positions()[:7].copy()
            if task.current_primitive == "place" and not can_place(arm):
                q_dots[arm] = None
                q_cmd_state[arm] = franka[arm].get_joint_positions().copy()
                continue
            keepout_dot = stack_keepout_velocity(arm, q)
            if keepout_dot is not None:
                q_dots[arm] = keepout_dot
                safety_hold_steps[arm] += 1
                continue
            if task.current_primitive in ik_primitives:
                x = q - task.q_goal
                q_dots[arm] = np.clip(
                    -args.ik_goal_gain * x,
                    -max_joint_vel,
                    max_joint_vel,
                )
                continue

            ds = ds_sets[arm][task.current_primitive]
            x = q - task.q_goal
            x_n = (x - ds["state_mean"]) / ds["state_std"]
            x_t = torch.tensor(x_n, dtype=torch.float32, device=device).unsqueeze(0)

            if args.use_safe:
                scale_factor = torch.tensor(
                    ds["vel_scale"] / ds["state_std"],
                    dtype=torch.float32, device=device).unsqueeze(0)
                qd_n = ds["model"].safe_velocity(x_t, scale_factor=scale_factor)
            else:
                with torch.no_grad():
                    qd_n = ds["model"](x_t)
            q_dots[arm] = args.ds_scale * qd_n.cpu().numpy().squeeze(0) * ds["vel_scale"]
            if DS_GOAL_GAIN > 0:
                q_dots[arm] = q_dots[arm] - DS_GOAL_GAIN * x
            q_dots[arm] = np.clip(q_dots[arm], -max_joint_vel, max_joint_vel)

        # Apply modulation between the two arms
        if not args.no_modulation:
            for arm, other in (("left", "right"), ("right", "left")):
                if q_dots[arm] is None:
                    continue
                if seq.tasks[arm].current_primitive in ik_primitives:
                    continue
                # Compute Jacobian by finite-diff (slow but version-stable)
                J = jacobian_finite_difference(franka[arm])
                q_dot_nom = q_dots[arm]
                q_dot_mod = mod.modulate_joint_velocity_points(
                    q_dot_nominal=q_dot_nom,
                    self_points=protected_points[arm],
                    obstacle_points=protected_points[other],
                    jacobian=J,
                )
                w = avoidance_weight(arm)
                q_dots[arm] = (1.0 - w) * q_dot_nom + w * q_dot_mod
                if arm != priority_arm:
                    q_dots[arm] *= float(args.yield_mod_speed_scale)
            if not args.no_lateral_order_modulation:
                for arm, other in (("left", "right"), ("right", "left")):
                    if q_dots[arm] is None:
                        continue
                    if not use_lateral_order_modulation(arm):
                        continue
                    side_axis = np.array(
                        [-1.0, 0.0, 0.0] if arm == "left" else [1.0, 0.0, 0.0]
                    )
                    J = jacobian_finite_difference(franka[arm])
                    q_dot_nom = q_dots[arm]
                    q_dot_mod = mod.modulate_joint_velocity_lateral_order(
                        q_dot_nominal=q_dot_nom,
                        self_points=protected_points[arm],
                        obstacle_points=protected_points[other],
                        jacobian=J,
                        side_axis=side_axis,
                        min_separation=lateral_order_min_separation,
                        reactivity=lateral_order_reactivity,
                        lambda_floor=lateral_order_lambda_floor,
                    )
                    w = float(lateral_order_mod_weight)
                    q_dots[arm] = (1.0 - w) * q_dot_nom + w * q_dot_mod

        if not args.no_link_safety_hold:
            link_dist = min_link_distance()
            if link_hold_arm is not None:
                release_dist = args.link_safety_radius + args.link_safety_hysteresis
                if link_dist > release_dist:
                    print(f"[SAFETY] link distance {link_dist:.3f} m > "
                          f"{release_dist:.3f}; releasing {link_hold_arm}")
                    link_hold_arm = None
            if link_hold_arm is None and link_dist < args.link_safety_radius:
                link_hold_arm = link_safety_hold_arm()
                other = "right" if link_hold_arm == "left" else "left"
                print(f"[SAFETY] link distance {link_dist:.3f} m < "
                      f"{args.link_safety_radius:.3f}; holding {link_hold_arm}, "
                      f"letting {other} clear")
            if link_hold_arm is not None:
                if q_dots.get(link_hold_arm) is not None:
                    q_dots[link_hold_arm] = np.zeros(7)
                    safety_hold_steps[link_hold_arm] += 1
                if (args.link_safety_print_every
                        and step % args.link_safety_print_every == 0):
                    other = "right" if link_hold_arm == "left" else "left"
                    print(f"[SAFETY] link distance {link_dist:.3f} m; "
                          f"holding {link_hold_arm}, letting {other} clear")

        # Apply commands and step
        for arm in ("left", "right"):
            if q_dots[arm] is None:
                continue
            q_cmd_state[arm][:7] = q_cmd_state[arm][:7] + q_dots[arm] * physics_dt
            apply_full_command(arm, q_cmd_state[arm][:7])

        step_env()
        carry_held_blocks()

        # Per-arm primitive completion checks
        for arm in ("left", "right"):
            task = seq.tasks[arm]
            if task.is_done():
                continue
            q = franka[arm].get_joint_positions()[:7]
            active_prim_steps = prim_steps[arm] - safety_hold_steps[arm]
            timed_out = active_prim_steps >= prim_timeout[task.current_primitive]
            if task.current_primitive in ik_primitives:
                ee_pos_now = ik_frame_position(arm)
                target_now = seq.cartesian_target(arm)
                cart_err = np.linalg.norm(ee_pos_now - target_now)
                joint_err = np.linalg.norm(q - task.q_goal)
                cart_tol = (
                    args.place_cart_done_tol
                    if task.current_primitive == "place" else args.cart_done_tol
                )
                if task.current_primitive == "place":
                    converged = (
                        cart_err < cart_tol
                        or joint_err < IK_JOINT_DONE_TOL
                    )
                else:
                    converged = (cart_err < cart_tol) or (joint_err < IK_JOINT_DONE_TOL)
                done_err = cart_err
                done_label = "||ee-target||"
            else:
                joint_err = np.linalg.norm(q - task.q_goal)
                cart_err = np.linalg.norm(ik_frame_position(arm) - seq.cartesian_target(arm))
                if (task.current_primitive == "reach"
                        and cart_err < DS_REACH_CART_DONE_TOL):
                    done_err = cart_err
                    converged = True
                    done_label = "||ee-target||"
                else:
                    done_err = joint_err
                    converged = done_err < joint_done_tol
                    done_label = "||q-q_goal||"
            if converged or timed_out:
                if timed_out and not converged:
                    cart_timeout_err = np.linalg.norm(
                        ik_frame_position(arm) - seq.cartesian_target(arm)
                    )
                    joint_timeout_err = np.linalg.norm(
                        franka[arm].get_joint_positions()[:7] - task.q_goal
                    )
                    print(f"[WARN] {arm}/{task.current_primitive} timed out "
                          f"after {active_prim_steps} active steps "
                          f"({prim_steps[arm]} wall steps, "
                          f"{safety_hold_steps[arm]} safety-held) "
                          f"({done_label}={done_err:.3f}, "
                          f"cart_err={cart_timeout_err:.3f}, "
                          f"joint_err={joint_timeout_err:.3f})")
                    if not args.advance_on_timeout:
                        print("[DEPLOY] Aborting instead of advancing. Use "
                              "--advance_on_timeout only for phase-flow debugging.")
                        recorder.close()
                        simulation_app.close()
                        return
                grip = gripper_action_for_primitive(task.current_primitive)
                if grip == "close":
                    hold_gripper(arm, 0.0, cfg["sim"]["gripper_steps"])
                    print_grasp_debug(arm, "after close", task.current_block)
                    if args.kinematic_carry:
                        ee = ik_frame_position(arm).copy()
                        block_pos = env.get_block_positions()[task.current_block].copy()
                        held_block[arm] = task.current_block
                        held_offset[arm] = block_pos - ee
                        carry_held_blocks()
                elif grip == "open":
                    if args.kinematic_carry:
                        # Keep the cube attached while the fingers open, then
                        # snap onto the reserved stack slot. This avoids a
                        # "teleport while still grasping" artifact.
                        hold_gripper(arm, 0.04, cfg["sim"]["gripper_steps"])
                        snap_held_block_to_stack(arm)
                        held_block[arm] = None
                    else:
                        hold_gripper(arm, 0.04, cfg["sim"]["gripper_steps"])
                        held_block[arm] = None
                    if args.priority_policy == "fixed":
                        priority_arm = other_arm(arm)
                        if not args.no_modulation:
                            print(f"[COORD] Priority passed to {priority_arm}")
                    joint_lula_move_to_cart(
                        arm,
                        seq.stack_clearance_target(arm),
                        steps=120,
                        cart_tol=0.03,
                        label="post-place stack clearance",
                    )
                elif args.debug_grasp and task.current_primitive == "lift":
                    print_grasp_debug(arm, "after lift", task.current_block)
                seq.primitive_complete(arm)
                prim_steps[arm] = 0
                safety_hold_steps[arm] = 0
                if seq.tasks[arm].is_done():
                    arm_parked[arm] = False
                    print(f"[DEPLOY] {arm} arm finished blocks; returning home.")

        done = all(seq.tasks[a].is_done() for a in ("left", "right"))
        parked = (
            args.no_return_home_after_done
            or all(arm_parked[a] for a in ("left", "right"))
        )
        if done and parked:
            print("[DEPLOY] Both arms finished stacking.")
            break

    print(f"[DEPLOY] Finished after {step + 1} steps.")
    recorder.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
