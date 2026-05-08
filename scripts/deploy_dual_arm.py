"""
Dual-arm deployment: transport-only Neural DS with inter-arm modulation.

Each arm:
  1. reach     — IK straight-line to hover above block
  2. grasp     — IK descend + close gripper
  3. lift      — IK raise to transport height
  4. transport — Neural DS drives q -> q_goal, modulated to avoid other arm
  5. place     — IK descend + open gripper

Both arms run their full pipeline concurrently. During transport, modulation
uses a simple hierarchy: the priority arm keeps its nominal DS velocity, while
the other arm yields by modulating around the priority arm's EE. Priority
alternates after each successful place. The can_place() yield gate prevents
both arms descending onto the stack simultaneously.

Usage:
  python scripts/deploy_dual_arm.py
  python scripts/deploy_dual_arm.py --use_safe
  python scripts/deploy_dual_arm.py --no_modulation   # ablation
"""

import os
import sys
import argparse
import pickle
import subprocess
import numpy as np
import torch
import yaml
from pathlib import Path
from enum import Enum, auto

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["CARB_LOG_LEVEL"] = "error"


class Stage(Enum):
    REACH     = auto()
    GRASP     = auto()
    LIFT      = auto()
    TRANSPORT = auto()
    PLACE     = auto()
    RETRACT   = auto()
    DONE      = auto()


def load_ds(ckpt_path, device):
    from src.neural_ds import StableNeuralDS, N_JOINTS
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
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


def load_ds_set(ckpt_dir, device, ckpt_arm="both"):
    """Load joint-space Neural DS checkpoints keyed by primitive.

    Kept as a small public helper because evaluate.py imports it.
    """
    primitives = ("reach", "grasp", "lift", "transport", "place")
    label = ckpt_arm
    if label not in ("both", "left", "right"):
        label = "both"
    return {
        primitive: load_ds(Path(ckpt_dir) / f"{label}_{primitive}.pt", device)
        for primitive in primitives
    }


def _articulation_action(positions):
    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core.utils.types import ArticulationAction
    return ArticulationAction(joint_positions=positions)


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
    """EE plus proxy spheres along the wrist/hand link before the EE."""
    points = [np.asarray(ee_pos, dtype=float)]
    if n_link_spheres <= 0:
        return np.asarray(points)

    # The Lula right_gripper frame uses local -Z as the direction back from the
    # gripper tip toward the wrist for the default down-facing grasp pose.
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
    """Add left/right finger-width proxy points on the gripper body."""
    points = [np.asarray(p, dtype=float) for p in points]
    if not lateral_offsets:
        return np.asarray(points)
    R = _quat_wxyz_to_matrix(ee_quat)
    # Franka's hand body is wide across the gripper-frame Y direction.
    lateral_axis = R @ np.array([0.0, 1.0, 0.0])
    if np.linalg.norm(lateral_axis) < 1e-9:
        lateral_axis = np.array([1.0, 0.0, 0.0])
    lateral_axis = lateral_axis / (np.linalg.norm(lateral_axis) + 1e-12)
    # The gripper body sits back from the EE/tool frame along local -Z for the
    # down-facing pose. Body offsets put the lateral protection on the mesh
    # instead of only at the tool centre.
    body_axis = R @ np.array([0.0, 0.0, -1.0])
    if np.linalg.norm(body_axis) < 1e-9:
        body_axis = np.array([0.0, 0.0, 1.0])
    body_axis = body_axis / (np.linalg.norm(body_axis) + 1e-12)
    ee_pos = np.asarray(ee_pos, dtype=float)
    body_offsets = body_offsets if body_offsets else [0.0]
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
    """Protected sphere centers sampled on distal FK frames.

    Uses Lula FK frame positions when available. If the configured frame names
    are not available in this Isaac install, falls back to EE-axis proxies.
    """
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


def _grasp_xy(block_xy, arm, cfg, cli_offset=None):
    offset = np.array(cfg["block"].get("grasp_xy_offset", {}).get(arm, [0.0, 0.0]),
                      dtype=float)
    if cli_offset is not None:
        offset = offset + np.asarray(cli_offset, dtype=float)
    return np.asarray(block_xy, dtype=float) + offset


def _load_deploy_config(path):
    if path is None:
        return {}
    with open(path, "r") as f:
        payload = yaml.safe_load(f) or {}
    return payload.get("deploy", payload)


def _arm_param(value, arm, default=None):
    """Read a scalar deploy parameter that may be overridden per arm."""
    if isinstance(value, dict):
        if arm in value:
            return value[arm]
        return value.get("default", default)
    return value


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


def _start_screen_record(path, fps=30, size="1280x720"):
    """Record the visible Isaac window/screen using ffmpeg/x11grab."""
    if not path:
        return None
    display = os.environ.get("DISPLAY")
    if not display:
        print("[WARN] DISPLAY is not set; cannot start Isaac viewport recording")
        return None
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-video_size", size,
        "-framerate", str(fps),
        "-f", "x11grab",
        "-i", f"{display}+0,0",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"[DEPLOY] Recording Isaac viewport to {out}")
        return proc
    except FileNotFoundError:
        print("[WARN] ffmpeg not found; Isaac viewport recording disabled")
        return None


def _stop_screen_record(proc):
    if proc is None:
        return
    try:
        proc.communicate(input=b"q", timeout=5)
    except Exception:
        proc.terminate()


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--deploy_config", type=str, default=None,
                     help="YAML file with deploy_dual_arm argument defaults")
    pre_args, _ = pre.parse_known_args()
    deploy_defaults = _load_deploy_config(pre_args.deploy_config)

    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy_config", type=str, default=pre_args.deploy_config,
                        help="YAML file with deploy_dual_arm argument defaults")
    parser.add_argument("--config",        type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt_arm",      type=str, default=None,
                        help="Checkpoint label for both arms (default: per-arm)")
    parser.add_argument("--max_transport", type=int, default=2000,
                        help="Max DS steps per transport")
    parser.add_argument("--use_safe",      action="store_true")
    parser.add_argument("--no_modulation", action="store_true")
    parser.add_argument("--modulation_space", type=str, default="cartesian",
                        choices=["cartesian", "jsdf"],
                        help="Avoidance space for neural joint-velocity transport. "
                             "cartesian uses the old EE velocity modulation; "
                             "jsdf uses a joint-space distance-field gradient "
                             "from protected sphere distances.")
    parser.add_argument("--priority_arm",  type=str, default="left",
                        choices=["left", "right"],
                        help="Arm that starts as the unmodulated priority arm")
    parser.add_argument("--priority_policy", type=str, default="fixed",
                        choices=["fixed", "closest_to_stack"],
                        help="How to choose which arm gets lower modulation weight")
    parser.add_argument("--priority_hysteresis", type=float, default=0.04,
                        help="Closest-to-stack priority switches only past this distance margin")
    parser.add_argument("--priority_mod_weight", type=float, default=0.25,
                        help="Blend weight for modulation on the priority arm")
    parser.add_argument("--yield_mod_weight", type=float, default=1.0,
                        help="Blend weight for modulation on the non-priority arm")
    parser.add_argument("--mod_radius", type=float, default=None,
                        help="Spherical modulation radius around each EE")
    parser.add_argument("--mod_reactivity", type=float, default=None,
                        help="Gamma exponent; lower values make modulation start earlier")
    parser.add_argument("--mod_isoline", type=float, default=None,
                        help="Gamma contour to track; >1 expands the effective modulation boundary")
    parser.add_argument("--mod_max_pairs", type=int, default=4,
                        help="Number of closest protected sphere pairs to apply. "
                             "Use 0 to apply all pairs.")
    parser.add_argument("--jsdf_influence_radius", type=float, default=None,
                        help="Protected sphere centre distance where JSDF-style "
                             "joint-space avoidance starts. Defaults to mod_radius.")
    parser.add_argument("--jsdf_gain", type=float, default=2.0,
                        help="Gain for joint-space distance-field avoidance.")
    parser.add_argument("--jsdf_max_joint_speed", type=float, default=0.45,
                        help="Norm cap for the JSDF-style avoidance joint velocity.")
    parser.add_argument("--jsdf_fd_eps", type=float, default=1e-3,
                        help="Finite-difference step in radians for the "
                             "joint-space distance gradient.")
    parser.add_argument("--jsdf_debug_every", type=int, default=0,
                        help="Print JSDF activation diagnostics every N sim "
                             "steps. 0 disables.")
    parser.add_argument("--no_preserve_mod_speed", action="store_true",
                        help="Do not rescale modulated Cartesian velocity back to nominal speed")
    parser.add_argument("--link_spheres", type=int, default=None,
                        help="Extra proxy spheres along the last link before the EE")
    parser.add_argument("--link_sphere_spacing", type=float, default=None,
                        help="Spacing in metres between wrist/last-link proxy spheres")
    parser.add_argument("--link_frames", nargs="+", default=None,
                        help="Lula FK frames used for distal-link proxy spheres")
    parser.add_argument("--link_samples_per_segment", type=int, default=None,
                        help="Extra proxy spheres between consecutive link_frames")
    parser.add_argument("--gripper_lateral_offsets", nargs="*", type=float,
                        default=None,
                        help="Extra gripper-body half-width proxy offsets in local gripper Y, metres")
    parser.add_argument("--gripper_lateral_body_offsets", nargs="*", type=float,
                        default=None,
                        help="Offsets back from the EE along local -Z for gripper-width proxies")
    parser.add_argument("--grasp_offset", type=float, nargs=2, default=None,
                        metavar=("DX", "DY"),
                        help="Extra world-frame XY pick offset in metres for both arms")
    parser.add_argument("--model",         type=str, default="neural",
                        choices=["neural", "lpvds"],
                        help="Transport DS model: neural joint-space or Cartesian LPVDS")
    parser.add_argument("--lookahead",     type=int, default=5,
                        help="LPVDS IK target = ee_pos + x_dot * lookahead * dt")
    parser.add_argument("--max_cart_speed", type=float, default=0.25,
                        help="Clip LPVDS Cartesian speed before IK retargeting")
    parser.add_argument("--yield_max_cart_speed", type=float, default=None,
                        help="Optional LPVDS speed cap applied only to the non-priority arm")
    parser.add_argument("--min_transport_progress", type=float, default=0.0,
                        help="Minimum fraction of nominal LPVDS progress preserved after modulation")
    parser.add_argument("--parked_obstacle_weight", type=float, default=1.0,
                        help="Scale modulation caused by an arm that is fully parked/DONE")
    parser.add_argument("--cart_gain", type=float, default=1.0,
                        help="Scale LPVDS Cartesian velocity before speed clipping")
    parser.add_argument("--speedup", type=float, default=1.0,
                        help="Global motion speed multiplier for IK timing and DS velocities")
    parser.add_argument("--ik_nullspace_seed_weight", type=float, default=0.0,
                        help="Blend IK warm-starts toward the configured default "
                             "joint pose. 0 uses previous IK solution; 1 uses "
                             "the default pose as the seed every solve.")
    parser.add_argument("--nullspace_home_gain", type=float, default=0.0,
                        help="Joint-space gain toward the configured default "
                             "pose, projected through the translational "
                             "Jacobian null-space during modulated joint "
                             "commands.")
    parser.add_argument("--sync_speeds", action="store_true",
                        help="Scale down the faster arm to match transport speed norms")
    parser.add_argument("--raw_lpvds",     action="store_true",
                        help="Use raw LPVDS velocity without stability projection")
    parser.add_argument("--no_workspace_clamp", action="store_true",
                        help="Do not clamp LPVDS IK targets to each arm's transport workspace")
    parser.add_argument("--z_margin", type=float, default=0.12,
                        help="LPVDS target z clamp around lift height when workspace clamp is enabled")
    parser.add_argument("--headless",      action="store_true")
    parser.add_argument("--done_tol",      type=float, default=0.05)
    parser.add_argument("--ik_done_tol", type=float, default=0.005,
                        help="Cartesian tolerance for finishing IK primitives before grasp/place")
    parser.add_argument("--ik_settle_steps", type=int, default=120,
                        help="Max extra ticks to hold final IK waypoint while waiting for tolerance")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for deploy-time block randomization")
    parser.add_argument("--no_randomize_blocks", action="store_true",
                        help="Use the scene's initial block positions")
    parser.add_argument("--diag_out", type=str, default=None,
                        help="Optional path to save LPVDS interaction diagnostics")
    parser.add_argument("--video_out", type=str, default=None,
                        help="Optional path to render DS interaction video after deploy")
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--video_stride", type=int, default=6)
    parser.add_argument("--video_views", choices=["original", "top", "both"],
                        default="original")
    parser.add_argument("--video_radial_field", action="store_true",
                        help="Show radial field samples in the generated DS video")
    parser.add_argument("--show_proxy_spheres", action="store_true",
                        help="Show protected modulation proxy spheres in Isaac viewport")
    parser.add_argument("--proxy_sphere_radius", type=float, default=0.025)
    parser.add_argument("--isaac_record_out", type=str, default=None,
                        help="Record the actual Isaac viewport with ffmpeg/x11grab")
    parser.add_argument("--isaac_record_fps", type=int, default=30)
    parser.add_argument("--isaac_record_size", type=str, default="1280x720")
    parser.add_argument("--status_every", type=int, default=240,
                        help="Print deployment progress every N sim steps")
    if deploy_defaults:
        parser.set_defaults(**deploy_defaults)
    args = parser.parse_args()
    if args.video_out and not args.diag_out:
        args.diag_out = "data/results/lpvds_interaction.pkl"

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless,
                                    "width": 1280, "height": 720})

    from src.env import DualArmEnv
    from src.ik_controller import IKController, _trapezoid_profile
    from src.franka_ik import FrankaIK
    from src.modulation import InterArmModulation, jacobian_finite_difference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DualArmEnv(config_path=args.config, arms=("left", "right"))
    cfg = env.cfg
    proxy_viz = IsaacProxySphereViz(radius=args.proxy_sphere_radius) \
        if args.show_proxy_spheres else None
    screen_recorder = None
    if args.isaac_record_out and args.headless:
        print("[WARN] --isaac_record_out needs non-headless rendering; disabling --headless recording")
    elif args.isaac_record_out:
        screen_recorder = _start_screen_record(
            args.isaac_record_out,
            fps=args.isaac_record_fps,
            size=args.isaac_record_size,
        )

    ARMS = ("left", "right")
    physics_dt  = cfg["sim"]["physics_dt"]
    steps       = cfg["sim"]["steps_per_primitive"]
    hover_h     = cfg["heights"]["hover"]
    lift_h      = cfg["heights"]["lift"]
    grasp_h     = cfg["heights"]["grasp"]
    block_h     = cfg["block"]["size"]
    goal_xy     = tuple(cfg["shared_goal"])
    yield_radius = cfg["coordination"].get("yield_radius", 0.12)
    ee_down     = np.array([0.0, 1.0, 0.0, 0.0])
    transport_pos = np.array([goal_xy[0], goal_xy[1], lift_h])
    lpv_min = {}
    lpv_max = {}
    for arm in ARMS:
        ws = cfg["block_workspace"][arm]
        lpv_min[arm] = np.array([
            min(ws["x_min"], goal_xy[0]) - 0.05,
            min(ws["y_min"], goal_xy[1]) - 0.05,
            lift_h - args.z_margin,
        ])
        lpv_max[arm] = np.array([
            max(ws["x_max"], goal_xy[0]) + 0.05,
            max(ws["y_max"], goal_xy[1]) + 0.05,
            lift_h + args.z_margin,
        ])

    # Per-arm objects
    franka      = {a: env.frankas[a] for a in ARMS}
    ik_motion   = {a: IKController(
                       franka[a], arm=a,
                       rest_q=np.array(cfg["arms"][f"default_joints_{a}"]),
                       nullspace_seed_weight=args.ik_nullspace_seed_weight)
                   for a in ARMS}
    ik_kin      = {a: FrankaIK(franka[a]) for a in ARMS}
    default_q   = {a: np.array(cfg["arms"][f"default_joints_{a}"]) for a in ARMS}

    if not args.no_randomize_blocks:
        rng = np.random.default_rng(args.seed)
        env.reset_blocks(render=not args.headless, rng=rng)
        print("[DEPLOY] Randomized block positions")

    # Load DS — one per arm (or shared if ckpt_arm specified)
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ds = {}
    lpv_model = {}
    if args.model == "lpvds":
        from src.lpv_ds import LPVDS
        for a in ARMS:
            label = args.ckpt_arm if args.ckpt_arm else a
            lpv_model[a] = LPVDS.load(ckpt_dir / f"{label}_transport_lpvds.pkl")
    else:
        for a in ARMS:
            label = args.ckpt_arm if args.ckpt_arm else a
            ds[a] = load_ds(ckpt_dir / f"{label}_transport.pt", device)

    # Transport q_goal — fixed target, same for both arms
    q_goal = {}
    if args.model == "neural":
        for a in ARMS:
            q, ok = ik_kin[a].solve(transport_pos, target_quat=ee_down,
                                     q_seed=default_q[a])
            if not ok:
                print(f"[WARN] IK failed for transport target on {a} arm")
            q_goal[a] = q

    mod_radius = args.mod_radius
    if mod_radius is None:
        mod_radius = cfg["coordination"].get("ee_safety_radius", 0.30)
    mod_reactivity = args.mod_reactivity
    if mod_reactivity is None:
        mod_reactivity = cfg["coordination"].get("modulation_reactivity", 2.0)
    mod_isoline = args.mod_isoline
    if mod_isoline is None:
        mod_isoline = cfg["coordination"].get("modulation_isoline", 1.0)
    jsdf_influence_radius = (
        mod_radius if args.jsdf_influence_radius is None
        else args.jsdf_influence_radius
    )
    mod = InterArmModulation(
        safe_radius=mod_radius,
        reactivity=mod_reactivity,
        preserve_speed=not args.no_preserve_mod_speed,
        isoline=mod_isoline,
        max_pairs=(None if args.mod_max_pairs == 0 else args.mod_max_pairs),
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
            "gripper_proxy_lateral_offsets", [-0.045, 0.045]
        )
    gripper_lateral_body_offsets = args.gripper_lateral_body_offsets
    if gripper_lateral_body_offsets is None:
        gripper_lateral_body_offsets = cfg["coordination"].get(
            "gripper_proxy_body_offsets", [0.055, 0.11]
        )

    block_names = {a: [b["name"] for b in cfg[f"{a}_blocks"]] for a in ARMS}
    goal_z      = cfg["table"]["height"] + block_h / 2
    block_idx   = {a: 0 for a in ARMS}

    # Per-arm state machine
    stage       = {a: Stage.REACH for a in ARMS}
    ee_grasp    = {a: ee_down.copy() for a in ARMS}
    transport_steps = {a: 0 for a in ARMS}
    ik_fail_streak = {a: 0 for a in ARMS}
    ik_plan_fail_streak = {a: 0 for a in ARMS}
    priority_arm = args.priority_arm
    diag_log = []

    def current_block(arm):
        idx = block_idx[arm]
        names = block_names[arm]
        return names[idx] if idx < len(names) else None

    def arm_done(arm):
        return block_idx[arm] >= len(block_names[arm])

    def other_arm(arm):
        return "right" if arm == "left" else "left"

    def avoidance_weight(arm):
        """Blend weight for this arm's smooth avoidance correction."""
        if args.no_modulation:
            return 0.0
        if stage[arm] == Stage.DONE:
            return 0.0
        other = other_arm(arm)
        obstacle_scale = (
            float(args.parked_obstacle_weight)
            if stage[other] == Stage.DONE else 1.0
        )
        if arm == priority_arm:
            return obstacle_scale * float(_arm_param(args.priority_mod_weight, arm, 0.25))
        return obstacle_scale * float(_arm_param(args.yield_mod_weight, arm, 1.0))

    def modulation_weight(arm):
        """Transport-only alias used by diagnostics and DS modulation."""
        if stage[arm] != Stage.TRANSPORT:
            return 0.0
        return avoidance_weight(arm)

    def update_priority():
        """Give right-of-way to the active arm closer to the shared stack."""
        nonlocal priority_arm
        if args.priority_policy != "closest_to_stack" or args.no_modulation:
            return
        candidates = [
            a for a in ARMS
            if stage[a] in (Stage.TRANSPORT, Stage.PLACE, Stage.RETRACT)
        ]
        if len(candidates) == 1:
            if priority_arm != candidates[0]:
                priority_arm = candidates[0]
                print(f"  [COORD] Priority -> {priority_arm} (sole active arm)")
            return
        if len(candidates) < 2:
            return
        goal = np.asarray(goal_xy, dtype=float)
        ee_xy = {
            a: ik_motion[a].ik.get_world_pose()[0][:2].copy()
            for a in candidates
        }
        dist = {a: float(np.linalg.norm(ee_xy[a] - goal)) for a in candidates}
        closest = min(candidates, key=lambda a: dist[a])
        if closest == priority_arm:
            return
        current = priority_arm if priority_arm in candidates else None
        margin = args.priority_hysteresis
        if current is None or dist[closest] + margin < dist[current]:
            priority_arm = closest
            print(f"  [COORD] Priority -> {priority_arm} "
                  f"(closer to stack: {dist[closest]:.3f}m)")

    def can_place(arm):
        """Yield if other arm is also near the stack goal."""
        other = other_arm(arm)
        if arm_done(other):
            return True
        if stage[other] == Stage.PLACE:
            return arm == priority_arm
        if stage[other] not in (Stage.TRANSPORT, Stage.RETRACT):
            return True
        ee_other, _ = ik_motion[other].ik.get_world_pose()
        gx, gy = goal_xy
        return np.linalg.norm(ee_other[:2] - np.array([gx, gy])) > yield_radius

    def sync_vector_norms(vectors, min_norm=1e-5):
        """Scale down faster active arms so paired transport speeds match."""
        if not args.sync_speeds:
            return
        active = [a for a in ARMS if vectors.get(a) is not None]
        if len(active) < 2:
            return
        norms = {a: float(np.linalg.norm(vectors[a])) for a in active}
        target = min(norms.values())
        if target < min_norm:
            return
        for arm in active:
            if norms[arm] > target:
                vectors[arm] *= target / norms[arm]

    def cap_yield_cart_speed(vectors):
        """Limit only non-priority Cartesian transport speed."""
        if args.yield_max_cart_speed is None:
            return
        cap = args.speedup * float(args.yield_max_cart_speed)
        for arm in ARMS:
            if arm == priority_arm or vectors.get(arm) is None:
                continue
            speed = float(np.linalg.norm(vectors[arm]))
            if speed > cap > 1e-9:
                vectors[arm] *= cap / speed

    def add_nullspace_home_velocity(arm, q_dot, jacobian, dt=None):
        """Bias joints toward default_q without changing translational EE velocity."""
        if args.nullspace_home_gain <= 0.0 or q_dot is None:
            return q_dot
        q_now = franka[arm].get_joint_positions()[:7].copy()
        J = jacobian[:3, :]
        JJt = J @ J.T
        damp = 0.05 ** 2 * np.eye(JJt.shape[0])
        J_pinv = J.T @ np.linalg.inv(JJt + damp)
        null_projector = np.eye(7) - J_pinv @ J
        q_home_dot = args.nullspace_home_gain * (default_q[arm] - q_now)
        scale = 1.0 if dt is None else float(dt)
        return q_dot + scale * (null_projector @ q_home_dot)

    def protected_points_for_q(arm, q):
        ee_pose_now = ik_motion[arm].ik.get_world_pose(q=q)
        return _protected_points_from_links(
            ik_motion[arm].ik, q, ee_pose_now[0], ee_pose_now[1],
            link_frames, link_samples_per_segment, max(0, link_spheres),
            link_sphere_spacing, gripper_lateral_offsets,
            gripper_lateral_body_offsets
        )

    def min_point_distance(self_points, obstacle_points):
        self_points = np.asarray(self_points, dtype=float).reshape(-1, 3)
        obstacle_points = np.asarray(obstacle_points, dtype=float).reshape(-1, 3)
        if len(self_points) == 0 or len(obstacle_points) == 0:
            return float("inf")
        deltas = self_points[:, None, :] - obstacle_points[None, :, :]
        return float(np.linalg.norm(deltas, axis=-1).min())

    def jsdf_avoidance_velocity(arm, q_dot_nominal, obstacle_points, self_points):
        """Joint-space distance-field repulsion from protected sphere distances."""
        if args.no_modulation or args.modulation_space != "jsdf":
            return np.zeros(7)
        d0 = min_point_distance(self_points, obstacle_points)
        if d0 >= jsdf_influence_radius:
            if args.jsdf_debug_every and global_step % args.jsdf_debug_every == 0:
                print(f"[JSDF] {arm} inactive d={d0:.3f} >= "
                      f"{jsdf_influence_radius:.3f}")
            return np.zeros(7)

        q0 = franka[arm].get_joint_positions()[:7].copy()
        grad = np.zeros(7)
        eps = max(float(args.jsdf_fd_eps), 1e-6)
        for j in range(7):
            q_eps = q0.copy()
            q_eps[j] += eps
            d_eps = min_point_distance(
                protected_points_for_q(arm, q_eps),
                obstacle_points,
            )
            grad[j] = (d_eps - d0) / eps

        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < 1e-9:
            if args.jsdf_debug_every and global_step % args.jsdf_debug_every == 0:
                print(f"[JSDF] {arm} inactive d={d0:.3f} grad_norm~0")
            return np.zeros(7)

        # Tail effect in joint space: if nominal motion is already increasing
        # the closest protected-sphere distance and we are not inside the core
        # safety radius, leave it alone.
        if d0 > mod_radius and float(np.dot(grad, q_dot_nominal)) >= 0.0:
            if args.jsdf_debug_every and global_step % args.jsdf_debug_every == 0:
                print(f"[JSDF] {arm} tail d={d0:.3f} "
                      f"grad_dot_qdot={float(np.dot(grad, q_dot_nominal)):.4f}")
            return np.zeros(7)

        activation = max(0.0, (jsdf_influence_radius - d0) / jsdf_influence_radius)
        q_avoid = args.jsdf_gain * activation * grad / grad_norm
        speed = float(np.linalg.norm(q_avoid))
        cap = max(float(args.jsdf_max_joint_speed), 1e-9)
        if speed > cap:
            q_avoid *= cap / speed
        if args.jsdf_debug_every and global_step % args.jsdf_debug_every == 0:
            print(f"[JSDF] {arm} active d={d0:.3f} "
                  f"activation={activation:.2f} "
                  f"grad_norm={grad_norm:.3f} "
                  f"|q_avoid|={np.linalg.norm(q_avoid):.3f}")
        return q_avoid

    def sped_steps(n_steps):
        return max(1, int(round(n_steps / max(args.speedup, 1e-6))))

    def print_status(prefix="[STATUS]"):
        ee = {a: ik_motion[a].ik.get_world_pose()[0].copy() for a in ARMS}
        dist = float(np.linalg.norm(ee["left"] - ee["right"]))
        parts = []
        for arm in ARMS:
            plan = ik_plan[arm]
            if plan is not None:
                progress = f"{plan['kind']}:{plan['i']}/{len(plan['s_values'])}"
            elif gripper_wait[arm] is not None:
                progress = f"gripper:{gripper_wait[arm]['remaining']}"
            elif joint_plan[arm] is not None:
                progress = f"home:{joint_plan[arm]['i']}/{len(joint_plan[arm]['s_values'])}"
            else:
                progress = "-"
            parts.append(
                f"{arm}={stage[arm].name} block={block_idx[arm]} "
                f"plan={progress} transport={transport_steps[arm]}"
            )
        print(f"{prefix} step={global_step} ee_dist={dist:.3f} " + " | ".join(parts))

    mod_mode = "OFF" if args.no_modulation else (
        f"weighted priority={priority_arm} "
        f"policy={args.priority_policy} "
        f"priority_w={args.priority_mod_weight} yield_w={args.yield_mod_weight}"
    )
    print(f"[DEPLOY] Dual-arm transport DS  model={args.model}  safe={args.use_safe}  "
          f"modulation={mod_mode}")
    print(f"[DEPLOY] Modulation sphere radius={mod_radius:.3f}m "
          f"reactivity={mod_reactivity:.2f} "
          f"isoline={mod_isoline:.2f} "
          f"preserve_speed={not args.no_preserve_mod_speed}")
    print(f"[DEPLOY] Protected points per arm={1 + max(0, link_spheres)} "
          f"fallback points; FK frames={link_frames}, "
          f"samples/segment={link_samples_per_segment})")
    print(f"[DEPLOY] Gripper-width proxy offsets={gripper_lateral_offsets}")
    print(f"[DEPLOY] Gripper-width body offsets={gripper_lateral_body_offsets}")
    if not args.no_modulation:
        if args.modulation_space == "jsdf":
            print("[DEPLOY] IK primitives and neural transport use "
                  "JSDF-style joint-space avoidance: "
                  f"influence={jsdf_influence_radius:.3f}m "
                  f"gain={args.jsdf_gain:.2f} "
                  f"max_speed={args.jsdf_max_joint_speed:.2f}rad/s")
        else:
            print("[DEPLOY] IK primitives use smooth Cartesian EE modulation")
    print(f"[DEPLOY] IK frame={ik_motion['left'].ik.ee_frame}")
    if args.model == "lpvds":
        for arm in ARMS:
            print(f"[DEPLOY] {arm} LPVDS goal={np.round(lpv_model[arm].x_goal, 4)}  "
                  f"config transport={np.round(transport_pos, 4)}  "
                  f"lookahead={args.lookahead}")
            if not args.no_workspace_clamp:
                print(f"[DEPLOY] {arm} LPVDS target clamp min={np.round(lpv_min[arm], 3)} "
                      f"max={np.round(lpv_max[arm], 3)}")

    # ── Nonblocking primitive motion helpers ────────────────────────────────
    # IK primitives advance one waypoint per simulation tick.  Both arms use
    # the same normalized trapezoid profile for a given primitive, so paired
    # reach/grasp/lift/place moves have matching speed shapes.
    ik_plan = {a: None for a in ARMS}
    joint_plan = {a: None for a in ARMS}
    gripper_wait = {a: None for a in ARMS}

    def start_ik_plan(arm, target, quat, n_steps, kind):
        if ik_plan[arm] is not None:
            return
        ee_start, _ = ik_motion[arm].ik.get_world_pose()
        ik_plan[arm] = {
            "start": np.asarray(ee_start, dtype=float).copy(),
            "end": np.asarray(target, dtype=float).copy(),
            "quat": quat.copy(),
            "s_values": _trapezoid_profile(n_steps, ik_motion[arm].vel_ramp_frac),
            "i": 0,
            "settle": 0,
            "kind": kind,
        }

    def start_joint_plan(arm, target_q, n_steps, kind):
        if joint_plan[arm] is not None:
            return
        joint_plan[arm] = {
            "start": franka[arm].get_joint_positions()[:7].copy(),
            "end": np.asarray(target_q, dtype=float).copy(),
            "s_values": _trapezoid_profile(n_steps, ik_motion[arm].vel_ramp_frac),
            "i": 0,
            "kind": kind,
        }

    def start_gripper_wait(arm, open_gripper, n_steps, next_stage, after=None):
        ik_motion[arm].set_gripper(open=open_gripper)
        gripper_wait[arm] = {
            "remaining": n_steps,
            "next_stage": next_stage,
            "after": after,
        }

    def finish_ik_plan(arm, kind):
        nonlocal goal_z, priority_arm
        if kind == "reach":
            stage[arm] = Stage.GRASP
        elif kind == "grasp":
            start_gripper_wait(arm, open_gripper=False,
                               n_steps=cfg["sim"]["gripper_steps"],
                               next_stage=Stage.LIFT)
        elif kind == "lift":
            transport_steps[arm] = 0
            ik_fail_streak[arm] = 0
            stage[arm] = Stage.TRANSPORT
        elif kind == "place":
            start_gripper_wait(arm, open_gripper=True,
                               n_steps=cfg["sim"]["gripper_steps"],
                               next_stage=Stage.RETRACT,
                               after="place_open")
        elif kind == "retract":
            block_idx[arm] += 1
            if arm_done(arm):
                print(f"  [{arm}] Post-place lift clear; moving to nominal pose")
                ik_motion[arm].set_gripper(open=True)
                start_joint_plan(arm, default_q[arm], n_steps=120, kind="home")
            else:
                stage[arm] = Stage.REACH

    def finish_gripper_wait(arm, wait):
        nonlocal goal_z, priority_arm
        if wait["after"] == "place_open":
            goal_z += block_h + 0.002
            if args.priority_policy == "fixed":
                priority_arm = other_arm(arm)
            if not args.no_modulation and args.priority_policy == "fixed":
                print(f"  [COORD] Priority passed to {priority_arm}")
        stage[arm] = wait["next_stage"]

    def step_nontransport_plans():
        """Apply all active IK/gripper/home actions for this simulation tick."""
        moved_arms = set()
        ee_pose_now = {a: ik_motion[a].ik.get_world_pose() for a in ARMS}
        ee_now = {a: ee_pose_now[a][0].copy() for a in ARMS}
        q_now = {a: franka[a].get_joint_positions()[:7].copy() for a in ARMS}
        protected_now = {
            a: _protected_points_from_links(
                ik_motion[a].ik, q_now[a], ee_now[a], ee_pose_now[a][1],
                link_frames, link_samples_per_segment, max(0, link_spheres),
                link_sphere_spacing, gripper_lateral_offsets,
                gripper_lateral_body_offsets
            )
            for a in ARMS
        }
        if proxy_viz is not None:
            proxy_viz.update(protected_now)
        next_waypoint = {}

        for arm in ARMS:
            plan = ik_plan[arm]
            if plan is None:
                next_waypoint[arm] = None
                continue
            s = plan["s_values"][plan["i"]]
            next_waypoint[arm] = plan["start"] + s * (plan["end"] - plan["start"])

        for arm in ARMS:
            plan = ik_plan[arm]
            if plan is None:
                continue
            waypoint = next_waypoint[arm]
            w = avoidance_weight(arm)
            if w > 0.0 and args.modulation_space != "jsdf":
                other = other_arm(arm)
                v_nom = waypoint - ee_now[arm]
                v_mod = mod.huber.modulate_cartesian_points(
                    v_nom, protected_now[arm], protected_now[other]
                )
                v_cmd = (1.0 - w) * v_nom + w * v_mod
                # Keep smooth avoidance from erasing primitive progress.
                if np.dot(v_cmd, v_nom) <= 0.0:
                    v_cmd = v_nom
                waypoint = ee_now[arm] + v_cmd
            if w > 0.0 and args.modulation_space == "jsdf":
                full_cmd, ik_ok = ik_motion[arm].command_for(
                    waypoint, target_quat=plan["quat"]
                )
                q_now_arm = q_now[arm]
                q_ik = full_cmd[:7].copy()
                q_step = q_ik - q_now_arm
                J = jacobian_finite_difference(franka[arm])
                q_avoid = jsdf_avoidance_velocity(
                    arm,
                    q_step / max(physics_dt, 1e-9),
                    obstacle_points=protected_now[other_arm(arm)],
                    self_points=protected_now[arm],
                )
                q_step_cmd = q_step + w * q_avoid * physics_dt
                if np.dot(q_step_cmd, q_step) <= 0.0:
                    q_step_cmd = q_step
                q_step_cmd = add_nullspace_home_velocity(
                    arm, q_step_cmd, J, dt=physics_dt
                )
                full_cmd[:7] = q_now_arm + q_step_cmd
                ik_motion[arm]._q_last = full_cmd[:7].copy()
                franka[arm].apply_action(_articulation_action(full_cmd))
            else:
                ik_ok = ik_motion[arm].step_to(waypoint, target_quat=plan["quat"])
            moved_arms.add(arm)
            if ik_ok:
                ik_plan_fail_streak[arm] = 0
                if plan["i"] < len(plan["s_values"]) - 1:
                    plan["i"] += 1
                else:
                    ee_err = np.linalg.norm(ee_now[arm] - plan["end"])
                    if ee_err <= args.ik_done_tol:
                        kind = plan["kind"]
                        ik_plan[arm] = None
                        finish_ik_plan(arm, kind)
                        continue
                    plan["settle"] += 1
                    if plan["settle"] == args.ik_settle_steps:
                        print(f"  [WARN] {arm} {plan['kind']} final EE error "
                              f"{ee_err:.4f}m > {args.ik_done_tol:.4f}m")
                    if plan["settle"] >= args.ik_settle_steps:
                        kind = plan["kind"]
                        ik_plan[arm] = None
                        finish_ik_plan(arm, kind)
                        continue
            else:
                ik_plan_fail_streak[arm] += 1
                if ik_plan_fail_streak[arm] == 25:
                    print(f"  [WARN] {arm} {plan['kind']} IK failed for 25 consecutive steps")
                if ik_plan_fail_streak[arm] >= 60:
                    fallback = next_waypoint[arm]
                    if ik_motion[arm].step_to(fallback, target_quat=plan["quat"]):
                        print(f"  [WARN] {arm} {plan['kind']} using unmodulated IK waypoint")
                        ik_plan_fail_streak[arm] = 0
                        if plan["i"] < len(plan["s_values"]) - 1:
                            plan["i"] += 1

        for arm in ARMS:
            wait = gripper_wait[arm]
            if wait is None:
                continue
            q = franka[arm].get_joint_positions()[:7].copy()
            finger = ik_motion[arm]._finger_width
            full_cmd = np.concatenate([q, [finger, finger]])
            franka[arm].apply_action(_articulation_action(full_cmd))
            moved_arms.add(arm)
            wait["remaining"] -= 1
            if wait["remaining"] <= 0:
                gripper_wait[arm] = None
                finish_gripper_wait(arm, wait)

        for arm in ARMS:
            plan = joint_plan[arm]
            if plan is None:
                continue
            s = plan["s_values"][plan["i"]]
            q_nom = plan["start"] + s * (plan["end"] - plan["start"])
            q_now_arm = franka[arm].get_joint_positions()[:7].copy()
            q_step = q_nom - q_now_arm
            w = avoidance_weight(arm)
            if w > 0.0 and np.linalg.norm(q_step) > 1e-9:
                other = other_arm(arm)
                J = jacobian_finite_difference(franka[arm])
                if args.modulation_space == "jsdf":
                    q_step_mod = q_step + physics_dt * jsdf_avoidance_velocity(
                        arm,
                        q_step / max(physics_dt, 1e-9),
                        obstacle_points=protected_now[other],
                        self_points=protected_now[arm],
                    )
                else:
                    q_step_mod = mod.modulate_joint_velocity_points(
                        q_dot_nominal=q_step,
                        self_points=protected_now[arm],
                        obstacle_points=protected_now[other],
                        jacobian=J,
                    )
                q_step_cmd = (1.0 - w) * q_step + w * q_step_mod
                if np.dot(q_step_cmd, q_step) <= 0.0:
                    q_step_cmd = q_step
                q_step_cmd = add_nullspace_home_velocity(
                    arm, q_step_cmd, J, dt=physics_dt
                )
                q = q_now_arm + q_step_cmd
            else:
                if args.nullspace_home_gain > 0.0:
                    J = jacobian_finite_difference(franka[arm])
                    q_step = add_nullspace_home_velocity(
                        arm, q_step, J, dt=physics_dt
                    )
                    q = q_now_arm + q_step
                else:
                    q = q_nom
            finger = ik_motion[arm]._finger_width
            full_cmd = np.concatenate([q, [finger, finger]])
            franka[arm].apply_action(_articulation_action(full_cmd))
            moved_arms.add(arm)
            plan["i"] += 1
            if plan["i"] >= len(plan["s_values"]):
                joint_plan[arm] = None
                ik_motion[arm].reset()
                stage[arm] = Stage.DONE

        return moved_arms

    # ── Main loop ─────────────────────────────────────────────────────────────
    # IK primitives and DS transport are both stepped tick-by-tick.  Each tick
    # gathers all arm commands first, then advances the world once.

    global_step = 0
    MAX_GLOBAL  = 100_000
    idle_ticks = 0

    while global_step < MAX_GLOBAL:
        if not simulation_app.is_running():
            break
        if all(stage[a] == Stage.DONE for a in ARMS):
            print("[DEPLOY] Both arms finished.")
            break
        update_priority()

        for arm in ARMS:
            if (arm_done(arm) or ik_plan[arm] is not None or
                    joint_plan[arm] is not None or gripper_wait[arm] is not None):
                continue

            blk = current_block(arm)
            if blk is None:
                continue
            bpos = env.get_block_positions()[blk].copy()
            bx, by = bpos[0], bpos[1]
            pick_xy = _grasp_xy([bx, by], arm, cfg, args.grasp_offset)

            # ── Start IK stages; execution happens below one tick at a time ──
            if stage[arm] == Stage.REACH:
                ee_grasp[arm] = env.get_block_grasp_quat(blk)
                start_ik_plan(arm, np.array([pick_xy[0], pick_xy[1], hover_h]),
                              ee_grasp[arm], sped_steps(steps["reach"]),
                              kind="reach")
                continue

            if stage[arm] == Stage.GRASP:
                start_ik_plan(arm, np.array([pick_xy[0], pick_xy[1], grasp_h]),
                              ee_grasp[arm], sped_steps(steps["grasp"]),
                              kind="grasp")
                continue

            if stage[arm] == Stage.LIFT:
                start_ik_plan(arm, np.array([pick_xy[0], pick_xy[1], lift_h]),
                              ee_down, sped_steps(steps["lift"]), kind="lift")
                continue

            if stage[arm] == Stage.PLACE:
                if not can_place(arm):
                    # Hover at transport position until coast is clear
                    pass
                else:
                    place_pos = np.array([goal_xy[0], goal_xy[1],
                                          goal_z + 0.02])
                    start_ik_plan(arm, place_pos, ee_down,
                                  sped_steps(steps["place"]), kind="place")
                continue

            if stage[arm] == Stage.RETRACT:
                post_place_lift = np.array([goal_xy[0], goal_xy[1], lift_h])
                start_ik_plan(arm, post_place_lift, ee_down,
                              sped_steps(60), kind="retract")
                continue

        moved_arms = step_nontransport_plans()

        # ── DS transport step (both arms simultaneously) ───────────────────
        # Snapshot EE and wrist/last-link proxy spheres before commands.
        ee_pose = {a: ik_motion[a].ik.get_world_pose() for a in ARMS}
        ee_pos = {a: ee_pose[a][0].copy() for a in ARMS}
        ee_quat = {a: ee_pose[a][1] for a in ARMS}
        q_snapshot = {a: franka[a].get_joint_positions()[:7].copy() for a in ARMS}
        protected_points = {
            a: _protected_points_from_links(
                ik_motion[a].ik, q_snapshot[a], ee_pos[a], ee_quat[a],
                link_frames, link_samples_per_segment, max(0, link_spheres),
                link_sphere_spacing, gripper_lateral_offsets,
                gripper_lateral_body_offsets
            )
            for a in ARMS
        }
        if proxy_viz is not None:
            proxy_viz.update(protected_points)

        if args.model == "lpvds":
            cart_vels = {}
            cart_nominal = {}
            cart_modulated = {}
            for arm in ARMS:
                if (stage[arm] != Stage.TRANSPORT or arm_done(arm) or
                        arm in moved_arms):
                    cart_vels[arm] = None
                    cart_nominal[arm] = None
                    cart_modulated[arm] = None
                    continue

                if np.linalg.norm(ee_pos[arm] - lpv_model[arm].x_goal) < args.done_tol:
                    print(f"  [{arm}] Transport done in {transport_steps[arm]} steps")
                    stage[arm] = Stage.PLACE
                    cart_vels[arm] = None
                    cart_nominal[arm] = None
                    cart_modulated[arm] = None
                    continue

                transport_steps[arm] += 1
                if transport_steps[arm] > args.max_transport:
                    print(f"  [WARN] {arm} transport hit max steps")
                    stage[arm] = Stage.PLACE
                    cart_vels[arm] = None
                    cart_nominal[arm] = None
                    cart_modulated[arm] = None
                    continue

                if args.raw_lpvds:
                    cart_vels[arm] = lpv_model[arm].predict(ee_pos[arm])
                else:
                    cart_vels[arm] = lpv_model[arm].safe_velocity(ee_pos[arm])
                cart_nominal[arm] = cart_vels[arm].copy()
                cart_modulated[arm] = cart_vels[arm].copy()

            for arm in ARMS:
                w = modulation_weight(arm)
                if w <= 0.0 or cart_vels[arm] is None:
                    continue
                other = other_arm(arm)
                v_nom = cart_vels[arm]
                v_mod = mod.huber.modulate_cartesian_points(
                    v_nom, protected_points[arm], protected_points[other]
                )
                cart_vels[arm] = (1.0 - w) * v_nom + w * v_mod
                nom_sq = float(np.dot(v_nom, v_nom))
                if nom_sq > 1e-12 and args.min_transport_progress > 0.0:
                    min_dot = float(args.min_transport_progress) * nom_sq
                    cmd_dot = float(np.dot(cart_vels[arm], v_nom))
                    if cmd_dot < min_dot:
                        cart_vels[arm] += ((min_dot - cmd_dot) / nom_sq) * v_nom
                cart_modulated[arm] = cart_vels[arm].copy()

            for arm in ARMS:
                if cart_vels[arm] is None:
                    continue
                cart_vels[arm] = args.speedup * args.cart_gain * cart_vels[arm]
                speed = np.linalg.norm(cart_vels[arm])
                max_cart_speed = args.speedup * args.max_cart_speed
                if speed > max_cart_speed:
                    cart_vels[arm] *= max_cart_speed / speed
            cap_yield_cart_speed(cart_vels)
            sync_vector_norms(cart_vels)

            any_transport = False
            for arm in ARMS:
                if cart_vels[arm] is None:
                    continue
                any_transport = True
                ee_target = ee_pos[arm] + cart_vels[arm] * args.lookahead * physics_dt
                if not args.no_workspace_clamp:
                    ee_target = np.clip(ee_target, lpv_min[arm], lpv_max[arm])
                ik_ok = ik_motion[arm].step_to(ee_target, target_quat=ee_down)
                other = other_arm(arm)
                comp = mod.huber.closest_components(
                    protected_points[arm], protected_points[other],
                    v_nom=cart_nominal[arm]
                )
                if comp is None:
                    comp = mod.huber.components(
                        ee_pos[arm], ee_pos[other], v_nom=cart_nominal[arm]
                    )
                diag_log.append({
                    "t": global_step * physics_dt,
                    "step": global_step,
                    "arm": arm,
                    "stage": stage[arm].name,
                    "priority_arm": priority_arm,
                    "priority_policy": args.priority_policy,
                    "mod_weight": float(modulation_weight(arm)),
                    "ee": ee_pos[arm].tolist(),
                    "ee_other": ee_pos[other].tolist(),
                    "protected_points": protected_points[arm].tolist(),
                    "protected_points_other": protected_points[other].tolist(),
                    "goal": lpv_model[arm].x_goal.tolist(),
                    "v_nom": cart_nominal[arm].tolist(),
                    "v_mod": cart_modulated[arm].tolist(),
                    "v_cmd": cart_vels[arm].tolist(),
                    "target": ee_target.tolist(),
                    "gamma": comp["gamma"],
                    "gamma_eff": comp["gamma_eff"],
                    "mod_isoline": comp["isoline"],
                    "lambda_r": comp["lambda_r"],
                    "lambda_t": comp["lambda_t"],
                    "tail_active": comp["tail_active"],
                    "preserve_speed": comp["preserve_speed"],
                    "closest_self_point_index": int(comp.get("self_point_index", 0)),
                    "closest_other_point_index": int(comp.get("obstacle_point_index", 0)),
                    "distance": float(comp.get(
                        "distance", np.linalg.norm(ee_pos[arm] - ee_pos[other])
                    )),
                    "ee_distance": float(np.linalg.norm(ee_pos[arm] - ee_pos[other])),
                    "safe_radius": float(mod_radius),
                    "reactivity": float(mod_reactivity),
                    "ik_ok": bool(ik_ok),
                })
                ik_fail_streak[arm] = 0 if ik_ok else ik_fail_streak[arm] + 1
                if ik_fail_streak[arm] == 25:
                    print(f"  [WARN] {arm} LPVDS IK has failed for 25 consecutive steps")

            if any_transport or moved_arms:
                env.step(render=not args.headless)
                global_step += 1
                idle_ticks = 0
                if args.status_every > 0 and global_step % args.status_every == 0:
                    print_status()
            else:
                idle_ticks += 1
                if idle_ticks % 240 == 0:
                    print_status(prefix="[IDLE]")
            continue

        q_dots = {}
        for arm in ARMS:
            if stage[arm] != Stage.TRANSPORT or arm_done(arm) or arm in moved_arms:
                q_dots[arm] = None
                continue

            q = franka[arm].get_joint_positions()[:7].copy()

            if np.linalg.norm(q - q_goal[arm]) < args.done_tol:
                print(f"  [{arm}] Transport done in {transport_steps[arm]} steps")
                stage[arm] = Stage.PLACE
                q_dots[arm] = None
                continue

            transport_steps[arm] += 1
            if transport_steps[arm] > args.max_transport:
                print(f"  [WARN] {arm} transport hit max steps")
                stage[arm] = Stage.PLACE
                q_dots[arm] = None
                continue

            x   = q - q_goal[arm]
            x_n = (x - ds[arm]["state_mean"]) / ds[arm]["state_std"]
            x_t = torch.tensor(x_n, dtype=torch.float32,
                               device=device).unsqueeze(0)

            if args.use_safe:
                scale_factor = torch.tensor(
                    ds[arm]["vel_scale"] / ds[arm]["state_std"],
                    dtype=torch.float32, device=device).unsqueeze(0)
                qd_n = ds[arm]["model"].safe_velocity(
                    x_t, scale_factor=scale_factor)
            else:
                with torch.no_grad():
                    qd_n = ds[arm]["model"](x_t)

            q_dots[arm] = (args.speedup * qd_n.cpu().numpy().squeeze(0) *
                           ds[arm]["vel_scale"])

        # Apply inter-arm modulation
        for arm in ARMS:
            w = modulation_weight(arm)
            if w <= 0.0 or q_dots[arm] is None:
                continue
            other = other_arm(arm)
            J = jacobian_finite_difference(franka[arm])
            q_dot_nom = q_dots[arm]
            if args.modulation_space == "jsdf":
                q_dot_mod = (
                    q_dot_nom
                    + jsdf_avoidance_velocity(
                        arm, q_dot_nom,
                        obstacle_points=protected_points[other],
                        self_points=protected_points[arm],
                    )
                )
            else:
                q_dot_mod = mod.modulate_joint_velocity_points(
                    q_dot_nominal=q_dot_nom,
                    self_points=protected_points[arm],
                    obstacle_points=protected_points[other],
                    jacobian=J,
                )
            q_dots[arm] = (1.0 - w) * q_dot_nom + w * q_dot_mod
            q_dots[arm] = add_nullspace_home_velocity(arm, q_dots[arm], J)

        sync_vector_norms(q_dots)

        # Command arms in transport
        any_transport = False
        for arm in ARMS:
            if q_dots[arm] is None:
                continue
            any_transport = True
            q = franka[arm].get_joint_positions()[:7].copy()
            q_cmd = q + q_dots[arm] * physics_dt
            finger = ik_motion[arm]._finger_width
            full_cmd = np.concatenate([q_cmd, [finger, finger]])
            franka[arm].apply_action(_articulation_action(full_cmd))

        if any_transport or moved_arms:
            env.step(render=not args.headless)
            global_step += 1
            idle_ticks = 0
            if args.status_every > 0 and global_step % args.status_every == 0:
                print_status()
        else:
            idle_ticks += 1
            if idle_ticks % 240 == 0:
                print_status(prefix="[IDLE]")

    print(f"[DEPLOY] Finished after {global_step} sim steps.")
    diag_path = Path(args.diag_out) if args.diag_out else None
    if diag_path:
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "wb") as f:
            pickle.dump({
                "config": {
                    "model": args.model,
                    "mod_radius": mod_radius,
                    "mod_reactivity": mod_reactivity,
                    "mod_isoline": mod_isoline,
                    "priority_mod_weight": args.priority_mod_weight,
                    "priority_policy": args.priority_policy,
                    "priority_hysteresis": args.priority_hysteresis,
                    "yield_mod_weight": args.yield_mod_weight,
                    "preserve_mod_speed": not args.no_preserve_mod_speed,
                    "link_spheres": int(max(0, link_spheres)),
                    "link_sphere_spacing": float(link_sphere_spacing),
                    "link_frames": list(link_frames),
                    "link_samples_per_segment": int(link_samples_per_segment),
                    "gripper_lateral_offsets": list(gripper_lateral_offsets),
                    "gripper_lateral_body_offsets": list(gripper_lateral_body_offsets),
                    "show_proxy_spheres": bool(args.show_proxy_spheres),
                    "proxy_sphere_radius": float(args.proxy_sphere_radius),
                    "isaac_record_out": args.isaac_record_out,
                    "grasp_xy_offset": cfg["block"].get("grasp_xy_offset", {}),
                    "grasp_offset_cli": args.grasp_offset,
                    "cart_gain": args.cart_gain,
                    "max_cart_speed": args.max_cart_speed,
                    "yield_max_cart_speed": args.yield_max_cart_speed,
                    "min_transport_progress": args.min_transport_progress,
                    "parked_obstacle_weight": args.parked_obstacle_weight,
                    "speedup": args.speedup,
                    "speed_sync": args.sync_speeds,
                },
                "rows": diag_log,
            }, f)
        print(f"[DEPLOY] Saved diagnostics to {diag_path}")
    _stop_screen_record(screen_recorder)
    simulation_app.close()
    if args.video_out:
        video_cmd = [
            sys.executable,
            "scripts/animate_lpvds_interaction.py",
            "--diag", str(diag_path),
            "--out", args.video_out,
            "--fps", str(args.video_fps),
            "--stride", str(args.video_stride),
            "--views", args.video_views,
        ]
        if args.video_radial_field:
            video_cmd.append("--radial_field")
        print(f"[DEPLOY] Rendering DS visualization video to {args.video_out}")
        subprocess.run(video_cmd, check=True)


if __name__ == "__main__":
    main()
