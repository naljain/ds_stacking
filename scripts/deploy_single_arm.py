"""
Deploy joint-space Neural DS on a single arm.

The DS produces q_dot directly. We integrate q_dot to get a target joint
position, send that as the command, and let Isaac Sim's articulation
controller handle the low-level torque tracking. IK is used only at primitive
transitions; inside a primitive, the closed-loop joint dynamics are the trained
DS modulo actuator dynamics and optional deployment ablations.

When a primitive transitions, we call Lula IK once to compute q_goal for the
new Cartesian target, then keep that q_goal fixed until the next transition.

Usage:
  python scripts/deploy_single_arm.py --arm left
  python scripts/deploy_single_arm.py --arm left --use_safe   # Lyapunov projection
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


def _load_deploy_config(path):
    if path is None:
        return {}
    with open(path, "r") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Deploy config must be a mapping: {path}")
    return payload.get("deploy", payload)


def load_ds(ckpt_path, device):
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


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--deploy_config", type=str, default=None,
                     help="YAML file with deploy argument defaults.")
    pre_args, _ = pre.parse_known_args()
    deploy_defaults = _load_deploy_config(pre_args.deploy_config)

    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--arm", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt_arm", type=str, default=None,
                        choices=["left", "right"],
                        help="Checkpoint prefix for the loaded DS. Defaults to "
                             "--arm; override only for ablations like running "
                             "the right-trained DS on the left arm.")
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--use_safe", action="store_true",
                        help="Apply Lyapunov projection at inference")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override Lyapunov decay rate at deployment "
                             "(higher = more aggressive projection / faster convergence).")
    parser.add_argument("--ds_scale", type=float, default=1.0,
                        help="Scale learned DS velocity. Set to 0 for a pure "
                             "joint-space attractor sanity check (then only "
                             "DS_GOAL_GAIN drives motion).")
    parser.add_argument("--max_joint_vel", type=float, default=None,
                        help="Deployment joint velocity clamp in rad/s. "
                             "Defaults to training.max_joint_vel from config.")
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
    parser.add_argument("--place_cart_done_tol", type=float, default=0.01,
                        help="Cartesian EE completion tolerance (meters) for place. "
                             "Tight by default so --kinematic_carry doesn't snap early.")
    parser.add_argument("--log_csv", type=str, default=None,
                        help="If set, write per-step diagnostics to this CSV "
                             "for post-mortem plotting.")
    parser.add_argument("--print_every", type=int, default=50,
                        help="Print diagnostic line every N steps (0 = off).")
    parser.add_argument("--debug_ik", action="store_true",
                        help="Print IK target, success flag, and resulting "
                             "joint goal at each primitive transition.")
    parser.add_argument("--debug_grasp", action="store_true",
                        help="Print finger joint positions and block height "
                             "after close/lift for physical grasp debugging.")
    parser.add_argument("--kinematic_carry", action="store_true",
                        help="After grasp, attach the active block to the EE "
                             "kinematically until place. Use this to debug the "
                             "DS/task pipeline separately from gripper contact.")
    parser.add_argument("--advance_on_timeout", action="store_true",
                        help="Legacy debug behavior: advance to the next "
                             "primitive on timeout even if q has not reached "
                             "q_goal. Leave this off for pickup tests.")
    parser.add_argument("--ik_primitives", type=str, default="grasp,lift,place",
                        help="Comma-separated primitives to execute with the "
                             "Lula joint-space controller instead of the "
                             "learned DS.")
    parser.add_argument("--ik_goal_gain", type=float, default=3.0,
                        help="Joint-space attraction gain for "
                             "Lula-controlled primitives.")
    parser.add_argument("--cartesian_ik_primitives", action="store_true",
                        help="Execute IK primitives as straight Cartesian "
                             "segments, matching the stephen/testing pre-pick "
                             "and pre-place behavior.")
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
                             "the selected arm.")
    parser.add_argument("--post_place_lift_steps", type=int, default=80,
                        help="After opening on place, lift the empty gripper "
                             "back to transport height for this many joint-Lula "
                             "steps before advancing to the next block. Use 0 "
                             "to disable.")
    parser.add_argument("--post_place_lift_tol", type=float, default=0.03,
                        help="Cartesian tolerance in meters for the post-place "
                             "empty-gripper lift.")
    parser.add_argument("--debug_post_place_lift", action="store_true",
                        help="Print post-place lift target and final EE error.")
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
        args.gif_out = str(Path(args.video_dir) / f"{args.arm}_neural.gif")
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
    from src.offscreen_recorder import OffscreenGifRecorder
    from src.primitives import (
        DS_PRIMITIVES,
        SCRIPTED_PRIMITIVES,
        gripper_action_for_primitive,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DualArmEnv(config_path=args.config, arms=(args.arm,))
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
        base = np.asarray(offsets.get(args.arm, [0.0, 0.0]), dtype=float)
        offsets[args.arm] = (base + np.asarray(args.grasp_offset, dtype=float)).tolist()
        print(f"[DEPLOY] {args.arm} grasp XY offset -> {offsets[args.arm]}")
    franka = env.frankas[args.arm]
    ik_kin = FrankaIK(franka)
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
    ckpt_arm = args.ckpt_arm if args.ckpt_arm is not None else args.arm
    ds_set = {p: load_ds(ckpt_dir / f"{ckpt_arm}_{p}.pt", device)
              for p in ds_primitives}
    print(f"[DEPLOY] Loaded DS checkpoints: {ckpt_arm}_*.pt for "
          f"primitives {ds_primitives}")

    # Override training alpha to drive faster Lyapunov decay at deployment
    if args.alpha is not None:
        for p in ds_set.values():
            p["model"].alpha = args.alpha
        print(f"[DEPLOY] Overriding alpha -> {args.alpha}")

    seq = TaskSequencer(env, cfg)
    physics_dt = cfg["sim"]["physics_dt"]
    max_joint_vel = (cfg["training"]["max_joint_vel"]
                     if args.max_joint_vel is None else args.max_joint_vel)

    # Compute initial q_goal for the first primitive
    def update_q_goal(arm):
        task = seq.tasks[arm]
        cart = seq.cartesian_target(arm)
        if cart is None:
            return None
        # Seed Lula IK from the previous primitive's q_goal when available.
        # The collection-time expert (direct_joint_set) settled to within
        # joint_done_tol of every q_goal, so seeding the *next* primitive's
        # IK from the previous q_goal matches what Lula saw during data
        # collection. Seeding from the lagging live articulation pose
        # instead lets a 0.1–0.2 rad tracking lag flip Lula to a different
        # IK branch — the resulting q_goal can be 2+ rad away in joint
        # space, which then drives the DS into OOD territory.
        prev_q_goal = getattr(task, "q_goal", None)
        if prev_q_goal is not None:
            q_seed = np.asarray(prev_q_goal, dtype=float).copy()
        else:
            q_seed = franka.get_joint_positions()[:7].copy()
        ee_quat = seq.ee_orientation(arm)
        q_goal, ok = ik_kin.solve(cart, target_quat=ee_quat, q_seed=q_seed)
        if args.debug_ik:
            print(f"[IK] {seq.tasks[arm].current_primitive:9s} ok={ok} "
                  f"cart={cart.round(3)} seed={q_seed.round(3)} "
                  f"q_goal={q_goal.round(3)} "
                  f"||seed-goal||={np.linalg.norm(q_seed - q_goal):.3f}")
        task = seq.tasks[arm]
        if task.current_primitive in ("transport", "place"):
            print(f"[STACK] {arm}/{task.current_primitive} "
                  f"slot={seq.stack_slot_index(arm)} "
                  f"block_center_z={task.reserved_goal_z:.3f} "
                  f"ee_target_z={cart[2]:.3f}")
        seq.tasks[arm].q_goal = q_goal
        return q_goal

    update_q_goal(args.arm)

    print(f"[DEPLOY] Joint-space DS on {args.arm} arm — safe={args.use_safe}, "
          f"ds_scale={args.ds_scale}, ds_goal_gain={DS_GOAL_GAIN}, "
          f"max_joint_vel={max_joint_vel}")

    last_primitive = seq.tasks[args.arm].current_primitive
    prim_steps = 0
    q_cmd_state = franka.get_joint_positions().copy()
    desired_finger_width = 0.04
    # 30× the collection budget — very generous so a slow-converging DS
    # has plenty of room before we give up and advance.
    prim_timeout = {p: s * 30
                    for p, s in cfg["sim"]["steps_per_primitive"].items()}

    def trapezoid_profile(n_steps, ramp_frac=0.2):
        n_steps = max(1, int(n_steps))
        ramp = max(1, int(n_steps * ramp_frac))
        cruise = max(0, n_steps - 2 * ramp)
        v = np.zeros(n_steps)
        for i in range(ramp):
            v[i] = (i + 1) / ramp
        for i in range(ramp, ramp + cruise):
            v[i] = 1.0
        for i in range(ramp + cruise, n_steps):
            v[i] = (n_steps - i) / ramp
        v /= max(v.sum(), 1e-12)
        return np.clip(np.cumsum(v), 0.0, 1.0)

    def apply_full_command(q7=None):
        full_cmd = franka.get_joint_positions().copy()
        if q7 is not None:
            full_cmd[:7] = q7[:7]
        full_cmd[7:9] = desired_finger_width
        franka.apply_action(ArticulationAction(joint_positions=full_cmd))

    def hold_gripper(width, steps):
        nonlocal desired_finger_width, q_cmd_state
        desired_finger_width = float(width)
        for _ in range(steps):
            q_now = franka.get_joint_positions().copy()
            q_now[7:9] = desired_finger_width
            franka.apply_action(ArticulationAction(joint_positions=q_now))
            step_env()
        q_cmd_state = franka.get_joint_positions().copy()
        q_cmd_state[7:9] = desired_finger_width

    hold_gripper(0.04, cfg["sim"]["gripper_steps"])

    # Let blocks settle before querying their positions
    for _ in range(60):
        step_env()
    q_cmd_state = franka.get_joint_positions().copy()

    csv_log = None
    if args.log_csv is not None:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_log = open(args.log_csv, "w")
        csv_log.write("step,primitive,prim_step,e_norm,V,"
                      "qd_raw_norm,qd_norm,proj_delta,cos_to_goal\n")
        print(f"[DEPLOY] Logging to {args.log_csv}")

    held_block = None
    held_offset = np.zeros(3)

    def carry_held_block():
        if held_block is None:
            return
        ee_pos = ik_frame_position().copy()
        obj = env.get_block_obj(held_block)
        obj.set_world_pose(position=ee_pos + held_offset,
                           orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))

    def print_grasp_debug(label, block_name):
        if not args.debug_grasp or block_name is None:
            return
        q_full = franka.get_joint_positions()
        block_pos = env.get_block_positions()[block_name]
        ee_pos = ik_frame_position()
        table_top = cfg["table"]["height"]
        print(f"[GRASP] {label}: block={block_name} "
              f"block_xy=({block_pos[0]:+.3f},{block_pos[1]:+.3f}) "
              f"ee_xy=({ee_pos[0]:+.3f},{ee_pos[1]:+.3f}) "
              f"xy_err={np.linalg.norm(ee_pos[:2] - block_pos[:2]):.4f} "
              f"z={block_pos[2]:.4f} dz_table={block_pos[2] - table_top:.4f} "
              f"fingers={q_full[7:9].round(4)} "
              f"target_width={desired_finger_width:.4f}")

    def snap_held_block_to_stack():
        if held_block is None:
            return
        obj = env.get_block_obj(held_block)
        obj.set_world_pose(position=seq.stack_target_position(args.arm),
                           orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))

    def ik_frame_position():
        try:
            return ik_kin.forward_position(franka.get_joint_positions()[:7].copy())
        except Exception as exc:
            if args.debug_ik:
                print(f"[WARN] Lula FK failed, using env EE pose: {exc}")
            return env.get_ee_pose(args.arm)[0]

    def joint_lula_move_to_cart(target_cart, steps, cart_tol=None, label="joint_lula"):
        if steps <= 0:
            return False
        if cart_tol is None:
            cart_tol = args.post_place_lift_tol
        start_ee = env.get_ee_pose(args.arm)[0].copy()
        q_seed = franka.get_joint_positions()[:7].copy()
        q_goal, ok = ik_kin.solve(target_cart, target_quat=None, q_seed=q_seed)
        if not ok:
            print(f"[WARN] {label} IK failed for target={target_cart.round(3)}")
            return False
        if args.debug_post_place_lift:
            print(f"[POST_PLACE] target={target_cart.round(3)} "
                  f"start_ee={start_ee.round(3)} "
                  f"||q-q_goal||={np.linalg.norm(q_seed - q_goal):.3f}")
        for _ in range(steps):
            q_now = franka.get_joint_positions()[:7].copy()
            ee_now = ik_frame_position()
            if np.linalg.norm(ee_now - target_cart) < cart_tol:
                break
            err = q_now - q_goal
            q_dot = np.clip(-args.ik_goal_gain * err, -max_joint_vel, max_joint_vel)
            q_cmd_state[:7] = q_cmd_state[:7] + q_dot * physics_dt
            apply_full_command(q_cmd_state[:7])
            step_env()
        final_ee = ik_frame_position().copy()
        q_cmd_state[:] = franka.get_joint_positions().copy()
        final_err = np.linalg.norm(final_ee - target_cart)
        if args.debug_post_place_lift:
            print(f"[POST_PLACE] final_ee={final_ee.round(3)} "
                  f"err={final_err:.3f}")
        return final_err < cart_tol

    def cartesian_lula_move_to_cart(target_cart, target_quat, steps,
                                    cart_tol=None, label="cartesian_ik"):
        nonlocal q_cmd_state
        if cart_tol is None:
            cart_tol = IK_CART_DONE_TOL
        start_ee = ik_frame_position().copy()
        target_cart = np.asarray(target_cart, dtype=float).copy()
        q_seed = franka.get_joint_positions()[:7].copy()
        q_last = q_seed.copy()
        ok_any = False
        for s in trapezoid_profile(steps):
            waypoint = start_ee + s * (target_cart - start_ee)
            q_goal, ok = ik_kin.solve(
                waypoint,
                target_quat=target_quat,
                q_seed=q_last,
            )
            if not ok:
                if args.debug_ik:
                    print(f"[WARN] {label} IK failed at waypoint={waypoint.round(3)}")
                continue
            ok_any = True
            q_last = q_goal.copy()
            q_cmd_state[:7] = q_goal
            apply_full_command(q_cmd_state[:7])
            step_env()
            carry_held_block()
        q_cmd_state[:] = franka.get_joint_positions().copy()
        final_err = np.linalg.norm(ik_frame_position() - target_cart)
        if not ok_any:
            print(f"[WARN] {label} IK failed for every waypoint "
                  f"target={target_cart.round(3)}")
            return False
        if final_err > cart_tol:
            print(f"[WARN] {label} final EE error {final_err:.4f} m "
                  f"> {cart_tol:.4f} m")
        return final_err < cart_tol

    for step in range(args.max_steps):
        if not simulation_app.is_running():
            break
        task = seq.tasks[args.arm]
        if task.is_done():
            print("[DEPLOY] All blocks placed.")
            break

        # If primitive changed since last step, refresh q_goal
        if task.current_primitive != last_primitive:
            update_q_goal(args.arm)
            q_cmd_state = franka.get_joint_positions().copy()
            q_goal_dbg = task.q_goal
            q_now_dbg  = franka.get_joint_positions()[:7]
            init_e     = np.linalg.norm(q_now_dbg - q_goal_dbg)
            print(f"[DEPLOY] -> {task.current_primitive:9s}  "
                  f"q_goal={q_goal_dbg.round(2)}  init ||e||={init_e:.3f}")
            last_primitive = task.current_primitive
            prim_steps = 0

        prim_steps += 1

        q      = franka.get_joint_positions()[:7].copy()
        q_goal = task.q_goal
        cart_target = seq.cartesian_target(args.arm)
        using_ik = task.current_primitive in ik_primitives

        if using_ik and args.cartesian_ik_primitives:
            steps = cfg["sim"]["steps_per_primitive"].get(task.current_primitive, 80)
            cartesian_lula_move_to_cart(
                cart_target,
                seq.ee_orientation(args.arm),
                steps=steps,
                cart_tol=IK_CART_DONE_TOL,
                label=task.current_primitive,
            )
            grip = gripper_action_for_primitive(task.current_primitive)
            if grip == "close":
                hold_gripper(0.0, cfg["sim"]["gripper_steps"])
                print_grasp_debug("after close", task.current_block)
                if args.kinematic_carry:
                    ee_pos = ik_frame_position().copy()
                    block_pos = env.get_block_positions()[task.current_block].copy()
                    held_block = task.current_block
                    held_offset = block_pos - ee_pos
                    carry_held_block()
            elif grip == "open":
                if args.kinematic_carry:
                    hold_gripper(0.04, cfg["sim"]["gripper_steps"])
                    snap_held_block_to_stack()
                    held_block = None
                else:
                    hold_gripper(0.04, cfg["sim"]["gripper_steps"])
                    held_block = None
                if args.post_place_lift_steps > 0:
                    joint_lula_move_to_cart(
                        seq.stack_clearance_target(args.arm),
                        args.post_place_lift_steps,
                        cart_tol=args.post_place_lift_tol,
                        label="post-place lift",
                    )
                    q_cmd_state = franka.get_joint_positions().copy()
            elif args.debug_grasp and task.current_primitive == "lift":
                print_grasp_debug("after lift", task.current_block)
            seq.primitive_complete(args.arm)
            prim_steps = 0
            continue

        if using_ik:
            V_val = 0.0
            x = q - q_goal
            e_norm = np.linalg.norm(x)
            ee_err = np.linalg.norm(ik_frame_position() - cart_target)
            q_dot_raw = -args.ik_goal_gain * x
            q_dot_clipped = np.clip(q_dot_raw, -max_joint_vel, max_joint_vel)
            qd_norm = np.linalg.norm(q_dot_clipped)
            cos_to_goal = (
                -np.dot(x, q_dot_clipped) / (e_norm * qd_norm + 1e-9)
                if e_norm * qd_norm > 1e-9 else 0.0
            )
            proj_correction = 0.0
        else:
            # Build state & query DS
            ds  = ds_set[task.current_primitive]
            x   = q - q_goal
            x_n = (x - ds["state_mean"]) / ds["state_std"]
            x_t = torch.tensor(x_n, dtype=torch.float32, device=device).unsqueeze(0)

            # Always evaluate raw f for logging — cheap (single forward pass)
            with torch.no_grad():
                qd_n_raw = ds["model"](x_t)
            if args.use_safe:
                scale_factor = torch.tensor(
                    ds["vel_scale"] / ds["state_std"],
                    dtype=torch.float32, device=device).unsqueeze(0)
                qd_n = ds["model"].safe_velocity(x_t, scale_factor=scale_factor)
            else:
                qd_n = qd_n_raw

            q_dot_raw = qd_n_raw.cpu().numpy().squeeze(0) * ds["vel_scale"]
            q_dot = args.ds_scale * qd_n.cpu().numpy().squeeze(0) * ds["vel_scale"]
            if DS_GOAL_GAIN > 0:
                q_dot = q_dot - DS_GOAL_GAIN * x
            q_dot_clipped = np.clip(q_dot, -max_joint_vel, max_joint_vel)

            # V value, useful for tracking convergence
            with torch.no_grad():
                V_val = ds["model"].V(x_t).item()

            # Cosine between q_dot and -e: +1 means heading straight to goal,
            # -1 means moving directly away. The single most diagnostic number.
            e_norm = np.linalg.norm(x)
            qd_norm = np.linalg.norm(q_dot_clipped)
            cos_to_goal = (
                -np.dot(x, q_dot_clipped) / (e_norm * qd_norm + 1e-9)
                if e_norm * qd_norm > 1e-9 else 0.0
            )
            # Use Lula's IK frame (right_gripper) for the diagnostic, NOT
            # Isaac's high-level end_effector (panda_hand). They differ by ~10
            # cm of gripper-hand offset, and cart_target is defined for the
            # Lula frame because that's what IK targets. Mixing them produces
            # a small ee_err in the print while the convergence check (which
            # correctly uses ik_frame_position) sees a much larger cart_err.
            ee_err = np.linalg.norm(ik_frame_position() - cart_target)

            proj_correction = float(np.linalg.norm(q_dot - q_dot_raw))

        if args.print_every and step % args.print_every == 0:
            q_cmd_preview = q_cmd_state[:7] + q_dot_clipped * physics_dt
            print(f"  step {step:5d} | {task.current_primitive:9s} | "
                  f"{'IK' if using_ik else 'DS'} | "
                  f"||e||={e_norm:.3f}  V={V_val:7.3f}  "
                  f"ee_err={ee_err:.3f}  "
                  f"||qd_raw||={np.linalg.norm(q_dot_raw):.2f}  "
                  f"||qd||={qd_norm:.2f}  "
                  f"proj_Δ={proj_correction:.2f}  "
                  f"cos→goal={cos_to_goal:+.2f}  "
                  f"max|qcmd-q|={np.max(np.abs(q_cmd_preview - q)):.4f}")

        if args.log_csv is not None:
            csv_log.write(
                f"{step},{task.current_primitive},{prim_steps},"
                f"{e_norm:.5f},{V_val:.5f},"
                f"{np.linalg.norm(q_dot_raw):.5f},"
                f"{qd_norm:.5f},{proj_correction:.5f},"
                f"{cos_to_goal:.5f}\n"
            )

        # Integrate velocity into a persistent command trajectory. Using
        # q_measured + q_dot*dt every tick can stall when the articulation
        # lags the tiny one-step target; q_cmd_state gives the low-level
        # position controller an actual trajectory to track.
        q_cmd_state[:7] = q_cmd_state[:7] + q_dot_clipped * physics_dt
        apply_full_command(q_cmd_state[:7])

        step_env()
        carry_held_block()

        # Primitive completion: only convergence means success. A timeout is a
        # controller failure for real pickup; advancing would close the gripper
        # from the wrong pose and cascade into misleading downstream failures.
        timed_out = prim_steps >= prim_timeout[task.current_primitive]
        if using_ik:
            ee_pos = ik_frame_position()
            cart_err = np.linalg.norm(ee_pos - cart_target)
            joint_err = np.linalg.norm(franka.get_joint_positions()[:7] - q_goal)
            done_err = cart_err
            cart_tol = (
                args.place_cart_done_tol
                if task.current_primitive == "place" else args.cart_done_tol
            )
            if task.current_primitive == "place":
                converged = cart_err < cart_tol
            else:
                converged = (cart_err < cart_tol) or (joint_err < IK_JOINT_DONE_TOL)
            done_label = "||ee-target||"
        else:
            joint_err = np.linalg.norm(q - q_goal)
            cart_err = np.linalg.norm(ik_frame_position() - cart_target)
            if task.current_primitive == "reach" and cart_err < DS_REACH_CART_DONE_TOL:
                converged = True
                done_err = cart_err
                done_label = "||ee-target||"
            else:
                converged = joint_err < joint_done_tol
                done_err = joint_err
                done_label = "||q-q_goal||"
        if converged or timed_out:
            if timed_out and not converged:
                print(f"[WARN] {task.current_primitive} timed out after "
                      f"{prim_steps} steps ({done_label}={done_err:.3f})")
                if not args.advance_on_timeout:
                    print("[DEPLOY] Aborting instead of advancing. Use "
                          "--advance_on_timeout only for phase-flow debugging.")
                    break
            grip = gripper_action_for_primitive(task.current_primitive)
            if grip == "close":
                hold_gripper(0.0, cfg["sim"]["gripper_steps"])
                print_grasp_debug("after close", task.current_block)
                if args.kinematic_carry:
                    ee_pos = ik_frame_position().copy()
                    block_pos = env.get_block_positions()[task.current_block].copy()
                    held_block = task.current_block
                    held_offset = block_pos - ee_pos
                    carry_held_block()
            elif grip == "open":
                if args.kinematic_carry:
                    hold_gripper(0.04, cfg["sim"]["gripper_steps"])
                    snap_held_block_to_stack()
                    held_block = None
                else:
                    hold_gripper(0.04, cfg["sim"]["gripper_steps"])
                    held_block = None
                if args.post_place_lift_steps > 0:
                    retract_cart = seq.stack_clearance_target(args.arm)
                    joint_lula_move_to_cart(
                        retract_cart,
                        args.post_place_lift_steps,
                        cart_tol=args.post_place_lift_tol,
                        label="post-place lift",
                    )
                    q_cmd_state = franka.get_joint_positions().copy()
            elif args.debug_grasp and task.current_primitive == "lift":
                print_grasp_debug("after lift", task.current_block)
            seq.primitive_complete(args.arm)
            prim_steps = 0

    print(f"[DEPLOY] Finished after {step + 1} steps.")
    if csv_log is not None:
        csv_log.close()
    recorder.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
