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
import numpy as np
import torch
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
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--mod_reactivity", type=float, default=4.0,
                        help="Exponent for inter-arm modulation. Lower values "
                             "make modulation act earlier over a wider range.")
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
    args = parser.parse_args()
    joint_done_tol = args.done_tol if args.joint_done_tol is None else args.joint_done_tol

    from isaacsim import SimulationApp
    _app_cfg = {"headless": args.headless}
    if not args.headless:
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
    from src.primitives import (
        DS_PRIMITIVES,
        SCRIPTED_PRIMITIVES,
        gripper_action_for_primitive,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DualArmEnv(config_path=args.config, arms=("left", "right"))
    cfg = env.cfg
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
    mod = InterArmModulation(
        safe_radius=(cfg["coordination"]["ee_safety_radius"]
                     if args.mod_safe_radius is None else args.mod_safe_radius),
        reactivity=args.mod_reactivity,
    )

    physics_dt = cfg["sim"]["physics_dt"]
    max_joint_vel = (cfg["training"]["max_joint_vel"]
                     if args.max_joint_vel is None else args.max_joint_vel)
    stagger_steps = (cfg["coordination"].get("start_stagger_steps", 0)
                     if args.stagger_steps is None else args.stagger_steps)
    arm_start_step = {"left": 0, "right": max(0, stagger_steps)}
    home_q = {arm: franka[arm].get_joint_positions()[:7].copy()
              for arm in ("left", "right")}
    q_cmd_state = {arm: franka[arm].get_joint_positions().copy()
                   for arm in ("left", "right")}
    arm_parked = {arm: False for arm in ("left", "right")}

    # Open both grippers
    for arm in franka:
        franka[arm].gripper.apply_action(
            ArticulationAction(joint_positions=np.array([0.04, 0.04]))
        )

    # Let blocks settle before querying their positions
    for _ in range(60):
        env.step(render=not args.headless)
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
    if ik_primitives:
        print(f"[DEPLOY] IK primitives: {', '.join(sorted(ik_primitives))}")
    if stagger_steps > 0:
        print(f"[DEPLOY] Initial stagger: right arm starts after "
              f"{stagger_steps} steps ({stagger_steps * physics_dt:.2f}s)")

    held_block = {"left": None, "right": None}
    held_offset = {"left": np.zeros(3), "right": np.zeros(3)}
    link_hold_arm = None
    safety_hold_steps = {"left": 0, "right": 0}

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
            ee = env.get_ee_pose(arm)[0].copy()
            obj = env.get_block_obj(held_block[arm])
            obj.set_world_pose(position=ee + held_offset[arm],
                               orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))

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
            full = franka[arm].get_joint_positions().copy()
            full[:7] = q_cmd_state[arm][:7]
            franka[arm].apply_action(ArticulationAction(joint_positions=full))
            env.step(render=not args.headless)
            carry_held_blocks()
        q_cmd_state[arm] = franka[arm].get_joint_positions().copy()
        return np.linalg.norm(ik_frame_position(arm) - target_cart) < cart_tol

    for step in range(args.max_steps):
        if not simulation_app.is_running():
            break

        # Cache EE positions BEFORE we move so modulation uses consistent state
        ee_pos = {arm: env.get_ee_pose(arm)[0].copy() for arm in ("left", "right")}

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
                    simulation_app.close()
                    return
                last_prim[arm] = task.current_primitive
                prim_steps[arm] = 0
                safety_hold_steps[arm] = 0
                q_cmd_state[arm] = franka[arm].get_joint_positions().copy()

            prim_steps[arm] += 1

            q = franka[arm].get_joint_positions()[:7].copy()
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
                q_dots[arm] = mod.modulate_joint_velocity(
                    q_dot_nominal=q_dots[arm],
                    ee_pos_self=ee_pos[arm],
                    ee_pos_other=ee_pos[other],
                    jacobian=J,
                )

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
            q_cmd_full = franka[arm].get_joint_positions().copy()
            q_cmd_full[:7] = q_cmd_state[arm][:7]
            franka[arm].apply_action(ArticulationAction(joint_positions=q_cmd_full))

        env.step(render=not args.headless)
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
                done_err = np.linalg.norm(ee_pos_now - target_now)
                joint_err = np.linalg.norm(q - task.q_goal)
                converged = (
                    done_err < IK_CART_DONE_TOL
                    or joint_err < IK_JOINT_DONE_TOL
                )
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
                        simulation_app.close()
                        return
                grip = gripper_action_for_primitive(task.current_primitive)
                if grip == "close":
                    franka[arm].gripper.apply_action(
                        ArticulationAction(joint_positions=np.array([0.0, 0.0]))
                    )
                    for _ in range(cfg["sim"]["gripper_steps"]):
                        env.step(render=not args.headless)
                    q_cmd_state[arm] = franka[arm].get_joint_positions().copy()
                    if args.kinematic_carry:
                        ee = env.get_ee_pose(arm)[0].copy()
                        block_pos = env.get_block_positions()[task.current_block].copy()
                        held_block[arm] = task.current_block
                        held_offset[arm] = block_pos - ee
                        carry_held_blocks()
                elif grip == "open":
                    if args.kinematic_carry:
                        snap_held_block_to_stack(arm)
                    held_block[arm] = None
                    franka[arm].gripper.apply_action(
                        ArticulationAction(joint_positions=np.array([0.04, 0.04]))
                    )
                    for _ in range(cfg["sim"]["gripper_steps"]):
                        env.step(render=not args.headless)
                    q_cmd_state[arm] = franka[arm].get_joint_positions().copy()
                    joint_lula_move_to_cart(
                        arm,
                        seq.stack_clearance_target(arm),
                        steps=120,
                        cart_tol=0.03,
                        label="post-place stack clearance",
                    )
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
    simulation_app.close()


if __name__ == "__main__":
    main()
