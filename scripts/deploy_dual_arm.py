"""
Dual-arm deployment with joint-space Neural DS + DS modulation for collision
avoidance.

Key difference from the previous (FSM-coordinator) version:
  - There is NO discrete hold/release logic. Both arms run their DS
    continuously at every timestep.
  - Inter-arm collision avoidance is handled by a state-dependent modulation
    matrix M(x_self, x_other) applied to each arm's velocity. The modulated
    velocity smoothly tangents along the safety-sphere of the other arm's EE.
  - Closed-loop: q̇_self = J_self^+ · M(ee_self, ee_other) · J_self · f(q - q_goal)
    The whole system is therefore a coupled dynamical system, not a hybrid
    system with discrete events.

We compute the Jacobian by finite differences (slow but version-portable).

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


def load_ds_set(ckpt_dir, ckpt_arm, device):
    from src.neural_ds import StableNeuralDS, N_JOINTS
    primitives = ["reach", "grasp", "lift", "transport", "place"]
    out = {}
    for p in primitives:
        ckpt = torch.load(ckpt_dir / f"{ckpt_arm}_{p}.pt", map_location=device, weights_only=False)
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
        out[p] = {
            "model":      model,
            "state_mean": ckpt["state_mean"],
            "state_std":  ckpt["state_std"],
            "vel_scale":  ckpt["vel_scale"],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt_arm", type=str, default="both")
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--use_safe", action="store_true")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override Lyapunov decay rate at deployment "
                             "(higher = more aggressive projection).")
    parser.add_argument("--goal_gain", type=float, default=0.0,
                        help="Add q_goal attraction term -gain*(q-q_goal) to "
                             "the learned DS before modulation. Use as a "
                             "stabilizing ablation when raw DS convergence is poor.")
    parser.add_argument("--ds_scale", type=float, default=1.0,
                        help="Scale learned DS velocity. Use 0 with "
                             "--goal_gain for a pure joint-space attractor "
                             "sanity check.")
    parser.add_argument("--max_joint_vel", type=float, default=None,
                        help="Deployment joint velocity clamp in rad/s. "
                             "Defaults to training.max_joint_vel from config.")
    parser.add_argument("--no_modulation", action="store_true",
                        help="Disable DS modulation (ablation).")
    parser.add_argument("--stagger_steps", type=int, default=None,
                        help="Initial right-arm launch delay in physics steps. "
                             "Defaults to coordination.start_stagger_steps.")
    parser.add_argument("--kinematic_carry", action="store_true",
                        help="After grasp, attach each active block to its EE "
                             "kinematically until place. Use this to debug the "
                             "DS/task pipeline separately from gripper contact.")
    parser.add_argument("--advance_on_timeout", action="store_true",
                        help="Legacy debug behavior: advance a primitive on "
                             "timeout even when q has not reached q_goal.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--done_tol", type=float, default=0.05)
    args = parser.parse_args()

    from isaacsim import SimulationApp
    _app_cfg = {"headless": args.headless}
    if not args.headless:
        _app_cfg.update({"width": 1280, "height": 720})
    simulation_app = SimulationApp(_app_cfg)

    from omni.isaac.core.utils.types import ArticulationAction
    from src.env import DualArmEnv
    from src.coordinator import TaskSequencer
    from src.franka_ik import FrankaIK
    from src.modulation import InterArmModulation, jacobian_finite_difference
    from src.primitives import gripper_action_for_primitive

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DualArmEnv(config_path=args.config, arms=("left", "right"))
    cfg = env.cfg
    franka = {"left": env.frankas["left"], "right": env.frankas["right"]}
    ik_kin = {arm: FrankaIK(franka[arm]) for arm in franka}

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ds_set = load_ds_set(ckpt_dir, args.ckpt_arm, device)

    # Override training alpha to drive faster Lyapunov decay at deployment
    if args.alpha is not None:
        for p in ds_set.values():
            p["model"].alpha = args.alpha
        print(f"[DEPLOY] Overriding alpha -> {args.alpha}")

    seq = TaskSequencer(env, cfg)
    mod = InterArmModulation(
        safe_radius=cfg["coordination"]["ee_safety_radius"],
        reactivity=4.0,
    )

    physics_dt = cfg["sim"]["physics_dt"]
    max_joint_vel = (cfg["training"]["max_joint_vel"]
                     if args.max_joint_vel is None else args.max_joint_vel)
    stagger_steps = (cfg["coordination"].get("start_stagger_steps", 0)
                     if args.stagger_steps is None else args.stagger_steps)
    arm_start_step = {"left": 0, "right": max(0, stagger_steps)}

    # Open both grippers
    for arm in franka:
        franka[arm].gripper.apply_action(
            ArticulationAction(joint_positions=np.array([0.04, 0.04]))
        )

    # Let blocks settle before querying their positions
    for _ in range(60):
        env.step(render=not args.headless)

    # Initialise q_goals per arm
    def update_q_goal(arm):
        cart = seq.cartesian_target(arm)
        if cart is None:
            return None
        q_seed = franka[arm].get_joint_positions()[:7].copy()
        ee_quat = seq.ee_orientation(arm)
        q_goal, _ = ik_kin[arm].solve(cart, target_quat=ee_quat, q_seed=q_seed)
        seq.tasks[arm].q_goal = q_goal
        return q_goal

    for arm in ("left", "right"):
        update_q_goal(arm)

    last_prim = {arm: seq.tasks[arm].current_primitive for arm in ("left", "right")}
    prim_steps = {"left": 0, "right": 0}
    # 30× the collection budget per primitive before we give up.
    prim_timeout = {p: s * 30
                    for p, s in cfg["sim"]["steps_per_primitive"].items()}

    print(f"[DEPLOY] Dual-arm joint-space DS — safe={args.use_safe}, "
          f"modulation={'OFF' if args.no_modulation else 'ON'}, "
          f"goal_gain={args.goal_gain}, ds_scale={args.ds_scale}, "
          f"max_joint_vel={max_joint_vel}")
    if stagger_steps > 0:
        print(f"[DEPLOY] Initial stagger: right arm starts after "
              f"{stagger_steps} steps ({stagger_steps * physics_dt:.2f}s)")

    held_block = {"left": None, "right": None}
    held_offset = {"left": np.zeros(3), "right": np.zeros(3)}

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

    for step in range(args.max_steps):
        if not simulation_app.is_running():
            break

        # Cache EE positions BEFORE we move so modulation uses consistent state
        ee_pos = {arm: env.get_ee_pose(arm)[0].copy() for arm in ("left", "right")}

        # Compute nominal q̇ for each arm (in parallel, before any commits)
        q_dots = {}
        for arm in ("left", "right"):
            task = seq.tasks[arm]
            if task.is_done() or step < arm_start_step[arm]:
                q_dots[arm] = None
                continue

            if task.current_primitive != last_prim[arm]:
                update_q_goal(arm)
                last_prim[arm] = task.current_primitive
                prim_steps[arm] = 0

            prim_steps[arm] += 1

            q = franka[arm].get_joint_positions()[:7].copy()
            ds = ds_set[task.current_primitive]
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
            if args.goal_gain > 0:
                q_dots[arm] = q_dots[arm] - args.goal_gain * x
            q_dots[arm] = np.clip(q_dots[arm], -max_joint_vel, max_joint_vel)

        # Apply modulation between the two arms
        if not args.no_modulation:
            for arm, other in (("left", "right"), ("right", "left")):
                if q_dots[arm] is None:
                    continue
                # Compute Jacobian by finite-diff (slow but version-stable)
                J = jacobian_finite_difference(franka[arm])
                q_dots[arm] = mod.modulate_joint_velocity(
                    q_dot_nominal=q_dots[arm],
                    ee_pos_self=ee_pos[arm],
                    ee_pos_other=ee_pos[other],
                    jacobian=J,
                )

        # Apply commands and step
        for arm in ("left", "right"):
            if q_dots[arm] is None:
                continue
            q = franka[arm].get_joint_positions()[:7].copy()
            q_cmd_full = franka[arm].get_joint_positions().copy()
            q_cmd_full[:7] = q + q_dots[arm] * physics_dt
            franka[arm].apply_action(ArticulationAction(joint_positions=q_cmd_full))

        env.step(render=not args.headless)
        carry_held_blocks()

        # Per-arm primitive completion checks
        for arm in ("left", "right"):
            task = seq.tasks[arm]
            if task.is_done():
                continue
            q = franka[arm].get_joint_positions()[:7]
            timed_out = prim_steps[arm] >= prim_timeout[task.current_primitive]
            converged = np.linalg.norm(q - task.q_goal) < args.done_tol
            if converged or timed_out:
                if timed_out and not converged:
                    print(f"[WARN] {arm}/{task.current_primitive} timed out "
                          f"after {prim_steps[arm]} steps "
                          f"(||q-q_goal||={np.linalg.norm(q - task.q_goal):.3f})")
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
                seq.primitive_complete(arm)
                prim_steps[arm] = 0

        if all(seq.tasks[a].is_done() for a in ("left", "right")):
            print("[DEPLOY] Both arms finished stacking.")
            break

    print(f"[DEPLOY] Finished after {step + 1} steps.")
    simulation_app.close()


if __name__ == "__main__":
    main()
