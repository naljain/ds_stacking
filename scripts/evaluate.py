"""
Evaluation harness — runs the joint-space DS + modulation pipeline under
various perturbation conditions and ablations.

Conditions:
  nominal             — no perturbations
  block_displacement  — random XY shift on a target block mid-task
  ee_disturbance      — force impulse on EE during transport
  arm_block           — freeze one arm during stacking
  combined            — all of the above

Ablations (per condition, controlled by flags):
  --no_modulation     — disable DS modulation (FSM-free becomes naive parallel)
  --no_lyapunov_proj  — use raw f(x) without projection (test soft training)

Metrics per condition:
  stack_completion_rate
  blocks_placed_avg
  avg_time_per_cube
  grasp_failure_rate
  collisions_avg          — number of EE proximity events
  recovery_success_rate

Results -> data/results/eval_<timestamp>.json.
"""

import os
import sys
import argparse
import yaml
import json
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["CARB_LOG_LEVEL"] = "error"


CONDITIONS = ["nominal", "block_displacement", "ee_disturbance",
              "arm_block", "combined"]


def run_one_trial(env, ds_set, seq, ik_kin, mod, franka, cfg,
                  condition, rng, perturbations,
                  use_modulation=True, use_safe=False,
                  done_tol=0.05, max_steps=8000, render=True,
                  stagger_steps=None, goal_gain=0.0, ds_scale=1.0,
                  max_joint_vel=None):
    """Run one full trial. Returns metrics dict."""
    from omni.isaac.core.utils.types import ArticulationAction
    from src.modulation import jacobian_finite_difference
    from src.primitives import gripper_action_for_primitive

    physics_dt = cfg["sim"]["physics_dt"]
    device = next(ds_set["reach"]["model"].parameters()).device
    if stagger_steps is None:
        stagger_steps = cfg["coordination"].get("start_stagger_steps", 0)
    arm_start_step = {"left": 0, "right": max(0, stagger_steps)}

    metrics = {
        "completed":      False,
        "blocks_placed":  0,
        "grasp_attempts": 0,
        "grasp_failures": 0,
        "collisions":     0,
        "timed_out":      False,
        "timeout_arm":    None,
        "timeout_primitive": None,
        "duration_s":     0.0,
    }

    # Per-step diagnostics for the modulation. List of dicts with keys:
    #   t, arm, gamma, v_cart_norm_nom, v_cart_norm_mod, radial_dot_nom,
    #   radial_dot_mod, ee_self, ee_other.
    # Skipped entirely when modulation is disabled.
    diagnostics_log = []

    # Schedule perturbation
    perturb_step  = None
    perturb_arm   = None
    perturb_block = None
    if condition != "nominal":
        perturb_step  = rng.integers(500, max_steps * 2 // 3)
        perturb_arm   = rng.choice(["left", "right"])
        all_blocks    = [b["name"] for b in cfg["left_blocks"]] + \
                        [b["name"] for b in cfg["right_blocks"]]
        perturb_block = rng.choice(all_blocks)

    # 30× the collection budget per primitive before we give up.
    prim_timeout = {p: s * 30
                    for p, s in cfg["sim"]["steps_per_primitive"].items()}
    prim_steps   = {"left": 0, "right": 0}
    max_joint_vel = cfg["training"]["max_joint_vel"] if max_joint_vel is None else max_joint_vel

    # Init q_goals
    def update_q_goal(arm):
        cart = seq.cartesian_target(arm)
        if cart is None:
            return
        q_seed  = franka[arm].get_joint_positions()[:7].copy()
        ee_quat = seq.ee_orientation(arm)
        q_goal, _ = ik_kin[arm].solve(cart, target_quat=ee_quat, q_seed=q_seed)
        seq.tasks[arm].q_goal = q_goal

    for arm in ("left", "right"):
        update_q_goal(arm)
    last_prim = {a: seq.tasks[a].current_primitive for a in ("left", "right")}

    for step in range(max_steps):
        ee_pos = {a: env.get_ee_pose(a)[0].copy() for a in ("left", "right")}
        if np.linalg.norm(ee_pos["left"] - ee_pos["right"]) < 0.08:
            metrics["collisions"] += 1

        # Apply perturbation at scheduled step
        if perturb_step is not None and step == perturb_step:
            if condition in ("block_displacement", "combined"):
                perturbations["block"].apply(env, perturb_block, rng=rng)
            if condition in ("ee_disturbance", "combined"):
                perturbations["ee"].apply(env, perturb_arm, rng=rng,
                                          physics_dt=physics_dt, render=render)
            if condition in ("arm_block", "combined"):
                perturbations["block_arm"].apply(
                    coordinator=seq, arm=perturb_arm, env=env,
                    update_q_goal_fn=update_q_goal,
                    franka=franka[perturb_arm],
                    physics_dt=physics_dt, render=render,
                )

        # Compute nominal q̇
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
            x_t = torch.tensor(x_n, dtype=torch.float32,
                               device=device).unsqueeze(0)
            if use_safe:
                scale_factor = torch.tensor(
                    ds["vel_scale"] / ds["state_std"],
                    dtype=torch.float32, device=device).unsqueeze(0)
                qd_n = ds["model"].safe_velocity(x_t, scale_factor=scale_factor)
            else:
                with torch.no_grad():
                    qd_n = ds["model"](x_t)
            q_dots[arm] = ds_scale * qd_n.cpu().numpy().squeeze(0) * ds["vel_scale"]
            if goal_gain > 0:
                q_dots[arm] = q_dots[arm] - goal_gain * x
            q_dots[arm] = np.clip(q_dots[arm], -max_joint_vel, max_joint_vel)

        # Modulation
        if use_modulation:
            for arm, other in (("left", "right"), ("right", "left")):
                if q_dots[arm] is None:
                    continue
                J = jacobian_finite_difference(franka[arm])

                # Log diagnostics BEFORE applying modulation so we capture
                # both nominal and modulated quantities at the same x.
                diag = mod.diagnostics(
                    q_dot_nominal=q_dots[arm],
                    ee_pos_self=ee_pos[arm],
                    ee_pos_other=ee_pos[other],
                    jacobian=J,
                )
                diag["t"] = step * physics_dt
                diag["arm"] = arm
                diag["ee_self"]  = ee_pos[arm].tolist()
                diag["ee_other"] = ee_pos[other].tolist()
                diagnostics_log.append(diag)

                q_dots[arm] = mod.modulate_joint_velocity(
                    q_dot_nominal=q_dots[arm],
                    ee_pos_self=ee_pos[arm],
                    ee_pos_other=ee_pos[other],
                    jacobian=J,
                )

        # Apply
        for arm in ("left", "right"):
            if q_dots[arm] is None:
                continue
            q = franka[arm].get_joint_positions()[:7].copy()
            full = franka[arm].get_joint_positions().copy()
            full[:7] = q + q_dots[arm] * physics_dt
            franka[arm].apply_action(ArticulationAction(joint_positions=full))

        env.step(render=render)

        # Primitive completion
        trial_failed = False
        for arm in ("left", "right"):
            task = seq.tasks[arm]
            if task.is_done():
                continue
            q = franka[arm].get_joint_positions()[:7]
            timed_out = prim_steps[arm] >= prim_timeout[task.current_primitive]
            converged = np.linalg.norm(q - task.q_goal) < done_tol
            if timed_out and not converged:
                metrics["timed_out"] = True
                metrics["timeout_arm"] = arm
                metrics["timeout_primitive"] = task.current_primitive
                trial_failed = True
                break
            if converged:
                grip = gripper_action_for_primitive(task.current_primitive)
                if task.current_primitive == "grasp":
                    metrics["grasp_attempts"] += 1
                if grip == "close":
                    franka[arm].gripper.apply_action(
                        ArticulationAction(joint_positions=np.array([0.0, 0.0])))
                    for _ in range(cfg["sim"]["gripper_steps"]):
                        env.step(render=render)
                elif grip == "open":
                    franka[arm].gripper.apply_action(
                        ArticulationAction(joint_positions=np.array([0.04, 0.04])))
                    for _ in range(cfg["sim"]["gripper_steps"]):
                        env.step(render=render)
                if task.current_primitive == "place":
                    metrics["blocks_placed"] += 1
                if task.current_primitive == "lift":
                    block_name = task.current_block
                    bp = env.get_block_positions()[block_name]
                    if bp[2] < cfg["table"]["height"] + cfg["block"]["size"]:
                        metrics["grasp_failures"] += 1
                seq.primitive_complete(arm)
                prim_steps[arm] = 0

        if trial_failed:
            break

        if all(seq.tasks[a].is_done() for a in ("left", "right")):
            metrics["completed"] = True
            break

    metrics["duration_s"] = (step + 1) * physics_dt
    return metrics, diagnostics_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt_arm", type=str, default="both")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--conditions", type=str, nargs="+", default=CONDITIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_modulation", action="store_true")
    parser.add_argument("--use_safe", action="store_true")
    parser.add_argument("--goal_gain", type=float, default=0.0,
                        help="Add q_goal attraction term -gain*(q-q_goal) to "
                             "the learned DS during evaluation.")
    parser.add_argument("--ds_scale", type=float, default=1.0,
                        help="Scale learned DS velocity. Use 0 with "
                             "--goal_gain for a pure joint-space attractor "
                             "sanity check.")
    parser.add_argument("--max_joint_vel", type=float, default=None,
                        help="Deployment/evaluation joint velocity clamp in "
                             "rad/s. Defaults to training.max_joint_vel.")
    parser.add_argument("--stagger_steps", type=int, default=None,
                        help="Initial right-arm launch delay in physics steps. "
                             "Defaults to coordination.start_stagger_steps.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--done_tol", type=float, default=0.05)
    args = parser.parse_args()

    from isaacsim import SimulationApp
    _app_cfg = {"headless": args.headless}
    if not args.headless:
        _app_cfg.update({"width": 1280, "height": 720})
    simulation_app = SimulationApp(_app_cfg)

    from src.env import DualArmEnv
    from src.coordinator import TaskSequencer
    from src.franka_ik import FrankaIK
    from src.modulation import InterArmModulation
    from src.perturbations import BlockDisplacement, EEDisturbance, ArmBlock
    from scripts.deploy_dual_arm import load_ds_set

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ds_set = load_ds_set(ckpt_dir, args.ckpt_arm, device)

    perturbations = {
        "block": BlockDisplacement(
            max_offset=cfg["perturbations"]["block_displacement"]["max_offset"]),
        "ee":    EEDisturbance(
            max_force=cfg["perturbations"]["ee_disturbance"]["max_force"],
            duration_s=cfg["perturbations"]["ee_disturbance"]["duration"]),
        "block_arm": ArmBlock(
            freeze_duration_s=cfg["perturbations"]["arm_block"]["freeze_duration"]),
    }

    rng = np.random.default_rng(args.seed)

    env = DualArmEnv(config_path=args.config, arms=("left", "right"))
    franka = {"left": env.frankas["left"], "right": env.frankas["right"]}
    ik_kin = {a: FrankaIK(franka[a]) for a in franka}
    mod = InterArmModulation(
        safe_radius=cfg["coordination"]["ee_safety_radius"],
        reactivity=4.0,
    )

    results = {cond: [] for cond in args.conditions}
    diag_log = {cond: [] for cond in args.conditions}

    for cond in args.conditions:
        print(f"\n=== Condition: {cond} ===")
        for trial in range(args.n_trials):
            env.reset_blocks(render=not args.headless)
            seq = TaskSequencer(env, cfg)
            print(f"  trial {trial + 1}/{args.n_trials}")
            m, diag = run_one_trial(env, ds_set, seq, ik_kin, mod, franka, cfg,
                                    cond, rng, perturbations,
                                    use_modulation=not args.no_modulation,
                                    use_safe=args.use_safe,
                                    done_tol=args.done_tol,
                                    stagger_steps=args.stagger_steps,
                                    goal_gain=args.goal_gain,
                                    ds_scale=args.ds_scale,
                                    max_joint_vel=args.max_joint_vel,
                                    render=not args.headless)
            results[cond].append(m)
            diag_log[cond].append(diag)

    summary = {}
    for cond, trials in results.items():
        n = len(trials)
        placed = sum(t["blocks_placed"] for t in trials)
        attempts = max(sum(t["grasp_attempts"] for t in trials), 1)
        failures = sum(t["grasp_failures"] for t in trials)
        time_total = sum(t["duration_s"] for t in trials)
        summary[cond] = {
            "stack_completion_rate":  sum(t["completed"] for t in trials) / max(n, 1),
            "blocks_placed_avg":      placed / max(n, 1),
            "avg_time_per_cube":      time_total / max(placed, 1),
            "grasp_failure_rate":     failures / attempts,
            "collisions_avg":         sum(t["collisions"] for t in trials) / max(n, 1),
            "recovery_success_rate":  sum(t["completed"] for t in trials) / max(n, 1)
                                       if cond != "nominal" else None,
        }
        print(f"\n[{cond}] {summary[cond]}")

    out_dir = Path(cfg["paths"]["results"])
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    out_path = out_dir / f"eval_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary,
                   "config":  {"no_modulation": args.no_modulation,
                               "use_safe":      args.use_safe,
                               "stagger_steps": args.stagger_steps
                                                if args.stagger_steps is not None
                                                else cfg["coordination"].get(
                                                    "start_stagger_steps", 0),
                               "goal_gain":     args.goal_gain,
                               "ds_scale":      args.ds_scale,
                               "max_joint_vel": args.max_joint_vel
                                                if args.max_joint_vel is not None
                                                else cfg["training"]["max_joint_vel"]},
                   "trials":  results},
                  f, indent=2, default=str)
    print(f"\n[EVAL] Saved results to {out_path}")

    # Diagnostics (potentially large) saved as pickle for plotting
    if not args.no_modulation:
        import pickle
        diag_path = out_dir / f"diag_{timestamp}.pkl"
        with open(diag_path, "wb") as f:
            pickle.dump(diag_log, f)
        print(f"[EVAL] Saved diagnostics to {diag_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
