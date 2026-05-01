"""
Deploy joint-space Neural DS on a single arm.

The DS produces q_dot directly. We integrate q_dot to get a target joint
position, send that as the command, and let Isaac Sim's articulation
controller handle the low-level torque tracking. There is no IK at runtime —
the closed-loop joint dynamics ARE the trained DS (modulo actuator dynamics).

When a primitive transitions, we call Lula IK once to compute q* for the
new Cartesian target, then keep that q* fixed until the next transition.

Usage:
  python scripts/deploy_single_arm.py --arm left
  python scripts/deploy_single_arm.py --arm left --use_safe   # Lyapunov projection
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


def load_ds(ckpt_path, device):
    from src.neural_ds import StableNeuralDS, N_JOINTS
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = StableNeuralDS(
        n_joints    = N_JOINTS,
        hidden_dim  = cfg["hidden_dim"],
        lyap_hidden = cfg["lyapunov_hidden"],
        alpha       = cfg["alpha"],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt_arm", type=str, default="both")
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--use_safe", action="store_true",
                        help="Apply Lyapunov projection at inference")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override Lyapunov decay rate at deployment "
                             "(higher = more aggressive projection / faster convergence).")
    parser.add_argument("--goal_gain", type=float, default=0.0,
                        help="Add q_goal attraction term -gain*(q-q_goal) to "
                             "the learned DS at deployment. Use this as a "
                             "stabilizing ablation if the learned field points "
                             "away from the attractor.")
    parser.add_argument("--ds_scale", type=float, default=1.0,
                        help="Scale learned DS velocity. Use 0 with "
                             "--goal_gain for a pure joint-space attractor "
                             "sanity check.")
    parser.add_argument("--max_joint_vel", type=float, default=None,
                        help="Deployment joint velocity clamp in rad/s. "
                             "Defaults to training.max_joint_vel from config.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--done_tol", type=float, default=0.05,
                        help="L2 tolerance in joint space to declare primitive done.")
    parser.add_argument("--log_csv", type=str, default=None,
                        help="If set, write per-step diagnostics to this CSV "
                             "for post-mortem plotting.")
    parser.add_argument("--print_every", type=int, default=50,
                        help="Print diagnostic line every N steps (0 = off).")
    parser.add_argument("--debug_ik", action="store_true",
                        help="Print IK target, success flag, and resulting "
                             "joint goal at each primitive transition.")
    parser.add_argument("--kinematic_carry", action="store_true",
                        help="After grasp, attach the active block to the EE "
                             "kinematically until place. Use this to debug the "
                             "DS/task pipeline separately from gripper contact.")
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
    from src.primitives import gripper_action_for_primitive

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DualArmEnv(config_path=args.config, arms=(args.arm,))
    cfg = env.cfg
    franka = env.frankas[args.arm]
    ik_kin = FrankaIK(franka)

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    primitives = ["reach", "grasp", "lift", "transport", "place"]
    ds_set = {p: load_ds(ckpt_dir / f"{args.ckpt_arm}_{p}.pt", device)
              for p in primitives}

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
        cart = seq.cartesian_target(arm)
        if cart is None:
            return None
        q_seed = franka.get_joint_positions()[:7].copy()
        ee_quat = seq.ee_orientation(arm)
        q_goal, ok = ik_kin.solve(cart, target_quat=ee_quat, q_seed=q_seed)
        if args.debug_ik:
            print(f"[IK] {seq.tasks[arm].current_primitive:9s} ok={ok} "
                  f"cart={cart.round(3)} seed={q_seed.round(3)} "
                  f"q_goal={q_goal.round(3)} "
                  f"||seed-goal||={np.linalg.norm(q_seed - q_goal):.3f}")
        seq.tasks[arm].q_goal = q_goal
        return q_goal

    update_q_goal(args.arm)

    print(f"[DEPLOY] Joint-space DS on {args.arm} arm — safe={args.use_safe}, "
          f"goal_gain={args.goal_gain}, ds_scale={args.ds_scale}, "
          f"max_joint_vel={max_joint_vel}")

    last_primitive = seq.tasks[args.arm].current_primitive
    prim_steps = 0
    # 30× the collection budget — very generous so a slow-converging DS
    # has plenty of room before we give up and advance.
    prim_timeout = {p: s * 30
                    for p, s in cfg["sim"]["steps_per_primitive"].items()}

    franka.gripper.apply_action(
        ArticulationAction(joint_positions=np.array([0.04, 0.04]))
    )

    # Let blocks settle before querying their positions
    for _ in range(60):
        env.step(render=not args.headless)

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
        ee_pos = env.get_ee_pose(args.arm)[0].copy()
        obj = env.get_block_obj(held_block)
        obj.set_world_pose(position=ee_pos + held_offset,
                           orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))

    def snap_held_block_to_stack():
        if held_block is None:
            return
        obj = env.get_block_obj(held_block)
        obj.set_world_pose(position=seq.stack_target_position(args.arm),
                           orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))

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
        q_dot     = args.ds_scale * qd_n.cpu().numpy().squeeze(0) * ds["vel_scale"]
        if args.goal_gain > 0:
            q_dot = q_dot - args.goal_gain * x
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

        proj_correction = float(np.linalg.norm(q_dot - q_dot_raw))

        if args.print_every and step % args.print_every == 0:
            q_cmd_preview = q + q_dot_clipped * physics_dt
            print(f"  step {step:5d} | {task.current_primitive:9s} | "
                  f"||e||={e_norm:.3f}  V={V_val:7.3f}  "
                  f"||qd_raw||={np.linalg.norm(q_dot_raw):.2f}  "
                  f"||qd||={qd_norm:.2f}  "
                  f"proj_Δ={proj_correction:.2f}  "
                  f"cos→goal={cos_to_goal:+.2f}  "
                  f"max|Δqcmd|={np.max(np.abs(q_cmd_preview - q)):.4f}")

        if args.log_csv is not None:
            csv_log.write(
                f"{step},{task.current_primitive},{prim_steps},"
                f"{e_norm:.5f},{V_val:.5f},"
                f"{np.linalg.norm(q_dot_raw):.5f},"
                f"{qd_norm:.5f},{proj_correction:.5f},"
                f"{cos_to_goal:.5f}\n"
            )

        q_dot = q_dot_clipped

        # Integrate to get joint position command
        q_cmd = q + q_dot * physics_dt
        full_cmd = franka.get_joint_positions().copy()
        full_cmd[:7] = q_cmd
        franka.apply_action(ArticulationAction(joint_positions=full_cmd))

        env.step(render=not args.headless)
        carry_held_block()

        # Primitive completion: joint-space convergence OR per-primitive timeout
        timed_out = prim_steps >= prim_timeout[task.current_primitive]
        converged = np.linalg.norm(q - q_goal) < args.done_tol
        if converged or timed_out:
            if timed_out and not converged:
                print(f"[WARN] {task.current_primitive} timed out after "
                      f"{prim_steps} steps (||q-q*||="
                      f"{np.linalg.norm(q - q_goal):.3f})")
            grip = gripper_action_for_primitive(task.current_primitive)
            if grip == "close":
                franka.gripper.apply_action(
                    ArticulationAction(joint_positions=np.array([0.0, 0.0]))
                )
                for _ in range(cfg["sim"]["gripper_steps"]):
                    env.step(render=not args.headless)
                if args.kinematic_carry:
                    ee_pos = env.get_ee_pose(args.arm)[0].copy()
                    block_pos = env.get_block_positions()[task.current_block].copy()
                    held_block = task.current_block
                    held_offset = block_pos - ee_pos
                    carry_held_block()
            elif grip == "open":
                if args.kinematic_carry:
                    snap_held_block_to_stack()
                held_block = None
                franka.gripper.apply_action(
                    ArticulationAction(joint_positions=np.array([0.04, 0.04]))
                )
                for _ in range(cfg["sim"]["gripper_steps"]):
                    env.step(render=not args.headless)
            seq.primitive_complete(args.arm)
            prim_steps = 0

    print(f"[DEPLOY] Finished after {step + 1} steps.")
    if csv_log is not None:
        csv_log.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
