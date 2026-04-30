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
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--use_safe", action="store_true",
                        help="Apply Lyapunov projection at inference")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--done_tol", type=float, default=0.05,
                        help="L2 tolerance in joint space to declare primitive done.")
    args = parser.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless,
                                    "width": 1280, "height": 720})

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

    seq = TaskSequencer(env, cfg)
    physics_dt = cfg["sim"]["physics_dt"]

    # Compute initial q_goal for the first primitive
    def update_q_goal(arm):
        cart = seq.cartesian_target(arm)
        if cart is None:
            return None
        q_seed = franka.get_joint_positions()[:7].copy()
        ee_quat = seq.ee_orientation(arm)
        q_goal, _ = ik_kin.solve(cart, target_quat=ee_quat, q_seed=q_seed)
        seq.tasks[arm].q_goal = q_goal
        return q_goal

    update_q_goal(args.arm)

    print(f"[DEPLOY] Joint-space DS on {args.arm} arm — safe={args.use_safe}")

    last_primitive = seq.tasks[args.arm].current_primitive
    prim_steps = 0
    # 4× the collection budget per primitive before we give up and advance
    prim_timeout = {p: s * 4
                    for p, s in cfg["sim"]["steps_per_primitive"].items()}

    franka.gripper.apply_action(
        ArticulationAction(joint_positions=np.array([0.04, 0.04]))
    )

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
            last_primitive = task.current_primitive
            prim_steps = 0

        prim_steps += 1

        q      = franka.get_joint_positions()[:7].copy()
        q_goal = task.q_goal

        # Build state & query DS
        ds  = ds_set[task.current_primitive]
        x   = np.concatenate([q, q_goal])
        x_n = (x - ds["state_mean"]) / ds["state_std"]
        x_t = torch.tensor(x_n, dtype=torch.float32, device=device).unsqueeze(0)
        if args.use_safe:
            qd_n = ds["model"].safe_velocity(x_t)
        else:
            with torch.no_grad():
                qd_n = ds["model"](x_t)
        q_dot = qd_n.cpu().numpy().squeeze(0) * ds["vel_scale"]
        q_dot = np.clip(q_dot, -cfg["training"]["max_joint_vel"],
                                cfg["training"]["max_joint_vel"])

        # Integrate to get joint position command
        q_cmd = q + q_dot * physics_dt
        full_cmd = franka.get_joint_positions().copy()
        full_cmd[:7] = q_cmd
        franka.apply_action(ArticulationAction(joint_positions=full_cmd))

        env.step(render=not args.headless)

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
            elif grip == "open":
                franka.gripper.apply_action(
                    ArticulationAction(joint_positions=np.array([0.04, 0.04]))
                )
                for _ in range(cfg["sim"]["gripper_steps"]):
                    env.step(render=not args.headless)
            seq.primitive_complete(args.arm)
            prim_steps = 0

    print(f"[DEPLOY] Finished after {step + 1} steps.")
    simulation_app.close()


if __name__ == "__main__":
    main()
