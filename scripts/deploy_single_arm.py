"""
Deploy transport-only Neural DS on a single arm.

Workflow per block:
  1. reach     — IK straight-line to hover above block
  2. grasp     — IK straight-line descend, close gripper
  3. lift      — IK straight-line raise to transport height
  4. transport — Neural DS drives q -> q_goal (above shared stack)  ← learned
  5. place     — IK straight-line descend, open gripper

Block positions are read from the sim at runtime (known perfectly).
The DS q_goal is computed once via Lula IK at the transport target.

Usage:
  python scripts/deploy_single_arm.py --arm left
  python scripts/deploy_single_arm.py --arm left --use_safe
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
    cfg  = ckpt["config"]
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
    parser.add_argument("--arm",       type=str, default="left", choices=["left", "right"])
    parser.add_argument("--config",    type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt_arm",  type=str, default=None,
                        help="Checkpoint arm label (default: same as --arm)")
    parser.add_argument("--max_steps", type=int, default=2000,
                        help="Max DS steps per transport primitive")
    parser.add_argument("--use_safe",  action="store_true",
                        help="Apply Lyapunov projection at inference (neural DS only)")
    parser.add_argument("--model",     type=str, default="neural",
                        choices=["neural", "lpvds"],
                        help="DS model type: neural (default) or lpvds")
    parser.add_argument("--headless",  action="store_true")
    parser.add_argument("--done_tol",  type=float, default=0.05,
                        help="EE distance (m) or joint L2 to declare transport done")
    args = parser.parse_args()

    if args.ckpt_arm is None:
        args.ckpt_arm = args.arm

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless,
                                    "width": 1280, "height": 720})

    from src.env import DualArmEnv
    from src.ik_controller import IKController
    from src.franka_ik import FrankaIK

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env    = DualArmEnv(config_path=args.config, arms=(args.arm,))
    cfg    = env.cfg
    franka = env.frankas[args.arm]

    default_joints = np.array(cfg["arms"][f"default_joints_{args.arm}"])
    ik_motion = IKController(franka, arm=args.arm, rest_q=default_joints)
    ik_kin    = FrankaIK(franka)

    # Load transport DS — neural or LPVDS
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    if args.model == "lpvds":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.lpv_ds import LPVDS
        lpv_model = LPVDS.load(ckpt_dir / f"{args.ckpt_arm}_transport_lpvds.pkl")
        ds = None  # not used
    else:
        ds = load_ds(ckpt_dir / f"{args.ckpt_arm}_transport.pt", device)
        lpv_model = None

    physics_dt = cfg["sim"]["physics_dt"]
    steps      = cfg["sim"]["steps_per_primitive"]
    hover_h    = cfg["heights"]["hover"]
    lift_h     = cfg["heights"]["lift"]
    grasp_h    = cfg["heights"]["grasp"]
    block_h    = cfg["block"]["size"]
    goal_xy    = tuple(cfg["shared_goal"])
    block_names = [b["name"] for b in cfg[f"{args.arm}_blocks"]]

    ee_down = np.array([0.0, 1.0, 0.0, 0.0])   # w,x,y,z straight down

    # Pre-place target: fixed position above the stack
    transport_pos = np.array([goal_xy[0], goal_xy[1], lift_h])

    # Compute transport q_goal once — it never changes (fixed target)
    q_goal, ok = ik_kin.solve(transport_pos, target_quat=ee_down,
                               q_seed=default_joints)
    if not ok:
        print("[WARN] IK failed for transport target, using seed")

    goal_z = cfg["table"]["height"] + block_h / 2

    print(f"[DEPLOY] Single-arm transport DS — arm={args.arm}  safe={args.use_safe}")

    def _articulation_action(positions):
        try:
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            from omni.isaac.core.utils.types import ArticulationAction
        return ArticulationAction(joint_positions=positions)

    for block_name in block_names:
        block_pos = env.get_block_positions()[block_name].copy()
        bx, by = block_pos[0], block_pos[1]
        print(f"  [DEPLOY] Block {block_name} at ({bx:.3f}, {by:.3f})")

        # ── 1. Reach ──────────────────────────────────────────────────────
        ee_grasp = env.get_block_grasp_quat(block_name)
        ik_motion.move_to(env.world, np.array([bx, by, hover_h]),
                          target_quat=ee_grasp,
                          steps=steps["reach"], render=not args.headless)

        # ── 2. Grasp ──────────────────────────────────────────────────────
        ik_motion.move_to(env.world, np.array([bx, by, grasp_h]),
                          target_quat=ee_grasp,
                          steps=steps["grasp"], render=not args.headless)
        ik_motion.set_gripper(open=False)
        for _ in range(cfg["sim"]["gripper_steps"]):
            env.world.step(render=not args.headless)

        # ── 3. Lift ───────────────────────────────────────────────────────
        ik_motion.move_to(env.world, np.array([bx, by, lift_h]),
                          target_quat=ee_down,
                          steps=steps["lift"], render=not args.headless)

        # ── 4. Transport — Neural DS ──────────────────────────────────────
        print(f"  [DEPLOY] Transport DS running...")
        for step in range(args.max_steps):
            if not simulation_app.is_running():
                break

            q = franka.get_joint_positions()[:7].copy()

            # Check convergence
            if args.model == "lpvds":
                ee_pos, _ = franka.end_effector.get_world_pose()
                conv_dist = np.linalg.norm(np.array(ee_pos) - lpv_model.x_goal)
                done = conv_dist < args.done_tol
            else:
                done = np.linalg.norm(q - q_goal) < args.done_tol
            if done:
                print(f"  [DEPLOY] Transport done in {step} steps")
                break

            # Query DS
            if args.model == "lpvds":
                # Cartesian DS: integrate desired EE velocity one step forward,
                # then solve IK for that target position. This avoids the
                # finite-difference Jacobian which perturbs sim state.
                ee_pos, _ = franka.end_effector.get_world_pose()
                ee_pos    = np.array(ee_pos)
                x_dot_des = lpv_model.predict(ee_pos)              # (3,) m/s
                ee_next   = ee_pos + x_dot_des * physics_dt        # (3,) target
                q_next, ok = ik_kin.solve(ee_next, target_quat=ee_down,
                                          q_seed=q)
                if not ok:
                    q_next = q  # hold if IK fails
                q_dot = (q_next - q) / physics_dt                  # (7,)
            else:
                x   = np.concatenate([q, q_goal])
                x_n = (x - ds["state_mean"]) / ds["state_std"]
                x_t = torch.tensor(x_n, dtype=torch.float32,
                                   device=device).unsqueeze(0)
                if args.use_safe:
                    qd_n = ds["model"].safe_velocity(x_t)
                else:
                    with torch.no_grad():
                        qd_n = ds["model"](x_t)
                q_dot = qd_n.cpu().numpy().squeeze(0) * ds["vel_scale"]

            # Integrate and command all 9 joints at once
            q_cmd = q + q_dot * physics_dt
            finger = ik_motion._finger_width
            full_cmd = np.concatenate([q_cmd, [finger, finger]])
            franka.apply_action(_articulation_action(full_cmd))
            env.step(render=not args.headless)
        else:
            print(f"  [WARN] Transport hit max_steps ({args.max_steps})")

        # ── 5. Place ──────────────────────────────────────────────────────
        place_pos = np.array([goal_xy[0], goal_xy[1], goal_z + 0.02])
        ik_motion.move_to(env.world, place_pos, target_quat=ee_down,
                          steps=steps["place"], render=not args.headless)
        ik_motion.set_gripper(open=True)
        for _ in range(cfg["sim"]["gripper_steps"]):
            env.world.step(render=not args.headless)

        # Retract before next block
        ik_motion.move_to(env.world, transport_pos, target_quat=ee_down,
                          steps=60, render=not args.headless)

        goal_z += block_h + 0.002

    print("[DEPLOY] All blocks placed.")
    simulation_app.close()


if __name__ == "__main__":
    main()
