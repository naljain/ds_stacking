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
                        help="EE dist (m) to declare transport done")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for deploy-time block randomization")
    parser.add_argument("--no_randomize_blocks", action="store_true",
                        help="Use the scene's initial block positions")
    parser.add_argument("--lookahead", type=int,   default=5,
                        help="IK target = ee_pos + x_dot * lookahead * dt")
    parser.add_argument("--max_cart_speed", type=float, default=0.25,
                        help="Clip LPVDS Cartesian speed before IK retargeting")
    parser.add_argument("--cart_gain", type=float, default=1.0,
                        help="Scale LPVDS Cartesian velocity before speed clipping")
    parser.add_argument("--raw_lpvds", action="store_true",
                        help="Use raw LPVDS velocity without stability projection")
    parser.add_argument("--no_workspace_clamp", action="store_true",
                        help="Do not clamp LPVDS IK targets to the transport workspace")
    parser.add_argument("--z_margin", type=float, default=0.12,
                        help="LPVDS target z clamp around lift height when workspace clamp is enabled")
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

    if not args.no_randomize_blocks:
        rng = np.random.default_rng(args.seed)
        env.reset_blocks(render=not args.headless, rng=rng)
        print("[DEPLOY] Randomized block positions")

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
    ws = cfg["block_workspace"][args.arm]
    lpv_min = np.array([
        min(ws["x_min"], goal_xy[0]) - 0.05,
        min(ws["y_min"], goal_xy[1]) - 0.05,
        lift_h - args.z_margin,
    ])
    lpv_max = np.array([
        max(ws["x_max"], goal_xy[0]) + 0.05,
        max(ws["y_max"], goal_xy[1]) + 0.05,
        lift_h + args.z_margin,
    ])

    # Compute transport q_goal once — it never changes (fixed target)
    q_goal, ok = ik_kin.solve(transport_pos, target_quat=ee_down,
                               q_seed=default_joints)
    if not ok:
        print("[WARN] IK failed for transport target, using seed")

    goal_z = cfg["table"]["height"] + block_h / 2

    print(f"[DEPLOY] Single-arm transport DS — arm={args.arm}  safe={args.use_safe}")
    print(f"[DEPLOY] IK frame={ik_motion.ik.ee_frame}")
    if args.model == "lpvds":
        print(f"[DEPLOY] LPVDS goal={np.round(lpv_model.x_goal, 4)}  "
              f"config transport={np.round(transport_pos, 4)}  "
              f"lookahead={args.lookahead}")
        if not args.no_workspace_clamp:
            print(f"[DEPLOY] LPVDS target clamp min={np.round(lpv_min, 3)} "
                  f"max={np.round(lpv_max, 3)}")

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

        # ── 4. Transport — DS ─────────────────────────────────────────────
        # Compute J once here: arm is in a stable post-lift pose and changes
        # slowly during transport, so one J per block is sufficient.
        # Computing J inside the loop disturbs the sim (7 FK perturbations).
        print(f"  [DEPLOY] Transport DS running...")
        ik_fail_streak = 0
        for step in range(args.max_steps):
            if not simulation_app.is_running():
                break

            q = franka.get_joint_positions()[:7].copy()

            # Check convergence
            if args.model == "lpvds":
                ee_pos_c, _ = ik_motion.ik.get_world_pose()
                done = np.linalg.norm(np.array(ee_pos_c) - lpv_model.x_goal) < args.done_tol
            else:
                done = np.linalg.norm(q - q_goal) < args.done_tol
            if done:
                print(f"  [DEPLOY] Transport done in {step} steps")
                break

            # Query DS and command robot
            if args.model == "lpvds":
                ee_pos, _ = ik_motion.ik.get_world_pose()
                ee_pos    = np.array(ee_pos)
                if args.raw_lpvds:
                    x_dot = lpv_model.predict(ee_pos)                # (3,) m/s
                else:
                    x_dot = lpv_model.safe_velocity(ee_pos)          # (3,) m/s
                x_dot = args.cart_gain * x_dot
                speed = np.linalg.norm(x_dot)
                if speed > args.max_cart_speed:
                    x_dot = x_dot * (args.max_cart_speed / speed)
                ee_target = ee_pos + x_dot * args.lookahead * physics_dt
                if not args.no_workspace_clamp:
                    ee_target = np.clip(ee_target, lpv_min, lpv_max)
                # Use ik_motion.step_to() instead of raw ik_kin.solve():
                #   - warm-starts from _q_last (previous solution) → same branch
                #   - handles ok=False by holding last good solution
                #   - applies the full 9-DOF command internally
                # DO NOT call franka.apply_action after this — step_to does it.
                ik_ok = ik_motion.step_to(ee_target, target_quat=ee_down)
                ik_fail_streak = 0 if ik_ok else ik_fail_streak + 1
                if step % 50 == 0:
                    dist = np.linalg.norm(ee_pos - lpv_model.x_goal)
                    print(f"    [DBG] step={step:>4} ee={np.round(ee_pos[:2],3)} "
                          f"z={ee_pos[2]:.3f} xdot={np.round(x_dot,3)} "
                          f"target={np.round(ee_target,3)} "
                          f"dist={dist:.3f}m ik_ok={ik_ok}")
                if ik_fail_streak == 25:
                    print("    [WARN] LPVDS IK has failed for 25 consecutive steps; "
                          "holding the last valid joint target.")
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
                q_dot  = qd_n.cpu().numpy().squeeze(0) * ds["vel_scale"]
                finger   = ik_motion._finger_width
                q_cmd    = q + q_dot * physics_dt
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
