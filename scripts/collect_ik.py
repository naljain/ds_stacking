"""
Trajectory collection for transport-only Neural DS training.

Workflow per demo:
  1. reach   — move above block (RMPflow/IK, not recorded)
  2. grasp   — descend and close gripper (not recorded)
  3. lift    — raise to transport height (not recorded)
  4. transport — move to fixed position above shared stack  ← RECORDED
  5. place   — descend and release (not recorded)

Only the transport segment is saved. The DS learns to map
[q, q_goal] -> q_dot for the transport primitive only.

Usage:
  python scripts/collect_ik.py --arm left  --n_demos 50
  python scripts/collect_ik.py --arm right --n_demos 50
"""

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["CARB_LOG_LEVEL"] = "error"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",      type=str, default="left", choices=["left", "right"])
    parser.add_argument("--n_demos",  type=int, default=50)
    parser.add_argument("--config",   type=str, default="configs/default.yaml")
    parser.add_argument("--headless", action="store_true")

    args = parser.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless,
                                    "width": 1280, "height": 720})

    from src.env import DualArmEnv
    from src.ik_controller import IKController
    from src.franka_ik import FrankaIK

    env = DualArmEnv(config_path=args.config, arms=(args.arm,))
    cfg = env.cfg
    franka = env.frankas[args.arm]

    # Default joint pose from config (raw radians) — IK seed and reset pose
    default_joints = np.array(cfg["arms"][f"default_joints_{args.arm}"])
    ik_motion = IKController(franka, arm=args.arm, rest_q=default_joints)
    ik_kin    = FrankaIK(franka)

    block_names = [b["name"] for b in cfg[f"{args.arm}_blocks"]]
    goal_xy     = tuple(cfg["shared_goal"])

    hover_h    = cfg["heights"]["hover"]
    lift_h     = cfg["heights"]["lift"]
    grasp_h    = cfg["heights"]["grasp"]
    base_z     = cfg["table"]["height"] + cfg["block"]["size"] / 2
    physics_dt = cfg["sim"]["physics_dt"]
    steps      = cfg["sim"]["steps_per_primitive"]

    out_dir = Path(cfg["paths"]["demos"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arm}_demos.pkl"

    # EE orientation for transport/lift/place — straight down, no yaw
    ee_down = np.array([0.0, 1.0, 0.0, 0.0])   # w,x,y,z

    rng = np.random.default_rng(seed=42)
    all_demos = []

    print(f"\n[INFO] Collecting {args.n_demos} demos ({args.arm} arm) — transport only.")

    for demo_idx in range(args.n_demos):
        print(f"  Demo {demo_idx + 1}/{args.n_demos}")
        env.reset_arms(render=not args.headless)
        env.reset_blocks(render=not args.headless, rng=rng)
        ik_motion.reset()
        ik_motion.set_gripper(open=True)

        demo_traj = []
        goal_z  = base_z   # reset stack height each demo

        for block_name in block_names:
            # Block position is known perfectly — no noise
            block_pos = env.get_block_positions()[block_name].copy()
            bx, by = block_pos[0], block_pos[1]

            # ── 1. Reach: hover above block, EE aligned to block face ──
            ee_grasp = env.get_block_grasp_quat(block_name)
            ik_motion.move_to(env.world, np.array([bx, by, hover_h]),
                              target_quat=ee_grasp,
                              steps=steps["reach"], render=not args.headless)

            # ── 2. Grasp: descend to exact block centre ───────────────────
            ik_motion.move_to(env.world, np.array([bx, by, grasp_h]),
                              target_quat=ee_grasp,
                              steps=steps["grasp"], render=not args.headless)
            ik_motion.set_gripper(open=False)
            for _ in range(cfg["sim"]["gripper_steps"]):
                env.world.step(render=not args.headless)

            # ── 3. Lift straight up to transport height ───────────────────
            ik_motion.move_to(env.world, np.array([bx, by, lift_h]),
                              target_quat=ee_down,
                              steps=steps["lift"], render=not args.headless)

            # ── 4. Transport: move to above stack  ← RECORD THIS ─────────
            transport_pos = np.array([goal_xy[0], goal_xy[1], lift_h])

            # Compute q_goal once via IK — fixed seed keeps solution branch
            q_goal, ok = ik_kin.solve(transport_pos, target_quat=ee_down,
                                      q_seed=default_joints)
            if not ok:
                print(f"    [WARN] IK failed for transport block={block_name}, "
                      f"using seed as q_goal")

            prev_q  = franka.get_joint_positions()[:7].copy()
            prev_ee = None   # for EE velocity finite difference

            def record():
                nonlocal prev_q, prev_ee
                q      = franka.get_joint_positions()[:7].copy()
                q_dot  = (q - prev_q) / physics_dt
                ee_pos = ik_kin.get_world_pose()[0].copy()
                ee_vel = (ee_pos - prev_ee) / physics_dt                          if prev_ee is not None else np.zeros(3)
                demo_traj.append({
                    "q":         q,
                    "q_dot":     q_dot,
                    "q_goal":    q_goal.copy(),
                    "ee_pos":    ee_pos,
                    "ee_vel":    ee_vel,
                    "physics_dt": physics_dt,
                    "primitive": "transport",
                    "block":     block_name,
                    "arm":       args.arm,
                    "target":    transport_pos.copy(),
                })
                prev_q  = q
                prev_ee = ee_pos

            ik_motion.move_to(env.world, transport_pos, target_quat=ee_down,
                              steps=steps["transport"],
                              record_callback=record,
                              render=not args.headless)

            # ── 5. Place: descend and release (not recorded) ──────────────
            place_pos = np.array([goal_xy[0], goal_xy[1], goal_z + 0.02])
            ik_motion.move_to(env.world, place_pos, target_quat=ee_down,
                              steps=steps["place"], render=not args.headless)
            ik_motion.set_gripper(open=True)
            for _ in range(cfg["sim"]["gripper_steps"]):
                env.world.step(render=not args.headless)

            # Retract to lift height before next block
            ik_motion.move_to(env.world,
                              np.array([goal_xy[0], goal_xy[1], lift_h]),
                              target_quat=ee_down, steps=60,
                              render=not args.headless)

            goal_z += cfg["block"]["size"] + 0.002

        all_demos.append({
            "arm":        args.arm,
            "demo_idx":   demo_idx,
            "trajectory": demo_traj,
            "success":    True,
        })

    with open(out_path, "wb") as f:
        pickle.dump(all_demos, f)

    n_steps = sum(len(d["trajectory"]) for d in all_demos)
    print(f"\n[INFO] Saved {len(all_demos)} demos ({n_steps} transport steps) to {out_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
