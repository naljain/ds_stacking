"""
Joint-space trajectory collection for joint-space Neural DS training.

For each demo, scripts the arm through the full pick-and-stack sequence on its
3 blocks using RMPflow as the underlying motion generator, but records:

    q       (7,)   joint positions  (excludes finger joints)
    q_dot   (7,)   joint velocities (finite-diff)
    q_goal  (7,)   target joint configuration for the active primitive
                   (the joint config where RMPflow actually settled — sampled
                    after move_to completes so labels are consistent with q_dot)

Also recorded for bookkeeping:
    primitive   str   one of {reach, grasp, lift, transport, place}
    block       str   active block name
    arm         str   {left, right}

Saved to data/demonstrations/{arm}_demos.pkl as a list of demos.

The DS will be trained to map [q, q_goal] -> q_dot, learning a goal-conditioned
joint-space velocity field.

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
    parser.add_argument("--noise",    type=float, default=0.02,
                        help="Max XY noise added to block-target positions per demo.")
    args = parser.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless,
                                    "width": 1280, "height": 720})

    from src.env import DualArmEnv
    from src.ik_controller import IKController
    from src.primitives import (primitive_target, PRIMITIVE_ORDER,
                                grasp_quat_from_block)

    env = DualArmEnv(config_path=args.config, arms=(args.arm,))
    cfg = env.cfg
    franka = env.frankas[args.arm]

    ik_motion = IKController(franka)

    block_names = [b["name"] for b in cfg[f"{args.arm}_blocks"]]
    goal_xy     = tuple(cfg["goals"][args.arm])

    hover_h = cfg["heights"]["hover"]
    lift_h  = cfg["heights"]["lift"]
    grasp_h = cfg["heights"]["grasp"]
    base_z  = cfg["table"]["height"] + cfg["block"]["size"] / 2
    physics_dt = cfg["sim"]["physics_dt"]

    steps = cfg["sim"]["steps_per_primitive"]

    out_dir = Path(cfg["paths"]["demos"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arm}_demos.pkl"

    rng = np.random.default_rng(seed=42)
    all_demos = []

    print(f"\n[INFO] Collecting {args.n_demos} demos ({args.arm} arm).")

    for demo_idx in range(args.n_demos):
        print(f"  Demo {demo_idx + 1}/{args.n_demos}")
        env.reset_blocks(render=not args.headless)
        ik_motion.reset()
        ik_motion.set_gripper(open=True)

        demo_traj = []
        prev_q = franka.get_joint_positions()[:7].copy()

        goal_z = base_z
        for block_idx, block_name in enumerate(block_names):
            block_pos, block_quat = env.get_block_poses()[block_name]
            block_pos = block_pos.copy()
            xy_noise  = rng.uniform(-args.noise, args.noise, size=2)
            noisy_pos = block_pos.copy()
            noisy_pos[:2] += xy_noise

            # Orientation aligned to the block's yaw, used for reach + grasp.
            aligned_quat = grasp_quat_from_block(block_quat)

            for primitive in PRIMITIVE_ORDER:
                # Cartesian target for RMPflow
                target_cart = primitive_target(
                    primitive=primitive,
                    block_pos=noisy_pos,
                    goal_xy=goal_xy,
                    goal_z=goal_z,
                    hover_h=hover_h,
                    lift_h=lift_h,
                    grasp_h=grasp_h,
                )

                # Use block-aligned orientation when approaching; straight down
                # for lift / transport / place (block orientation no longer matters).
                target_quat = aligned_quat if primitive in ("reach", "grasp") else None

                # Buffer this primitive's steps; q_goal is filled in retroactively
                # below once we know where RMPflow actually settled.
                prim_buf = []

                def record():
                    nonlocal prev_q
                    q = franka.get_joint_positions()[:7].copy()
                    q_dot = (q - prev_q) / physics_dt
                    prim_buf.append({
                        "q":         q,
                        "q_dot":     q_dot,
                        "ee_pos":    franka.end_effector.get_world_pose()[0].copy(),
                        "primitive": primitive,
                        "block":     block_name,
                        "arm":       args.arm,
                        "target":    target_cart.copy(),
                    })
                    prev_q = q

                ik_motion.move_to(
                    world=env.world,
                    target_pos=target_cart,
                    target_quat=target_quat,
                    steps=steps[primitive],
                    record_callback=record,
                    render=not args.headless,
                )

                # Label q_goal with where the arm actually settled — this is
                # the joint config that q_dot was pointing toward throughout
                # the primitive, so the DS training target is self-consistent.
                q_goal = franka.get_joint_positions()[:7].copy()
                for step in prim_buf:
                    step["q_goal"] = q_goal
                demo_traj.extend(prim_buf)

                if primitive == "grasp":
                    ik_motion.set_gripper(open=False)
                    for _ in range(cfg["sim"]["gripper_steps"]):
                        env.world.step(render=not args.headless)
                elif primitive == "place":
                    ik_motion.set_gripper(open=True)
                    for _ in range(cfg["sim"]["gripper_steps"]):
                        env.world.step(render=not args.headless)

            # Retract before next block (not recorded)
            retract_cart = np.array([goal_xy[0], goal_xy[1], lift_h])
            ik_motion.move_to(env.world, retract_cart, steps=60,
                              record_callback=None,
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
    print(f"\n[INFO] Saved {len(all_demos)} demos ({n_steps} timesteps) to {out_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
