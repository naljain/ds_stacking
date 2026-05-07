"""
Joint-space trajectory collection for joint-space Neural DS training.

For each demo, scripts the arm through the full pick-and-stack sequence on its
3 blocks. It computes the same Lula q_goal used at deployment and records a
joint-space expert moving toward that attractor.

    q       (7,)   joint positions  (excludes finger joints)
    q_dot   (7,)   joint velocities (finite-diff)
    q_goal  (7,)   target joint configuration for the active primitive.
                   By default this is the Lula target used by the joint-space
                   expert and by deployment.

Also recorded for bookkeeping:
    primitive   str   one of {reach, grasp, lift, transport, place}
    block       str   active block name
    arm         str   {left, right}

Saved to data/demonstrations/{arm}_demos.pkl as a list of demos.

The DS will be trained to map q - q_goal -> q_dot, learning an error-space
joint velocity field for each primitive.

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
    parser.add_argument("--noise",    type=float, default=0.0,
                        help="Legacy target XY noise in metres. Keep this at "
                             "0 for reliable grasp demos; nonzero values aim "
                             "beside the observed block.")
    parser.add_argument("--block_xy_jitter", type=float, default=0.0,
                        help="Max XY jitter applied to the physical block "
                             "positions per demo. This widens the data without "
                             "commanding grasps away from the block.")
    parser.add_argument("--start_jitter", type=float, default=0.15,
                        help="Per-joint random start-pose perturbation (rad). "
                             "Widens the demonstrated trajectory manifold so "
                             "the DS doesn't overfit to one path.")
    parser.add_argument("--settle_extra_steps", type=int, default=60,
                        help="Max extra controller steps used to settle at each "
                             "primitive target.")
    parser.add_argument("--inter_primitive_pause_steps", type=int, default=None,
                        help="Non-recorded hold steps inserted after each "
                             "primitive so segment boundaries are clean. "
                             "Defaults to sim.inter_primitive_pause_steps.")
    parser.add_argument("--lift_check_margin", type=float, default=0.01,
                        help="Block must rise above table + block size + this "
                             "margin after lift, otherwise the demo is retried.")
    parser.add_argument("--physical_grasp", action="store_true",
                        help="Require Isaac gripper contact to lift the block. "
                             "Default collection kinematically carries the "
                             "active block after grasp so DS demos are not "
                             "discarded due contact-grasp flakiness.")
    parser.add_argument("--joint_goal_gain", type=float, default=2.0,
                        help="First-order Lula joint-space expert gain.")
    parser.add_argument("--collection_max_joint_vel", type=float, default=1.2,
                        help="Per-joint velocity cap in rad/s for collection "
                             "experts. Kept separate from training.max_joint_vel "
                             "so demos can be slower without changing "
                             "deployment clamps.")
    parser.add_argument("--joint_done_tol", type=float, default=0.03,
                        help="Joint-space tolerance for early stopping in "
                             "the Lula joint-space expert.")
    parser.add_argument("--max_q_goal_error", type=float, default=0.08,
                        help="Discard a demo attempt if a joint_lula primitive "
                             "settles farther than this L2 joint error from "
                             "its Lula q_goal.")
    parser.add_argument("--direct_joint_set", action="store_true", default=True,
                        help="Set joint "
                             "positions directly along the generated demo "
                             "trajectory instead of relying on articulation "
                             "position-controller tracking. Useful when Isaac "
                             "tracking lags make q_settled differ from q_goal.")
    parser.add_argument("--track_joint_commands", dest="direct_joint_set",
                        action="store_false",
                        help="Use articulation "
                             "position command tracking instead of direct joint "
                             "sets. This is diagnostic only; tracking lag can "
                             "produce inconsistent q_goal labels.")
    args = parser.parse_args()

    from isaacsim import SimulationApp
    _app_cfg = {"headless": args.headless}
    if not args.headless:
        _app_cfg.update({"width": 1280, "height": 720})
    simulation_app = SimulationApp(_app_cfg)

    from src.env import DualArmEnv
    from src.franka_ik import FrankaIK
    from src.primitives import (primitive_target, PRIMITIVE_ORDER,
                                grasp_quat_from_block)
    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core.utils.types import ArticulationAction

    env = DualArmEnv(config_path=args.config, arms=(args.arm,))
    cfg = env.cfg
    franka = env.frankas[args.arm]

    ik_kin    = FrankaIK(franka)

    block_names = [b["name"] for b in cfg[f"{args.arm}_blocks"]]
    goal_xy     = tuple(cfg["goals"][args.arm])

    hover_h = cfg["heights"]["hover"]
    lift_h  = cfg["heights"]["lift"]
    grasp_h = cfg["heights"]["grasp"]
    base_z  = cfg["table"]["height"] + cfg["block"]["size"] / 2
    block_h = cfg["block"]["size"] + 0.002
    stack_clearance = cfg.get("stack", {}).get("clearance_above_top", 0.12)
    physics_dt = cfg["sim"]["physics_dt"]

    steps = cfg["sim"]["steps_per_primitive"]
    pause_steps = (cfg["sim"].get("inter_primitive_pause_steps", 0)
                   if args.inter_primitive_pause_steps is None
                   else args.inter_primitive_pause_steps)

    out_dir = Path(cfg["paths"]["demos"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arm}_demos.pkl"

    rng = np.random.default_rng(seed=42)
    all_demos = []

    print(f"\n[INFO] Collecting {args.n_demos} demos ({args.arm} arm).")

    home_q = franka.get_joint_positions().copy()
    if args.block_xy_jitter <= 0 and args.noise <= 0:
        print("[INFO] Collecting conservative demos: no block jitter, no target noise.")
    if not args.physical_grasp:
        print("[INFO] Kinematic block carry is ON for collection "
              "(use --physical_grasp to require contact grasps).")
    if pause_steps > 0:
        print(f"[INFO] Holding {pause_steps} steps between primitives "
              "without recording.")
    print(f"[INFO] Lula collection gain={args.joint_goal_gain}, "
          f"max_joint_vel={args.collection_max_joint_vel} rad/s")

    def reset_arm_to_start():
        franka.set_joint_positions(home_q)
        if hasattr(franka, "set_joint_velocities"):
            franka.set_joint_velocities(np.zeros_like(home_q))
        franka.gripper.apply_action(
            ArticulationAction(joint_positions=np.array([0.04, 0.04]))
        )
        for _ in range(30):
            env.world.step(render=not args.headless)

    max_attempts = max(args.n_demos * 5, args.n_demos)
    attempts = 0
    while len(all_demos) < args.n_demos and attempts < max_attempts:
        attempts += 1
        demo_idx = len(all_demos)
        print(f"  Demo {demo_idx + 1}/{args.n_demos} (attempt {attempts})")
        env.reset_blocks(render=not args.headless)
        reset_arm_to_start()

        if args.block_xy_jitter > 0:
            for block_name in block_names:
                pos, quat = env.get_block_poses()[block_name]
                pos = pos.copy()
                pos[:2] += rng.uniform(-args.block_xy_jitter,
                                       args.block_xy_jitter, size=2)
                obj = env.get_block_obj(block_name)
                obj.set_world_pose(position=pos, orientation=quat)
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))
            for _ in range(30):
                env.world.step(render=not args.headless)

        # Randomise the starting joint pose so each demo's first reach starts
        # from a different config. Without this, every demo's path is
        # essentially the same and the DS overfits to one trajectory shape.
        if args.start_jitter > 0:
            jitter = rng.uniform(-args.start_jitter, args.start_jitter,
                                 size=7).astype(np.float32)
            jittered = home_q.copy()
            jittered[:7] += jitter
            franka.set_joint_positions(jittered)
            if hasattr(franka, "set_joint_velocities"):
                franka.set_joint_velocities(np.zeros_like(jittered))
            for _ in range(20):
                env.world.step(render=not args.headless)

        demo_traj = []
        prev_q = franka.get_joint_positions()[:7].copy()
        demo_failed = False
        held_block = None
        held_offset = np.zeros(3)

        def carry_held_block():
            if held_block is None:
                return
            ee_pos = franka.end_effector.get_world_pose()[0].copy()
            obj = env.get_block_obj(held_block)
            obj.set_world_pose(position=ee_pos + held_offset,
                               orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            obj.set_linear_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))

        def hold_between_primitives():
            if pause_steps <= 0:
                return
            if hasattr(franka, "set_joint_velocities"):
                full_v = np.zeros_like(franka.get_joint_positions())
                franka.set_joint_velocities(full_v)
            hold_q = franka.get_joint_positions().copy()
            for _ in range(pause_steps):
                franka.apply_action(ArticulationAction(joint_positions=hold_q))
                env.world.step(render=not args.headless)
                carry_held_block()

        goal_z = base_z
        for block_idx, block_name in enumerate(block_names):
            stack_slot = block_idx
            block_pos, block_quat = env.get_block_poses()[block_name]
            block_pos = block_pos.copy()
            xy_noise  = rng.uniform(-args.noise, args.noise, size=2)
            noisy_pos = block_pos.copy()
            noisy_pos[:2] += xy_noise

            # Orientation aligned to the block's yaw, used for reach + grasp.
            aligned_quat = grasp_quat_from_block(block_quat)

            for primitive in PRIMITIVE_ORDER:
                # Cartesian target for this primitive.
                target_cart = primitive_target(
                    primitive=primitive,
                    block_pos=noisy_pos,
                    goal_xy=goal_xy,
                    goal_z=goal_z,
                    hover_h=hover_h,
                    lift_h=lift_h,
                    grasp_h=grasp_h,
                )
                if primitive == "transport":
                    existing_stack_top = goal_z - cfg["block"]["size"] / 2
                    target_cart[2] = max(
                        target_cart[2],
                        existing_stack_top + stack_clearance,
                    )

                # Use block-aligned orientation when approaching; straight down
                # for lift / transport / place (block orientation no longer matters).
                target_quat = aligned_quat if primitive in ("reach", "grasp") else None

                # Buffer this primitive's steps; q_goal is filled in once the
                # primitive attractor is known.
                prim_buf = []

                def record(q_override=None, q_dot_override=None):
                    nonlocal prev_q
                    q = (franka.get_joint_positions()[:7].copy()
                         if q_override is None else q_override.copy())
                    q_dot = ((q - prev_q) / physics_dt
                             if q_dot_override is None else q_dot_override.copy())
                    prim_buf.append({
                        "q":         q,
                        "q_dot":     q_dot,
                        "ee_pos":    franka.end_effector.get_world_pose()[0].copy(),
                        "primitive": primitive,
                        "block":     block_name,
                        "arm":       args.arm,
                        "target":    target_cart.copy(),
                        "stack_slot": stack_slot,
                        "stack_goal_z": goal_z,
                        "stack_clearance_above_top": stack_clearance,
                        "collection_joint_goal_gain": args.joint_goal_gain,
                        "collection_max_joint_vel": args.collection_max_joint_vel,
                    })
                    prev_q = q

                # Reset prev_q so the first q_dot of this primitive is a clean
                # forward-difference within the primitive, not contaminated by
                # the retract or gripper-action motion that came before it.
                prev_q = franka.get_joint_positions()[:7].copy()

                q_seed = franka.get_joint_positions()[:7].copy()
                q_goal_lula, ok = ik_kin.solve(
                    target_cart, target_quat=target_quat, q_seed=q_seed)
                if not ok:
                    print(f"    [WARN] Lula IK failed for {primitive}; "
                          "using current q as q_goal.")
                    q_goal_lula = q_seed

                # Proportional-decay Lula expert: q_dot = -k * (q - q_goal),
                # clipped to max_joint_vel. Velocity scales with the error and
                # goes to zero at the attractor, which matches f(0)=0.
                max_joint_vel = args.collection_max_joint_vel
                k = float(args.joint_goal_gain)
                q_start = franka.get_joint_positions().copy()
                e0 = np.linalg.norm(q_start[:7] - q_goal_lula)
                if e0 > args.joint_done_tol and k > 1e-3:
                    decay_steps = int(np.ceil(
                        np.log(e0 / args.joint_done_tol) / (k * physics_dt)
                    ))
                else:
                    decay_steps = 0
                n_steps = max(steps[primitive], decay_steps) + args.settle_extra_steps
                n_steps = max(n_steps, 2)
                q_demo = q_start[:7].copy()
                prev_q = q_demo.copy()

                for _ in range(1, n_steps + 1):
                    q_now = (q_demo.copy() if args.direct_joint_set
                             else franka.get_joint_positions()[:7].copy())
                    e = q_now - q_goal_lula
                    if np.linalg.norm(e) < args.joint_done_tol:
                        break
                    q_dot_des = np.clip(-k * e, -max_joint_vel, max_joint_vel)
                    q_cmd = franka.get_joint_positions().copy()
                    q_cmd[:7] = q_now + q_dot_des * physics_dt
                    if args.direct_joint_set:
                        q_demo = q_cmd[:7].copy()
                        franka.set_joint_positions(q_cmd)
                    else:
                        franka.apply_action(
                            ArticulationAction(joint_positions=q_cmd)
                        )
                    env.world.step(render=not args.headless)
                    carry_held_block()
                    if args.direct_joint_set:
                        record(q_override=q_demo, q_dot_override=q_dot_des)
                        prev_q = q_demo.copy()
                    else:
                        record()
                q_settled = (q_demo.copy() if args.direct_joint_set
                             else franka.get_joint_positions()[:7].copy())

                q_goal = q_goal_lula
                q_goal_lula_error = float(np.linalg.norm(q_settled - q_goal_lula))
                if q_goal_lula_error > args.max_q_goal_error:
                    print(f"    [WARN] {primitive} settled {q_goal_lula_error:.3f} "
                          f"rad from Lula q_goal; discarding this demo attempt")
                    demo_failed = True
                    break
                if q_goal_lula_error > 0.25:
                    print(f"    [WARN] {primitive} Lula q_goal differs from "
                          f"expert settled q by {q_goal_lula_error:.3f} rad; "
                          "label_source=lula")
                for step in prim_buf:
                    step["q_goal"] = q_goal
                    step["q_goal_source"] = "lula"
                    step["motion_source"] = "lula"
                    step["q_goal_settled"] = q_settled
                    step["q_goal_lula"] = q_goal_lula
                    step["q_goal_lula_ok"] = bool(ok)
                    step["q_goal_lula_error"] = q_goal_lula_error
                demo_traj.extend(prim_buf)

                if primitive == "grasp":
                    for _ in range(10):
                        env.world.step(render=not args.headless)
                    franka.gripper.apply_action(
                        ArticulationAction(joint_positions=np.array([0.0, 0.0]))
                    )
                    for _ in range(cfg["sim"]["gripper_steps"]):
                        env.world.step(render=not args.headless)
                    if not args.physical_grasp:
                        ee_pos = franka.end_effector.get_world_pose()[0].copy()
                        held_block = block_name
                        held_offset = block_pos - ee_pos
                        carry_held_block()
                elif primitive == "place":
                    held_block = None
                    franka.gripper.apply_action(
                        ArticulationAction(joint_positions=np.array([0.04, 0.04]))
                    )
                    for _ in range(cfg["sim"]["gripper_steps"]):
                        env.world.step(render=not args.headless)

                if primitive == "lift" and args.physical_grasp:
                    lifted_pos = env.get_block_positions()[block_name]
                    min_lift_z = (cfg["table"]["height"]
                                  + cfg["block"]["size"]
                                  + args.lift_check_margin)
                    if lifted_pos[2] < min_lift_z:
                        ee_pos = franka.end_effector.get_world_pose()[0].copy()
                        print(f"    [WARN] missed grasp on {block_name}; "
                              "discarding this demo attempt")
                        print(f"           block_z={lifted_pos[2]:.3f}, "
                              f"needed>{min_lift_z:.3f}, "
                              f"ee={ee_pos.round(3)}, "
                              f"block={lifted_pos.round(3)}")
                        demo_failed = True
                        break

                hold_between_primitives()

            if demo_failed:
                break

            # Retract before next block (not recorded)
            existing_stack_top = goal_z - cfg["block"]["size"] / 2
            retract_z = max(lift_h, existing_stack_top + stack_clearance)
            retract_cart = np.array([goal_xy[0], goal_xy[1], retract_z])
            q_seed = franka.get_joint_positions()[:7].copy()
            q_goal_retract, ok = ik_kin.solve(
                retract_cart, target_quat=None, q_seed=q_seed)
            if ok:
                for _ in range(60 + args.settle_extra_steps):
                    q_now = franka.get_joint_positions()[:7].copy()
                    e = q_now - q_goal_retract
                    if np.linalg.norm(e) < args.joint_done_tol:
                        break
                    q_dot = np.clip(
                        -args.joint_goal_gain * e,
                        -args.collection_max_joint_vel,
                        args.collection_max_joint_vel,
                    )
                    q_cmd = franka.get_joint_positions().copy()
                    q_cmd[:7] = q_now + q_dot * physics_dt
                    franka.apply_action(ArticulationAction(joint_positions=q_cmd))
                    env.world.step(render=not args.headless)
                    carry_held_block()
            goal_z += block_h

        if not demo_failed:
            all_demos.append({
                "arm":        args.arm,
                "demo_idx":   demo_idx,
                "trajectory": demo_traj,
                "inter_primitive_pause_steps": pause_steps,
                "collection_joint_goal_gain": args.joint_goal_gain,
                "collection_max_joint_vel": args.collection_max_joint_vel,
                "success":    True,
            })

    if len(all_demos) < args.n_demos:
        print(f"\n[WARN] Collected only {len(all_demos)}/{args.n_demos} "
              f"successful demos after {attempts} attempts.")

    with open(out_path, "wb") as f:
        pickle.dump(all_demos, f)

    n_steps = sum(len(d["trajectory"]) for d in all_demos)
    print(f"\n[INFO] Saved {len(all_demos)} demos ({n_steps} timesteps) to {out_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
