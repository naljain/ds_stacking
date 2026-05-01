"""
Dual-arm deployment: transport-only Neural DS with inter-arm modulation.

Each arm:
  1. reach     — IK straight-line to hover above block
  2. grasp     — IK descend + close gripper
  3. lift      — IK raise to transport height
  4. transport — Neural DS drives q -> q_goal, modulated to avoid other arm
  5. place     — IK descend + open gripper

Both arms run their full pipeline concurrently. During transport, each arm's
DS velocity is modulated by the Huber framework using the other arm's EE as
a moving spherical obstacle. The can_place() yield gate prevents both arms
descending onto the stack simultaneously.

Usage:
  python scripts/deploy_dual_arm.py
  python scripts/deploy_dual_arm.py --use_safe
  python scripts/deploy_dual_arm.py --no_modulation   # ablation
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from enum import Enum, auto

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
os.environ["CARB_LOG_LEVEL"] = "error"


class Stage(Enum):
    REACH     = auto()
    GRASP     = auto()
    LIFT      = auto()
    TRANSPORT = auto()
    PLACE     = auto()
    RETRACT   = auto()
    DONE      = auto()


def load_ds(ckpt_path, device):
    from src.neural_ds import StableNeuralDS, N_JOINTS
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
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


def _articulation_action(positions):
    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core.utils.types import ArticulationAction
    return ArticulationAction(joint_positions=positions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt_arm",      type=str, default=None,
                        help="Checkpoint label for both arms (default: per-arm)")
    parser.add_argument("--max_transport", type=int, default=2000,
                        help="Max DS steps per transport")
    parser.add_argument("--use_safe",      action="store_true")
    parser.add_argument("--no_modulation", action="store_true")
    parser.add_argument("--headless",      action="store_true")
    parser.add_argument("--done_tol",      type=float, default=0.05)
    args = parser.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless,
                                    "width": 1280, "height": 720})

    from src.env import DualArmEnv
    from src.ik_controller import IKController
    from src.franka_ik import FrankaIK
    from src.modulation import InterArmModulation, jacobian_finite_difference

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = DualArmEnv(config_path=args.config, arms=("left", "right"))
    cfg = env.cfg

    ARMS = ("left", "right")
    physics_dt  = cfg["sim"]["physics_dt"]
    steps       = cfg["sim"]["steps_per_primitive"]
    hover_h     = cfg["heights"]["hover"]
    lift_h      = cfg["heights"]["lift"]
    grasp_h     = cfg["heights"]["grasp"]
    block_h     = cfg["block"]["size"]
    goal_xy     = tuple(cfg["shared_goal"])
    yield_radius = cfg["coordination"].get("yield_radius", 0.12)
    ee_down     = np.array([0.0, 1.0, 0.0, 0.0])
    transport_pos = np.array([goal_xy[0], goal_xy[1], lift_h])

    # Per-arm objects
    franka      = {a: env.frankas[a] for a in ARMS}
    ik_motion   = {a: IKController(franka[a], arm=a,
                                   rest_q=np.array(cfg["arms"][f"default_joints_{a}"]))
                   for a in ARMS}
    ik_kin      = {a: FrankaIK(franka[a]) for a in ARMS}
    default_q   = {a: np.array(cfg["arms"][f"default_joints_{a}"]) for a in ARMS}

    # Load DS — one per arm (or shared if ckpt_arm specified)
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ds = {}
    for a in ARMS:
        label = args.ckpt_arm if args.ckpt_arm else a
        ds[a] = load_ds(ckpt_dir / f"{label}_transport.pt", device)

    # Transport q_goal — fixed target, same for both arms
    q_goal = {}
    for a in ARMS:
        q, ok = ik_kin[a].solve(transport_pos, target_quat=ee_down,
                                 q_seed=default_q[a])
        if not ok:
            print(f"[WARN] IK failed for transport target on {a} arm")
        q_goal[a] = q

    mod = InterArmModulation(
        safe_radius=cfg["coordination"]["ee_safety_radius"],
        reactivity=4.0,
    )

    block_names = {a: [b["name"] for b in cfg[f"{a}_blocks"]] for a in ARMS}
    goal_z      = {a: cfg["table"]["height"] + block_h / 2 for a in ARMS}
    block_idx   = {a: 0 for a in ARMS}

    # Per-arm state machine
    stage       = {a: Stage.REACH for a in ARMS}
    ee_grasp    = {a: ee_down.copy() for a in ARMS}
    transport_steps = {a: 0 for a in ARMS}

    def current_block(arm):
        idx = block_idx[arm]
        names = block_names[arm]
        return names[idx] if idx < len(names) else None

    def arm_done(arm):
        return block_idx[arm] >= len(block_names[arm])

    def can_place(arm):
        """Yield if other arm is also near the stack goal."""
        other = "right" if arm == "left" else "left"
        if arm_done(other):
            return True
        if stage[other] not in (Stage.TRANSPORT, Stage.PLACE):
            return True
        ee_other, _ = env.get_ee_pose(other)
        gx, gy = goal_xy
        return np.linalg.norm(ee_other[:2] - np.array([gx, gy])) > yield_radius

    print(f"[DEPLOY] Dual-arm transport DS  safe={args.use_safe}  "
          f"modulation={'OFF' if args.no_modulation else 'ON'}")

    # ── IK move helper (runs synchronously, blocks until done) ────────────────
    def ik_move(arm, target, quat, n_steps):
        ik_motion[arm].move_to(env.world, target, target_quat=quat,
                               steps=n_steps, render=not args.headless)

    # ── Main loop ─────────────────────────────────────────────────────────────
    # Arms that are in IK stages (reach/grasp/lift/place/retract) run
    # synchronously inside this loop iteration. Arms in transport run the
    # DS step-by-step so both can be modulated against each other.

    global_step = 0
    MAX_GLOBAL  = 100_000

    while global_step < MAX_GLOBAL:
        if not simulation_app.is_running():
            break
        if all(arm_done(a) for a in ARMS):
            print("[DEPLOY] Both arms finished.")
            break

        for arm in ARMS:
            if arm_done(arm):
                continue

            blk = current_block(arm)
            bpos = env.get_block_positions()[blk].copy()
            bx, by = bpos[0], bpos[1]

            # ── IK stages (synchronous, consume many sim steps internally) ──
            if stage[arm] == Stage.REACH:
                ee_grasp[arm] = env.get_block_grasp_quat(blk)
                ik_move(arm, np.array([bx, by, hover_h]),
                        ee_grasp[arm], steps["reach"])
                stage[arm] = Stage.GRASP
                continue

            if stage[arm] == Stage.GRASP:
                ik_move(arm, np.array([bx, by, grasp_h]),
                        ee_grasp[arm], steps["grasp"])
                ik_motion[arm].set_gripper(open=False)
                for _ in range(cfg["sim"]["gripper_steps"]):
                    env.world.step(render=not args.headless)
                stage[arm] = Stage.LIFT
                continue

            if stage[arm] == Stage.LIFT:
                ik_move(arm, np.array([bx, by, lift_h]),
                        ee_down, steps["lift"])
                transport_steps[arm] = 0
                stage[arm] = Stage.TRANSPORT
                continue

            if stage[arm] == Stage.PLACE:
                if not can_place(arm):
                    # Hover at transport position until coast is clear
                    pass
                else:
                    place_pos = np.array([goal_xy[0], goal_xy[1],
                                          goal_z[arm] + 0.02])
                    ik_move(arm, place_pos, ee_down, steps["place"])
                    ik_motion[arm].set_gripper(open=True)
                    for _ in range(cfg["sim"]["gripper_steps"]):
                        env.world.step(render=not args.headless)
                    stage[arm] = Stage.RETRACT
                continue

            if stage[arm] == Stage.RETRACT:
                ik_move(arm, transport_pos, ee_down, 60)
                goal_z[arm] += block_h + 0.002
                block_idx[arm] += 1
                stage[arm] = Stage.REACH
                continue

        # ── DS transport step (both arms simultaneously) ───────────────────
        # Snapshot EE positions before any commands this tick
        ee_pos = {a: env.get_ee_pose(a)[0].copy() for a in ARMS}

        q_dots = {}
        for arm in ARMS:
            if stage[arm] != Stage.TRANSPORT or arm_done(arm):
                q_dots[arm] = None
                continue

            q = franka[arm].get_joint_positions()[:7].copy()

            if np.linalg.norm(q - q_goal[arm]) < args.done_tol:
                print(f"  [{arm}] Transport done in {transport_steps[arm]} steps")
                stage[arm] = Stage.PLACE
                q_dots[arm] = None
                continue

            transport_steps[arm] += 1
            if transport_steps[arm] > args.max_transport:
                print(f"  [WARN] {arm} transport hit max steps")
                stage[arm] = Stage.PLACE
                q_dots[arm] = None
                continue

            x   = np.concatenate([q, q_goal[arm]])
            x_n = (x - ds[arm]["state_mean"]) / ds[arm]["state_std"]
            x_t = torch.tensor(x_n, dtype=torch.float32,
                               device=device).unsqueeze(0)

            if args.use_safe:
                qd_n = ds[arm]["model"].safe_velocity(x_t)
            else:
                with torch.no_grad():
                    qd_n = ds[arm]["model"](x_t)

            q_dots[arm] = qd_n.cpu().numpy().squeeze(0) * ds[arm]["vel_scale"]

        # Apply inter-arm modulation
        if not args.no_modulation:
            for arm, other in (("left", "right"), ("right", "left")):
                if q_dots[arm] is None:
                    continue
                J = jacobian_finite_difference(franka[arm])
                q_dots[arm] = mod.modulate_joint_velocity(
                    q_dot_nominal=q_dots[arm],
                    ee_pos_self=ee_pos[arm],
                    ee_pos_other=ee_pos[other],
                    jacobian=J,
                )

        # Command arms in transport
        any_transport = False
        for arm in ARMS:
            if q_dots[arm] is None:
                continue
            any_transport = True
            q = franka[arm].get_joint_positions()[:7].copy()
            q_cmd = q + q_dots[arm] * physics_dt
            finger = ik_motion[arm]._finger_width
            full_cmd = np.concatenate([q_cmd, [finger, finger]])
            franka[arm].apply_action(_articulation_action(full_cmd))

        if any_transport:
            env.step(render=not args.headless)
            global_step += 1

    print(f"[DEPLOY] Finished after {global_step} DS steps.")
    simulation_app.close()


if __name__ == "__main__":
    main()
