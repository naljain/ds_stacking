"""
Dual-arm deployment: transport-only Neural DS with inter-arm modulation.

Each arm:
  1. reach     — IK straight-line to hover above block
  2. grasp     — IK descend + close gripper
  3. lift      — IK raise to transport height
  4. transport — Neural DS drives q -> q_goal, modulated to avoid other arm
  5. place     — IK descend + open gripper

Both arms run their full pipeline concurrently. During transport, modulation
uses a simple hierarchy: the priority arm keeps its nominal DS velocity, while
the other arm yields by modulating around the priority arm's EE. Priority
alternates after each successful place. The can_place() yield gate prevents
both arms descending onto the stack simultaneously.

Usage:
  python scripts/deploy_dual_arm.py
  python scripts/deploy_dual_arm.py --use_safe
  python scripts/deploy_dual_arm.py --no_modulation   # ablation
"""

import os
import sys
import argparse
import pickle
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


def load_ds_set(ckpt_dir, device, ckpt_arm="both"):
    """Load joint-space Neural DS checkpoints keyed by primitive.

    Kept as a small public helper because evaluate.py imports it.
    """
    primitives = ("reach", "grasp", "lift", "transport", "place")
    label = ckpt_arm
    if label not in ("both", "left", "right"):
        label = "both"
    return {
        primitive: load_ds(Path(ckpt_dir) / f"{label}_{primitive}.pt", device)
        for primitive in primitives
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
    parser.add_argument("--priority_arm",  type=str, default="left",
                        choices=["left", "right"],
                        help="Arm that starts as the unmodulated priority arm")
    parser.add_argument("--priority_mod_weight", type=float, default=0.25,
                        help="Blend weight for modulation on the priority arm")
    parser.add_argument("--yield_mod_weight", type=float, default=1.0,
                        help="Blend weight for modulation on the non-priority arm")
    parser.add_argument("--mod_radius", type=float, default=None,
                        help="Spherical modulation radius around each EE")
    parser.add_argument("--mod_reactivity", type=float, default=None,
                        help="Gamma exponent; lower values make modulation start earlier")
    parser.add_argument("--model",         type=str, default="neural",
                        choices=["neural", "lpvds"],
                        help="Transport DS model: neural joint-space or Cartesian LPVDS")
    parser.add_argument("--lookahead",     type=int, default=5,
                        help="LPVDS IK target = ee_pos + x_dot * lookahead * dt")
    parser.add_argument("--max_cart_speed", type=float, default=0.25,
                        help="Clip LPVDS Cartesian speed before IK retargeting")
    parser.add_argument("--cart_gain", type=float, default=1.0,
                        help="Scale LPVDS Cartesian velocity before speed clipping")
    parser.add_argument("--raw_lpvds",     action="store_true",
                        help="Use raw LPVDS velocity without stability projection")
    parser.add_argument("--no_workspace_clamp", action="store_true",
                        help="Do not clamp LPVDS IK targets to each arm's transport workspace")
    parser.add_argument("--z_margin", type=float, default=0.12,
                        help="LPVDS target z clamp around lift height when workspace clamp is enabled")
    parser.add_argument("--headless",      action="store_true")
    parser.add_argument("--done_tol",      type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for deploy-time block randomization")
    parser.add_argument("--no_randomize_blocks", action="store_true",
                        help="Use the scene's initial block positions")
    parser.add_argument("--diag_out", type=str, default=None,
                        help="Optional path to save LPVDS interaction diagnostics")
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
    lpv_min = {}
    lpv_max = {}
    for arm in ARMS:
        ws = cfg["block_workspace"][arm]
        lpv_min[arm] = np.array([
            min(ws["x_min"], goal_xy[0]) - 0.05,
            min(ws["y_min"], goal_xy[1]) - 0.05,
            lift_h - args.z_margin,
        ])
        lpv_max[arm] = np.array([
            max(ws["x_max"], goal_xy[0]) + 0.05,
            max(ws["y_max"], goal_xy[1]) + 0.05,
            lift_h + args.z_margin,
        ])

    # Per-arm objects
    franka      = {a: env.frankas[a] for a in ARMS}
    ik_motion   = {a: IKController(franka[a], arm=a,
                                   rest_q=np.array(cfg["arms"][f"default_joints_{a}"]))
                   for a in ARMS}
    ik_kin      = {a: FrankaIK(franka[a]) for a in ARMS}
    default_q   = {a: np.array(cfg["arms"][f"default_joints_{a}"]) for a in ARMS}

    if not args.no_randomize_blocks:
        rng = np.random.default_rng(args.seed)
        env.reset_blocks(render=not args.headless, rng=rng)
        print("[DEPLOY] Randomized block positions")

    # Load DS — one per arm (or shared if ckpt_arm specified)
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ds = {}
    lpv_model = {}
    if args.model == "lpvds":
        from src.lpv_ds import LPVDS
        for a in ARMS:
            label = args.ckpt_arm if args.ckpt_arm else a
            lpv_model[a] = LPVDS.load(ckpt_dir / f"{label}_transport_lpvds.pkl")
    else:
        for a in ARMS:
            label = args.ckpt_arm if args.ckpt_arm else a
            ds[a] = load_ds(ckpt_dir / f"{label}_transport.pt", device)

    # Transport q_goal — fixed target, same for both arms
    q_goal = {}
    if args.model == "neural":
        for a in ARMS:
            q, ok = ik_kin[a].solve(transport_pos, target_quat=ee_down,
                                     q_seed=default_q[a])
            if not ok:
                print(f"[WARN] IK failed for transport target on {a} arm")
            q_goal[a] = q

    mod_radius = args.mod_radius
    if mod_radius is None:
        mod_radius = cfg["coordination"].get("ee_safety_radius", 0.30)
    mod_reactivity = args.mod_reactivity
    if mod_reactivity is None:
        mod_reactivity = cfg["coordination"].get("modulation_reactivity", 2.0)
    mod = InterArmModulation(
        safe_radius=mod_radius,
        reactivity=mod_reactivity,
    )

    block_names = {a: [b["name"] for b in cfg[f"{a}_blocks"]] for a in ARMS}
    goal_z      = cfg["table"]["height"] + block_h / 2
    block_idx   = {a: 0 for a in ARMS}

    # Per-arm state machine
    stage       = {a: Stage.REACH for a in ARMS}
    ee_grasp    = {a: ee_down.copy() for a in ARMS}
    transport_steps = {a: 0 for a in ARMS}
    ik_fail_streak = {a: 0 for a in ARMS}
    priority_arm = args.priority_arm
    diag_log = []

    def current_block(arm):
        idx = block_idx[arm]
        names = block_names[arm]
        return names[idx] if idx < len(names) else None

    def arm_done(arm):
        return block_idx[arm] >= len(block_names[arm])

    def other_arm(arm):
        return "right" if arm == "left" else "left"

    def modulation_weight(arm):
        """Blend weight for this arm's avoidance correction."""
        if args.no_modulation:
            return 0.0
        if stage[arm] != Stage.TRANSPORT or arm_done(arm):
            return 0.0
        other = other_arm(arm)
        if arm_done(other):
            return 0.0
        if arm == priority_arm:
            return args.priority_mod_weight
        return args.yield_mod_weight

    def can_place(arm):
        """Yield if other arm is also near the stack goal."""
        other = other_arm(arm)
        if arm_done(other):
            return True
        if stage[other] == Stage.PLACE:
            return arm == priority_arm
        if stage[other] not in (Stage.TRANSPORT, Stage.RETRACT):
            return True
        ee_other, _ = ik_motion[other].ik.get_world_pose()
        gx, gy = goal_xy
        return np.linalg.norm(ee_other[:2] - np.array([gx, gy])) > yield_radius

    mod_mode = "OFF" if args.no_modulation else (
        f"weighted priority={priority_arm} "
        f"priority_w={args.priority_mod_weight} yield_w={args.yield_mod_weight}"
    )
    print(f"[DEPLOY] Dual-arm transport DS  model={args.model}  safe={args.use_safe}  "
          f"modulation={mod_mode}")
    print(f"[DEPLOY] Modulation sphere radius={mod_radius:.3f}m "
          f"reactivity={mod_reactivity:.2f}")
    print(f"[DEPLOY] IK frame={ik_motion['left'].ik.ee_frame}")
    if args.model == "lpvds":
        for arm in ARMS:
            print(f"[DEPLOY] {arm} LPVDS goal={np.round(lpv_model[arm].x_goal, 4)}  "
                  f"config transport={np.round(transport_pos, 4)}  "
                  f"lookahead={args.lookahead}")
            if not args.no_workspace_clamp:
                print(f"[DEPLOY] {arm} LPVDS target clamp min={np.round(lpv_min[arm], 3)} "
                      f"max={np.round(lpv_max[arm], 3)}")

    # ── IK move helper (runs synchronously, blocks until done) ────────────────
    def ik_move(arm, target, quat, n_steps):
        ik_motion[arm].move_to(env.world, target, target_quat=quat,
                               steps=n_steps, render=not args.headless)

    def joint_move(arm, target_q, n_steps=120):
        q0 = franka[arm].get_joint_positions()[:7].copy()
        finger = ik_motion[arm]._finger_width
        for s in np.linspace(0.0, 1.0, n_steps):
            q = q0 + s * (target_q - q0)
            full = np.concatenate([q, [finger, finger]])
            franka[arm].apply_action(_articulation_action(full))
            env.world.step(render=not args.headless)
        ik_motion[arm].reset()

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
                ik_fail_streak[arm] = 0
                stage[arm] = Stage.TRANSPORT
                continue

            if stage[arm] == Stage.PLACE:
                if not can_place(arm):
                    # Hover at transport position until coast is clear
                    pass
                else:
                    place_pos = np.array([goal_xy[0], goal_xy[1],
                                          goal_z + 0.02])
                    ik_move(arm, place_pos, ee_down, steps["place"])
                    ik_motion[arm].set_gripper(open=True)
                    for _ in range(cfg["sim"]["gripper_steps"]):
                        env.world.step(render=not args.headless)
                    goal_z += block_h + 0.002
                    priority_arm = other_arm(arm)
                    if not args.no_modulation:
                        print(f"  [COORD] Priority passed to {priority_arm}")
                    stage[arm] = Stage.RETRACT
                continue

            if stage[arm] == Stage.RETRACT:
                ik_move(arm, transport_pos, ee_down, 60)
                block_idx[arm] += 1
                if arm_done(arm):
                    print(f"  [{arm}] All cubes done; moving to nominal pose")
                    ik_motion[arm].set_gripper(open=True)
                    joint_move(arm, default_q[arm], n_steps=120)
                    stage[arm] = Stage.DONE
                else:
                    stage[arm] = Stage.REACH
                continue

        # ── DS transport step (both arms simultaneously) ───────────────────
        # Snapshot EE positions before any commands this tick
        ee_pos = {a: ik_motion[a].ik.get_world_pose()[0].copy() for a in ARMS}

        if args.model == "lpvds":
            cart_vels = {}
            cart_nominal = {}
            cart_modulated = {}
            for arm in ARMS:
                if stage[arm] != Stage.TRANSPORT or arm_done(arm):
                    cart_vels[arm] = None
                    cart_nominal[arm] = None
                    cart_modulated[arm] = None
                    continue

                if np.linalg.norm(ee_pos[arm] - lpv_model[arm].x_goal) < args.done_tol:
                    print(f"  [{arm}] Transport done in {transport_steps[arm]} steps")
                    stage[arm] = Stage.PLACE
                    cart_vels[arm] = None
                    cart_nominal[arm] = None
                    cart_modulated[arm] = None
                    continue

                transport_steps[arm] += 1
                if transport_steps[arm] > args.max_transport:
                    print(f"  [WARN] {arm} transport hit max steps")
                    stage[arm] = Stage.PLACE
                    cart_vels[arm] = None
                    cart_nominal[arm] = None
                    cart_modulated[arm] = None
                    continue

                if args.raw_lpvds:
                    cart_vels[arm] = lpv_model[arm].predict(ee_pos[arm])
                else:
                    cart_vels[arm] = lpv_model[arm].safe_velocity(ee_pos[arm])
                cart_nominal[arm] = cart_vels[arm].copy()
                cart_modulated[arm] = cart_vels[arm].copy()

            for arm in ARMS:
                w = modulation_weight(arm)
                if w <= 0.0 or cart_vels[arm] is None:
                    continue
                other = other_arm(arm)
                v_nom = cart_vels[arm]
                v_mod = mod.huber.modulate_cartesian(v_nom, ee_pos[arm], ee_pos[other])
                cart_vels[arm] = (1.0 - w) * v_nom + w * v_mod
                cart_modulated[arm] = cart_vels[arm].copy()

            for arm in ARMS:
                if cart_vels[arm] is None:
                    continue
                cart_vels[arm] = args.cart_gain * cart_vels[arm]
                speed = np.linalg.norm(cart_vels[arm])
                if speed > args.max_cart_speed:
                    cart_vels[arm] *= args.max_cart_speed / speed

            any_transport = False
            for arm in ARMS:
                if cart_vels[arm] is None:
                    continue
                any_transport = True
                ee_target = ee_pos[arm] + cart_vels[arm] * args.lookahead * physics_dt
                if not args.no_workspace_clamp:
                    ee_target = np.clip(ee_target, lpv_min[arm], lpv_max[arm])
                ik_ok = ik_motion[arm].step_to(ee_target, target_quat=ee_down)
                other = other_arm(arm)
                diag_log.append({
                    "t": global_step * physics_dt,
                    "step": global_step,
                    "arm": arm,
                    "stage": stage[arm].name,
                    "priority_arm": priority_arm,
                    "mod_weight": float(modulation_weight(arm)),
                    "ee": ee_pos[arm].tolist(),
                    "ee_other": ee_pos[other].tolist(),
                    "goal": lpv_model[arm].x_goal.tolist(),
                    "v_nom": cart_nominal[arm].tolist(),
                    "v_mod": cart_modulated[arm].tolist(),
                    "v_cmd": cart_vels[arm].tolist(),
                    "target": ee_target.tolist(),
                    "gamma": float(mod.huber.gamma(ee_pos[arm], ee_pos[other])),
                    "distance": float(np.linalg.norm(ee_pos[arm] - ee_pos[other])),
                    "safe_radius": float(mod_radius),
                    "reactivity": float(mod_reactivity),
                    "ik_ok": bool(ik_ok),
                })
                ik_fail_streak[arm] = 0 if ik_ok else ik_fail_streak[arm] + 1
                if ik_fail_streak[arm] == 25:
                    print(f"  [WARN] {arm} LPVDS IK has failed for 25 consecutive steps")

            if any_transport:
                env.step(render=not args.headless)
                global_step += 1
            continue

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
        for arm in ARMS:
            w = modulation_weight(arm)
            if w <= 0.0 or q_dots[arm] is None:
                continue
            other = other_arm(arm)
            J = jacobian_finite_difference(franka[arm])
            q_dot_nom = q_dots[arm]
            q_dot_mod = mod.modulate_joint_velocity(
                q_dot_nominal=q_dot_nom,
                ee_pos_self=ee_pos[arm],
                ee_pos_other=ee_pos[other],
                jacobian=J,
            )
            q_dots[arm] = (1.0 - w) * q_dot_nom + w * q_dot_mod

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
    if args.diag_out:
        diag_path = Path(args.diag_out)
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "wb") as f:
            pickle.dump({
                "config": {
                    "model": args.model,
                    "mod_radius": mod_radius,
                    "mod_reactivity": mod_reactivity,
                    "priority_mod_weight": args.priority_mod_weight,
                    "yield_mod_weight": args.yield_mod_weight,
                    "cart_gain": args.cart_gain,
                    "max_cart_speed": args.max_cart_speed,
                },
                "rows": diag_log,
            }, f)
        print(f"[DEPLOY] Saved diagnostics to {diag_path}")
    simulation_app.close()


if __name__ == "__main__":
    main()
