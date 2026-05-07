"""
Lightweight keyboard teleoperation for manual demonstration collection.

Controls:
  W/S       — move EE in +Y / -Y
  A/D       — move EE in -X / +X
  Q/E       — move EE in +Z / -Z
  F         — toggle gripper
  R         — start recording a primitive segment
  P         — commit segment: labels q_goal = current joint config (retroactive,
              same as collect_ik.py) then starts a fresh segment
  Backspace — discard current segment
  ESC       — save all demos to disk and quit

Each P-commit produces one primitive segment with q, q_dot, q_goal in joint
space — the same format train_ds.py expects. Label the primitive name with
--primitive when launching; run once per primitive type.

Usage:
  python scripts/teleop.py --arm left --primitive reach
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
    parser.add_argument("--arm",       type=str, default="left", choices=["left", "right"])
    parser.add_argument("--primitive", type=str, default="reach",
                        choices=["reach", "grasp", "lift", "transport", "place"])
    parser.add_argument("--config",    type=str, default="configs/default.yaml")
    parser.add_argument("--step",      type=float, default=0.005, help="metres per key tick")
    args = parser.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

    import carb
    import omni.appwindow

    try:
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core.utils.types import ArticulationAction
    from src.env import DualArmEnv
    from src.franka_ik import FrankaIK

    env = DualArmEnv(config_path=args.config, arms=(args.arm,))
    cfg = env.cfg
    franka = env.frankas[args.arm]
    ik = FrankaIK(franka)
    franka.gripper.apply_action(
        ArticulationAction(joint_positions=np.array([0.04, 0.04]))
    )

    out_dir = Path(cfg["paths"]["demos"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arm}_teleop_demos.pkl"
    physics_dt = cfg["sim"]["physics_dt"]

    # Mutable state — needs nonlocal-style sharing with key callback
    state = {
        "gripper_open": True,
        "recording":    False,
        "current_seg":  [],   # steps for the current primitive segment
        "all_demos":    [],
        "keys_held":    set(),
        "should_quit":  False,
        "prev_q":       None,
    }

    KEY_DELTAS = {
        carb.input.KeyboardInput.W: np.array([ 0,  1, 0]),
        carb.input.KeyboardInput.S: np.array([ 0, -1, 0]),
        carb.input.KeyboardInput.A: np.array([-1,  0, 0]),
        carb.input.KeyboardInput.D: np.array([ 1,  0, 0]),
        carb.input.KeyboardInput.Q: np.array([ 0,  0, 1]),
        carb.input.KeyboardInput.E: np.array([ 0,  0,-1]),
    }

    def on_key(event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            state["keys_held"].add(event.input)

            if event.input == carb.input.KeyboardInput.F:
                state["gripper_open"] = not state["gripper_open"]
                width = 0.04 if state["gripper_open"] else 0.0
                franka.gripper.apply_action(
                    ArticulationAction(joint_positions=np.array([width, width]))
                )
                print(f"[TELEOP] Gripper {'open' if state['gripper_open'] else 'closed'}")

            elif event.input == carb.input.KeyboardInput.R:
                state["recording"] = not state["recording"]
                if state["recording"]:
                    state["current_seg"] = []
                    state["prev_q"]      = None
                    print("[TELEOP] Recording started")
                else:
                    print(f"[TELEOP] Recording paused — {len(state['current_seg'])} steps so far")

            elif event.input == carb.input.KeyboardInput.P:
                if state["current_seg"]:
                    # Retroactive q_goal: where the arm settled at end of segment
                    q_goal = franka.get_joint_positions()[:7].copy()
                    for step in state["current_seg"]:
                        step["q_goal"] = q_goal
                    state["all_demos"].append({
                        "arm":        args.arm,
                        "demo_idx":   len(state["all_demos"]),
                        "trajectory": state["current_seg"],
                        "success":    True,
                    })
                    print(f"[TELEOP] Segment committed ({args.primitive}, "
                          f"{len(state['current_seg'])} steps) — "
                          f"total segments: {len(state['all_demos'])}")
                state["current_seg"] = []
                state["recording"]   = False
                state["prev_q"]      = None

            elif event.input == carb.input.KeyboardInput.BACKSPACE:
                state["current_seg"] = []
                state["recording"]   = False
                state["prev_q"]      = None
                print("[TELEOP] Current segment discarded")

            elif event.input == carb.input.KeyboardInput.ESCAPE:
                state["should_quit"] = True

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            state["keys_held"].discard(event.input)
        return True

    appwindow = omni.appwindow.get_default_app_window()
    appwindow.get_keyboard().subscribe_to_keyboard_events(on_key)

    target_pos = franka.end_effector.get_world_pose()[0].copy()

    print(f"\n[TELEOP] arm={args.arm}  primitive={args.primitive}")
    print("[TELEOP] WASD+QE move | F gripper | R record | P commit segment | BkSp drop | ESC quit\n")

    while simulation_app.is_running() and not state["should_quit"]:
        for key, direction in KEY_DELTAS.items():
            if key in state["keys_held"]:
                target_pos = target_pos + direction * args.step

        q_seed = franka.get_joint_positions()[:7].copy()
        q_goal, ok = ik.solve(target_pos, q_seed=q_seed)
        if ok:
            q_dot = np.clip(
                -3.0 * (q_seed - q_goal),
                -cfg["training"]["max_joint_vel"],
                cfg["training"]["max_joint_vel"],
            )
            q_cmd = franka.get_joint_positions().copy()
            q_cmd[:7] = q_seed + q_dot * physics_dt
            franka.apply_action(ArticulationAction(joint_positions=q_cmd))
        env.step(render=True)

        if state["recording"]:
            q = franka.get_joint_positions()[:7].copy()
            if state["prev_q"] is None:
                state["prev_q"] = q.copy()
            q_dot = (q - state["prev_q"]) / physics_dt
            state["current_seg"].append({
                "q":         q,
                "q_dot":     q_dot,
                "ee_pos":    franka.end_effector.get_world_pose()[0].copy(),
                "primitive": args.primitive,
                "block":     "unknown",
                "arm":       args.arm,
                "target":    target_pos.copy(),
                # q_goal filled in retroactively when P is pressed
            })
            state["prev_q"] = q

    with open(out_path, "wb") as f:
        pickle.dump(state["all_demos"], f)
    print(f"[TELEOP] Saved {len(state['all_demos'])} demos to {out_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
