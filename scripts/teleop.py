"""
Lightweight keyboard teleoperation for manual demonstration collection.

Controls:
  W/S       — move EE in +Y / -Y
  A/D       — move EE in -X / +X
  Q/E       — move EE in +Z / -Z
  F         — toggle gripper
  R         — start/stop recording the current demo
  P         — push current demo into the buffer (then start a new one)
  Backspace — discard current demo
  ESC       — save all demos to disk and quit

Recorded format matches collect_ik.py so downstream training is identical.

Usage:
  python scripts/teleop.py --arm left
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
    parser.add_argument("--arm",    type=str, default="left", choices=["left", "right"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--step",   type=float, default=0.005, help="metres per key tick")
    args = parser.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

    import carb
    import omni.appwindow

    from src.env import DualArmEnv
    from src.ik_controller import IKController

    env = DualArmEnv(config_path=args.config, arms=(args.arm,))
    cfg = env.cfg
    franka = env.frankas[args.arm]
    ik = IKController(franka)
    ik.set_gripper(open=True)

    out_dir = Path(cfg["paths"]["demos"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arm}_teleop_demos.pkl"

    # Mutable state — needs nonlocal-style sharing with key callback
    state = {
        "gripper_open": True,
        "recording":    False,
        "current_demo": [],
        "all_demos":    [],
        "keys_held":    set(),
        "should_quit":  False,
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
                ik.set_gripper(open=state["gripper_open"])
                print(f"[TELEOP] Gripper {'open' if state['gripper_open'] else 'closed'}")

            elif event.input == carb.input.KeyboardInput.R:
                state["recording"] = not state["recording"]
                if state["recording"]:
                    state["current_demo"] = []
                    print("[TELEOP] Recording started")
                else:
                    print(f"[TELEOP] Recording paused — {len(state['current_demo'])} steps so far")

            elif event.input == carb.input.KeyboardInput.P:
                if state["current_demo"]:
                    state["all_demos"].append({
                        "arm": args.arm,
                        "demo_idx": len(state["all_demos"]),
                        "trajectory": state["current_demo"],
                        "success": True,
                    })
                    print(f"[TELEOP] Demo committed — total: {len(state['all_demos'])}")
                state["current_demo"] = []
                state["recording"] = False

            elif event.input == carb.input.KeyboardInput.BACKSPACE:
                state["current_demo"] = []
                state["recording"]    = False
                print("[TELEOP] Current demo discarded")

            elif event.input == carb.input.KeyboardInput.ESCAPE:
                state["should_quit"] = True

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            state["keys_held"].discard(event.input)
        return True

    appwindow = omni.appwindow.get_default_app_window()
    appwindow.get_keyboard().subscribe_to_keyboard_events(on_key)

    target_pos = franka.end_effector.get_world_pose()[0].copy()
    prev_ee_pos = target_pos.copy()

    print("\n[TELEOP] Controls: WASD + QE move | F gripper | R record | P commit | BkSp drop | ESC quit\n")

    while simulation_app.is_running() and not state["should_quit"]:
        # Apply held-key deltas to target
        for key, direction in KEY_DELTAS.items():
            if key in state["keys_held"]:
                target_pos = target_pos + direction * args.step

        ee_pos, ee_rot = franka.end_effector.get_world_pose()

        if state["recording"]:
            ee_vel = (ee_pos - prev_ee_pos) * (1.0 / cfg["sim"]["physics_dt"])
            state["current_demo"].append({
                "ee_pos":    ee_pos.copy(),
                "ee_vel":    ee_vel.copy(),
                "ee_rot":    ee_rot.copy(),
                "joints":    franka.get_joint_positions().copy(),
                "block_pos": np.zeros(3),    # teleop sequences don't have a single "active block" — caller can post-label
                "goal_pos":  target_pos.copy(),
                "primitive": "teleop",
                "block":     "unknown",
                "arm":       args.arm,
                "gripper":   0.04 if state["gripper_open"] else 0.0,
            })
        prev_ee_pos = ee_pos.copy()

        ik.step_to(target_pos)
        env.step(render=True)

    with open(out_path, "wb") as f:
        pickle.dump(state["all_demos"], f)
    print(f"[TELEOP] Saved {len(state['all_demos'])} demos to {out_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
