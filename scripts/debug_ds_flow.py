"""
Offline diagnostics for a trained joint-space DS checkpoint.

This does not require Isaac Sim. It checks whether a learned primitive velocity
field points toward its attractor on:
  1. the demo states used for that primitive, and
  2. random states around the training error distribution.

Example:
  python scripts/debug_ds_flow.py data/checkpoints/both_transport.pt \
      --demo data/demonstrations/left_demos.pkl \
      --demo data/demonstrations/right_demos.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.neural_ds import StableNeuralDS, N_JOINTS


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = StableNeuralDS(
        n_joints=N_JOINTS,
        hidden_dim=cfg["hidden_dim"],
        lyap_hidden=cfg["lyapunov_hidden"],
        alpha=cfg["alpha"],
        stable_skip_gain=cfg.get("stable_skip_gain", 0.0),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return ckpt, model


def load_demo_states(demo_paths, primitive):
    states = []
    velocities = []
    for path in demo_paths:
        with open(path, "rb") as f:
            demos = pickle.load(f)
        for demo in demos:
            for step in demo["trajectory"]:
                if step["primitive"] != primitive:
                    continue
                states.append(np.asarray(step["q"]) - np.asarray(step["q_goal"]))
                velocities.append(np.asarray(step["q_dot"]))
    if not states:
        return None, None
    return np.stack(states), np.stack(velocities)


def velocity_stats(label, errors, velocities):
    e_norm = np.linalg.norm(errors, axis=1)
    v_norm = np.linalg.norm(velocities, axis=1)
    cos = -np.sum(errors * velocities, axis=1) / (e_norm * v_norm + 1e-9)
    edot = np.sum(errors * velocities, axis=1)
    print(label)
    print(
        "  cos(qdot,-e) mean/med/p10/p01 : "
        f"{cos.mean():+.3f} / {np.median(cos):+.3f} / "
        f"{np.quantile(cos, 0.10):+.3f} / {np.quantile(cos, 0.01):+.3f}"
    )
    print(f"  moving away fraction          : {np.mean(cos < 0.0):.3f}")
    print(f"  e_dot positive fraction       : {np.mean(edot > 0.0):.3f}")
    print(
        "  ||qdot|| median/p95/max       : "
        f"{np.median(v_norm):.3f} / {np.quantile(v_norm, 0.95):.3f} / {v_norm.max():.3f}"
    )


def model_velocity(model, errors, state_std, vel_scale, use_safe):
    x = torch.tensor(errors / state_std, dtype=torch.float32)
    with torch.no_grad():
        if use_safe:
            scale_factor = torch.tensor(
                vel_scale / state_std, dtype=torch.float32
            ).unsqueeze(0)
            v_n = model.safe_velocity(x, scale_factor=scale_factor)
        else:
            v_n = model(x)
    return v_n.numpy() * vel_scale


def random_field_check(model, state_std, vel_scale, spans, n):
    rng = np.random.default_rng(1)
    for span in spans:
        errors = rng.normal(size=(n, N_JOINTS)) * span * state_std
        for use_safe in (False, True):
            velocities = model_velocity(model, errors, state_std, vel_scale, use_safe)
            label = f"random span={span:g} {'safe' if use_safe else 'raw'}"
            velocity_stats(label, errors, velocities)


def rollout_check(model, state_std, vel_scale, n, steps, dt, max_joint_vel):
    rng = np.random.default_rng(2)
    for use_safe in (False, True):
        errors = rng.uniform(-1.0, 1.0, size=(n, N_JOINTS)) * 2.5 * state_std
        start_norm = np.linalg.norm(errors, axis=1)
        for _ in range(steps):
            velocities = model_velocity(model, errors, state_std, vel_scale, use_safe)
            velocities = np.clip(velocities, -max_joint_vel, max_joint_vel)
            errors = errors + velocities * dt
        end_norm = np.linalg.norm(errors, axis=1)
        label = "safe" if use_safe else "raw"
        print(f"rollout {label}")
        print(
            "  median ||e|| start -> end    : "
            f"{np.median(start_norm):.3f} -> {np.median(end_norm):.3f}"
        )
        print(f"  fraction grew                : {np.mean(end_norm > start_norm):.3f}")
        print(
            "  end ||e|| p95/max            : "
            f"{np.quantile(end_norm, 0.95):.3f} / {end_norm.max():.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--demo", type=Path, action="append", default=[])
    parser.add_argument("--primitive", type=str, default=None)
    parser.add_argument("--random_n", type=int, default=2000)
    parser.add_argument("--rollout_n", type=int, default=100)
    parser.add_argument("--rollout_steps", type=int, default=3000)
    parser.add_argument("--dt", type=float, default=0.00833)
    parser.add_argument("--max_joint_vel", type=float, default=1.5)
    args = parser.parse_args()

    ckpt, model = load_model(args.checkpoint)
    primitive = args.primitive or ckpt.get("primitive")
    state_std = np.asarray(ckpt["state_std"], dtype=float)
    vel_scale = np.asarray(ckpt["vel_scale"], dtype=float)

    print(f"checkpoint : {args.checkpoint}")
    print(f"primitive  : {primitive}")
    print(f"state_std  : {np.round(state_std, 4)}")
    print(f"vel_scale  : {np.round(vel_scale, 4)}")
    print(f"config     : {ckpt['config']}")

    if args.demo:
        errors, demo_velocities = load_demo_states(args.demo, primitive)
        if errors is not None:
            velocity_stats("demo velocities on demo states", errors, demo_velocities)
            velocity_stats(
                "raw model on demo states",
                errors,
                model_velocity(model, errors, state_std, vel_scale, use_safe=False),
            )
            velocity_stats(
                "safe model on demo states",
                errors,
                model_velocity(model, errors, state_std, vel_scale, use_safe=True),
            )

    random_field_check(model, state_std, vel_scale, spans=(0.5, 1.0, 2.0), n=args.random_n)
    rollout_check(
        model,
        state_std,
        vel_scale,
        n=args.rollout_n,
        steps=args.rollout_steps,
        dt=args.dt,
        max_joint_vel=args.max_joint_vel,
    )


if __name__ == "__main__":
    main()
