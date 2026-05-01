"""
Train a joint-space Neural DS + Lyapunov network for one primitive.

Loads demos from data/demonstrations/{arm}_demos.pkl, filters to the requested
primitive, forms (state, q_dot) pairs where state = [q (7), q_goal (7)] and
q_dot is the demonstrated joint velocity.

Saves a checkpoint per (arm, primitive) at data/checkpoints/{arm}_{primitive}.pt.

Usage:
  python scripts/train_ds.py --primitive reach --arm both
"""

import os
import sys
import argparse
import pickle
import yaml
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.neural_ds import StableNeuralDS, total_loss, N_JOINTS


def load_trajectories(demo_paths, primitive):
    """Concatenate (state=q-q_goal, q_dot) pairs for one primitive."""
    states = []
    velocities = []
    for path in demo_paths:
        with open(path, "rb") as f:
            demos = pickle.load(f)
        for demo in demos:
            for step in demo["trajectory"]:
                if step["primitive"] != primitive:
                    continue
                x = step["q"] - step["q_goal"]  # error-based: 7-dim
                states.append(x)
                velocities.append(step["q_dot"])
    if not states:
        raise RuntimeError(f"No samples found for primitive '{primitive}' in {demo_paths}")
    return np.stack(states), np.stack(velocities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primitive", type=str, required=True,
                        choices=["reach", "grasp", "lift", "transport", "place"])
    parser.add_argument("--arm", type=str, default="both",
                        choices=["left", "right", "both"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["training"]
    epochs = args.epochs or train_cfg["epochs"]
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")

    demos_dir = Path(cfg["paths"]["demos"])
    if args.arm == "both":
        demo_paths = [demos_dir / "left_demos.pkl", demos_dir / "right_demos.pkl"]
    else:
        demo_paths = [demos_dir / f"{args.arm}_demos.pkl"]

    print(f"[TRAIN] Loading {args.primitive} samples from {demo_paths}")
    states, velocities = load_trajectories(demo_paths, args.primitive)
    print(f"[TRAIN] Got {len(states)} samples, state dim {states.shape[1]}")

    # Defensive clip — RMPflow respects max_joint_vel during collection, so
    # anything above this is a finite-difference artifact (e.g. across an
    # un-recorded primitive boundary). Letting one outlier through poisons
    # vel_scale and crushes the model's effective output resolution.
    max_v = train_cfg["max_joint_vel"]
    n_clipped = int((np.abs(velocities) > max_v).sum())
    if n_clipped > 0:
        print(f"[TRAIN] clipping {n_clipped} velocity components above {max_v} rad/s")
    velocities = np.clip(velocities, -max_v, max_v)

    # ── Normalisation per joint ───────────────────────────────────────────────
    # Zero-mean: goal (e=0) maps to x_n=0 so the -x skip connection points at it.
    # Floor on std: prevents OOD amplification for joints whose error barely
    # varies in training (e.g. wrist joints RMPflow holds in a fixed null-space).
    # Without this, a deployment q_goal differing by 0.05 rad in such a joint
    # produces x_n=10+ and saturates Tanh, destroying the velocity output.
    state_mean = np.zeros(N_JOINTS)
    state_std  = np.maximum(states.std(0), 0.1) + 1e-6
    vel_scale  = np.maximum(np.abs(velocities).max(axis=0), 1e-3)  # shape (7,)

    states_n     = (states - state_mean) / state_std
    velocities_n = velocities / vel_scale

    # Diagnostics — everything you'd want to know about the dataset before training
    print(f"[TRAIN] ── data stats for primitive '{args.primitive}' ──")
    print(f"[TRAIN]   |e| (rad)            : "
          f"min={np.abs(states).min(axis=0).round(3)}  "
          f"max={np.abs(states).max(axis=0).round(3)}")
    print(f"[TRAIN]   |q_dot| (rad/s)      : "
          f"min={np.abs(velocities).min(axis=0).round(3)}  "
          f"max={np.abs(velocities).max(axis=0).round(3)}")
    print(f"[TRAIN]   state_std            : {state_std.round(3)}")
    print(f"[TRAIN]   vel_scale            : {vel_scale.round(3)}")
    print(f"[TRAIN]   scale_factor (vs/ss) : {(vel_scale/state_std).round(3)}")
    print(f"[TRAIN]   |x_n| range          : "
          f"[{np.abs(states_n).min():.2f}, {np.abs(states_n).max():.2f}]")
    print(f"[TRAIN]   |v_n| range          : "
          f"[{np.abs(velocities_n).min():.2f}, {np.abs(velocities_n).max():.2f}]")

    X = torch.tensor(states_n, dtype=torch.float32)
    V = torch.tensor(velocities_n, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, V),
                        batch_size=train_cfg["batch_size"], shuffle=True)

    model = StableNeuralDS(
        n_joints    = N_JOINTS,
        hidden_dim  = train_cfg["hidden_dim"],
        lyap_hidden = train_cfg["lyapunov_hidden"],
        alpha       = train_cfg["alpha"],
    ).to(device)

    optim     = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    # Rescaling between model output (normalised by vel_scale) and actual
    # rate of change of x_n (normalised by state_std). Used so the stability
    # loss enforces dV/dt ≤ -αV on the real dynamics, not just on a dot
    # product in mismatched coordinates.
    scale_factor = torch.tensor(vel_scale / state_std,
                                dtype=torch.float32, device=device)

    best = float("inf")
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.arm}_{args.primitive}.pt"

    history = {"total": [], "imit": [], "stab": []}

    for epoch in range(epochs):
        running = {"total": 0.0, "imit": 0.0, "stab": 0.0, "n": 0}
        for x_batch, v_batch in loader:
            x_batch = x_batch.to(device)
            v_batch = v_batch.to(device)

            optim.zero_grad()
            loss, l_imit, l_stab = total_loss(
                model, x_batch, v_batch,
                alpha=train_cfg["alpha"],
                lambda_stab=train_cfg["lambda_stab"],
                scale_factor=scale_factor,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            n = len(x_batch)
            running["total"] += loss.item() * n
            running["imit"]  += l_imit * n
            running["stab"]  += l_stab * n
            running["n"]     += n

        avg = {k: running[k] / running["n"] for k in ("total", "imit", "stab")}
        for k in ("total", "imit", "stab"):
            history[k].append(avg[k])

        if avg["total"] < best:
            best = avg["total"]
            torch.save({
                "state_dict":  model.state_dict(),
                "state_mean":  state_mean,
                "state_std":   state_std,
                "vel_scale":   vel_scale,
                "primitive":   args.primitive,
                "arm":         args.arm,
                "config":      train_cfg,
                "n_joints":    N_JOINTS,
                "history":     history,
            }, ckpt_path)

        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d} | total {avg['total']:.5f}  "
                  f"imit {avg['imit']:.5f}  stab {avg['stab']:.5f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

    print(f"[TRAIN] Best loss {best:.5f}, checkpoint -> {ckpt_path}")

    # ── Post-training diagnostics ────────────────────────────────────────────
    # Reload the BEST checkpoint and evaluate on the full training set.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X_all = X.to(device)
    V_all = V.to(device)
    with torch.no_grad():
        pred = model(X_all)
        per_step_err = (pred - V_all).pow(2).sum(-1).sqrt()  # ||v_pred - v_demo||
        imit_mse = (pred - V_all).pow(2).mean().item()

    # Stability check: how often does dV/dt + αV exceed 0 on training data?
    x_grad = X_all.clone().requires_grad_(True)
    V_val  = model.V(x_grad)
    grad   = torch.autograd.grad(V_val.sum(), x_grad)[0]
    gV_eff = grad * scale_factor
    with torch.no_grad():
        v_out = model.f(X_all)
        dV_dt = (gV_eff * v_out).sum(-1)
        violates = (dV_dt + train_cfg["alpha"] * V_val) > 0
        stab_violation_rate = violates.float().mean().item()

    print(f"[TRAIN] ── post-training quality on training set ──")
    print(f"[TRAIN]   imitation MSE         : {imit_mse:.5f}")
    print(f"[TRAIN]   per-step ||v_err||    : "
          f"median={per_step_err.median().item():.4f}  "
          f"p95={per_step_err.quantile(0.95).item():.4f}")
    print(f"[TRAIN]   V(0)                  : "
          f"{model.V(torch.zeros(1, N_JOINTS, device=device)).item():.6f}  "
          f"(should be ~0)")
    print(f"[TRAIN]   ||f(0)||              : "
          f"{model(torch.zeros(1, N_JOINTS, device=device)).norm().item():.6f}  "
          f"(should be ~0)")
    print(f"[TRAIN]   stability violation %  : "
          f"{100 * stab_violation_rate:.2f}%  (lower = better)")


if __name__ == "__main__":
    main()
