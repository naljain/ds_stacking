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
    """Concatenate (state=[q, q_goal], q_dot) pairs for one primitive."""
    states = []
    velocities = []
    for path in demo_paths:
        with open(path, "rb") as f:
            demos = pickle.load(f)
        for demo in demos:
            for step in demo["trajectory"]:
                if step["primitive"] != primitive:
                    continue
                x = np.concatenate([step["q"], step["q_goal"]])
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

    # ── Normalisation per joint ───────────────────────────────────────────────
    state_mean = states.mean(0)
    state_std  = states.std(0) + 1e-6
    vel_scale  = np.maximum(np.abs(velocities).max(axis=0), 1e-3)  # shape (7,)

    states_n     = (states - state_mean) / state_std
    velocities_n = velocities / vel_scale

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

    optim = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    best = float("inf")
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.arm}_{args.primitive}.pt"

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
            }, ckpt_path)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d} | total {avg['total']:.5f}  "
                  f"imit {avg['imit']:.5f}  stab {avg['stab']:.5f}")

    print(f"[TRAIN] Best loss {best:.5f}, checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
