"""
Train a joint-space Neural DS + Lyapunov network for one primitive.

Loads demos from data/demonstrations/{arm}_demos.pkl, filters to the requested
primitive, forms (state, q_dot) pairs where state = q - q_goal and q_dot is
the demonstrated joint velocity.

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
from src.primitives import DS_PRIMITIVES


def load_trajectories(demo_paths, primitive):
    """Concatenate (state=q-q_goal, q_dot) pairs for one primitive.

    Also returns a manifest describing exactly which saved demonstrations
    contributed samples. This keeps the data split explicit: each DS checkpoint
    is trained from one primitive label, optionally pooled across arms.
    """
    states = []
    velocities = []
    manifest = {
        "primitive": primitive,
        "demo_files": [str(p) for p in demo_paths],
        "total_samples": 0,
        "per_file": {},
        "per_arm": {},
        "per_block": {},
        "per_stack_slot": {},
        "per_q_goal_source": {},
        "per_q_goal_seed_policy": {},
        "per_motion_source": {},
        "q_goal_lula_error": {},
    }
    for path in demo_paths:
        path_key = str(path)
        manifest["per_file"][path_key] = {
            "demos": 0,
            "samples": 0,
            "blocks": {},
        }
        with open(path, "rb") as f:
            demos = pickle.load(f)
        manifest["per_file"][path_key]["demos"] = len(demos)
        for demo in demos:
            for step in demo["trajectory"]:
                if step["primitive"] != primitive:
                    continue
                q = np.asarray(step["q"], dtype=float)
                q_goal = np.asarray(step["q_goal"], dtype=float)
                x = q - q_goal
                states.append(x)
                velocities.append(np.asarray(step["q_dot"], dtype=float))
                arm = step.get("arm", demo.get("arm", "unknown"))
                block = step.get("block", "unknown")
                stack_slot = step.get("stack_slot", "unknown")
                q_goal_source = step.get("q_goal_source", "legacy_unknown")
                q_goal_seed_policy = step.get("q_goal_seed_policy", "legacy_unknown")
                motion_source = step.get("motion_source", "legacy_unknown")
                manifest["total_samples"] += 1
                manifest["per_file"][path_key]["samples"] += 1
                manifest["per_arm"][arm] = manifest["per_arm"].get(arm, 0) + 1
                manifest["per_block"][block] = manifest["per_block"].get(block, 0) + 1
                manifest["per_stack_slot"][stack_slot] = (
                    manifest["per_stack_slot"].get(stack_slot, 0) + 1
                )
                manifest["per_q_goal_source"][q_goal_source] = (
                    manifest["per_q_goal_source"].get(q_goal_source, 0) + 1
                )
                manifest["per_q_goal_seed_policy"][q_goal_seed_policy] = (
                    manifest["per_q_goal_seed_policy"].get(q_goal_seed_policy, 0) + 1
                )
                manifest["per_motion_source"][motion_source] = (
                    manifest["per_motion_source"].get(motion_source, 0) + 1
                )
                if "q_goal_lula_error" in step:
                    err_stats = manifest["q_goal_lula_error"]
                    err_stats["count"] = err_stats.get("count", 0) + 1
                    err_stats["sum"] = err_stats.get("sum", 0.0) + float(step["q_goal_lula_error"])
                    err_stats["max"] = max(err_stats.get("max", 0.0),
                                           float(step["q_goal_lula_error"]))
                blocks = manifest["per_file"][path_key]["blocks"]
                blocks[block] = blocks.get(block, 0) + 1
    if not states:
        raise RuntimeError(f"No samples found for primitive '{primitive}' in {demo_paths}")
    return np.stack(states), np.stack(velocities), manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primitive", type=str, required=True,
                        choices=DS_PRIMITIVES)
    parser.add_argument("--arm", type=str, default="both",
                        choices=["left", "right", "both"])
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--demos",  type=str, default=None,
                        help="Path to a specific demos pkl (e.g. cleaned). "
                             "Overrides the default per-arm path.")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["training"]
    epochs = args.epochs or train_cfg["epochs"]
    device = torch.device(train_cfg["device"] if torch.cuda.is_available() else "cpu")

    demos_dir = Path(cfg["paths"]["demos"])
    if args.demos:
        demo_paths = [Path(args.demos)]
    elif args.arm == "both":
        demo_paths = [demos_dir / "left_demos.pkl", demos_dir / "right_demos.pkl"]
    else:
        demo_paths = [demos_dir / f"{args.arm}_demos.pkl"]

    print(f"[TRAIN] Loading {args.primitive} samples from {demo_paths}")
    states, velocities, data_manifest = load_trajectories(demo_paths, args.primitive)
    print(f"[TRAIN] Got {len(states)} samples, state dim {states.shape[1]}")
    print("[TRAIN] -- data partition used for this DS --")
    print(f"[TRAIN]   checkpoint           : {args.arm}_{args.primitive}.pt")
    print(f"[TRAIN]   primitive label      : {args.primitive}")
    print(f"[TRAIN]   source demo files    : {data_manifest['demo_files']}")
    print(f"[TRAIN]   samples by arm       : {data_manifest['per_arm']}")
    print(f"[TRAIN]   samples by block     : {data_manifest['per_block']}")
    print(f"[TRAIN]   samples by stack slot: {data_manifest['per_stack_slot']}")
    print(f"[TRAIN]   q_goal source counts : {data_manifest['per_q_goal_source']}")
    print(f"[TRAIN]   q_goal seed policies : {data_manifest['per_q_goal_seed_policy']}")
    print(f"[TRAIN]   motion source counts : {data_manifest['per_motion_source']}")
    if data_manifest["q_goal_lula_error"].get("count", 0):
        err = data_manifest["q_goal_lula_error"]
        print("[TRAIN]   Lula-settled mismatch: "
              f"mean={err['sum'] / err['count']:.3f}, max={err['max']:.3f}")

    max_v = train_cfg.get("max_joint_vel")
    if max_v is not None:
        n_clipped = int((np.abs(velocities) > max_v).sum())
        if n_clipped > 0:
            print(f"[TRAIN] clipping {n_clipped} velocity components above {max_v} rad/s")
        velocities = np.clip(velocities, -max_v, max_v)

    # ── Normalisation per joint ───────────────────────────────────────────────
    state_mean = np.zeros(N_JOINTS)
    state_std  = np.maximum(states.std(0), 0.1) + 1e-6
    vel_scale  = np.maximum(np.abs(velocities).max(axis=0), 1e-3)

    states_n     = (states - state_mean) / state_std
    velocities_n = velocities / vel_scale

    print(f"[TRAIN] -- data stats for primitive '{args.primitive}' --")
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
        stable_skip_gain = train_cfg.get("stable_skip_gain", 0.0),
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
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
                "data_manifest": data_manifest,
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
