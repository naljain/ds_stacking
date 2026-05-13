"""
Extract post-training Neural DS diagnostics from saved checkpoints.

This is intentionally narrow: it reproduces the "post-training quality on
training set" block from scripts/train_ds.py without retraining.

Example:
  python scripts/extract_training_diagnostics.py \
    --ckpt_dir data/checkpoints \
    --demo_dir data/demonstrations \
    --out_dir data/results \
    --arms left right \
    --primitives reach transport
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_ds import load_trajectories
from src.neural_ds import StableNeuralDS, N_JOINTS


@dataclass
class TrainingDiagnostics:
    arm: str
    primitive: str
    checkpoint: str
    train_samples: int
    val_samples: int
    best_epoch: int
    best_metric: str
    best_value: float
    imitation_mse_norm: float
    per_step_v_err_median_norm: float
    per_step_v_err_p95_norm: float
    physical_v_err_median_rad_s: float
    physical_v_err_p95_rad_s: float
    V_at_zero: float
    f_at_zero_norm: float
    stability_violation_pct: float
    final_val_imitation_mse_norm: float


def load_model(ckpt, device):
    cfg = ckpt["config"]
    model = StableNeuralDS(
        n_joints=N_JOINTS,
        hidden_dim=cfg["hidden_dim"],
        lyap_hidden=cfg["lyapunov_hidden"],
        alpha=cfg["alpha"],
        stable_skip_gain=cfg.get("stable_skip_gain", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def demo_paths_for(args, ckpt, arm):
    manifest = ckpt.get("data_manifest", {})
    paths = manifest.get("demo_files")
    if paths:
        return [Path(p) for p in paths]
    if arm == "both":
        return [
            Path(args.demo_dir) / "left_demos.pkl",
            Path(args.demo_dir) / "right_demos.pkl",
        ]
    return [Path(args.demo_dir) / f"{arm}_demos.pkl"]


def compute_one(args, arm, primitive, device):
    ckpt_path = Path(args.ckpt_dir) / f"{arm}_{primitive}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = load_model(ckpt, device)
    demo_paths = demo_paths_for(args, ckpt, arm)

    states, velocities, val_states, val_velocities, _ = load_trajectories(
        demo_paths, primitive
    )
    max_v = ckpt["config"].get("max_joint_vel", 1.5)
    velocities = np.clip(velocities, -max_v, max_v)
    if len(val_velocities):
        val_velocities = np.clip(val_velocities, -max_v, max_v)

    state_mean = np.asarray(ckpt["state_mean"])
    state_std = np.asarray(ckpt["state_std"])
    vel_scale = np.asarray(ckpt["vel_scale"])

    states_n = (states - state_mean) / state_std
    velocities_n = velocities / vel_scale
    x = torch.tensor(states_n, dtype=torch.float32, device=device)
    v = torch.tensor(velocities_n, dtype=torch.float32, device=device)
    vel_scale_t = torch.tensor(vel_scale, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)
    scale_factor = vel_scale_t / state_std_t

    with torch.no_grad():
        pred = model(x)
        err = pred - v
        per_step_err = err.pow(2).sum(-1).sqrt()
        imit_mse = err.pow(2).mean().item()
        physical_err = (err * vel_scale_t).pow(2).sum(-1).sqrt()

    x_grad = x.clone().requires_grad_(True)
    V_val = model.V(x_grad)
    grad = torch.autograd.grad(V_val.sum(), x_grad)[0]
    with torch.no_grad():
        v_out = model.f(x)
        dV_dt = (grad * scale_factor * v_out).sum(-1)
        alpha = ckpt["config"].get("alpha", 1.0)
        violates = (dV_dt + alpha * V_val) > 0
        stab_violation_pct = 100.0 * violates.float().mean().item()

    zero = torch.zeros(1, N_JOINTS, device=device)
    with torch.no_grad():
        V0 = model.V(zero).item()
        f0 = model(zero).norm().item()

    if len(val_states):
        val_states_n = (val_states - state_mean) / state_std
        val_velocities_n = val_velocities / vel_scale
        x_val = torch.tensor(val_states_n, dtype=torch.float32, device=device)
        v_val = torch.tensor(val_velocities_n, dtype=torch.float32, device=device)
        with torch.no_grad():
            val_imit = (model(x_val) - v_val).pow(2).mean().item()
    else:
        val_imit = float("nan")

    return TrainingDiagnostics(
        arm=arm,
        primitive=primitive,
        checkpoint=str(ckpt_path),
        train_samples=int(len(states)),
        val_samples=int(len(val_states)),
        best_epoch=int(ckpt.get("best_epoch", -1)),
        best_metric=str(ckpt.get("best_metric", "")),
        best_value=float(ckpt.get("best_value", float("nan"))),
        imitation_mse_norm=float(imit_mse),
        per_step_v_err_median_norm=float(per_step_err.median().item()),
        per_step_v_err_p95_norm=float(per_step_err.quantile(0.95).item()),
        physical_v_err_median_rad_s=float(physical_err.median().item()),
        physical_v_err_p95_rad_s=float(physical_err.quantile(0.95).item()),
        V_at_zero=float(V0),
        f_at_zero_norm=float(f0),
        stability_violation_pct=float(stab_violation_pct),
        final_val_imitation_mse_norm=float(val_imit),
    )


def print_table(rows):
    header = (
        f"{'Arm':<6} {'Primitive':<10} {'N':>7} {'ValN':>6} "
        f"{'MSE':>10} {'MedErr':>10} {'P95Err':>10} "
        f"{'V0':>9} {'f0':>9} {'StabViol':>9} {'ValMSE':>10}"
    )
    print("\nPOST-TRAINING NEURAL DS DIAGNOSTICS (TRAINING SET)")
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for r in rows:
        print(
            f"{r.arm:<6} {r.primitive:<10} {r.train_samples:>7d} "
            f"{r.val_samples:>6d} {r.imitation_mse_norm:>10.6f} "
            f"{r.per_step_v_err_median_norm:>10.6f} "
            f"{r.per_step_v_err_p95_norm:>10.6f} "
            f"{r.V_at_zero:>9.2e} {r.f_at_zero_norm:>9.2e} "
            f"{r.stability_violation_pct:>8.2f}% "
            f"{r.final_val_imitation_mse_norm:>10.6f}"
        )
    print("=" * len(header))


def write_outputs(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"training_diagnostics_{ts}.json"
    csv_path = out_dir / f"training_diagnostics_{ts}.csv"
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(TrainingDiagnostics.__dataclass_fields__)
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    print("\nWrote:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", default="data/checkpoints")
    parser.add_argument("--demo_dir", default="data/demonstrations")
    parser.add_argument("--out_dir", default="data/results")
    parser.add_argument("--arms", nargs="+", default=["left", "right"])
    parser.add_argument("--primitives", nargs="+", default=["reach", "transport"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = []
    for arm in args.arms:
        for primitive in args.primitives:
            rows.append(compute_one(args, arm, primitive, device))
    print_table(rows)
    write_outputs(rows, Path(args.out_dir))


if __name__ == "__main__":
    main()
