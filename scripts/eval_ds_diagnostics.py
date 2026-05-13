"""
eval_ds_diagnostics.py
======================
Offline diagnostic script for trained Neural DS checkpoints.
Computes all metrics in Table 2 of the paper and writes them to JSON + CSV.

Does NOT require Isaac Sim — runs on the saved checkpoint files and
pre-collected demonstration data.

Metrics computed
----------------
  imitation_mse        : MSE of f_theta(e) vs q_dot_demo on training set
  stab_violation_rate  : fraction of training samples where dV/dt + alpha*V > 0
  f_at_zero            : ||f_theta(0)|| (structural guarantee check)
  rollout_conv_rate     : fraction of N_ROLLOUT random ICs that converge
  rollout_median_time  : median time to convergence in rollout simulations

Rollout convergence is tested in a pure DS integration loop (no Isaac Sim).
Error norms are tracked until below done_tol or until timeout.

Usage
-----
  python scripts/eval_ds_diagnostics.py \
      --ckpt_dir data/checkpoints \
      --demo_dir data/demonstrations \
      --out_dir  data/results \
      --arms left right \
      --primitives reach transport \
      --n_rollouts 50 \
      --rollout_steps 600 \
      --use_safe

Outputs
-------
  data/results/ds_diagnostics_<timestamp>.json
  data/results/ds_diagnostics_<timestamp>.csv
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    print("[ERROR] PyTorch not found. Run: pip install torch")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class DSCheckpointDiagnostics:
    model_type: str
    arm: str
    primitive: str
    # Table 2 metrics
    imitation_mse: float = float("nan")
    stab_violation_rate: float = float("nan")
    f_at_zero_norm: float = float("nan")
    rollout_conv_rate: float = float("nan")
    rollout_median_conv_time: float = float("nan")
    # Extra diagnostics (not in paper table but useful)
    imitation_mae: float = float("nan")
    median_vel_error: float = float("nan")
    n_train_samples: int = 0
    n_rollouts: int = 0
    physics_dt: float = float("nan")
    done_tol: float = float("nan")


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str) -> Dict:
    """Load a Neural DS checkpoint (as saved by train_ds.py)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt


def build_neural_ds(ckpt: Dict):
    """Reconstruct the Neural DS model from a checkpoint dict.

    Current `train_ds.py` checkpoints save a `StableNeuralDS` state under
    `state_dict`, plus array-valued `state_mean`, `state_std`, and `vel_scale`.
    Older checkpoints used `model_state_dict` and scalar normalization. Support
    both so archived runs remain inspectable.
    """
    # Add src/ to path so we can import without installing
    src_dir = Path(__file__).parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from neural_ds import NeuralDS, StableNeuralDS, N_JOINTS

    model_cfg = ckpt.get("config", ckpt.get("model_config", {}))
    hidden_dim = model_cfg.get("hidden_dim", 128)
    stable_skip_gain = model_cfg.get("stable_skip_gain", 0.0)
    alpha = model_cfg.get("alpha", 1.0)
    lyap_hidden = model_cfg.get("lyapunov_hidden", 64)
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict"))
    if state_dict is None:
        raise KeyError("checkpoint has neither 'state_dict' nor 'model_state_dict'")

    if any(k.startswith("f.") or k.startswith("V.") for k in state_dict):
        model = StableNeuralDS(
            n_joints=ckpt.get("n_joints", N_JOINTS),
            hidden_dim=hidden_dim,
            lyap_hidden=lyap_hidden,
            alpha=alpha,
            stable_skip_gain=stable_skip_gain,
        )
    else:
        model = NeuralDS(
            state_dim=N_JOINTS,
            hidden_dim=hidden_dim,
            stable_skip_gain=stable_skip_gain,
        )
    model.eval()
    model.load_state_dict(state_dict)

    state_mean = np.asarray(ckpt.get("state_mean", np.zeros(N_JOINTS)), dtype=np.float32)
    state_std = np.asarray(ckpt.get("state_std", np.ones(N_JOINTS)), dtype=np.float32)
    vel_scale = np.asarray(ckpt.get("vel_scale", np.ones(N_JOINTS)), dtype=np.float32)
    if state_mean.ndim == 0:
        state_mean = np.full(N_JOINTS, float(state_mean), dtype=np.float32)
    if state_std.ndim == 0:
        state_std = np.full(N_JOINTS, float(state_std), dtype=np.float32)
    if vel_scale.ndim == 0:
        vel_scale = np.full(N_JOINTS, float(vel_scale), dtype=np.float32)

    return model, state_mean, state_std, vel_scale, float(alpha)


def iter_demo_steps(demos):
    """Yield step dicts from current and legacy demo pickle layouts."""
    for demo in demos:
        if isinstance(demo, dict) and "trajectory" in demo:
            yield from demo["trajectory"]
        elif isinstance(demo, list):
            yield from demo


# ---------------------------------------------------------------------------
# Training-set metrics
# ---------------------------------------------------------------------------

def compute_training_metrics(
    model,
    demo_pkl: str,
    arm: str,
    primitive: str,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    vel_scale: np.ndarray,
    alpha: float = 1.0,
    device: str = "cpu",
) -> Dict:
    """
    Load demonstrations and compute imitation MSE, stability violation rate,
    and f(0) norm on the full training set.
    """
    import pickle

    with open(demo_pkl, "rb") as f:
        demos = pickle.load(f)

    samples = [
        s for s in iter_demo_steps(demos)
        if s.get("arm") == arm and s.get("primitive") == primitive
    ]

    if not samples:
        print(f"  [WARN] No samples found for arm={arm} primitive={primitive} in {demo_pkl}")
        return {}

    q      = np.array([s["q"]      for s in samples], dtype=np.float32)
    q_dot  = np.array([s["q_dot"]  for s in samples], dtype=np.float32)
    q_goal = np.array([s["q_goal"] for s in samples], dtype=np.float32)

    e       = q - q_goal
    e_n     = (e - state_mean) / state_std
    q_dot_n = q_dot / vel_scale

    e_n_t     = torch.tensor(e_n,     dtype=torch.float32, device=device)
    q_dot_n_t = torch.tensor(q_dot_n, dtype=torch.float32, device=device)
    q_dot_t   = torch.tensor(q_dot,   dtype=torch.float32, device=device)
    vel_scale_t = torch.tensor(vel_scale, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)
    scale_factor = vel_scale_t / state_std_t

    with torch.no_grad():
        f_pred_n = model(e_n_t)           # (N, 7) normalised velocity

        # Imitation loss
        err_phys = f_pred_n * vel_scale_t - q_dot_t
        mse_phys = float(torch.mean(err_phys ** 2).item())
        mae_phys = float(torch.mean(torch.abs(err_phys)).item())
        med_vel = float(torch.median(torch.norm(err_phys, dim=1)).item())

        # f(0) structural check
        zero_input = torch.zeros(1, 7, device=device)
        f_zero     = model(zero_input)
        f_zero_norm = float(torch.norm(f_zero * vel_scale_t).item())

    # Stability needs autograd through V. Use the same scale factor as training:
    # dx_n/dt = q_dot / state_std = f_n * vel_scale / state_std.
    x_grad = e_n_t.detach().clone().requires_grad_(True)
    V_val = model.V(x_grad) if hasattr(model, "V") else torch.sum(x_grad ** 2, dim=1)
    grad = torch.autograd.grad(V_val.sum(), x_grad)[0]
    with torch.no_grad():
        f_pred_n = model(x_grad)
        dV_dt = torch.sum((grad * scale_factor) * f_pred_n, dim=1)
        violated = (dV_dt + alpha * V_val) > 0.0
        stab_viol_rate = float(violated.float().mean().item())

    return {
        "imitation_mse":         mse_phys,
        "imitation_mae":         mae_phys,
        "median_vel_error":      med_vel,
        "stab_violation_rate":   stab_viol_rate,
        "f_at_zero_norm":        f_zero_norm,
        "n_train_samples":       len(samples),
    }


# ---------------------------------------------------------------------------
# Rollout convergence
# ---------------------------------------------------------------------------

def rollout_ds(
    model,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    vel_scale: np.ndarray,
    n_rollouts: int = 50,
    n_steps: int = 600,
    physics_dt: float = 1 / 60,
    done_tol: float = 0.05,
    init_error_scale: float = 1.5,
    use_safe: bool = False,
    alpha: float = 1.0,
    device: str = "cpu",
) -> Dict:
    """
    Simulate N_ROLLOUTS integration trajectories from random initial errors.
    Returns convergence rate and median convergence time.
    """
    conv_count = 0
    conv_times = []
    state_mean_t = torch.tensor(state_mean, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(state_std, dtype=torch.float32, device=device)
    vel_scale_t = torch.tensor(vel_scale, dtype=torch.float32, device=device)
    scale_factor_t = (vel_scale_t / state_std_t).unsqueeze(0)

    for _ in range(n_rollouts):
        # Random initial error within ±init_error_scale * state_std per joint
        e = (
            np.random.uniform(-init_error_scale, init_error_scale, size=(7,))
            .astype(np.float32)
            * state_std
        )

        conv_time = float("nan")
        for step in range(n_steps):
            e_t = torch.tensor(e, dtype=torch.float32, device=device)
            e_n_t = ((e_t - state_mean_t) / state_std_t).unsqueeze(0)
            with torch.no_grad():
                if use_safe:
                    if hasattr(model, "safe_velocity"):
                        old_alpha = model.alpha
                        model.alpha = alpha
                        f_n = model.safe_velocity(
                            e_n_t, scale_factor=scale_factor_t
                        ).squeeze(0)
                        model.alpha = old_alpha
                    else:
                        # Legacy fallback for old NeuralDS-only checkpoints.
                        f_n = model(e_n_t).squeeze(0)
                        e_n = e_n_t.squeeze(0)
                        v = torch.dot(e_n, e_n)
                        dv_dt = 2.0 * torch.dot(e_n, f_n)
                        violation = dv_dt + alpha * v
                        if violation > 0:
                            grad_v = 2.0 * e_n
                            correction = violation / (
                                torch.dot(grad_v, grad_v) + 1e-8
                            )
                            f_n = f_n - correction * grad_v
                    q_dot_n = f_n.cpu().numpy()
                else:
                    q_dot_n = model(e_n_t).squeeze(0).cpu().numpy()

            q_dot = q_dot_n * vel_scale
            e = e + q_dot * physics_dt       # simple Euler integration

            if np.linalg.norm(e) < done_tol:
                conv_time = (step + 1) * physics_dt
                break

        if not math.isnan(conv_time):
            conv_count += 1
            conv_times.append(conv_time)

    conv_rate   = conv_count / n_rollouts
    median_time = float(np.median(conv_times)) if conv_times else float("nan")
    return {
        "rollout_conv_rate":          conv_rate,
        "rollout_median_conv_time":   median_time,
    }


# ---------------------------------------------------------------------------
# Demo file discovery
# ---------------------------------------------------------------------------

def find_demo_pkl(demo_dir: str, arm: str) -> Optional[str]:
    """Return the path to the demo pickle for a given arm."""
    candidates = [
        Path(demo_dir) / f"{arm}_demos.pkl",
        Path(demo_dir) / f"demos_{arm}.pkl",
        Path(demo_dir) / f"{arm}.pkl",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Fallback: any pkl containing the arm name
    for p in Path(demo_dir).glob("*.pkl"):
        if arm in p.stem:
            return str(p)
    return None


def find_checkpoint(ckpt_dir: str, arm: str, primitive: str) -> Optional[str]:
    """Return the path to the checkpoint for a given (arm, primitive)."""
    candidates = [
        Path(ckpt_dir) / f"{arm}_{primitive}.pt",
        Path(ckpt_dir) / f"{primitive}_{arm}.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def write_diagnostics(results: List[DSCheckpointDiagnostics], out_dir: Path):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"ds_diagnostics_{ts}.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    csv_path = out_dir / f"ds_diagnostics_{ts}.csv"
    fields = list(DSCheckpointDiagnostics.__dataclass_fields__.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print(f"\nDiagnostics written to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


def print_diagnostics_table(results: List[DSCheckpointDiagnostics]):
    header = (
        f"{'Model':<10} {'Arm':<6} {'Primitive':<12} "
        f"{'Imit.MSE':>10} {'StabViol%':>10} "
        f"{'f(0)norm':>10} {'ConvRate':>9} {'MedConvT':>9}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        sv = f"{r.stab_violation_rate*100:.1f}%" if not math.isnan(r.stab_violation_rate) else "  —  "
        mse = f"{r.imitation_mse:.5f}" if not math.isnan(r.imitation_mse) else "  —  "
        f0  = f"{r.f_at_zero_norm:.2e}" if not math.isnan(r.f_at_zero_norm) else "  —  "
        cr  = f"{r.rollout_conv_rate:.2f}" if not math.isnan(r.rollout_conv_rate) else "  —  "
        mt  = f"{r.rollout_median_conv_time:.2f}s" if not math.isnan(r.rollout_median_conv_time) else "  —  "
        print(
            f"{r.model_type:<10} {r.arm:<6} {r.primitive:<12} "
            f"{mse:>10} {sv:>10} {f0:>10} {cr:>9} {mt:>9}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline Neural DS diagnostic metrics (Table 2)."
    )
    p.add_argument("--ckpt_dir",   default="data/checkpoints")
    p.add_argument("--demo_dir",   default="data/demonstrations")
    p.add_argument("--out_dir",    default="data/results")
    p.add_argument("--arms",       nargs="+", default=["left", "right"])
    p.add_argument("--primitives", nargs="+", default=["reach", "transport"])
    p.add_argument("--n_rollouts",    type=int,   default=50)
    p.add_argument("--rollout_steps", type=int,   default=600)
    p.add_argument("--rollout_dt",    type=float, default=1/60)
    p.add_argument("--done_tol",      type=float, default=0.05)
    p.add_argument("--init_error_scale", type=float, default=1.5)
    p.add_argument("--alpha",      type=float, default=1.0)
    p.add_argument("--use_safe",   action="store_true",
                   help="Apply Lyapunov projection during rollout simulations")
    p.add_argument("--device",     default="cpu")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()
    np.random.seed(0)
    torch.manual_seed(0)

    results: List[DSCheckpointDiagnostics] = []

    for arm in args.arms:
        demo_pkl = find_demo_pkl(args.demo_dir, arm)
        if demo_pkl is None:
            print(f"[WARN] No demo file found for arm={arm} in {args.demo_dir}")

        for primitive in args.primitives:
            ckpt_path = find_checkpoint(args.ckpt_dir, arm, primitive)
            if ckpt_path is None:
                print(f"[SKIP] No checkpoint for {arm}/{primitive} in {args.ckpt_dir}")
                continue

            print(f"\n── {arm} / {primitive} ──────────────────────────────")
            print(f"  checkpoint : {ckpt_path}")
            if demo_pkl:
                print(f"  demos      : {demo_pkl}")

            diag = DSCheckpointDiagnostics(
                model_type="neural",
                arm=arm,
                primitive=primitive,
                n_rollouts=args.n_rollouts,
                physics_dt=args.rollout_dt,
                done_tol=args.done_tol,
            )

            # Load model
            ckpt = load_checkpoint(ckpt_path)
            model, state_mean, state_std, vel_scale, ckpt_alpha = build_neural_ds(ckpt)
            model = model.to(args.device)
            alpha = args.alpha if args.alpha is not None else ckpt_alpha

            # Training-set metrics (requires demos)
            if demo_pkl and os.path.isfile(demo_pkl):
                tm = compute_training_metrics(
                    model=model,
                    demo_pkl=demo_pkl,
                    arm=arm,
                    primitive=primitive,
                    state_mean=state_mean,
                    state_std=state_std,
                    vel_scale=vel_scale,
                    alpha=alpha,
                    device=args.device,
                )
                diag.imitation_mse        = tm.get("imitation_mse",       float("nan"))
                diag.imitation_mae        = tm.get("imitation_mae",        float("nan"))
                diag.median_vel_error     = tm.get("median_vel_error",     float("nan"))
                diag.stab_violation_rate  = tm.get("stab_violation_rate",  float("nan"))
                diag.f_at_zero_norm       = tm.get("f_at_zero_norm",       float("nan"))
                diag.n_train_samples      = tm.get("n_train_samples",      0)
                print(f"  samples    : {diag.n_train_samples}")
                print(f"  imit. MSE  : {diag.imitation_mse:.6f}")
                print(f"  stab viol  : {diag.stab_violation_rate*100:.1f}%")
                print(f"  ||f(0)||   : {diag.f_at_zero_norm:.2e}")
            else:
                print("  [SKIP] training-set metrics (no demo file)")

            # Rollout convergence (offline, no Isaac Sim needed)
            print(f"  running {args.n_rollouts} rollouts ...")
            rc = rollout_ds(
                model=model,
                state_mean=state_mean,
                state_std=state_std,
                vel_scale=vel_scale,
                n_rollouts=args.n_rollouts,
                n_steps=args.rollout_steps,
                physics_dt=args.rollout_dt,
                done_tol=args.done_tol,
                init_error_scale=args.init_error_scale,
                use_safe=args.use_safe,
                alpha=alpha,
                device=args.device,
            )
            diag.rollout_conv_rate        = rc["rollout_conv_rate"]
            diag.rollout_median_conv_time = rc["rollout_median_conv_time"]
            print(f"  conv. rate : {diag.rollout_conv_rate:.2f}")
            print(f"  median t   : {diag.rollout_median_conv_time:.2f}s")

            results.append(diag)

    if not results:
        print("[ERROR] No results computed. Check --ckpt_dir and --demo_dir.")
        sys.exit(1)

    print_diagnostics_table(results)
    write_diagnostics(results, Path(args.out_dir))


if __name__ == "__main__":
    main()
