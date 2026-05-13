"""
analyze_deployment_logs.py
==========================
Reads the deployment CSV logs produced by deploy_single_arm.py --log_csv
and the checkpoint plots produced by plot_ds.py, and computes every metric
cited in the paper.  No Isaac Sim required.

Expected inputs (from a normal deploy + plot run):
  data/results/left_ds_lula_scripted.csv   (or right_ds_lula_scripted.csv)
  data/results/ds_plots/left/01_loss.png       (existence check only)
  data/results/ds_plots/left/04_rollouts.png   (existence check only)
  data/checkpoints/left_reach.pt
  data/checkpoints/left_transport.pt

What this script produces
--------------------------
  data/results/analysis_<timestamp>.json   — all metrics, machine-readable
  data/results/analysis_<timestamp>.csv    — flat summary table
  data/results/fig_deploy_trace_<arm>.png  — per-primitive e_norm + cos trace
  data/results/fig_convergence_<arm>.png   — convergence waterfall per primitive

Usage
-----
  python scripts/analyze_deployment_logs.py \
      --logs data/results/left_ds_lula_scripted.csv \
             data/results/right_ds_lula_scripted.csv \
      --ckpt_dir data/checkpoints \
      --out_dir  data/results

The --logs flag accepts one or more CSV paths.  Each is parsed independently
and metrics are reported per arm/run.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not found — skipping figure output")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] torch not found — skipping checkpoint diagnostics")


# ── Primitive color palette (matches plot_ds.py) ───────────────────────────
PRIM_COLORS = {
    "reach":     "#4477bb",
    "grasp":     "#cc4444",
    "lift":      "#44aa44",
    "transport": "#dd8833",
    "place":     "#9944cc",
}
DS_PRIMITIVES = {"reach", "transport"}


# ══════════════════════════════════════════════════════════════════════════════
# CSV parsing
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> List[Dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def parse_rows(rows: List[Dict]) -> Dict:
    """
    Convert raw CSV rows into typed numpy arrays, split by primitive segment.
    Returns a dict with:
      steps, e_norm, V, qd_raw, qd, proj_delta, cos_to_goal, primitive (arrays)
      segments: list of {primitive, rows, steps, e_norm, cos_to_goal, ...}
    """
    steps      = np.array([int(r["step"])         for r in rows])
    e_norm     = np.array([float(r["e_norm"])      for r in rows])
    V          = np.array([float(r["V"])           for r in rows])
    qd_raw     = np.array([float(r["qd_raw_norm"]) for r in rows])
    qd         = np.array([float(r["qd_norm"])     for r in rows])
    proj_delta = np.array([float(r["proj_delta"])  for r in rows])
    cos_goal   = np.array([float(r["cos_to_goal"]) for r in rows])
    primitives = [r["primitive"] for r in rows]

    # Split into contiguous primitive segments
    segments = []
    cur_prim = primitives[0]
    seg_start = 0
    for i, p in enumerate(primitives + [None]):
        if p != cur_prim:
            mask = slice(seg_start, i)
            segments.append({
                "primitive":  cur_prim,
                "step_start": int(steps[seg_start]),
                "step_end":   int(steps[i - 1]),
                "n_steps":    i - seg_start,
                "e_norm":     e_norm[mask],
                "V":          V[mask],
                "qd_raw":     qd_raw[mask],
                "qd":         qd[mask],
                "proj_delta": proj_delta[mask],
                "cos_goal":   cos_goal[mask],
            })
            cur_prim = p
            seg_start = i

    return {
        "steps":      steps,
        "e_norm":     e_norm,
        "V":          V,
        "qd_raw":     qd_raw,
        "qd":         qd,
        "proj_delta": proj_delta,
        "cos_goal":   cos_goal,
        "primitives": primitives,
        "segments":   segments,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-segment metrics
# ══════════════════════════════════════════════════════════════════════════════

DONE_TOL = 0.05   # rad, matches deployment done_tol

def segment_metrics(seg: Dict, physics_dt: float = 1 / 60) -> Dict:
    """Compute convergence and quality metrics for one primitive segment."""
    e   = seg["e_norm"]
    cos = seg["cos_goal"]
    qd  = seg["qd"]
    pd  = seg["proj_delta"]
    n   = len(e)
    is_ds = seg["primitive"] in DS_PRIMITIVES

    # Convergence
    converged = bool(e[-1] < DONE_TOL)
    conv_step = next((i for i, v in enumerate(e) if v < DONE_TOL), None)
    conv_time = (conv_step + 1) * physics_dt if conv_step is not None else float("nan")

    # cos→goal statistics (only meaningful for DS primitives)
    cos_mean    = float(np.mean(cos))
    cos_neg_frac = float(np.mean(cos < 0.0))  # fraction pointing away from goal

    # Velocity statistics
    qd_mean = float(np.mean(qd))
    qd_max  = float(np.max(qd))

    # Safe projection activity (non-zero proj_delta means projection fired)
    proj_active_frac = float(np.mean(pd > 1e-6))
    proj_mean_mag    = float(np.mean(pd[pd > 1e-6])) if np.any(pd > 1e-6) else 0.0

    # Monotonicity of e_norm (good DS should decrease mostly monotonically)
    diffs = np.diff(e)
    mono_frac = float(np.mean(diffs <= 0.0))  # fraction of steps where e decreased

    return {
        "primitive":         seg["primitive"],
        "is_ds":             is_ds,
        "n_steps":           n,
        "duration_s":        n * physics_dt,
        "e_start":           float(e[0]),
        "e_end":             float(e[-1]),
        "e_min":             float(np.min(e)),
        "converged":         converged,
        "conv_time_s":       conv_time,
        "cos_mean":          cos_mean,
        "cos_neg_frac":      cos_neg_frac,
        "qd_mean":           qd_mean,
        "qd_max":            qd_max,
        "proj_active_frac":  proj_active_frac,
        "proj_mean_mag":     proj_mean_mag,
        "e_monotone_frac":   mono_frac,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Run-level summary
# ══════════════════════════════════════════════════════════════════════════════

def run_summary(parsed: Dict, label: str, physics_dt: float = 1 / 60) -> Dict:
    """Aggregate segment metrics into a single run summary."""
    seg_metrics = [segment_metrics(s, physics_dt) for s in parsed["segments"]]

    ds_segs  = [m for m in seg_metrics if m["is_ds"]]
    ik_segs  = [m for m in seg_metrics if not m["is_ds"]]
    all_segs = seg_metrics

    def _mean(lst, key):
        vals = [x[key] for x in lst if not (isinstance(x[key], float) and math.isnan(x[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    # DS-primitive convergence rate
    ds_conv_rate = sum(m["converged"] for m in ds_segs) / max(len(ds_segs), 1)

    # Mean cos→goal across all DS steps (directional quality of learned field)
    ds_cos_all = np.concatenate([
        parsed["segments"][i]["cos_goal"]
        for i, s in enumerate(parsed["segments"])
        if s["primitive"] in DS_PRIMITIVES
    ]) if ds_segs else np.array([])
    cos_mean_global = float(np.mean(ds_cos_all)) if len(ds_cos_all) > 0 else float("nan")
    cos_neg_global  = float(np.mean(ds_cos_all < 0.0)) if len(ds_cos_all) > 0 else float("nan")

    # Safe projection activity
    proj_all = np.concatenate([
        parsed["segments"][i]["proj_delta"]
        for i, s in enumerate(parsed["segments"])
        if s["primitive"] in DS_PRIMITIVES
    ]) if ds_segs else np.array([])
    proj_active_frac = float(np.mean(proj_all > 1e-6)) if len(proj_all) > 0 else float("nan")

    # Total run duration
    total_steps = int(parsed["steps"][-1] - parsed["steps"][0] + 1)

    # Primitive-level convergence table
    prim_table = {
        m["primitive"]: {
            "converged":    m["converged"],
            "conv_time_s":  m["conv_time_s"],
            "e_start":      m["e_start"],
            "e_end":        m["e_end"],
            "cos_mean":     m["cos_mean"] if m["is_ds"] else None,
            "n_steps":      m["n_steps"],
        }
        for m in all_segs
    }

    return {
        "label":               label,
        "total_steps":         total_steps,
        "total_duration_s":    total_steps * physics_dt,
        "n_ds_primitives":     len(ds_segs),
        "n_ik_primitives":     len(ik_segs),
        "ds_convergence_rate": ds_conv_rate,
        "ds_cos_mean":         cos_mean_global,
        "ds_cos_neg_frac":     cos_neg_global,
        "ds_proj_active_frac": proj_active_frac,
        "ds_mean_conv_time_s": _mean(ds_segs, "conv_time_s"),
        "ds_e_monotone_frac":  _mean(ds_segs, "e_monotone_frac"),
        "per_primitive":       prim_table,
        "segment_metrics":     seg_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint diagnostics (offline, no Isaac Sim)
# ══════════════════════════════════════════════════════════════════════════════

def checkpoint_diagnostics(ckpt_path: str, demo_path: Optional[str] = None) -> Dict:
    """
    Load a Neural DS checkpoint and compute:
      - ||f(0)||  (structural guarantee)
      - rollout convergence rate + median time
    Optionally, if a demo pickle is provided, compute training-set imitation MSE.
    """
    if not HAS_TORCH:
        return {"error": "torch not available"}

    src_dir = Path(ckpt_path).parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"error": str(e)}

    try:
        from neural_ds import NeuralDS
    except ImportError:
        return {"error": "could not import NeuralDS from src/"}

    model_cfg = ckpt.get("model_config", {})
    model = NeuralDS(
        state_dim=7,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        stable_skip_gain=model_cfg.get("stable_skip_gain", 0.5),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    state_std = float(ckpt.get("state_std", 1.0))
    vel_scale_raw = ckpt.get("vel_scale", 1.0)
    vel_scale = float(np.mean(vel_scale_raw)) if hasattr(vel_scale_raw, "__len__") else float(vel_scale_raw)

    results = {}

    # f(0) check
    with torch.no_grad():
        f0 = model(torch.zeros(1, 7))
        results["f_at_zero_norm"] = float(torch.norm(f0).item()) * vel_scale

    # Offline rollout convergence
    rng = np.random.default_rng(42)
    n_rollouts, n_steps, dt, done_tol = 100, 600, 1/60, 0.05
    conv_count = 0
    conv_times = []
    max_jv = float(ckpt.get("max_joint_vel", 2.0))

    for _ in range(n_rollouts):
        e = rng.uniform(-1.5, 1.5, size=(7,)).astype(np.float32) * state_std
        conv_t = float("nan")
        for step in range(n_steps):
            with torch.no_grad():
                e_n = torch.tensor(e / state_std, dtype=torch.float32).unsqueeze(0)
                qdot_n = model(e_n).squeeze(0).numpy()
            qdot = np.clip(qdot_n * vel_scale, -max_jv, max_jv)
            e = e + qdot * dt
            if np.linalg.norm(e) < done_tol:
                conv_t = (step + 1) * dt
                break
        if not math.isnan(conv_t):
            conv_count += 1
            conv_times.append(conv_t)

    results["rollout_conv_rate"]       = conv_count / n_rollouts
    results["rollout_median_conv_time"] = float(np.median(conv_times)) if conv_times else float("nan")
    results["state_std"]  = state_std
    results["vel_scale"]  = vel_scale

    # Training-set imitation MSE (if demo file provided)
    if demo_path and Path(demo_path).exists():
        try:
            import pickle
            with open(demo_path, "rb") as f:
                demos = pickle.load(f)

            primitive = Path(ckpt_path).stem.split("_", 1)[-1]   # e.g. left_reach -> reach
            arm       = Path(ckpt_path).stem.split("_")[0]

            samples = [s for d in demos for s in d
                       if s.get("arm") == arm and s.get("primitive") == primitive]
            if samples:
                q      = np.array([s["q"]      for s in samples], dtype=np.float32)
                q_dot  = np.array([s["q_dot"]  for s in samples], dtype=np.float32)
                q_goal = np.array([s["q_goal"] for s in samples], dtype=np.float32)
                e      = (q - q_goal) / state_std
                qdot_n = q_dot / vel_scale
                with torch.no_grad():
                    pred = model(torch.tensor(e)).numpy()
                mse = float(np.mean((pred - qdot_n) ** 2)) * (vel_scale ** 2)
                results["imitation_mse"] = mse
                results["n_train_samples"] = len(samples)

                # Stability violation rate
                dv_dt = 2.0 * np.sum(e * pred, axis=1)
                v     = np.sum(e ** 2, axis=1)
                alpha = 1.0
                viol  = np.mean((dv_dt + alpha * v) > 0.0)
                results["stab_violation_rate"] = float(viol)
        except Exception as ex:
            results["imitation_mse_error"] = str(ex)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def plot_deployment_trace(parsed: Dict, label: str, out_path: Path):
    """
    Three-panel figure:
      Top:    ||e|| over time, shaded by primitive, done_tol line
      Middle: cos→goal over time (DS primitives only)
      Bottom: ||q̇|| raw vs post-projection
    """
    if not HAS_MPL:
        return

    steps   = parsed["steps"]
    e_norm  = parsed["e_norm"]
    cos_g   = parsed["cos_goal"]
    qd_raw  = parsed["qd_raw"]
    qd      = parsed["qd"]
    prims   = parsed["primitives"]
    dt      = 1 / 60
    t       = steps * dt

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # ── Shade primitive regions ───────────────────────────────────────────
    def shade_prims(ax):
        cur = prims[0]; s = 0
        for i, p in enumerate(prims + [None]):
            if p != cur:
                col = PRIM_COLORS.get(cur, "#cccccc")
                ax.axvspan(t[s], t[min(i, len(t)-1)-1], alpha=0.12, color=col)
                cur = p; s = i if i < len(t) else len(t)-1

    # ── Panel 0: error norm ───────────────────────────────────────────────
    shade_prims(axes[0])
    axes[0].plot(t, e_norm, color="#333333", lw=0.8)
    axes[0].axhline(DONE_TOL, ls="--", color="red", alpha=0.7, label=f"done_tol={DONE_TOL}")
    axes[0].set_ylabel(r"$\|\mathbf{e}\|$ (rad)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.25)

    # ── Panel 1: cos→goal ─────────────────────────────────────────────────
    shade_prims(axes[1])
    # Mask IK steps (cos is 0 for IK, not meaningful)
    cos_masked = np.where(
        np.array([p in DS_PRIMITIVES for p in prims]), cos_g, np.nan
    )
    axes[1].plot(t, cos_masked, color="#1166cc", lw=0.7, alpha=0.85)
    axes[1].axhline(0.0, ls="--", color="orange", alpha=0.6, label="0 = perpendicular")
    axes[1].axhline(1.0, ls=":",  color="green",  alpha=0.5, label="1 = straight to goal")
    axes[1].set_ylim(-1.15, 1.15)
    axes[1].set_ylabel(r"$\cos(\dot{\mathbf{q}}, -\mathbf{e})$")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(True, alpha=0.25)

    # ── Panel 2: velocity norms ───────────────────────────────────────────
    shade_prims(axes[2])
    axes[2].plot(t, qd_raw, color="#888888", lw=0.6, alpha=0.7, label="raw DS")
    axes[2].plot(t, qd,     color="#cc4400", lw=0.8,             label="post-projection")
    axes[2].set_ylabel(r"$\|\dot{\mathbf{q}}\|$ (rad/s)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.25)

    # Primitive legend
    patches = [mpatches.Patch(color=PRIM_COLORS[p], alpha=0.5, label=p)
               for p in PRIM_COLORS]
    axes[0].legend(handles=patches + [
        plt.Line2D([0], [0], ls="--", color="red", label=f"done_tol={DONE_TOL}")
    ], fontsize=7, loc="upper right", ncol=3)

    fig.suptitle(f"Deployment trace — {label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_convergence_waterfall(parsed: Dict, label: str, out_path: Path):
    """
    One subplot per DS primitive activation showing e_norm vs time.
    Highlights whether convergence was achieved and how quickly.
    """
    if not HAS_MPL:
        return

    ds_segs = [s for s in parsed["segments"] if s["primitive"] in DS_PRIMITIVES]
    if not ds_segs:
        return

    n = len(ds_segs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    dt = 1 / 60
    for ax, seg in zip(axes, ds_segs):
        t_seg = np.arange(len(seg["e_norm"])) * dt
        color = PRIM_COLORS.get(seg["primitive"], "#333333")
        ax.plot(t_seg, seg["e_norm"], color=color, lw=1.2)
        ax.axhline(DONE_TOL, ls="--", color="red", alpha=0.7, lw=0.8)
        m = segment_metrics(seg, dt)
        title = seg["primitive"]
        if m["converged"]:
            ax.axvline(m["conv_time_s"], ls=":", color="green", alpha=0.8)
            title += f"\n✓ {m['conv_time_s']:.2f}s"
        else:
            title += "\n✗ timeout"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$\|\mathbf{e}\|$ (rad)")
    fig.suptitle(f"DS primitive convergence — {label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════════════════

def write_results(all_summaries: List[Dict], out_dir: Path):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full JSON
    json_path = out_dir / f"analysis_{ts}.json"
    # segment_metrics contains numpy bools — convert
    def _clean(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(_clean(all_summaries), f, indent=2)

    # Flat CSV (one row per run, key metrics only)
    csv_path = out_dir / f"analysis_{ts}_summary.csv"
    flat_keys = [
        "label", "total_duration_s", "ds_convergence_rate",
        "ds_cos_mean", "ds_cos_neg_frac", "ds_proj_active_frac",
        "ds_mean_conv_time_s", "ds_e_monotone_frac",
        "n_ds_primitives", "n_ik_primitives",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
        writer.writeheader()
        for s in all_summaries:
            writer.writerow({k: s.get(k, "") for k in flat_keys})

    print(f"\nResults written to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")


def print_summary(summaries: List[Dict]):
    header = (
        f"{'Run':<35} {'Conv%':>6} {'cos_mean':>9} "
        f"{'cos_neg%':>9} {'proj%':>6} {'mono%':>6}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for s in summaries:
        def _f(v, fmt=".2f"):
            return f"{v:{fmt}}" if not (isinstance(v, float) and math.isnan(v)) else "  —  "
        print(
            f"{s['label']:<35} "
            f"{_f(s['ds_convergence_rate']*100, '.0f')+'%':>6} "
            f"{_f(s['ds_cos_mean']):>9} "
            f"{_f(s['ds_cos_neg_frac']*100, '.0f')+'%':>9} "
            f"{_f(s['ds_proj_active_frac']*100, '.0f')+'%':>6} "
            f"{_f(s['ds_e_monotone_frac']*100, '.0f')+'%':>6}"
        )
    print("=" * len(header))
    print("  Conv%    = fraction of DS primitive activations that converge")
    print("  cos_mean = mean cos(q̇, -e) during DS steps (>0 = toward goal)")
    print("  cos_neg% = fraction of DS steps pointing away from goal")
    print("  proj%    = fraction of DS steps where safe projection fired")
    print("  mono%    = fraction of DS steps where ||e|| decreased")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", nargs="+", required=True,
                   help="Path(s) to deployment CSV logs")
    p.add_argument("--ckpt_dir", default="data/checkpoints",
                   help="Directory containing .pt checkpoint files")
    p.add_argument("--demo_dir", default="data/demonstrations",
                   help="Directory containing demo .pkl files (optional)")
    p.add_argument("--out_dir",  default="data/results")
    p.add_argument("--physics_dt", type=float, default=1/60)
    p.add_argument("--no_ckpt_diag", action="store_true",
                   help="Skip checkpoint diagnostics (faster)")
    p.add_argument("--no_figures", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    all_summaries = []

    for log_path in args.logs:
        log_path = Path(log_path)
        if not log_path.exists():
            print(f"[SKIP] {log_path} not found")
            continue

        # Infer arm from filename (left_ds_lula_scripted.csv -> left)
        label = log_path.stem
        arm   = label.split("_")[0] if label.split("_")[0] in ("left", "right") else "unknown"

        print(f"\n── {label} ({'arm=' + arm}) ──────────────────────────────")

        rows   = load_csv(str(log_path))
        parsed = parse_rows(rows)
        summ   = run_summary(parsed, label=label, physics_dt=args.physics_dt)
        all_summaries.append(summ)

        # Per-primitive table
        print(f"  total steps : {summ['total_steps']}  "
              f"({summ['total_duration_s']:.1f}s)")
        print(f"  DS conv rate: {summ['ds_convergence_rate']*100:.0f}%  "
              f"mean_t={summ['ds_mean_conv_time_s']:.2f}s")
        print(f"  cos→goal    : mean={summ['ds_cos_mean']:.3f}  "
              f"neg_frac={summ['ds_cos_neg_frac']*100:.1f}%")
        print(f"  proj fired  : {summ['ds_proj_active_frac']*100:.1f}% of DS steps")
        print(f"  monotone    : {summ['ds_e_monotone_frac']*100:.1f}% of DS steps")
        for prim, pm in summ["per_primitive"].items():
            conv_str = (f"✓ {pm['conv_time_s']:.2f}s"
                        if pm["converged"] else "✗ timeout")
            cos_str  = (f"  cos={pm['cos_mean']:.2f}"
                        if pm["cos_mean"] is not None else "")
            print(f"    {prim:<12} {conv_str}{cos_str}  "
                  f"e: {pm['e_start']:.3f} → {pm['e_end']:.3f}")

        # Checkpoint diagnostics
        if not args.no_ckpt_diag and HAS_TORCH:
            for prim in ("reach", "transport"):
                ckpt_path = Path(args.ckpt_dir) / f"{arm}_{prim}.pt"
                if not ckpt_path.exists():
                    continue
                demo_path = next(
                    (str(p) for p in Path(args.demo_dir).glob(f"{arm}*.pkl")), None
                ) if Path(args.demo_dir).exists() else None
                print(f"\n  checkpoint: {ckpt_path.name}")
                cd = checkpoint_diagnostics(str(ckpt_path), demo_path)
                summ.setdefault("checkpoints", {})[f"{arm}_{prim}"] = cd
                print(f"    ||f(0)||            = {cd.get('f_at_zero_norm', 'n/a'):.2e}")
                print(f"    rollout conv rate   = {cd.get('rollout_conv_rate', 0)*100:.0f}%")
                print(f"    rollout median time = {cd.get('rollout_median_conv_time', float('nan')):.2f}s")
                if "imitation_mse" in cd:
                    print(f"    imitation MSE       = {cd['imitation_mse']:.5f}")
                if "stab_violation_rate" in cd:
                    print(f"    stab violation rate = {cd['stab_violation_rate']*100:.1f}%")

        # Figures
        if not args.no_figures and HAS_MPL:
            plot_deployment_trace(
                parsed, label,
                out_dir / f"fig_deploy_trace_{label}.png"
            )
            plot_convergence_waterfall(
                parsed, label,
                out_dir / f"fig_convergence_{label}.png"
            )

    if not all_summaries:
        print("[ERROR] No logs processed.")
        return

    print_summary(all_summaries)
    write_results(all_summaries, out_dir)


if __name__ == "__main__":
    main()