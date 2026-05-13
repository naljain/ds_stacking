"""
Diagnostic plots for a trained Neural DS checkpoint. Pure-PyTorch + matplotlib;
does NOT require Isaac Sim, so you can run it on a laptop after copying a .pt
file.

Produces 4 figures per checkpoint:

  1. Loss curves            total / imit / stab over epochs
  2. Phase portrait         streamlines of f(x) in a 2D slice
  3. Lyapunov landscape     V(x) heatmap + -∇V quiver overlay
  4. Velocity profile       ||q_dot||(t) along several rollouts from random x_0

The 2D slice fixes 5 of the 7 error joints to zero and varies two of them
(default: joints 0 and 1 — the shoulder pair, usually the biggest movers).

Usage:
  python scripts/plot_ds.py data/checkpoints/left_reach.pt
  python scripts/plot_ds.py --all                       # every *.pt in ckpt_dir
  python scripts/plot_ds.py --all --ckpt_arm left       # left_reach, left_transport
  python scripts/plot_ds.py data/checkpoints/left_reach.pt --joints 0 3
  python scripts/plot_ds.py data/checkpoints/right_transport.pt --no-rollouts
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.neural_ds import StableNeuralDS, N_JOINTS


def load_checkpoint(ckpt_path, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = StableNeuralDS(
        n_joints    = N_JOINTS,
        hidden_dim  = cfg["hidden_dim"],
        lyap_hidden = cfg["lyapunov_hidden"],
        alpha       = cfg["alpha"],
        stable_skip_gain = cfg.get("stable_skip_gain", 0.0),
    ).to(device)
    incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    if unexpected or missing:
        only_old_lyap = (
            unexpected
            and all(k.startswith("V.g.") for k in unexpected)
            and not missing
        )
        if only_old_lyap:
            print(
                "[plot] old learned-Lyapunov checkpoint detected; "
                "loaded velocity field and will use the current quadratic V "
                "for Lyapunov/safe plots."
            )
        else:
            print(
                f"[plot] checkpoint/model key mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
    model.eval()
    return ckpt, model


def make_grid(joint_a, joint_b, n=40, span=3.0):
    """Make an (n*n, 7) grid of normalised states with all joints zero
    except joint_a and joint_b which sweep across [-span, span]."""
    a = np.linspace(-span, span, n)
    b = np.linspace(-span, span, n)
    A, B = np.meshgrid(a, b)
    xs = np.zeros((n * n, N_JOINTS), dtype=np.float32)
    xs[:, joint_a] = A.flatten()
    xs[:, joint_b] = B.flatten()
    return A, B, xs


def model_velocity(model, x_t, use_safe=False, vel_scale=None, state_std=None):
    """Evaluate the checkpoint velocity in normalized coordinates."""
    if not use_safe:
        with torch.no_grad():
            return model(x_t)
    if vel_scale is None or state_std is None:
        raise ValueError("safe velocity plotting requires vel_scale and state_std")
    scale_factor = torch.tensor(
        vel_scale / state_std,
        dtype=torch.float32,
        device=x_t.device,
    ).unsqueeze(0)
    return model.safe_velocity(x_t, scale_factor=scale_factor)


def plot_loss(history, title, out):
    """history: dict with 'total', 'imit', 'stab' lists."""
    if history is None or not history.get("total"):
        print(f"[plot] {title}: no history saved in checkpoint, skipping loss plot")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["total"], label="total")
    ax.plot(history["imit"],  label="imitation")
    ax.plot(history["stab"],  label="stability")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log)")
    ax.set_title(f"{title} — training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out}")


def plot_phase_portrait(model, joint_a, joint_b, title, out, device, span=3.0,
                        use_safe=False, vel_scale=None, state_std=None):
    """Streamlines of f(x) projected onto (joint_a, joint_b) plane."""
    A, B, xs = make_grid(joint_a, joint_b, n=40, span=span)
    v = model_velocity(
        model,
        torch.from_numpy(xs).to(device),
        use_safe=use_safe,
        vel_scale=vel_scale,
        state_std=state_std,
    ).detach().cpu().numpy()
    U = v[:, joint_a].reshape(A.shape)
    V = v[:, joint_b].reshape(A.shape)
    speed = np.sqrt(U ** 2 + V ** 2)

    fig, ax = plt.subplots(figsize=(7, 7))
    strm = ax.streamplot(A, B, U, V, color=speed, cmap="viridis",
                         density=1.4, linewidth=1)
    fig.colorbar(strm.lines, ax=ax, label="||f|| (normalised)")
    ax.scatter([0], [0], color="red", s=80, zorder=5, label="goal (e=0)")
    ax.set_xlabel(f"x_n[{joint_a}]   (= e[{joint_a}] / state_std[{joint_a}])")
    ax.set_ylabel(f"x_n[{joint_b}]")
    suffix = "safe f(x)" if use_safe else "raw f(x)"
    ax.set_title(f"{title} — phase portrait of {suffix}")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out}")


def plot_lyapunov(model, joint_a, joint_b, title, out, device, span=3.0):
    """Heatmap of V(x) with -∇V quiver overlay."""
    A, B, xs = make_grid(joint_a, joint_b, n=40, span=span)
    x_t = torch.from_numpy(xs).to(device).requires_grad_(True)
    V_val = model.V(x_t)
    grad = torch.autograd.grad(V_val.sum(), x_t)[0].detach().cpu().numpy()
    V_np = V_val.detach().cpu().numpy().reshape(A.shape)

    fig, ax = plt.subplots(figsize=(7, 7))
    cs = ax.contourf(A, B, V_np, levels=20, cmap="magma")
    fig.colorbar(cs, ax=ax, label="V(x)")

    # Quiver of -∇V — should point toward goal (origin) everywhere.
    n = 40
    skip = 3
    Ux = -grad[:, joint_a].reshape(A.shape)
    Uy = -grad[:, joint_b].reshape(A.shape)
    ax.quiver(A[::skip, ::skip], B[::skip, ::skip],
              Ux[::skip, ::skip], Uy[::skip, ::skip],
              color="white", alpha=0.7, scale=80, width=0.003)

    ax.scatter([0], [0], color="lime", s=80, zorder=5,
               edgecolors="black", label="goal (e=0)")
    ax.set_xlabel(f"x_n[{joint_a}]")
    ax.set_ylabel(f"x_n[{joint_b}]")
    ax.set_title(f"{title} — Lyapunov V(x) and -∇V")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out}")


def rollouts(model, vel_scale, state_std, title, out, device,
             n_traj=12, n_steps=400, dt=0.00833,
             max_joint_vel=2.0, span=2.5, use_safe=False):
    """Forward-simulate the closed-loop DS in REAL joint space (not normalised)
    from random initial errors. This is the same Euler integrator the deploy
    scripts use, so what you see here is what you'll get at deployment (modulo
    physics + Isaac Sim actuator dynamics)."""
    rng = np.random.default_rng(0)
    e0 = rng.uniform(-1.0, 1.0, size=(n_traj, N_JOINTS)) * (span * state_std)
    e_traj = np.zeros((n_traj, n_steps + 1, N_JOINTS), dtype=np.float32)
    e_traj[:, 0] = e0
    speeds = np.zeros((n_traj, n_steps), dtype=np.float32)

    with torch.no_grad():
        for t in range(n_steps):
            x_n = e_traj[:, t] / state_std
            x_t = torch.from_numpy(x_n).to(device)
            v_n = model_velocity(
                model, x_t,
                use_safe=use_safe,
                vel_scale=vel_scale,
                state_std=state_std,
            ).detach().cpu().numpy()
            q_dot = np.clip(v_n * vel_scale, -max_joint_vel, max_joint_vel)
            speeds[:, t] = np.linalg.norm(q_dot, axis=-1)
            e_traj[:, t + 1] = e_traj[:, t] + q_dot * dt

    norms = np.linalg.norm(e_traj, axis=-1)  # (n_traj, n_steps+1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ts = np.arange(n_steps + 1) * dt
    for i in range(n_traj):
        axes[0].plot(ts, norms[i], alpha=0.7)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("||e(t)||  (rad)")
    axes[0].set_title("error norm (should decay to 0)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0.05, ls="--", c="red", alpha=0.5,
                    label="done_tol = 0.05")
    axes[0].legend()

    ts2 = np.arange(n_steps) * dt
    for i in range(n_traj):
        axes[1].plot(ts2, speeds[i], alpha=0.7)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("||q_dot(t)||  (rad/s)")
    axes[1].axhline(max_joint_vel * np.sqrt(N_JOINTS), ls="--", c="red",
                    alpha=0.5, label=f"clip-saturated max")
    axes[1].set_title("velocity norm")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    suffix = "safe DS" if use_safe else "raw DS"
    fig.suptitle(f"{title} — {suffix} rollouts ({n_traj} initial conditions)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out}")


def plot_deploy_log(csv_path, out_path):
    """Plot the per-step CSV trace produced by deploy_single_arm.py --log_csv."""
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print(f"[plot] {csv_path} is empty"); return

    steps = np.array([int(r["step"]) for r in rows])
    e_norm = np.array([float(r["e_norm"]) for r in rows])
    V = np.array([float(r["V"]) for r in rows])
    qd_raw = np.array([float(r["qd_raw_norm"]) for r in rows])
    qd = np.array([float(r["qd_norm"]) for r in rows])
    proj_d = np.array([float(r["proj_delta"]) for r in rows])
    cos_g = np.array([float(r["cos_to_goal"]) for r in rows])
    prims = [r["primitive"] for r in rows]

    fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
    axes[0].plot(steps, e_norm); axes[0].set_ylabel("||e||")
    axes[0].axhline(0.05, ls="--", c="red", alpha=0.5, label="done_tol")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, V); axes[1].set_ylabel("V(x)")
    axes[1].set_yscale("log"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(steps, qd_raw, label="raw"); axes[2].plot(steps, qd, label="post-projection")
    axes[2].set_ylabel("||q̇||"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    axes[3].plot(steps, proj_d); axes[3].set_ylabel("projection Δ")
    axes[3].grid(True, alpha=0.3)
    axes[4].plot(steps, cos_g); axes[4].set_ylabel("cos(q̇, -e)")
    axes[4].axhline(1.0, ls="--", c="green", alpha=0.5, label="=1: straight to goal")
    axes[4].axhline(0.0, ls="--", c="orange", alpha=0.5, label="=0: perpendicular")
    axes[4].set_ylim(-1.1, 1.1); axes[4].legend(); axes[4].grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")

    # Color-band primitives along the bottom
    cur_prim = prims[0]; start = 0
    colors = {"reach":"#aaccff", "grasp":"#ffaaaa", "lift":"#aaffaa",
              "transport":"#ffddaa", "place":"#ddaaff"}
    for i, p in enumerate(prims + [None]):
        if p != cur_prim:
            for ax in axes:
                ax.axvspan(steps[start], steps[i-1] if i > 0 else steps[-1],
                           alpha=0.15, color=colors.get(cur_prim, "#cccccc"))
            cur_prim = p; start = i if i < len(steps) else 0

    fig.suptitle(f"Deployment trace — {csv_path}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def checkpoint_paths(args):
    if args.all:
        from src.primitives import DS_PRIMITIVES
        ckpt_dir = Path(args.ckpt_dir)
        if args.ckpt_arm == "all":
            paths = sorted(ckpt_dir.glob("*.pt"))
        else:
            paths = [
                ckpt_dir / f"{args.ckpt_arm}_{primitive}.pt"
                for primitive in DS_PRIMITIVES
            ]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "missing checkpoints: " + ", ".join(str(p) for p in missing)
            )
        return paths
    if args.ckpt is None:
        raise ValueError("provide a checkpoint path or use --all")
    return [Path(args.ckpt)]


def plot_checkpoint(ckpt_path, args, device):
    assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"

    out_dir = Path(args.out_dir) / ckpt_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt, model = load_checkpoint(ckpt_path, device)
    title = f"{ckpt.get('arm', '?')} / {ckpt.get('primitive', '?')}"
    vel_scale = np.array(ckpt["vel_scale"], dtype=np.float32)
    state_std = np.array(ckpt["state_std"], dtype=np.float32)
    print(f"[plot] {ckpt_path}")
    print(f"[plot] state_std: {state_std.round(3)}")
    print(f"[plot] vel_scale: {vel_scale.round(3)}")
    if "data_manifest" in ckpt:
        print(f"[plot] data_manifest: {ckpt['data_manifest']}")

    plot_loss(ckpt.get("history"), title,
              out_dir / "01_loss.png")
    plot_phase_portrait(model, args.joints[0], args.joints[1], title,
                        out_dir / "02_phase_portrait.png", device, args.span,
                        use_safe=args.use_safe,
                        vel_scale=vel_scale,
                        state_std=state_std)
    plot_lyapunov(model, args.joints[0], args.joints[1], title,
                  out_dir / "03_lyapunov.png", device, args.span)
    if not args.no_rollouts:
        rollouts(model,
                 vel_scale=vel_scale,
                 state_std=state_std,
                 title=title,
                 out=out_dir / "04_rollouts.png",
                 device=device,
                 max_joint_vel=ckpt["config"].get("max_joint_vel", 2.0),
                 span=args.span * 0.8,
                 use_safe=args.use_safe)

    print(f"[plot] all figures saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, nargs="?",
                        help="Path to checkpoint .pt file")
    parser.add_argument("--all", action="store_true",
                        help="Plot every checkpoint for --ckpt_arm in --ckpt_dir.")
    parser.add_argument("--ckpt_dir", type=str, default="data/checkpoints")
    parser.add_argument("--ckpt_arm", type=str, default="all",
                        choices=["left", "right", "both", "all"],
                        help="Checkpoint prefix to plot with --all. 'all' "
                             "globs every *.pt in --ckpt_dir; left/right "
                             "selects per-arm checkpoints.")
    parser.add_argument("--joints", type=int, nargs=2, default=[0, 1],
                        help="Two joint indices to slice for 2D plots")
    parser.add_argument("--out_dir", type=str, default="data/results/ds_plots",
                        help="Output directory for figures")
    parser.add_argument("--span", type=float, default=3.0,
                        help="Half-width of the 2D state-space slice "
                             "(in units of std)")
    parser.add_argument("--no-rollouts", action="store_true")
    parser.add_argument("--use_safe", action="store_true",
                        help="Plot the Lyapunov-projected velocity field and rollouts.")
    parser.add_argument("--deploy_log", type=str, default=None,
                        help="Path to deploy CSV log; if set, plots that "
                             "trace instead of (or in addition to) checkpoint.")
    args = parser.parse_args()

    if args.deploy_log is not None:
        out_path = Path(args.out_dir) / (Path(args.deploy_log).stem + "_trace.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plot_deploy_log(args.deploy_log, out_path)
        if args.ckpt is None:
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = checkpoint_paths(args)
    for ckpt_path in paths:
        plot_checkpoint(ckpt_path, args, device)


if __name__ == "__main__":
    main()
