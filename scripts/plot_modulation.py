"""
Plotting script for modulation visualisations.

Produces three figures useful for the paper:

  1. modulation_field.png   — 2D vector-field slice showing the nominal and
                              modulated DS around an obstacle. The classic
                              "Huber-style" figure.

  2. gamma_timeseries.png   — Γ(t) and ||v|| nominal vs modulated over the
                              course of evaluation trials. Shows when
                              modulation activates and how it deflects motion.

  3. radial_dot.png         — radial component of velocity (v · r) before and
                              after modulation, demonstrating the tail-effect
                              and convergence guarantee.

Figures 2 and 3 require evaluate.py to have been run (with modulation enabled)
to produce a diag_*.pkl file. Figure 1 is synthetic and runs standalone.

Usage:
  python scripts/plot_modulation.py field
  python scripts/plot_modulation.py timeseries --diag data/results/diag_<ts>.pkl
  python scripts/plot_modulation.py radial    --diag data/results/diag_<ts>.pkl
  python scripts/plot_modulation.py all       --diag data/results/diag_<ts>.pkl
"""

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modulation import HuberModulation


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: 2D modulation vector field around a synthetic obstacle
# ──────────────────────────────────────────────────────────────────────────────
def plot_modulation_field(out_path,
                          attractor=np.array([0.6, 0.0]),
                          obstacle=np.array([0.3, 0.0]),
                          safe_radius=0.10,
                          grid_n=32,
                          grid_extent=0.6):
    """Generate a 2D field figure: nominal DS toward attractor + modulation
    around obstacle. Uses Huber 2019 modulation in 2D (we just freeze the
    third component to zero — the modulation construction is the same)."""

    huber = HuberModulation(
        safe_radius=safe_radius,
        reactivity=2.0,
        tail_effect=True,
        eta_min=0.05,
    )

    # Grid in the XY plane
    xs = np.linspace(-grid_extent / 4, grid_extent, grid_n)
    ys = np.linspace(-grid_extent / 2, grid_extent / 2, grid_n)
    X, Y = np.meshgrid(xs, ys)

    Vx_nom = np.zeros_like(X)
    Vy_nom = np.zeros_like(Y)
    Vx_mod = np.zeros_like(X)
    Vy_mod = np.zeros_like(Y)

    # Linear attractor field f_nom(x) = -k * (x - x*)
    k = 1.0

    for i in range(grid_n):
        for j in range(grid_n):
            x = np.array([X[i, j], Y[i, j], 0.0])

            # Nominal velocity toward attractor
            v_nom_2d = -k * (np.array([X[i, j], Y[i, j]]) - attractor)
            v_nom = np.array([v_nom_2d[0], v_nom_2d[1], 0.0])

            # Embed obstacle in 3D for Huber call (z=0)
            obs_3d = np.array([obstacle[0], obstacle[1], 0.0])
            v_mod = huber.modulate_cartesian(v_nom, x, obs_3d)

            Vx_nom[i, j] = v_nom[0]
            Vy_nom[i, j] = v_nom[1]
            Vx_mod[i, j] = v_mod[0]
            Vy_mod[i, j] = v_mod[1]

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, Vx, Vy, title in [
        (axes[0], Vx_nom, Vy_nom, "Nominal DS"),
        (axes[1], Vx_mod, Vy_mod, "Modulated DS (Huber 2019)"),
    ]:
        speed = np.sqrt(Vx ** 2 + Vy ** 2)
        ax.streamplot(X, Y, Vx, Vy, color=speed, cmap="viridis",
                      density=1.6, linewidth=1.0, arrowsize=1.0)

        circle = plt.Circle(obstacle, safe_radius, color="crimson",
                            alpha=0.25, zorder=2, label="Safety sphere")
        ax.add_patch(circle)
        ax.plot(*obstacle, "rx", markersize=10, mew=2.2,
                label="Other arm EE")

        ax.plot(*attractor, "go", markersize=12, label="Attractor (q*)")

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("x [m]")
        ax.set_aspect("equal")
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("y [m]")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Inter-Arm Modulation Vector Field — 2D Slice",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Gamma + speed time series from a recorded trial
# ──────────────────────────────────────────────────────────────────────────────
def plot_gamma_timeseries(diag_log, out_path,
                          condition="nominal", trial=0, arm="left"):
    """Plot Γ(t), ||v_nom||, ||v_mod|| over the course of one trial."""
    if condition not in diag_log:
        print(f"[PLOT] Condition '{condition}' not in diag log — available: "
              f"{list(diag_log.keys())}")
        return
    if trial >= len(diag_log[condition]):
        print(f"[PLOT] Trial {trial} not in condition '{condition}' "
              f"(only {len(diag_log[condition])} trials)")
        return

    trial_log = diag_log[condition][trial]
    rows = [d for d in trial_log if d["arm"] == arm]
    if not rows:
        print(f"[PLOT] No diagnostic entries for arm={arm}")
        return

    t        = np.array([d["t"]               for d in rows])
    gamma    = np.array([d["gamma"]           for d in rows])
    v_nom    = np.array([d["v_cart_norm_nom"] for d in rows])
    v_mod    = np.array([d["v_cart_norm_mod"] for d in rows])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t, gamma, color="C3", lw=1.4, label=r"$\Gamma(t)$")
    axes[0].axhline(1.0, ls="--", color="black", alpha=0.5,
                    label=r"$\Gamma=1$ (boundary)")
    axes[0].set_ylabel(r"Obstacle level-set $\Gamma$")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title(
        f"Modulation Activity — condition: {condition}, arm: {arm}, trial: {trial}"
    )

    axes[1].plot(t, v_nom, color="C0", lw=1.2, label="nominal $\\|v\\|$")
    axes[1].plot(t, v_mod, color="C1", lw=1.2, label="modulated $\\|v\\|$")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("Cartesian EE speed [m/s]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Radial velocity component before / after modulation
# ──────────────────────────────────────────────────────────────────────────────
def plot_radial_dot(diag_log, out_path,
                    condition="nominal", trial=0, arm="left"):
    """Plot v·r for nominal vs modulated, shaded when Γ < threshold (close to
    the obstacle). Demonstrates the tail effect: modulation only damps inward
    radial motion, leaving outward (positive v·r) motion untouched."""
    if condition not in diag_log:
        print(f"[PLOT] Condition '{condition}' not in diag log")
        return
    if trial >= len(diag_log[condition]):
        print(f"[PLOT] Trial {trial} OOB")
        return

    rows = [d for d in diag_log[condition][trial] if d["arm"] == arm]
    if not rows:
        return

    t        = np.array([d["t"]              for d in rows])
    gamma    = np.array([d["gamma"]          for d in rows])
    rdot_nom = np.array([d["radial_dot_nom"] for d in rows])
    rdot_mod = np.array([d["radial_dot_mod"] for d in rows])

    fig, ax = plt.subplots(figsize=(10, 4))

    # Shade regions where modulation is active (Γ < 5)
    active = gamma < 5.0
    in_seg = False
    seg_start = None
    shaded_label_used = False
    for i, a in enumerate(active):
        if a and not in_seg:
            seg_start = t[i]
            in_seg = True
        elif (not a) and in_seg:
            label = r"$\Gamma < 5$" if not shaded_label_used else None
            ax.axvspan(seg_start, t[i], color="crimson", alpha=0.10,
                       label=label)
            shaded_label_used = True
            in_seg = False
    if in_seg:
        label = r"$\Gamma < 5$" if not shaded_label_used else None
        ax.axvspan(seg_start, t[-1], color="crimson", alpha=0.10, label=label)

    ax.axhline(0.0, color="black", ls="-", lw=0.7, alpha=0.6)
    ax.plot(t, rdot_nom, color="C0", lw=1.2, label=r"nominal $v \cdot r$")
    ax.plot(t, rdot_mod, color="C1", lw=1.2, label=r"modulated $v \cdot r$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"radial velocity component $v \cdot r$")
    ax.set_title(
        f"Tail-effect demonstration — condition: {condition}, arm: {arm}\n"
        r"Negative $v\cdot r$ means inward motion. In shaded regions, modulation drives it toward 0."
    )
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["field", "timeseries", "radial", "all"])
    parser.add_argument("--diag", type=str, default=None,
                        help="Path to diag_*.pkl (required for timeseries/radial).")
    parser.add_argument("--out", type=str, default="data/results")
    parser.add_argument("--condition", type=str, default="nominal")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--arm", type=str, default="left")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.which in ("field", "all"):
        plot_modulation_field(out_dir / "modulation_field.png")

    if args.which in ("timeseries", "radial", "all"):
        if args.diag is None:
            print("[PLOT] --diag is required for timeseries/radial plots")
            return
        with open(args.diag, "rb") as f:
            diag_log = pickle.load(f)

        if args.which in ("timeseries", "all"):
            plot_gamma_timeseries(
                diag_log,
                out_dir / "gamma_timeseries.png",
                condition=args.condition,
                trial=args.trial,
                arm=args.arm,
            )
        if args.which in ("radial", "all"):
            plot_radial_dot(
                diag_log,
                out_dir / "radial_dot.png",
                condition=args.condition,
                trial=args.trial,
                arm=args.arm,
            )


if __name__ == "__main__":
    main()
