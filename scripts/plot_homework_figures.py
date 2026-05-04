"""
Generate report-style figures for the dual-arm LPVDS stacking project.

The plots mirror the homework/reference style in the neighboring DS projects:
demonstration trajectories, GMM/LPVDS structure, vector-field modulation, and
time-series interaction diagnostics.

Usage:
  MPLCONFIGDIR=/tmp/mpl python scripts/plot_homework_figures.py
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lpv_ds import LPVDS
from src.modulation import HuberModulation


ARM_COLORS = {"left": "crimson", "right": "royalblue"}


def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def load_demos(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def transport_xyz(demo):
    pts = [
        np.asarray(step["ee_pos"], dtype=float)[:3]
        for step in demo["trajectory"]
        if step.get("primitive", "transport") == "transport"
    ]
    return np.asarray(pts)


def load_model(path):
    return LPVDS.load(path)


def plot_trajectory_atlas(demos_by_arm, cfg, out):
    import matplotlib.pyplot as plt

    goal_xy = np.asarray(cfg["shared_goal"], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, view in zip(axes, ("xy", "xz")):
        for arm, demos in demos_by_arm.items():
            color = ARM_COLORS[arm]
            for demo in demos:
                xyz = transport_xyz(demo)
                if len(xyz) == 0:
                    continue
                if view == "xy":
                    ax.plot(xyz[:, 0], xyz[:, 1], color=color, alpha=0.20, lw=1.0)
                    ax.scatter(xyz[0, 0], xyz[0, 1], color=color, s=10, alpha=0.35)
                    ax.scatter(xyz[-1, 0], xyz[-1, 1], color=color, s=14, alpha=0.55)
                else:
                    ax.plot(xyz[:, 0], xyz[:, 2], color=color, alpha=0.20, lw=1.0)
                    ax.scatter(xyz[0, 0], xyz[0, 2], color=color, s=10, alpha=0.35)
                    ax.scatter(xyz[-1, 0], xyz[-1, 2], color=color, s=14, alpha=0.55)

        if view == "xy":
            ax.scatter(goal_xy[0], goal_xy[1], marker="*", s=180, c="black", label="shared stack")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_title("Transport demonstrations: table plane")
        else:
            ax.axhline(cfg["heights"]["lift"], color="black", ls="--", lw=1.2, alpha=0.6, label="lift height")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
            ax.set_title("Transport demonstrations: height consistency")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")

    handles = [
        plt.Line2D([0], [0], color=ARM_COLORS["left"], lw=2, label="left demos"),
        plt.Line2D([0], [0], color=ARM_COLORS["right"], lw=2, label="right demos"),
        plt.Line2D([0], [0], marker="*", color="black", lw=0, markersize=12, label="shared stack"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out, dpi=220)
    plt.close(fig)


def ellipsoid(ax, mean, cov, color, scale=1.2, alpha=0.16):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    radii = scale * np.sqrt(vals)
    u = np.linspace(0.0, 2.0 * np.pi, 28)
    v = np.linspace(0.0, np.pi, 14)
    xyz = np.stack([
        radii[0] * np.outer(np.cos(u), np.sin(v)),
        radii[1] * np.outer(np.sin(u), np.sin(v)),
        radii[2] * np.outer(np.ones_like(u), np.cos(v)),
    ], axis=-1)
    xyz = xyz @ vecs.T + mean
    ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2],
                    color=color, alpha=alpha, linewidth=0, shade=True)


def plot_lpvds_structure(demos_by_arm, models, out):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for arm, demos in demos_by_arm.items():
        color = ARM_COLORS[arm]
        for demo in demos[::2]:
            xyz = transport_xyz(demo)
            if len(xyz) > 0:
                ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, alpha=0.18, lw=1.0)

        model = models[arm]
        for k in range(len(model.priors)):
            mean = model.x_mean + model.x_scale * model.mus[:, k]
            cov = (model.x_scale[:, None] * model.sigmas[:, :, k]) * model.x_scale[None, :]
            ellipsoid(ax, mean, cov, color=color)
            ax.text(mean[0], mean[1], mean[2] + 0.015, f"{arm[0]}{k+1}",
                    color=color, fontsize=8)
        ax.scatter(*model.x_goal, marker="*", s=160, color=color, label=f"{arm} attractor")

    ax.set_title("LPVDS transport structure: demonstrations and GMM regions")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=28, azim=-42)
    ax.legend(loc="upper left")
    set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_modulation_slice(cfg, out, radius=0.35, reactivity=1.5):
    import matplotlib.pyplot as plt

    obstacle = np.array([0.0, cfg["shared_goal"][1], cfg["heights"]["lift"]])
    goal = np.array([0.0, cfg["shared_goal"][1] - 0.22, cfg["heights"]["lift"]])
    huber = HuberModulation(safe_radius=radius, reactivity=reactivity)

    xs = np.linspace(-0.55, 0.55, 25)
    ys = np.linspace(0.20, 0.72, 25)
    X, Y = np.meshgrid(xs, ys)
    U_nom = np.zeros_like(X)
    V_nom = np.zeros_like(Y)
    U_mod = np.zeros_like(X)
    V_mod = np.zeros_like(Y)
    radial_nom = np.zeros_like(X)
    radial_mod = np.zeros_like(X)

    for idx in np.ndindex(X.shape):
        p = np.array([X[idx], Y[idx], obstacle[2]])
        v = goal - p
        v[2] = 0.0
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v = 0.14 * v / norm
        vm = huber.modulate_cartesian(v, p, obstacle)
        U_nom[idx], V_nom[idx] = v[:2]
        U_mod[idx], V_mod[idx] = vm[:2]
        r = huber.reference_direction(p, obstacle)
        radial_nom[idx] = float(np.dot(v, r))
        radial_mod[idx] = float(np.dot(vm, r))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, U, V, title in [
        (axes[0], U_nom, V_nom, "Nominal DS velocity"),
        (axes[1], U_mod, V_mod, "Modulated velocity around other EE"),
    ]:
        speed = np.sqrt(U ** 2 + V ** 2)
        ax.streamplot(X, Y, U, V, color=speed, cmap="viridis", density=1.15, linewidth=1.2)
        circle = plt.Circle(obstacle[:2], radius, color="crimson", alpha=0.14)
        ax.add_patch(circle)
        ax.scatter(obstacle[0], obstacle[1], color="crimson", s=60, label="other EE")
        ax.scatter(goal[0], goal[1], marker="*", color="black", s=120, label="attractor")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.grid(alpha=0.18)

    delta = radial_mod - radial_nom
    im = axes[2].contourf(X, Y, delta, levels=24, cmap="coolwarm")
    axes[2].add_patch(plt.Circle(obstacle[:2], radius, color="crimson", alpha=0.14))
    axes[2].scatter(obstacle[0], obstacle[1], color="crimson", s=60)
    axes[2].scatter(goal[0], goal[1], marker="*", color="black", s=120)
    axes[2].set_title(r"Radial change: $(v_{mod}-v_{nom})\cdot r$")
    axes[2].set_xlabel("x [m]")
    fig.colorbar(im, ax=axes[2], shrink=0.9)
    axes[0].set_ylabel("y [m]")
    axes[0].legend(loc="upper left", frameon=False)
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_interaction_report(diag_path, out):
    import matplotlib.pyplot as plt

    with open(diag_path, "rb") as f:
        payload = pickle.load(f)
    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ("left", "right")}

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    for arm, arm_rows in by_arm.items():
        if not arm_rows:
            continue
        color = ARM_COLORS[arm]
        t = np.array([r["t"] for r in arm_rows])
        dist = np.array([r["distance"] for r in arm_rows])
        gamma = np.array([r["gamma"] for r in arm_rows])
        w = np.array([r["mod_weight"] for r in arm_rows])
        v_nom = np.array([r["v_nom"] for r in arm_rows])
        v_mod = np.array([r["v_mod"] for r in arm_rows])
        v_cmd = np.array([r["v_cmd"] for r in arm_rows])
        ee = np.array([r["ee"] for r in arm_rows])
        other = np.array([r["ee_other"] for r in arm_rows])
        rhat = ee - other
        rhat /= np.linalg.norm(rhat, axis=1, keepdims=True) + 1e-9
        axes[0].plot(t, dist, color=color, label=arm)
        axes[1].plot(t, gamma, color=color, label=arm)
        axes[2].plot(t, np.linalg.norm(v_nom, axis=1), color=color, ls=":", label=f"{arm} nominal")
        axes[2].plot(t, np.linalg.norm(v_mod, axis=1), color=color, ls="--", label=f"{arm} modulated")
        axes[2].plot(t, np.linalg.norm(v_cmd, axis=1), color=color, label=f"{arm} command")
        axes[3].plot(t, np.sum(v_nom * rhat, axis=1), color=color, ls=":", label=f"{arm} nominal")
        axes[3].plot(t, np.sum(v_mod * rhat, axis=1), color=color, label=f"{arm} modulated")
        axes[4].plot(t, w, color=color, label=arm)

    axes[0].set_ylabel("EE distance [m]")
    axes[1].set_ylabel(r"$\Gamma$")
    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color="black", ls="--", alpha=0.45)
    axes[2].set_ylabel("speed [m/s]")
    axes[3].set_ylabel(r"radial velocity $v\cdot r$")
    axes[3].axhline(0.0, color="black", ls="--", alpha=0.35)
    axes[4].set_ylabel("blend weight")
    axes[4].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.suptitle("Dual-arm interaction diagnostics: modulation, distance, and radial safety")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="data/results/homework_plots")
    parser.add_argument("--diag", default="data/results/lpvds_interaction.pkl")
    parser.add_argument("--use_clean", action="store_true", default=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    suffix = "_clean" if args.use_clean else ""
    demos_by_arm = {
        arm: load_demos(Path(cfg["paths"]["demos"]) / f"{arm}_demos{suffix}.pkl")
        for arm in ("left", "right")
    }
    models = {
        arm: load_model(Path(cfg["paths"]["checkpoints"]) / f"{arm}_transport_lpvds.pkl")
        for arm in ("left", "right")
    }

    outputs = [
        out / "fig1_transport_demo_atlas.png",
        out / "fig2_lpvds_gmm_structure.png",
        out / "fig3_modulation_slice.png",
    ]
    plot_trajectory_atlas(demos_by_arm, cfg, outputs[0])
    plot_lpvds_structure(demos_by_arm, models, outputs[1])
    plot_modulation_slice(cfg, outputs[2])
    if Path(args.diag).exists():
        outputs.append(out / "fig4_interaction_diagnostics.png")
        plot_interaction_report(args.diag, outputs[-1])

    for path in outputs:
        print(f"[PLOT] Saved {path}")


if __name__ == "__main__":
    main()
