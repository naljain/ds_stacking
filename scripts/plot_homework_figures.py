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


def mean_demo_target(demos):
    targets = []
    for demo in demos:
        for step in demo["trajectory"]:
            if step.get("primitive", "transport") == "transport" and "target" in step:
                targets.append(np.asarray(step["target"], dtype=float)[:3])
                break
    if not targets:
        return None
    return np.vstack(targets).mean(axis=0)


def select_demos_for_model(arm, demo_dir, model, variant="auto"):
    candidates = {
        "raw": demo_dir / f"{arm}_demos.pkl",
        "clean": demo_dir / f"{arm}_demos_clean.pkl",
    }
    if variant in ("raw", "clean"):
        demos = load_demos(candidates[variant])
        return demos, candidates[variant], mean_demo_target(demos)

    scored = []
    for name, path in candidates.items():
        if not path.exists():
            continue
        demos = load_demos(path)
        target = mean_demo_target(demos)
        if target is None:
            score = np.inf
        else:
            score = float(np.linalg.norm(target - model.x_goal))
        scored.append((score, name, path, demos, target))
    if not scored:
        raise FileNotFoundError(f"No demonstrations found for {arm} in {demo_dir}")
    scored.sort(key=lambda item: item[0])
    _, _, path, demos, target = scored[0]
    return demos, path, target


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


def _demo_xy_and_vel(demos, dt):
    pts, vels = [], []
    for demo in demos:
        xyz = transport_xyz(demo)
        if len(xyz) < 2:
            continue
        vel = np.diff(xyz, axis=0) / dt
        pts.append(xyz[:-1])
        vels.append(vel)
    if not pts:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.vstack(pts), np.vstack(vels)


def _confidence_ellipse(ax, mean, cov, color, nsig=1.8, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-10)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * nsig * np.sqrt(vals)
    patch = Ellipse(mean, width, height, angle=angle, facecolor=color,
                    edgecolor=color, alpha=0.16, lw=1.5, **kwargs)
    ax.add_patch(patch)
    return patch


def plot_hw_style_flow(arm, demos, model, cfg, out):
    import matplotlib.pyplot as plt

    ws = cfg["block_workspace"][arm]
    goal = np.asarray(model.x_goal, dtype=float)
    z = goal[2]
    x_min = min(ws["x_min"], goal[0]) - 0.04
    x_max = max(ws["x_max"], goal[0]) + 0.04
    y_min = min(ws["y_min"], goal[1]) - 0.04
    y_max = max(ws["y_max"], goal[1]) + 0.04

    xs = np.linspace(x_min, x_max, 45)
    ys = np.linspace(y_min, y_max, 45)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    speed = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        vel = model.safe_velocity(np.array([X[idx], Y[idx], z]))
        U[idx], V[idx] = vel[:2]
        speed[idx] = np.linalg.norm(vel)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.streamplot(X, Y, U, V, color=speed, cmap="viridis", density=1.35, linewidth=1.1)
    for demo in demos:
        xyz = transport_xyz(demo)
        if len(xyz) > 0:
            ax.plot(xyz[:, 0], xyz[:, 1], color="black", alpha=0.18, lw=0.9)
    ax.scatter(goal[0], goal[1], marker="*", s=180, color="crimson", label="attractor")
    ax.set_title(f"{arm.capitalize()} LPVDS flow with demonstrations")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_hw_style_gaussians(arm, demos, model, cfg, out):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    color = ARM_COLORS[arm]
    for demo in demos:
        xyz = transport_xyz(demo)
        if len(xyz) > 0:
            ax.plot(xyz[:, 0], xyz[:, 1], color="0.20", alpha=0.16, lw=0.9)

    for k in range(len(model.priors)):
        mean = model.x_mean + model.x_scale * model.mus[:, k]
        cov = (model.x_scale[:, None] * model.sigmas[:, :, k]) * model.x_scale[None, :]
        _confidence_ellipse(ax, mean[:2], cov[:2, :2], color)
        ax.scatter(mean[0], mean[1], color=color, s=20)
        ax.text(mean[0], mean[1], str(k + 1), fontsize=8, color=color,
                ha="center", va="center")

    ax.scatter(model.x_goal[0], model.x_goal[1], marker="*", s=180,
               color="crimson", label="attractor")
    ax.set_title(f"{arm.capitalize()} LPVDS Gaussian mixture regions")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_hw_style_velocity_compare(arm, demos, model, cfg, out):
    import matplotlib.pyplot as plt

    dt = float(cfg["sim"]["physics_dt"])
    pts, demo_vel = _demo_xy_and_vel(demos, dt)
    if len(pts) == 0:
        return
    stride = max(1, len(pts) // 3500)
    pts = pts[::stride]
    demo_vel = demo_vel[::stride]
    pred_vel = np.array([model.safe_velocity(p) for p in pts])

    demo_speed = np.linalg.norm(demo_vel, axis=1)
    pred_speed = np.linalg.norm(pred_vel, axis=1)
    lim = max(float(demo_speed.max()), float(pred_speed.max()), 1e-6)
    rmse = np.sqrt(np.mean(np.sum((pred_vel - demo_vel) ** 2, axis=1)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    labels = [("x velocity [m/s]", 0), ("y velocity [m/s]", 1), ("speed [m/s]", None)]
    for ax, (title, idx) in zip(axes, labels):
        if idx is None:
            x, y = demo_speed, pred_speed
        else:
            x, y = demo_vel[:, idx], pred_vel[:, idx]
        mn = min(float(x.min()), float(y.min()))
        mx = max(float(x.max()), float(y.max()))
        pad = 0.06 * max(mx - mn, 1e-6)
        ax.scatter(x, y, s=5, alpha=0.18, color=ARM_COLORS[arm], edgecolors="none")
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", lw=1.0)
        ax.set_xlim(mn - pad, mx + pad)
        ax.set_ylim(mn - pad, mx + pad)
        ax.set_title(title)
        ax.set_xlabel("demonstration")
        ax.set_ylabel("LPVDS prediction")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{arm.capitalize()} velocity reproduction, RMSE={rmse*1000:.1f} mm/s")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_hw_style_lyapunov(arm, model, cfg, out_v, out_vdot):
    import matplotlib.pyplot as plt

    ws = cfg["block_workspace"][arm]
    goal = np.asarray(model.x_goal, dtype=float)
    z = goal[2]
    x_min = min(ws["x_min"], goal[0]) - 0.05
    x_max = max(ws["x_max"], goal[0]) + 0.05
    y_min = min(ws["y_min"], goal[1]) - 0.05
    y_max = max(ws["y_max"], goal[1]) + 0.05
    xs = np.linspace(x_min, x_max, 90)
    ys = np.linspace(y_min, y_max, 90)
    X, Y = np.meshgrid(xs, ys)
    V = np.zeros_like(X)
    Vdot = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        p = np.array([X[idx], Y[idx], z])
        err = p - goal
        vel = model.safe_velocity(p)
        V[idx] = np.dot(err, err)
        Vdot[idx] = 2.0 * np.dot(err, vel)

    for arr, path, title, cmap in [
        (V, out_v, f"{arm.capitalize()} quadratic Lyapunov candidate", "magma"),
        (Vdot, out_vdot, f"{arm.capitalize()} Lyapunov derivative along LPVDS", "coolwarm"),
    ]:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        if path == out_vdot:
            vmax = np.percentile(np.abs(arr), 98)
            im = ax.contourf(X, Y, arr, levels=32, cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.contour(X, Y, arr, levels=[0.0], colors="black", linewidths=1.2)
        else:
            im = ax.contourf(X, Y, arr, levels=32, cmap=cmap)
        ax.scatter(goal[0], goal[1], marker="*", s=180, color="lime", edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        fig.colorbar(im, ax=ax, shrink=0.9)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)


def generate_hw_style_set(demos_by_arm, models, cfg, out_root):
    hw_dir = out_root / "hw_style"
    hw_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for arm in ("left", "right"):
        specs = [
            (plot_hw_style_flow, hw_dir / f"project_lpvds_{arm}_flow.png"),
            (plot_hw_style_gaussians, hw_dir / f"project_lpvds_{arm}_gaussians.png"),
            (plot_hw_style_velocity_compare, hw_dir / f"project_lpvds_{arm}_velocity_compare.png"),
        ]
        for func, path in specs:
            func(arm, demos_by_arm[arm], models[arm], cfg, path)
            outputs.append(path)
        v_path = hw_dir / f"project_lpvds_{arm}_lyapunov.png"
        vd_path = hw_dir / f"project_lpvds_{arm}_lyapunov_d.png"
        plot_hw_style_lyapunov(arm, models[arm], cfg, v_path, vd_path)
        outputs.extend([v_path, vd_path])
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="data/results/homework_plots")
    parser.add_argument("--diag", default="data/results/lpvds_interaction.pkl")
    parser.add_argument("--demo_variant", choices=["auto", "raw", "clean"], default="auto",
                        help="Which demonstrations to plot; auto matches demo target to LPVDS attractor")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models = {
        arm: load_model(Path(cfg["paths"]["checkpoints"]) / f"{arm}_transport_lpvds.pkl")
        for arm in ("left", "right")
    }
    demos_by_arm = {}
    demo_dir = Path(cfg["paths"]["demos"])
    for arm in ("left", "right"):
        demos, demo_path, target = select_demos_for_model(
            arm, demo_dir, models[arm], variant=args.demo_variant
        )
        demos_by_arm[arm] = demos
        target_str = "unknown" if target is None else np.round(target, 4)
        print(
            f"[DATA] {arm}: using {demo_path} "
            f"target={target_str} model_goal={np.round(models[arm].x_goal, 4)}"
        )

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
    outputs.extend(generate_hw_style_set(demos_by_arm, models, cfg, out))

    for path in outputs:
        print(f"[PLOT] Saved {path}")


if __name__ == "__main__":
    main()
