"""
Visualize the learned 3D Cartesian LPVDS.

Produces a 3D quiver plot over the arm workspace. Optionally overlays the
other arm's EE as a spherical obstacle and shows the modulated vector field.

Usage:
  python scripts/plot_lpvds_3d.py --arm left
  python scripts/plot_lpvds_3d.py --arm right --modulated --other_ee 0.0 0.45 0.99
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lpv_ds import LPVDS
from src.modulation import HuberModulation


def _set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * np.max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def _sphere(ax, center, radius, color="crimson", alpha=0.12):
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 14)
    xs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    ys = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    zs = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--grid", type=int, default=7)
    parser.add_argument("--z_span", type=float, default=0.18)
    parser.add_argument("--speed_scale", type=float, default=0.12,
                        help="Visual arrow length scale")
    parser.add_argument("--modulated", action="store_true")
    parser.add_argument("--other_ee", type=float, nargs=3, default=None,
                        help="Other EE obstacle center for modulated field")
    parser.add_argument("--mod_radius", type=float, default=None)
    parser.add_argument("--mod_reactivity", type=float, default=None)
    parser.add_argument("--mod_weight", type=float, default=1.0)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt = Path(args.checkpoint) if args.checkpoint else (
        Path(cfg["paths"]["checkpoints"]) / f"{args.arm}_transport_lpvds.pkl"
    )
    model = LPVDS.load(ckpt)

    ws = cfg["block_workspace"][args.arm]
    goal = np.asarray(model.x_goal)
    xs = np.linspace(min(ws["x_min"], goal[0]), max(ws["x_max"], goal[0]), args.grid)
    ys = np.linspace(min(ws["y_min"], goal[1]), max(ws["y_max"], goal[1]), args.grid)
    zs = np.linspace(goal[2] - args.z_span / 2, goal[2] + args.z_span / 2, max(3, args.grid // 2))

    points = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    v_nom = np.array([model.safe_velocity(p) for p in points])
    vectors = v_nom.copy()

    other = np.array(args.other_ee, dtype=float) if args.other_ee is not None else None
    radius = args.mod_radius if args.mod_radius is not None else cfg["coordination"]["ee_safety_radius"]
    reactivity = args.mod_reactivity
    if reactivity is None:
        reactivity = cfg["coordination"].get("modulation_reactivity", 2.0)

    if args.modulated:
        if other is None:
            other = np.array([0.0, cfg["shared_goal"][1], goal[2]])
        huber = HuberModulation(safe_radius=radius, reactivity=reactivity)
        v_mod = np.array([huber.modulate_cartesian(v, p, other)
                          for p, v in zip(points, v_nom)])
        vectors = (1.0 - args.mod_weight) * v_nom + args.mod_weight * v_mod

    speeds = np.linalg.norm(vectors, axis=1)
    scale = args.speed_scale / max(float(np.percentile(speeds, 95)), 1e-6)
    vec_plot = vectors * scale

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(speeds.min()), vmax=float(speeds.max()))
    colors = cmap(norm(speeds))
    ax.quiver(points[:, 0], points[:, 1], points[:, 2],
              vec_plot[:, 0], vec_plot[:, 1], vec_plot[:, 2],
              colors=colors, length=1.0, normalize=False)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(speeds)
    fig.colorbar(mappable, ax=ax, shrink=0.72, label="EE speed [m/s]")
    ax.scatter([goal[0]], [goal[1]], [goal[2]], c="limegreen", s=90, label="LPVDS attractor")

    if other is not None:
        ax.scatter([other[0]], [other[1]], [other[2]], c="crimson", s=70, label="other EE obstacle")
        _sphere(ax, other, radius)

    ax.set_title(f"{args.arm} 3D Cartesian LPVDS" + (" with modulation" if args.modulated else ""))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper left")
    _set_axes_equal(ax)

    out = Path(args.out) if args.out else Path(cfg["paths"]["results"]) / f"{args.arm}_lpvds_3d.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[PLOT] Saved {out}")


if __name__ == "__main__":
    main()
