"""
Animate dual-arm LPVDS modulation over time in 3D.

Input is produced by:
  python scripts/deploy_dual_arm.py --model lpvds --diag_out data/results/lpvds_interaction.pkl

Usage:
  python scripts/animate_lpvds_interaction.py --diag data/results/lpvds_interaction.pkl
  python scripts/animate_lpvds_interaction.py --diag data/results/lpvds_interaction.pkl --out data/results/interaction.gif
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


COLORS = {"left": "tab:red", "right": "tab:blue"}


def _sphere_wire(center, radius, n=18):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2)
    xs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    ys = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    zs = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return xs, ys, zs


def _radial_field(center, radius, n_theta=10, n_phi=4):
    dirs = []
    for phi in np.linspace(0.25 * np.pi, 0.75 * np.pi, n_phi):
        for theta in np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False):
            dirs.append([
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi),
            ])
    dirs = np.asarray(dirs)
    origins = center[None, :] + 0.45 * radius * dirs
    vecs = 0.35 * radius * dirs
    return origins, vecs


def _axis_limits(rows):
    pts = []
    for r in rows:
        pts.append(r["ee"])
        pts.append(r["ee_other"])
        pts.append(r["goal"])
    pts = np.asarray(pts, dtype=float)
    pad = 0.12
    mins = pts.min(axis=0) - pad
    maxs = pts.max(axis=0) + pad
    span = max(maxs - mins)
    center = 0.5 * (mins + maxs)
    mins = center - span / 2
    maxs = center + span / 2
    return mins, maxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--stride", type=int, default=3,
                        help="Use every Nth diagnostic timestep")
    parser.add_argument("--trail", type=int, default=180,
                        help="Number of recent samples to show as trail")
    parser.add_argument("--arrow_scale", type=float, default=0.35)
    parser.add_argument("--radial_field", action="store_true",
                        help="Show radial field samples around each moving EE obstacle")
    parser.add_argument("--views", choices=["original", "top", "both"],
                        default="original",
                        help="Render original 3D view, top-down XY view, or both side-by-side")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    with open(args.diag, "rb") as f:
        payload = pickle.load(f)
    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    if not rows:
        raise ValueError("Diagnostic log is empty.")

    # Pair rows by simulation step so left/right update together.
    steps = sorted(set(r["step"] for r in rows))
    frames = []
    for step in steps[::max(args.stride, 1)]:
        frame_rows = {r["arm"]: r for r in rows if r["step"] == step}
        if frame_rows:
            frames.append((step, frame_rows))

    mins, maxs = _axis_limits(rows)
    radius = float(rows[0].get("safe_radius", 0.30))

    if args.views == "both":
        fig = plt.figure(figsize=(16, 7))
        ax_3d = fig.add_subplot(121, projection="3d")
        ax_top = fig.add_subplot(122)
    elif args.views == "top":
        fig, ax_top = plt.subplots(figsize=(8, 8))
        ax_3d = None
    else:
        fig = plt.figure(figsize=(10, 8))
        ax_3d = fig.add_subplot(111, projection="3d")
        ax_top = None

    def _draw_original(ax, frame_rows, step):
        ax.clear()
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

        text_lines = []
        for arm in ("left", "right"):
            arm_all = [r for r in rows if r["arm"] == arm and r["step"] <= step]
            if not arm_all:
                continue
            arm_trail = arm_all[-args.trail:]
            ee = np.array([r["ee"] for r in arm_trail])
            current = frame_rows.get(arm, arm_all[-1])
            p = np.array(current["ee"])
            other = np.array(current["ee_other"])
            goal = np.array(current["goal"])
            v_nom = np.array(current.get("v_nom", current["v_cmd"]))
            v_mod = np.array(current.get("v_mod", current["v_cmd"]))
            v_cmd = np.array(current["v_cmd"])

            color = COLORS[arm]
            ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=color, lw=2.0, alpha=0.85)
            ax.scatter([p[0]], [p[1]], [p[2]], color=color, s=55)
            ax.scatter([goal[0]], [goal[1]], [goal[2]], color=color, marker="*", s=115)

            xs, ys, zs = _sphere_wire(other, radius)
            ax.plot_wireframe(xs, ys, zs, color=color, alpha=0.12, linewidth=0.6)
            if args.radial_field:
                origins, vecs = _radial_field(other, radius)
                ax.quiver(origins[:, 0], origins[:, 1], origins[:, 2],
                          vecs[:, 0], vecs[:, 1], vecs[:, 2],
                          length=1.0, normalize=False, color=color,
                          alpha=0.18, linewidth=0.6)

            ax.quiver(p[0], p[1], p[2], v_nom[0], v_nom[1], v_nom[2],
                      length=args.arrow_scale, normalize=False, color=color,
                      linestyle="dotted", alpha=0.35)
            ax.quiver(p[0], p[1], p[2], v_mod[0], v_mod[1], v_mod[2],
                      length=args.arrow_scale, normalize=False, color=color,
                      linestyle="dashed", alpha=0.65)
            ax.quiver(p[0], p[1], p[2], v_cmd[0], v_cmd[1], v_cmd[2],
                      length=args.arrow_scale, normalize=False, color=color,
                      alpha=1.0)

            text_lines.append(
                f"{arm}: Gamma={current['gamma']:.2f}, "
                f"d={current['distance']:.3f}m, w={current['mod_weight']:.2f}"
            )

        t = max(r["t"] for r in frame_rows.values())
        ax.set_title("Original 3D view\n"
                     f"t={t:.2f}s, step={step}\n" + "\n".join(text_lines))
        ax.view_init(elev=25, azim=-55)

    def _draw_top(ax, frame_rows, step):
        ax.clear()
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.25)

        for arm in ("left", "right"):
            arm_all = [r for r in rows if r["arm"] == arm and r["step"] <= step]
            if not arm_all:
                continue
            arm_trail = arm_all[-args.trail:]
            ee = np.array([r["ee"] for r in arm_trail])
            current = frame_rows.get(arm, arm_all[-1])
            p = np.array(current["ee"])
            other = np.array(current["ee_other"])
            goal = np.array(current["goal"])
            v_nom = np.array(current.get("v_nom", current["v_cmd"]))
            v_mod = np.array(current.get("v_mod", current["v_cmd"]))
            v_cmd = np.array(current["v_cmd"])
            color = COLORS[arm]

            ax.plot(ee[:, 0], ee[:, 1], color=color, lw=2.0, alpha=0.85)
            ax.scatter([p[0]], [p[1]], color=color, s=55)
            ax.scatter([goal[0]], [goal[1]], color=color, marker="*", s=115)
            ax.add_patch(plt.Circle(other[:2], radius, color=color, alpha=0.08))

            if "protected_points_other" in current:
                pts = np.asarray(current["protected_points_other"])
                ax.scatter(pts[:, 0], pts[:, 1], color=color, s=18, alpha=0.25)

            ax.quiver(p[0], p[1], v_nom[0], v_nom[1],
                      angles="xy", scale_units="xy", scale=1.0 / args.arrow_scale,
                      color=color, alpha=0.25)
            ax.quiver(p[0], p[1], v_mod[0], v_mod[1],
                      angles="xy", scale_units="xy", scale=1.0 / args.arrow_scale,
                      color=color, alpha=0.55)
            ax.quiver(p[0], p[1], v_cmd[0], v_cmd[1],
                      angles="xy", scale_units="xy", scale=1.0 / args.arrow_scale,
                      color=color, alpha=1.0)

        t = max(r["t"] for r in frame_rows.values())
        ax.set_title(f"Top-down XY view\n"
                     f"t={t:.2f}s, step={step}")

    def draw(frame_idx):
        _, frame_rows = frames[frame_idx]
        step = frames[frame_idx][0]
        if ax_3d is not None:
            _draw_original(ax_3d, frame_rows, step)
        if ax_top is not None:
            _draw_top(ax_top, frame_rows, step)
        fig.tight_layout()
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / args.fps)

    out = Path(args.out) if args.out else Path(args.diag).with_suffix(".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".gif":
        anim.save(out, writer=PillowWriter(fps=args.fps))
    else:
        anim.save(out, writer=FFMpegWriter(fps=args.fps, bitrate=1800))
    plt.close(fig)
    print(f"[ANIM] Saved {out}")


if __name__ == "__main__":
    main()
