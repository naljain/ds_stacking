"""
Plot dual-arm LPVDS interaction diagnostics.

Input is produced by:
  python scripts/deploy_dual_arm.py --model lpvds --diag_out data/results/interaction.pkl

Outputs:
  interaction_3d.png       3D EE trajectories with obstacle spheres
  interaction_timeseries.png  distance, Gamma, speed, modulation weights
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def _sphere(ax, center, radius, color, alpha=0.09):
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 12)
    xs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    ys = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    zs = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0)


def _rows_by_arm(rows):
    return {arm: [r for r in rows if r["arm"] == arm] for arm in ("left", "right")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sphere_stride", type=int, default=60)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    with open(args.diag, "rb") as f:
        payload = pickle.load(f)
    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    by_arm = _rows_by_arm(rows)
    out_dir = Path(args.out) if args.out else Path(args.diag).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {"left": "tab:red", "right": "tab:blue"}
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for arm, arm_rows in by_arm.items():
        if not arm_rows:
            continue
        ee = np.array([r["ee"] for r in arm_rows])
        goal = np.array(arm_rows[0]["goal"])
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=colors[arm], lw=2.0, label=f"{arm} EE")
        ax.scatter([ee[0, 0]], [ee[0, 1]], [ee[0, 2]], color=colors[arm], marker="o", s=45)
        ax.scatter([goal[0]], [goal[1]], [goal[2]], color=colors[arm], marker="*", s=120)

    # Draw obstacle spheres from each arm's perspective at sparse timesteps.
    for arm, arm_rows in by_arm.items():
        for r in arm_rows[::max(args.sphere_stride, 1)]:
            _sphere(ax, np.array(r["ee_other"]), r["safe_radius"], colors[arm])

    ax.set_title("Dual-arm LPVDS interaction: EEs and moving obstacle spheres")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "interaction_3d.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for arm, arm_rows in by_arm.items():
        if not arm_rows:
            continue
        t = np.array([r["t"] for r in arm_rows])
        gamma = np.array([r["gamma"] for r in arm_rows])
        dist = np.array([r["distance"] for r in arm_rows])
        speed_nom = np.linalg.norm(np.array([r.get("v_nom", r["v_cmd"]) for r in arm_rows]), axis=1)
        speed_mod = np.linalg.norm(np.array([r.get("v_mod", r["v_cmd"]) for r in arm_rows]), axis=1)
        speed_cmd = np.linalg.norm(np.array([r["v_cmd"] for r in arm_rows]), axis=1)
        weight = np.array([r["mod_weight"] for r in arm_rows])
        axes[0].plot(t, dist, color=colors[arm], label=arm)
        axes[1].plot(t, gamma, color=colors[arm], label=arm)
        axes[2].plot(t, speed_nom, color=colors[arm], ls=":", label=f"{arm} nominal")
        axes[2].plot(t, speed_mod, color=colors[arm], ls="--", label=f"{arm} modulated")
        axes[2].plot(t, speed_cmd, color=colors[arm], label=f"{arm} command")
        axes[3].plot(t, weight, color=colors[arm], label=arm)

    axes[0].set_ylabel("EE distance [m]")
    axes[1].set_ylabel(r"$\Gamma=(d/R)^p$")
    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color="black", ls="--", alpha=0.5)
    axes[2].set_ylabel("command speed [m/s]")
    axes[3].set_ylabel("mod weight")
    axes[3].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "interaction_timeseries.png", dpi=180)
    plt.close(fig)

    print(f"[PLOT] Saved {out_dir / 'interaction_3d.png'}")
    print(f"[PLOT] Saved {out_dir / 'interaction_timeseries.png'}")


if __name__ == "__main__":
    main()
