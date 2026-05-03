"""
Filter demonstration data before DS training.

For each demo, computes per-block-transport segments and removes those whose
path length or smoothness lies beyond `--sigma` standard deviations from the
median. Saves a cleaned pkl alongside the original.

Usage:
  python scripts/clean_demos.py --arm left
  python scripts/clean_demos.py --arm left --sigma 2.0 --plot
"""

import argparse
import pickle
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def segment_stats(traj):
    """Return (path_length, smoothness) for a list of timestep dicts."""
    qs    = np.stack([t["q"]     for t in traj])
    qdots = np.stack([t["q_dot"] for t in traj])

    # Path length in joint space
    dq = np.diff(qs, axis=0)
    path_length = np.linalg.norm(dq, axis=1).sum()

    # Smoothness: sum of squared jerk (finite diff of q_dot)
    qddot = np.diff(qdots, axis=0)
    smoothness = (qddot ** 2).sum()

    return path_length, smoothness


def extract_transport_segments(demo):
    """Split a demo trajectory into per-block transport segments."""
    segments = {}
    current_block = None
    buf = []
    for step in demo["trajectory"]:
        if step["primitive"] != "transport":
            if buf and current_block is not None:
                segments[current_block] = buf
                buf = []
                current_block = None
            continue
        if step["block"] != current_block:
            if buf and current_block is not None:
                segments[current_block] = buf
            current_block = step["block"]
            buf = []
        buf.append(step)
    if buf and current_block is not None:
        segments[current_block] = buf
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",    type=str, default="left")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--sigma",  type=float, default=2.5,
                        help="Reject segments beyond this many MADs from median")
    parser.add_argument("--plot",   action="store_true",
                        help="Show path-length and smoothness distributions")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    demos_dir = Path(cfg["paths"]["demos"])
    in_path   = demos_dir / f"{args.arm}_demos.pkl"
    out_path  = demos_dir / f"{args.arm}_demos_clean.pkl"

    with open(in_path, "rb") as f:
        all_demos = pickle.load(f)

    print(f"[CLEAN] Loaded {len(all_demos)} demos from {in_path}")

    # ── Collect per-segment stats ─────────────────────────────────────────────
    # Each entry: (demo_idx, block_name, path_length, smoothness, segment)
    records = []
    for demo in all_demos:
        segs = extract_transport_segments(demo)
        for block_name, seg in segs.items():
            if len(seg) < 3:
                continue
            pl, sm = segment_stats(seg)
            records.append({
                "demo_idx":   demo["demo_idx"],
                "block":      block_name,
                "path_length": pl,
                "smoothness":  sm,
                "segment":    seg,
            })

    if not records:
        print("[CLEAN] No transport segments found — is your data collected?")
        return

    path_lengths = np.array([r["path_length"] for r in records])
    smoothnesses = np.array([r["smoothness"]  for r in records])

    # ── Robust outlier rejection using MAD ───────────────────────────────────
    # Using median + MAD rather than mean + std so a few extreme outliers
    # don't inflate the threshold and let other outliers through.
    def mad_mask(values, sigma):
        median = np.median(values)
        mad    = np.median(np.abs(values - median))
        mad    = max(mad, 1e-9)
        z      = np.abs(values - median) / (1.4826 * mad)  # 1.4826 ≈ 1/Φ^{-1}(0.75)
        return z <= sigma

    keep_pl = mad_mask(path_lengths, args.sigma)
    keep_sm = mad_mask(smoothnesses, args.sigma)
    keep    = keep_pl & keep_sm

    n_total   = len(records)
    n_keep    = keep.sum()
    n_reject  = n_total - n_keep

    print(f"[CLEAN] Segments total:    {n_total}")
    print(f"[CLEAN] Rejected (path):   {(~keep_pl).sum()}")
    print(f"[CLEAN] Rejected (jerk):   {(~keep_sm).sum()}")
    print(f"[CLEAN] Rejected (either): {n_reject}  ({100*n_reject/n_total:.1f}%)")
    print(f"[CLEAN] Keeping:           {n_keep}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(path_lengths[keep],  bins=20, label="kept",     alpha=0.7)
            axes[0].hist(path_lengths[~keep], bins=20, label="rejected", alpha=0.7, color="red")
            axes[0].set_title("Path length (joint space)")
            axes[0].legend()
            axes[1].hist(smoothnesses[keep],  bins=20, label="kept",     alpha=0.7)
            axes[1].hist(smoothnesses[~keep], bins=20, label="rejected", alpha=0.7, color="red")
            axes[1].set_title("Smoothness (sum sq jerk)")
            axes[1].legend()
            plt.tight_layout()
            plt.savefig(demos_dir / f"{args.arm}_segment_stats.png", dpi=120)
            print(f"[CLEAN] Plot saved to {demos_dir / f'{args.arm}_segment_stats.png'}")
            plt.show()
        except ImportError:
            print("[CLEAN] matplotlib not available, skipping plot")

    # ── Rebuild demo list with only clean transport steps ─────────────────────
    # Keep a demo if at least one of its segments survived; replace its
    # trajectory with only the kept transport steps (other primitives not
    # recorded so trajectory is transport-only anyway).
    kept_by_demo = {}
    for r, k in zip(records, keep):
        if not k:
            continue
        idx = r["demo_idx"]
        if idx not in kept_by_demo:
            kept_by_demo[idx] = []
        kept_by_demo[idx].extend(r["segment"])

    # Rebuild demo dicts
    demo_by_idx = {d["demo_idx"]: d for d in all_demos}
    clean_demos = []
    for idx, steps in kept_by_demo.items():
        d = dict(demo_by_idx[idx])
        d["trajectory"] = steps
        clean_demos.append(d)

    clean_demos.sort(key=lambda d: d["demo_idx"])

    with open(out_path, "wb") as f:
        pickle.dump(clean_demos, f)

    n_steps_in  = sum(len(d["trajectory"]) for d in all_demos if d["trajectory"])
    n_steps_out = sum(len(d["trajectory"]) for d in clean_demos)
    print(f"[CLEAN] Saved {len(clean_demos)} demos "
          f"({n_steps_out} steps, was {n_steps_in}) → {out_path}")
    print(f"[CLEAN] To train on clean data:")
    print(f"  python scripts/train_lpvds.py --arm {args.arm} --demos {out_path}")


if __name__ == "__main__":
    main()
