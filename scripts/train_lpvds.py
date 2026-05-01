"""
Train a Cartesian-space LPVDS transport model from collected demos.

Usage:
  python scripts/train_lpvds.py --arm left
  python scripts/train_lpvds.py --arm left --K_max 8 --demos data/demonstrations/left_demos_clean.pkl
"""

import sys
import argparse
import pickle
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm",     type=str, default="left")
    parser.add_argument("--config",  type=str, default="configs/default.yaml")
    parser.add_argument("--demos",   type=str, default=None)
    parser.add_argument("--K_max",   type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    demos_dir = Path(cfg["paths"]["demos"])
    demo_path = Path(args.demos) if args.demos else \
                demos_dir / f"{args.arm}_demos_clean.pkl"
    if not demo_path.exists():
        demo_path = demos_dir / f"{args.arm}_demos.pkl"

    print(f"[TRAIN] Loading demos from {demo_path}")
    with open(demo_path, 'rb') as f:
        demos = pickle.load(f)

    # x_goal = EE position at the transport target (same across all demos)
    x_goal = None
    for demo in demos:
        for step in demo['trajectory']:
            if step.get('primitive', 'transport') == 'transport':
                # 'target' is the Cartesian transport target stored in collect_ik
                x_goal = np.asarray(step['target'])
                break
        if x_goal is not None:
            break

    if x_goal is None:
        raise ValueError("No transport steps found. Run collect_ik.py first.")

    print(f"[TRAIN] x_goal (EE) = {np.round(x_goal, 4)} m")

    # Warn if ee_vel missing — will fall back to finite difference
    sample = demos[0]['trajectory'][0]
    if 'ee_vel' not in sample:
        print("[WARN] 'ee_vel' not in demos — will use finite-difference fallback.")
        print("       Re-collect with updated collect_ik.py for better accuracy.")

    from src.lpv_ds import LPVDS, evaluate_lpvds

    model = LPVDS.fit(demos, x_goal=x_goal, K_max=args.K_max,
                      epsilon=args.epsilon, verbose=args.verbose)

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / f"{args.arm}_transport_lpvds.pkl"
    model.save(out_path)

    evaluate_lpvds(model, demos)
    print(f"\n[TRAIN] Done. Deploy with:")
    print(f"  python scripts/deploy_single_arm.py --arm {args.arm} --model lpvds")


if __name__ == "__main__":
    main()
