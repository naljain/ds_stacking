"""Train 3D Cartesian LPVDS. Usage: python scripts/train_lpvds.py --arm left"""

import sys, argparse, pickle, numpy as np, yaml
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
    parser.add_argument("--use_clean", action="store_true",
                        help="Use data/demonstrations/{arm}_demos_clean.pkl")
    parser.add_argument("--allow_target_mismatch", action="store_true",
                        help="Train even if demo targets differ from current config shared_goal")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    demos_dir = Path(cfg["paths"]["demos"])
    if args.demos:
        demo_path = Path(args.demos)
    elif args.use_clean:
        demo_path = demos_dir / f"{args.arm}_demos_clean.pkl"
    else:
        demo_path = demos_dir / f"{args.arm}_demos.pkl"

    if not demo_path.exists():
        raise FileNotFoundError(
            f"{demo_path} does not exist. Collect demos first, or pass --demos."
        )

    print(f"[TRAIN] Loading demos from {demo_path}")
    with open(demo_path, 'rb') as f:
        demos = pickle.load(f)

    targets = []
    for demo in demos:
        for step in demo["trajectory"]:
            if step.get("primitive", "transport") == "transport" and "target" in step:
                targets.append(np.asarray(step["target"], dtype=float))
                break

    if not targets:
        raise ValueError("No transport targets found. Re-run collect_ik.py.")

    x_goal = np.mean(np.stack(targets), axis=0)
    goal_xy = cfg["shared_goal"]
    cfg_goal = np.array([goal_xy[0], goal_xy[1], cfg["heights"]["lift"]])
    target_error = np.linalg.norm(x_goal - cfg_goal)
    if target_error > 1e-3:
        msg = (f"Demo target {np.round(x_goal, 4)} differs from config target "
               f"{np.round(cfg_goal, 4)}. Re-collect demos if you intended to "
               "train for the config target.")
        if not args.allow_target_mismatch:
            raise ValueError(msg + " Use --allow_target_mismatch only for experiments.")
        print(f"[WARN] {msg}")
    print(f"[TRAIN] x_goal (EE): {np.round(x_goal, 4)}")

    from src.lpv_ds import LPVDS, evaluate_lpvds
    model = LPVDS.fit(demos, x_goal=x_goal, K_max=args.K_max,
                      epsilon=args.epsilon, verbose=args.verbose)

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / f"{args.arm}_transport_lpvds.pkl"
    model.save(out_path)
    evaluate_lpvds(model, demos)

    print(f"\n[TRAIN] Deploy with:")
    print(f"  python scripts/deploy_single_arm.py --arm {args.arm} --model lpvds")

if __name__ == "__main__":
    main()
