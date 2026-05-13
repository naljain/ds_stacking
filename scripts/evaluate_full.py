"""
Nominal evaluation for the current Neural DS stacking pipeline.

This script intentionally keeps the task-level evaluation small and aligned
with the maintained deployment entry points:

  1. nominal single-arm deployment
  2. nominal dual-arm deployment with protected-point modulation
  3. optional nominal dual-arm deployment with modulation disabled
  4. optional nominal dual-arm deployment with a start stagger

It does not inject perturbations. It launches deploy_single_arm.py and
deploy_dual_arm.py as subprocesses, then parses their logs for task success,
blocks placed, timeout/failure mode, approximate simulated runtime, and
minimum reported inter-arm distance.
"""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrialMetrics:
    case: str
    trial_idx: int
    command: str
    returncode: int
    success: bool
    blocks_placed: int
    expected_blocks: int
    stack_completion_rate: float
    simulated_time_s: float
    time_per_cube_s: float
    min_inter_arm_distance_m: float
    modulation_events: int
    failure_mode: str
    log_path: str


@dataclass
class AggregateMetrics:
    case: str
    n_trials: int
    success_rate: float
    blocks_placed_mean: float
    blocks_placed_std: float
    completion_rate_mean: float
    time_per_cube_mean: float
    min_inter_arm_distance_mean: float
    modulation_events_mean: float


def parse_steps(output: str) -> int | None:
    match = re.search(r"\[DEPLOY\]\s+Finished after\s+(\d+)\s+steps", output)
    return int(match.group(1)) if match else None


def parse_place_slots(output: str) -> int:
    slots = {
        int(m.group(1))
        for m in re.finditer(r"\[STACK\]\s+\w+/place\s+slot=(\d+)", output)
    }
    return len(slots)


def parse_min_distance(output: str) -> float:
    # Current dual-arm safety logs report "link distance X m". If link hold is
    # disabled, this may be unavailable; leave as NaN rather than inventing it.
    dists = [
        float(m.group(1))
        for m in re.finditer(r"link distance\s+([0-9.]+)\s*m", output)
    ]
    return min(dists) if dists else float("nan")


def classify_failure(output: str, returncode: int, success: bool) -> str:
    if success:
        return "none"
    if returncode != 0:
        if "ModuleNotFoundError" in output or "ImportError" in output:
            return "import_error"
        return f"returncode_{returncode}"
    if "timed out" in output:
        timeout = re.search(r"\[WARN\]\s+(.+?)\s+timed out", output)
        return f"timeout:{timeout.group(1)}" if timeout else "timeout"
    if "IK failed" in output or "ok=False" in output:
        return "ik_failure"
    if "Aborting" in output:
        return "aborted"
    return "incomplete"


def run_command(cmd: List[str], log_path: Path) -> tuple[int, str, float]:
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    wall_time = time.time() - start
    output = proc.stdout or ""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output)
    return proc.returncode, output, wall_time


def build_single_command(args, arm: str) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "deploy_single_arm.py"),
        "--arm", arm,
        "--ckpt_arm", arm,
        "--deploy_config", args.single_deploy_config,
        "--config", args.config,
    ]
    if args.headless:
        cmd.append("--headless")
    if args.kinematic_carry:
        cmd.append("--kinematic_carry")
    if args.use_safe:
        cmd.append("--use_safe")
    if args.print_every:
        cmd += ["--print_every", str(args.print_every)]
    return cmd


def build_dual_command(args, case: str) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "deploy_dual_arm.py"),
        "--deploy_config", args.dual_deploy_config,
        "--config", args.config,
    ]
    if args.headless:
        cmd.append("--headless")
    if args.kinematic_carry:
        cmd.append("--kinematic_carry")
    if args.use_safe:
        cmd.append("--use_safe")
    if case == "dual_nomod":
        cmd.append("--no_modulation")
    if case == "dual_stagger":
        cmd += ["--stagger_steps", str(args.stagger_steps)]
    if args.link_safety_hold:
        cmd.append("--link_safety_hold")
    if args.print_every:
        cmd += ["--print_every", str(args.print_every)]
    return cmd


def evaluate_case(args, cfg, case: str, trial_idx: int, out_dir: Path) -> TrialMetrics:
    if case.startswith("single_"):
        arm = case.split("_", 1)[1]
        expected_blocks = len(cfg[f"{arm}_blocks"])
        cmd = build_single_command(args, arm)
    else:
        expected_blocks = len(cfg["left_blocks"]) + len(cfg["right_blocks"])
        cmd = build_dual_command(args, case)

    log_path = out_dir / "logs" / f"{case}_trial{trial_idx + 1:02d}.log"
    returncode, output, wall_time = run_command(cmd, log_path)

    if case.startswith("single_"):
        success = "[DEPLOY] All blocks placed." in output
    else:
        success = "[DEPLOY] Both arms finished stacking." in output

    steps = parse_steps(output)
    sim_time = (
        steps * float(cfg["sim"]["physics_dt"])
        if steps is not None else wall_time
    )
    blocks_placed = expected_blocks if success else parse_place_slots(output)
    completion = blocks_placed / max(expected_blocks, 1)
    time_per_cube = sim_time / blocks_placed if blocks_placed > 0 else float("nan")
    min_dist = parse_min_distance(output) if case.startswith("dual_") else float("nan")
    mod_events = output.count("[SAFETY]") if case.startswith("dual_") else 0

    return TrialMetrics(
        case=case,
        trial_idx=trial_idx,
        command=" ".join(cmd),
        returncode=returncode,
        success=success,
        blocks_placed=blocks_placed,
        expected_blocks=expected_blocks,
        stack_completion_rate=completion,
        simulated_time_s=sim_time,
        time_per_cube_s=time_per_cube,
        min_inter_arm_distance_m=min_dist,
        modulation_events=mod_events,
        failure_mode=classify_failure(output, returncode, success),
        log_path=str(log_path),
    )


def aggregate(case: str, trials: List[TrialMetrics]) -> AggregateMetrics:
    def mean(vals):
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def std(vals):
        vals = [v for v in vals if not math.isnan(v)]
        if len(vals) <= 1:
            return 0.0 if vals else float("nan")
        m = mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    return AggregateMetrics(
        case=case,
        n_trials=len(trials),
        success_rate=mean([1.0 if t.success else 0.0 for t in trials]),
        blocks_placed_mean=mean([t.blocks_placed for t in trials]),
        blocks_placed_std=std([t.blocks_placed for t in trials]),
        completion_rate_mean=mean([t.stack_completion_rate for t in trials]),
        time_per_cube_mean=mean([t.time_per_cube_s for t in trials]),
        min_inter_arm_distance_mean=mean([t.min_inter_arm_distance_m for t in trials]),
        modulation_events_mean=mean([float(t.modulation_events) for t in trials]),
    )


def write_outputs(out_dir: Path, trials: List[TrialMetrics],
                  aggregates: List[AggregateMetrics]):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"nominal_eval_{ts}.json"
    json_path.write_text(json.dumps({
        "trials": [asdict(t) for t in trials],
        "aggregates": [asdict(a) for a in aggregates],
    }, indent=2))

    trials_csv = out_dir / f"nominal_eval_{ts}_trials.csv"
    with trials_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TrialMetrics.__dataclass_fields__))
        writer.writeheader()
        for row in trials:
            writer.writerow(asdict(row))

    agg_csv = out_dir / f"nominal_eval_{ts}_summary.csv"
    with agg_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(AggregateMetrics.__dataclass_fields__))
        writer.writeheader()
        for row in aggregates:
            writer.writerow(asdict(row))

    print("\nWrote evaluation outputs:")
    print(f"  {json_path}")
    print(f"  {trials_csv}")
    print(f"  {agg_csv}")


def print_summary(aggregates: List[AggregateMetrics]):
    header = (
        f"{'Case':<14} {'N':>3} {'Success':>8} {'Blocks':>8} "
        f"{'Compl.':>8} {'Time/Cube':>10} {'MinDist':>8} {'Events':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for a in aggregates:
        print(
            f"{a.case:<14} {a.n_trials:>3d} "
            f"{a.success_rate:>8.2f} "
            f"{a.blocks_placed_mean:>8.2f} "
            f"{a.completion_rate_mean:>8.2f} "
            f"{a.time_per_cube_mean:>10.2f} "
            f"{a.min_inter_arm_distance_mean:>8.3f} "
            f"{a.modulation_events_mean:>8.1f}"
        )
    print("=" * len(header))


def build_parser():
    p = argparse.ArgumentParser(
        description="Nominal task-level evaluation for current Neural DS deploy scripts."
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out_dir", default="data/results")
    p.add_argument("--n_trials", type=int, default=1)
    p.add_argument(
        "--cases", nargs="+",
        default=["single_left", "single_right", "dual_mod"],
        choices=["single_left", "single_right", "dual_mod", "dual_nomod", "dual_stagger"],
    )
    # Backward-compatible aliases from the earlier evaluator. Conditions are
    # ignored because this script is nominal-only. Neural variants are mapped to
    # current cases; LPV variants are dropped because LPV code is not present on
    # this branch.
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--single_deploy_config",
                   default="configs/deploy_single_neural_physical.yaml")
    p.add_argument("--dual_deploy_config",
                   default="configs/deploy_neural_physical.yaml")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--kinematic_carry", action="store_true")
    p.add_argument("--use_safe", action="store_true")
    p.add_argument("--link_safety_hold", action="store_true")
    p.add_argument("--stagger_steps", type=int, default=720)
    p.add_argument("--print_every", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    if args.conditions:
        print("[WARN] --conditions is ignored; this evaluator is nominal-only.")
    if args.variants:
        mapped_cases = []
        for variant in args.variants:
            if variant == "neural_mod":
                mapped_cases.append("dual_mod")
            elif variant == "neural_nomod":
                mapped_cases.append("dual_nomod")
            elif "lpv" in variant:
                print(f"[WARN] dropping {variant}: LPV is not available on this branch.")
            else:
                print(f"[WARN] unknown variant {variant!r}; ignoring.")
        if mapped_cases:
            args.cases = list(dict.fromkeys(mapped_cases))
            print(f"[INFO] mapped --variants to --cases {' '.join(args.cases)}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    trials: List[TrialMetrics] = []
    aggregates: List[AggregateMetrics] = []
    total = len(args.cases) * args.n_trials
    run_idx = 0

    for case in args.cases:
        case_trials = []
        for trial_idx in range(args.n_trials):
            run_idx += 1
            print(f"\n[{run_idx}/{total}] case={case} trial={trial_idx + 1}")
            trial = evaluate_case(args, cfg, case, trial_idx, out_dir)
            trials.append(trial)
            case_trials.append(trial)
            print(
                f"  success={trial.success} "
                f"blocks={trial.blocks_placed}/{trial.expected_blocks} "
                f"time_per_cube={trial.time_per_cube_s:.2f}s "
                f"failure={trial.failure_mode}"
            )
        aggregates.append(aggregate(case, case_trials))

    print_summary(aggregates)
    write_outputs(out_dir, trials, aggregates)


if __name__ == "__main__":
    main()
