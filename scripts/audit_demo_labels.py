"""
Audit collected demonstration labels for joint-space DS training.

The learned DS is trained on:

    e = q - q_goal
    q_dot = demonstrated velocity

For each primitive, q_dot should usually point toward -e, and the final sample
of each primitive segment should be close to q_goal. If this is not true, the
primitive label may be fine but the q_goal attractor label is inconsistent with
the demonstrated motion.

Usage:
  python scripts/audit_demo_labels.py data/demonstrations/left_demos.pkl
  python scripts/audit_demo_labels.py data/demonstrations/left_demos.pkl data/demonstrations/right_demos.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


PRIMITIVES = ["reach", "grasp", "lift", "transport", "place"]


def segment_iter(traj):
    i = 0
    while i < len(traj):
        primitive = traj[i]["primitive"]
        j = i + 1
        while j < len(traj) and traj[j]["primitive"] == primitive:
            j += 1
        yield primitive, traj[i:j]
        i = j


def audit_file(path):
    with open(path, "rb") as f:
        demos = pickle.load(f)

    stats = {
        p: {
            "samples": 0,
            "segments": 0,
            "cos": [],
            "edot": [],
            "start_e": [],
            "final_e": [],
            "lengths": [],
            "zero_qdot": 0,
            "lula_error": [],
            "motion_source": {},
            "q_goal_source": {},
        }
        for p in PRIMITIVES
    }
    boundary_jumps = []

    for demo in demos:
        traj = demo.get("trajectory", [])
        previous_q = None
        previous_primitive = None
        for primitive, segment in segment_iter(traj):
            if primitive not in stats:
                continue
            bucket = stats[primitive]
            bucket["segments"] += 1
            bucket["lengths"].append(len(segment))

            first = segment[0]
            last = segment[-1]
            start_e = np.linalg.norm(np.asarray(first["q"]) - np.asarray(first["q_goal"]))
            final_e = np.linalg.norm(np.asarray(last["q"]) - np.asarray(last["q_goal"]))
            bucket["start_e"].append(start_e)
            bucket["final_e"].append(final_e)

            if previous_q is not None:
                jump = np.linalg.norm(np.asarray(first["q"]) - previous_q)
                boundary_jumps.append((previous_primitive, primitive, jump))

            for step in segment:
                q = np.asarray(step["q"])
                q_goal = np.asarray(step["q_goal"])
                q_dot = np.asarray(step["q_dot"])
                e = q - q_goal
                e_norm = np.linalg.norm(e)
                v_norm = np.linalg.norm(q_dot)
                bucket["samples"] += 1
                if v_norm < 1e-9:
                    bucket["zero_qdot"] += 1
                    continue
                if "q_goal_lula_error" in step:
                    bucket["lula_error"].append(float(step["q_goal_lula_error"]))
                motion_source = step.get("motion_source", "legacy_unknown")
                q_goal_source = step.get("q_goal_source", "legacy_unknown")
                bucket["motion_source"][motion_source] = (
                    bucket["motion_source"].get(motion_source, 0) + 1
                )
                bucket["q_goal_source"][q_goal_source] = (
                    bucket["q_goal_source"].get(q_goal_source, 0) + 1
                )
                if e_norm < 1e-9:
                    continue
                bucket["cos"].append(float(np.dot(q_dot, -e) / (v_norm * e_norm)))
                bucket["edot"].append(float(np.dot(e, q_dot)))

            previous_q = np.asarray(last["q"])
            previous_primitive = primitive

    print(f"\n{path}")
    print(f"demos: {len(demos)}")

    for primitive in PRIMITIVES:
        bucket = stats[primitive]
        if bucket["samples"] == 0:
            continue

        cos = np.asarray(bucket["cos"], dtype=float)
        edot = np.asarray(bucket["edot"], dtype=float)
        start_e = np.asarray(bucket["start_e"], dtype=float)
        final_e = np.asarray(bucket["final_e"], dtype=float)
        lengths = np.asarray(bucket["lengths"], dtype=int)
        lula_error = np.asarray(bucket["lula_error"], dtype=float)

        print(f"\n{primitive}")
        print(f"  samples/segments       : {bucket['samples']} / {bucket['segments']}")
        print(f"  motion source counts   : {bucket['motion_source']}")
        print(f"  q_goal source counts   : {bucket['q_goal_source']}")
        print(f"  segment lengths        : {sorted(set(lengths.tolist()))}")
        print(f"  start ||q-q_goal||     : mean={start_e.mean():.3f} med={np.median(start_e):.3f}")
        print(f"  final ||q-q_goal||     : mean={final_e.mean():.3f} med={np.median(final_e):.3f} max={final_e.max():.3f}")
        print(f"  final < 0.10 rad       : {np.mean(final_e < 0.10):.3f}")
        if len(cos):
            print(f"  cos(q_dot, -error)    : mean={cos.mean():.3f} med={np.median(cos):.3f} p10={np.quantile(cos, 0.10):.3f}")
            print(f"  fraction moving away   : {np.mean(cos < 0.0):.3f}")
            print(f"  fraction e_dot positive: {np.mean(edot > 0.0):.3f}")
        print(f"  zero q_dot samples     : {bucket['zero_qdot']}")
        if len(lula_error):
            print(f"  Lula-settled mismatch  : mean={lula_error.mean():.3f} med={np.median(lula_error):.3f} max={lula_error.max():.3f}")

    if boundary_jumps:
        jumps = np.asarray([x[2] for x in boundary_jumps], dtype=float)
        print("\nboundary q jump")
        print(f"  median={np.median(jumps):.3f} max={jumps.max():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_files", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.demo_files:
        if not path.exists():
            print(f"{path}: missing")
            continue
        audit_file(path)


if __name__ == "__main__":
    main()
