# Reactive Dual-Arm Cube Stacking with Learned DS

MEAM 6230 Final Project — Nalini Jain, Shivank Gupta, Thomas Stephen Felix

This project explores learned dynamical-system (DS) motion primitives for
dual-arm cube stacking in Isaac Sim. The final system is intentionally hybrid:
learned joint-space Neural DS controllers handle the long free-space motions
(`reach` and `transport`), while scripted Lula IK handles the short constrained
motions (`grasp`, `lift`, and `place`).

![Dual-arm neural DS stacking](data/results/dual_neural.gif)

## Story

The first goal was to learn a Neural DS for the full pick-and-place behavior.
That failed for practical reasons: noisy demonstrations, inconsistent
primitive boundaries, data processing issues, and spurious attractors in the
learned vector field. A spurious attractor is a false goal where the robot can
stall or curve away from the intended target.

The final approach keeps the parts that worked:

- use Lula IK to produce consistent joint-space goals, `q_goal`;
- train the DS in joint-error coordinates, `e = q - q_goal`;
- learn only `reach` and `transport`, where reactive free-space motion helps;
- script `grasp`, `lift`, and `place`, where contact and precision dominate;
- use Cartesian modulation so the two arms can share the stack region.

## Results

| Condition | Blocks placed | Completion | Simulated time | Notes |
|---|---:|---:|---:|---|
| Modulation on | 6 / 6 | 100% | 14.4 s | Both arms returned home |
| Modulation off | 0 / 6 | 0% | 37.5 s timeout | Failed in first shared-stack transport |

Other observations:

- Data quality mattered more than the exact neural architecture.
- Learning all five primitives was unstable and unnecessary.
- Lyapunov safe projection changed failures from divergence to stall.
- End-effector modulation was necessary for the shared stack, but sampled-link
  safety was still useful because elbows and forearms are not protected by an
  end-effector sphere alone.
- LPV-DS was a useful stable-by-construction reference for transport, while
  Neural DS was more expressive in joint space.

The full writeup is in [`report/report.tex`](report/report.tex), with a built
PDF at [`report/report.pdf`](report/report.pdf).

## Repository Structure

```text
ds_stacking/
├── configs/
│   ├── default.yaml                  # Main scene, controller, and training config
│   ├── deploy_neural_physical.yaml   # Physical/contact deployment variant
│   └── deploy_single_neural_physical.yaml
├── data/
│   ├── demonstrations/               # Saved Lula-labeled demo pickles
│   ├── checkpoints/                  # Trained Neural DS checkpoints
│   └── results/                      # Evaluation artifacts and final GIF
├── report/
│   ├── report.tex                    # Project report source
│   └── report.pdf                    # Built report
├── scripts/
│   ├── smoke_test.py                 # Verify Isaac Sim scene loads
│   ├── collect_ik.py                 # Collect joint-space demonstrations
│   ├── audit_demo_labels.py          # Check q_goal / velocity consistency
│   ├── train_ds.py                   # Train one DS primitive
│   ├── train_all.sh                  # Train reach and transport checkpoints
│   ├── deploy_single_arm.py          # Single-arm hybrid DS + Lula deployment
│   ├── deploy_dual_arm.py            # Dual-arm deployment with modulation
│   ├── evaluate.py                   # Evaluation and ablations
│   ├── plot_ds.py                    # DS diagnostics and rollouts
│   └── plot_modulation.py            # Modulation diagnostics
└── src/
    ├── env.py                        # Isaac Sim scene construction
    ├── primitives.py                 # Primitive targets and completion checks
    ├── franka_ik.py                  # Lula IK wrapper
    ├── neural_ds.py                  # Neural DS and Lyapunov network
    ├── modulation.py                 # Huber-style DS modulation
    ├── coordinator.py                # Primitive sequencing and stack slots
    └── perturbations.py              # Evaluation perturbations
```

## Setup

```bash
conda create -n franka_isaac python=3.11 -y
conda activate franka_isaac
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
pip install torch numpy pyyaml tqdm matplotlib
```

If Isaac Sim cannot resolve the Franka assets, set a local USD path in
`configs/default.yaml`:

```yaml
assets:
  franka_usd: /path/to/Assets/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd
```

## Main Pipeline

Run all commands from the repository root.

```bash
# 1. Confirm the Isaac Sim scene loads.
python scripts/smoke_test.py

# 2. Collect Lula-labeled demonstrations.
python scripts/collect_ik.py --arm left --n_demos 50 --headless --block_xy_jitter 0.02 --start_jitter 0.15
python scripts/collect_ik.py --arm right --n_demos 50 --headless --block_xy_jitter 0.02 --start_jitter 0.15

# 3. Audit labels before training.
python scripts/audit_demo_labels.py data/demonstrations/left_demos.pkl data/demonstrations/right_demos.pkl

# 4. Train learned DS checkpoints.
bash scripts/train_all.sh

# 5. Validate one arm.
python scripts/deploy_single_arm.py --arm left --kinematic_carry --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25 --cart_done_tol 0.02 --place_cart_done_tol 0.01 --print_every 25 --debug_ik --log_csv data/results/left_ds_lula_scripted.csv

# 6. Deploy both arms with modulation.
python scripts/deploy_dual_arm.py --kinematic_carry --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25 --cart_done_tol 0.02 --place_cart_done_tol 0.01 --mod_safe_radius 0.25 --mod_reactivity 2.0 --link_safety_radius 0.20
```

For ablations:

```bash
python scripts/deploy_dual_arm.py --no_modulation --kinematic_carry --use_safe
python scripts/evaluate.py --n_trials 10 --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25
python scripts/evaluate.py --n_trials 10 --no_modulation
```

## Build the Report

```bash
cd report
latexmk -pdf -interaction=nonstopmode report.tex
```

The report source uses robust figure placeholders, so it still builds if a
local image artifact is missing.

## Useful Diagnostics

```bash
# Plot learned vector fields, Lyapunov landscapes, and rollouts.
python scripts/plot_ds.py --all --ckpt_arm both --use_safe --joints 0 1 --out_dir data/results/ds_plots

# Plot modulation diagnostics after an evaluation run.
python scripts/plot_modulation.py all --diag data/results/diag_<timestamp>.pkl
```

Important checks:

- `cos(q_dot, -error)` should usually be positive in demos and deployment.
- Large final `||q - q_goal||` in `audit_demo_labels.py` means the DS is being
  trained with inconsistent attractor labels.
- `--goal_gain > 0`, `--ds_scale < 1`, and `--ds_scale 0` are diagnostics, not
  the learned-DS method.
- Deployment timeouts should be treated as failures unless intentionally using
  `--advance_on_timeout` for debugging.

## Implementation Notes

- The DS is trained on `q_dot = f(e)`, where `e = q - q_goal`.
- The Neural DS structurally enforces `f(0) = 0` by subtracting the network
  output at the origin.
- The Lyapunov network defines
  `V(e) = ||g(e) - g(0)||^2 + epsilon ||e||^2`.
- `--use_safe` projects the commanded velocity so `dV/dt <= 0` at deployment.
- End-effector modulation bends each arm's Cartesian velocity around the other
  arm, then maps the correction back to joint space through a damped Jacobian
  pseudoinverse.
- The sampled-link hold is a practical safety guard for elbow and forearm
  proximity; it is separate from the continuous modulation guarantee.
