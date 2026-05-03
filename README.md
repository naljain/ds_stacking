# Reactive Dual-Arm Cube Stacking with Joint-Space Neural DS

MEAM 6230 Final Project — Nalini Jain, Shivank Gupta, Thomas Stephen Felix

## Overview

This project trains **joint-space Neural Dynamical System (DS)** motion
primitives from demonstrations to enable two Franka Panda arms to stack
cubes independently in a shared workspace. Inter-arm collision avoidance is
handled by **DS modulation** (Khansari-Zadeh & Billard 2012-style state-
dependent velocity shaping) rather than a discrete coordinator, keeping the
closed-loop a pure dynamical system.

## Why joint-space DS?

The DS is trained directly on `q̇ = f(e)` where
`e = q - q_goal`, `q` is the 7-DOF Franka joint configuration, and `q_goal`
is the target joint configuration for the current primitive. At deployment,
Lula IK is queried once at each primitive transition to compute `q_goal`; then
the network output is integrated as joint-position commands until the primitive
converges. There is no Cartesian controller or IK inside the control loop.

The model is parameterized as a stable error-space DS:

```text
f(e_n) = residual_theta(e_n) - stable_skip_gain * e_n
```

where `e_n` is normalized joint error. The stable skip is part of the learned
DS architecture and is trained together with the residual; it is different from
deployment-only `--goal_gain`, which is only a diagnostic controller.

This means:
  - Lyapunov stability claims hold in the actual control space.
  - No singularity / null-space issues from a Cartesian DS + Jacobian inverse.
  - The learned vector field is what's running, not RMPflow.

## Why DS modulation instead of a coordinator FSM?

A discrete "if EEs are close, hold one arm" coordinator would make the system
hybrid — and stability of hybrid systems requires extra dwell-time analysis.
Instead, we apply a state-dependent modulation matrix `M(x_self, x_other)`
that smoothly deflects each arm's velocity around a safety sphere centred on
the other arm's EE. The closed loop is therefore a coupled DS rather than a
hybrid system.

We use the **Huber, Billard & Slotine (2019)** modulation framework
(*"Avoidance of Convex and Concave Obstacles with Convergence Ensured Through
Contraction"*, IEEE RA-L), which extends Khansari-Zadeh & Billard 2012 with:

  - **Reference-direction basis** — `M = E·D·E^{-1}` is built around the
    direction from a reference point inside the obstacle to the agent, rather
    than the surface normal. This eliminates the antipodal saddle-point that
    can trap agents on the back side of an obstacle.
  - **Tail-effect gating** — when the nominal velocity already points away
    from the obstacle, no radial damping is applied, so the agent can leave
    the contention zone naturally.
  - **Contraction-based convergence guarantee** — under the construction
    above, the modulated DS still converges to its attractor.

```
q̇_self = q̇_nom + J_trans^+ · ( M(ee_self, ee_other) - I ) · J_trans · q̇_nom
```

When far apart, `M ≈ I` and each arm tracks its nominal DS exactly.
When close and approaching, `M` damps the radial component and boosts the
tangential component, deflecting motion around the other arm's EE.
When close but already moving outward, `M` falls back to identity (tail
effect), letting the arm separate cleanly.

## Directory Layout

```
ds_stacking/
├── src/
│   ├── env.py               # Isaac Sim scene with two Frankas + table + blocks
│   ├── primitives.py        # Primitive Cartesian targets + completion checks
│   ├── ik_controller.py     # RMPflow wrapper (legacy collection option)
│   ├── franka_ik.py         # Lula IK wrapper for q_goal lookup
│   ├── neural_ds.py         # Joint-space Neural DS + Lyapunov network
│   ├── modulation.py        # DS modulation matrices for collision avoidance
│   ├── coordinator.py       # Slim primitive sequencer (no FSM)
│   └── perturbations.py     # Perturbation injectors for evaluation
├── scripts/
│   ├── smoke_test.py        # Verify the scene loads
│   ├── collect_ik.py        # Generate (q, q_goal, q̇) joint-space demos
│   ├── audit_demo_labels.py # Check primitive/q_goal label consistency
│   ├── teleop.py            # Manual demo collection
│   ├── train_ds.py          # Train Neural DS for one primitive
│   ├── train_all.sh         # Train all 5 primitives sequentially
│   ├── deploy_single_arm.py # Run learned DS on one arm
│   ├── deploy_dual_arm.py   # Both arms with modulation
│   └── evaluate.py          # Full evaluation suite + ablations
├── configs/
│   └── default.yaml         # All hyperparameters and constants
├── data/
│   ├── demonstrations/      # Saved trajectory data (.pkl)
│   ├── checkpoints/         # Trained model weights (.pt)
│   └── results/             # Evaluation JSON files
├── requirements.txt
└── README.md
```

## Pipeline

```bash
# 1. Smoke test — confirm scene loads
python scripts/smoke_test.py

# 2. Collect demos (joint-space)
python scripts/collect_ik.py --arm left  --n_demos 50 --headless
python scripts/collect_ik.py --arm right --n_demos 50 --headless

# 3. Audit demo labels before training
python scripts/audit_demo_labels.py data/demonstrations/left_demos.pkl data/demonstrations/right_demos.pkl

# 4. Train all 5 primitives from pooled left+right demos
bash scripts/train_all.sh

# 5. Pure learned-DS validation on one arm
python scripts/deploy_single_arm.py --arm left --kinematic_carry --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25 --print_every 25 --debug_ik --log_csv data/results/left_pure_ds.csv

# 6. Deploy dual-arm with modulation
python scripts/deploy_dual_arm.py --kinematic_carry --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25

# 7. Run pure learned-DS evaluation
python scripts/evaluate.py --n_trials 10 --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25

# 8. Plot all learned primitive DS fields and rollouts
python scripts/plot_ds.py --all --ckpt_arm both --use_safe --joints 0 1 --out_dir data/results/ds_plots

# 9. Ablations for the writeup
python scripts/evaluate.py --no_modulation                # collision-avoidance ablation
python scripts/evaluate.py --use_safe                     # Lyapunov projection
```

If collection settings change, rerun from step 2 onward. The trained DS is only
as good as the saved demonstrations, so demos with missed grasps, target-noise
artifacts, or jerky primitive transitions should not be reused.

For focused single-arm debugging, train left-only checkpoints before judging the
learned DS:

```bash
python scripts/train_ds.py --primitive reach --arm left
python scripts/train_ds.py --primitive grasp --arm left
python scripts/train_ds.py --primitive lift --arm left
python scripts/train_ds.py --primitive transport --arm left
python scripts/train_ds.py --primitive place --arm left

python scripts/deploy_single_arm.py --arm left --ckpt_arm left --kinematic_carry --use_safe --ds_scale 1.0 --goal_gain 0.0 --done_tol 0.25 --print_every 25 --debug_ik --log_csv data/results/left_pure_ds.csv
```

Timeouts are failures in deployment and evaluation unless
`--advance_on_timeout` is explicitly passed for phase-flow debugging.

Diagnostic controllers such as `--goal_gain > 0`, `--ds_scale < 1`, or
`--ds_scale 0` are useful for isolating IK/actuation problems, but they are not
the learned-DS method and should not be reported as the main result.

To debug the learned vector fields directly, plot every primitive checkpoint:

```bash
python scripts/plot_ds.py --all --ckpt_arm both --use_safe --joints 0 1 --out_dir data/results/ds_plots
```

This creates loss curves, 2D phase portraits, Lyapunov landscapes, and closed-loop
rollouts for `reach`, `grasp`, `lift`, `transport`, and `place`. Use `--ckpt_arm
left` after left-only retraining.

## Setup

```bash
conda create -n franka_isaac python=3.11 -y
conda activate franka_isaac
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
pip install torch numpy pyyaml tqdm matplotlib
```

Isaac Sim's Franka helper resolves the robot USD through the configured Isaac
assets root. If your install cannot reach the default assets root, set a local
Franka USD in `configs/default.yaml`:

```yaml
assets:
  franka_usd: /path/to/Assets/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd
```

The ground plane is built from a local cuboid, so it does not require the Isaac
asset server.

## Data Collection Notes

`collect_ik.py` now defaults to `--motion_source joint_lula`. For each
primitive it computes the same Lula `q_goal` used by deployment, then records a
joint-space expert trajectory moving toward that attractor. This keeps the
training label `q_goal` and the demonstrated `q_dot` in the same joint-space DS.

The older Cartesian RMPflow collection path is still available with
`--motion_source rmpflow`, but it is diagnostic only for the pure joint-space DS
pipeline because RMPflow and Lula can choose different redundant-arm null-space
solutions for the same Cartesian target.

The default collection path is conservative: it targets the observed block pose
directly, uses no block jitter, and uses no target noise. Keep legacy target
noise `--noise` at `0` for reliable grasp demos; nonzero target noise commands
the gripper beside the observed block and can produce failed grasps.

After the base grasp is reliable, `--block_xy_jitter` can be used to move the
physical blocks and widen the data distribution without commanding grasps away
from the cube.

With `--motion_source joint_lula`, collection uses extra joint-space settling
steps before switching primitives. With `--motion_source rmpflow`, those same
extra steps are RMPflow settling steps. By default, after `grasp` collection
kinematically carries the active block with the EE so clean joint-space demos
are not discarded because Isaac contact grasping is flaky. Use `--physical_grasp`
when you explicitly want to test whether the parallel gripper/contact setup can
lift the cube; in that mode failed lifts are discarded.

`q_goal` labels matter as much as primitive labels. The default collection mode
uses `--motion_source joint_lula --q_goal_source lula`, so each sample's
attractor is the same style of Lula joint target used at deployment. The
collector also stores the terminal expert state as metadata (`q_goal_settled`,
`q_goal_lula_error`) so mismatches can be audited. If you use the legacy
`--motion_source rmpflow` path, large `q_goal_lula_error` values mean RMPflow is
moving toward a different null-space solution than deployment will use.

Before training, run:

```bash
python scripts/audit_demo_labels.py data/demonstrations/left_demos.pkl data/demonstrations/right_demos.pkl
```

For each primitive, the final `||q - q_goal||` should be small and
`cos(q_dot, -error)` should usually be positive. Large final errors or a high
fraction of negative cosines mean the DS is being trained on inconsistent
attractor labels.

## Data Partitioning for DS Training

The demonstration pickle for an arm contains full pick-and-stack trajectories,
not one file per primitive. Every recorded timestep stores a `primitive` label.
`scripts/train_ds.py` partitions the data by filtering on that label:

```text
reach samples     -> *_reach.pt
grasp samples     -> *_grasp.pt
lift samples      -> *_lift.pt
transport samples -> *_transport.pt
place samples     -> *_place.pt
```

The `--arm` flag controls which files are pooled:

| Training command | Source demos | Checkpoint prefix |
|---|---|---|
| `--arm left` | `left_demos.pkl` | `left_*` |
| `--arm right` | `right_demos.pkl` | `right_*` |
| `--arm both` | left + right demos | `both_*` |

Each checkpoint now stores a `data_manifest` with the primitive label, source
demo files, total sample count, samples by file, samples by arm, and samples by
block. The same information is printed during training so it is explicit which
data trained each DS.

## Primitives

Each pick-and-stack sequence decomposes into 5 learned primitives:

| Primitive | Cartesian goal | Joint goal |
|---|---|---|
| `reach`     | hover above source block | IK of hover pose |
| `grasp`     | descend to block surface | IK of grasp pose |
| `lift`      | raise to transport altitude | IK of lift pose |
| `transport` | move above stacking goal | IK of transport pose |
| `place`     | descend onto stack | IK of place pose |

Each primitive has its own `f_theta` and `V_phi` networks. At a primitive
transition, Lula IK is queried once to compute the new `q_goal`; the DS then
drives `q -> q_goal` smoothly until the next transition.

## Dual-Arm Coordination

The coordinator remains deliberately slim: it chooses block order, primitive
order, and stack slots. It does not perform close-range hold/release collision
logic. Inter-arm collision avoidance remains the responsibility of continuous
DS modulation.

Dual-arm deployment uses a small initial phase offset by default:
`coordination.start_stagger_steps: 30`, roughly 0.25 s at 120 Hz. The left arm
starts immediately and the right arm starts after the stagger. This avoids a
perfectly symmetric race into the shared stack while preserving the DS +
modulation framing. Override it with `--stagger_steps`.

Shared stack layers are reserved before `transport/place`, so both arms cannot
target the same stack height when they arrive at the goal area concurrently.
The reserved height is the desired block center height. During kinematic-carry
debug deployment, the carried block is snapped to that reserved stack slot when
`place` opens the gripper; otherwise the visual block pose can reflect the EE
carry offset instead of the coordinator's stack layer.

If a newly trained DS does not converge at deployment, inspect
`cos→goal` in `deploy_single_arm.py`. Negative values mean the learned vector
field is pointing away from `q_goal`. The pure learned-DS setting is
`--ds_scale 1.0 --goal_gain 0.0`. Any nonzero `--goal_gain` adds a hand-coded
linear attractor and is diagnostic only.

Single-arm deployment aborts on primitive timeout by default. This prevents a
failed `reach` from cascading into a false `grasp`. Use `--advance_on_timeout`
only for phase-flow debugging.

## Stability

Per-primitive: `V(e)` is positive-definite around `e = q - q_goal = 0` by construction;
training enforces `dV/dt + α·V <= 0` on the data distribution; inference can
optionally project `f(x)` onto the half-space where this holds exactly
(`--use_safe`).

Across primitives: switched-system stability via dwell-time. We measure the
per-primitive convergence time empirically and report it.

Across arms: modulation preserves the stability of each arm's DS individually
because `M(x)` is diagonal in the basis aligned with the obstacle normal and
the radial scaling is non-negative — so `dV/dt` of each arm's individual
Lyapunov function remains non-positive after modulation.
