# Reactive Dual-Arm Cube Stacking with Joint-Space Neural DS

MEAM 6230 Final Project — Nalini Jain, Shivank Gupta, Thomas Stephen Felix

## Overview

This project trains **joint-space Neural Dynamical System (DS)** motion
primitives from demonstrations to enable two Franka Panda arms to stack cubes
in a shared workspace. The default learned DS primitives are `reach` and
`transport`. The short constrained motions, `grasp`, `lift`, and `place`, are
executed with a Lula joint-space controller. Physical deployment presets can
also run `reach` through Lula so pickup starts from a deterministic pre-grasp
hover while `transport` remains learned.

Dual-arm deployment combines protected-point DS modulation, optional
sampled-link safety hold, dynamic stack-height clearance, and return-to-home
parking for completed arms.

## Why joint-space DS?

The DS is trained directly on `q̇ = f(e)` where
`e = q - q_goal`, `q` is the 7-DOF Franka joint configuration, and `q_goal`
is the target joint configuration for the current primitive. At deployment,
Lula IK is queried at each primitive transition to compute `q_goal`. For
`reach` and `transport`, the learned network output is integrated as
joint-position commands. For `grasp`, `lift`, and `place`, the same Lula
`q_goal` is followed by a clamped joint-space controller.

The model is parameterized as a stable error-space DS:

```text
f(e_n) = residual_theta(e_n) - stable_skip_gain * e_n
```

where `e_n` is normalized joint error. The stable skip is part of the learned
DS architecture and is trained together with the residual. The deploy scripts
keep the external `DS_GOAL_GAIN` constant at `0.0`; change it in code only as a
diagnostic fallback.

The Lyapunov candidate is now the quadratic `V(e_n)=||e_n||^2`. Training uses a
uniform scalar `state_std` across all joints, so decreasing `V` corresponds to
decreasing unnormalized joint-space error. The older learned Lyapunov feature
network has been removed.

This means:
  - Lyapunov stability claims hold in the actual control space.
  - No singularity / null-space issues from a Cartesian DS + Jacobian inverse.
  - The learned vector field runs only for reach and transport.

## Dual-Arm Safety

The nominal collision-avoidance layer is protected-point modulation. It applies
a state-dependent matrix `M(x_self, x_other)` that deflects Cartesian velocity
around proxy spheres on the other arm, then maps that Cartesian correction back
to joint space.

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

End-effector-only modulation was not enough for this task because elbows,
wrists, and gripper bodies can collide even when grippers avoid each other. The
current dual-arm deployment builds protected points from distal Lula FK frames
and gripper proxy offsets, modulates the closest protected-point pairs, and can
add lateral-order modulation to keep the left/right arms on their own sides.
A sampled-link hold is still available via `--link_safety_hold`, but it is
disabled by default and should be treated as a pragmatic safety guard, not part
of the continuous modulation proof.

## Directory Layout

```
ds_stacking/
├── src/
│   ├── env.py               # Isaac Sim scene with two Frankas + table + blocks
│   ├── primitives.py        # Primitive Cartesian targets + completion checks
│   ├── franka_ik.py         # Lula IK wrapper for q_goal lookup
│   ├── neural_ds.py         # Joint-space Neural DS + quadratic Lyapunov helper
│   ├── modulation.py        # Huber modulation + protected-point wrappers
│   ├── coordinator.py       # Primitive sequencing, stack slots, return-home
│   └── perturbations.py     # Perturbation injectors for evaluation
├── scripts/
│   ├── smoke_test.py        # Verify the scene loads
│   ├── collect_ik.py        # Generate (q, q_goal, q̇) joint-space demos
│   ├── audit_demo_labels.py # Check primitive/q_goal label consistency
│   ├── teleop.py            # Manual demo collection
│   ├── train_ds.py          # Train Neural DS for one primitive
│   ├── train_all.sh         # Train reach/transport DS checkpoints
│   ├── deploy_single_arm.py # One arm with DS + scripted Lula primitives
│   ├── deploy_dual_arm.py   # Dual arm with modulation + link safety
│   └── evaluate.py          # Full evaluation suite + ablations
├── configs/
│   ├── default.yaml                         # Scene, training, coordination constants
│   ├── deploy_single_neural_physical.yaml   # Single-arm deploy defaults
│   └── deploy_neural_physical.yaml          # Dual-arm deploy defaults
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
python scripts/collect_ik.py --arm left --n_demos 50 --headless --block_xy_jitter 0.02 --start_jitter 0.15
python scripts/collect_ik.py --arm right --n_demos 50 --headless --block_xy_jitter 0.02 --start_jitter 0.15

# 3. Audit demo labels before training
python scripts/audit_demo_labels.py data/demonstrations/left_demos.pkl data/demonstrations/right_demos.pkl

# 4. Train per-arm reach/transport DS primitives
bash scripts/train_all.sh

# 5. Hybrid DS + Lula validation on one arm
python scripts/deploy_single_arm.py --arm left --ckpt_arm left --deploy_config configs/deploy_single_neural_physical.yaml --kinematic_carry --print_every 25 --debug_ik --log_csv data/results/left_ds_lula_scripted.csv

# 6. Deploy dual-arm with modulation
python scripts/deploy_dual_arm.py --deploy_config configs/deploy_neural_physical.yaml

# 7. Run hybrid evaluation
python scripts/evaluate.py --n_trials 10 --use_safe --ds_scale 1.0 --done_tol 0.05

# 8. Plot learned DS fields and rollouts
python scripts/plot_ds.py --all --ckpt_arm left --use_safe --joints 0 1 --out_dir data/results/ds_plots/left
python scripts/plot_ds.py --all --ckpt_arm right --use_safe --joints 0 1 --out_dir data/results/ds_plots/right

# 9. Ablations for the writeup
python scripts/evaluate.py --no_modulation                # collision-avoidance ablation
python scripts/evaluate.py --use_safe                     # Lyapunov projection
```

If collection settings change, rerun from step 2 onward. The trained DS is only
as good as the saved demonstrations, so demos with missed grasps, target-noise
artifacts, or jerky primitive transitions should not be reused.

For focused single-arm debugging, use the same arm's checkpoints:

```bash
python scripts/train_ds.py --primitive reach --arm left
python scripts/train_ds.py --primitive transport --arm left

python scripts/deploy_single_arm.py --arm left --ckpt_arm left --deploy_config configs/deploy_single_neural_physical.yaml --kinematic_carry --print_every 25 --debug_ik --log_csv data/results/left_ds_lula_scripted.csv
```

Timeouts are failures in deployment and evaluation unless
`--advance_on_timeout` is explicitly passed for phase-flow debugging.

Diagnostic settings such as `--ds_scale < 1` or `--ds_scale 0` are useful for
isolating DS contribution, but they are not the learned-DS method and should
not be reported as the main result.

To debug the learned vector fields directly, plot the learned DS checkpoints:

```bash
python scripts/plot_ds.py --all --ckpt_arm left --use_safe --joints 0 1 --out_dir data/results/ds_plots/left
python scripts/plot_ds.py --all --ckpt_arm right --use_safe --joints 0 1 --out_dir data/results/ds_plots/right
```

This creates loss curves, 2D phase portraits, Lyapunov landscapes, and closed-loop
rollouts for `reach` and `transport`.

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

`collect_ik.py` computes the same Lula `q_goal` used by deployment, then
records a joint-space expert trajectory moving toward that attractor. This
keeps the training label `q_goal` and the demonstrated `q_dot` in the same
joint-space convention.

Collection defaults are intentionally slower and cleaner than early runs:

- `--joint_goal_gain 2.0`
- `--collection_max_joint_vel 1.2`
- `sim.inter_primitive_pause_steps: 120`

The pause is non-recorded. It lets the arm settle between primitives without
teaching the DS to stop at non-goal states.

The default collection path is conservative: it targets the observed block pose
directly, uses no block jitter, and uses no target noise. Keep legacy target
noise `--noise` at `0` for reliable grasp demos; nonzero target noise commands
the gripper beside the observed block and can produce failed grasps.

After the base grasp is reliable, `--block_xy_jitter` can be used to move the
physical blocks and widen the data distribution without commanding grasps away
from the cube.

Collection uses extra joint-space settling steps plus a non-recorded pause
between primitives. Transport targets use the same dynamic stack clearance as
deployment:

```text
transport_z = max(lift_h, existing_stack_top + stack.clearance_above_top)
```

By default, after `grasp` collection kinematically carries the active block with
the EE so clean joint-space demos are not discarded because Isaac contact
grasping is flaky. Use `--physical_grasp` when you explicitly want to test
whether the parallel gripper/contact setup can lift the cube; in that mode
failed lifts are discarded.

`q_goal` labels matter as much as primitive labels. The default collection mode
uses Lula, so each sample's attractor is the same style of joint target used at
deployment. The
collector also stores the terminal expert state as metadata (`q_goal_settled`,
`q_goal_lula_error`) so mismatches can be audited.

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
transport samples -> *_transport.pt
```

`grasp`, `lift`, and `place` remain labeled in the demonstrations for auditing,
but they are executed with the Lula joint-space controller rather than trained
as DS checkpoints.

The `--arm` flag controls which files are loaded:

| Training command | Source demos | Checkpoint prefix |
|---|---|---|
| `--arm left` | `left_demos.pkl` | `left_*` |
| `--arm right` | `right_demos.pkl` | `right_*` |
| `--arm both` | left + right demos | `both_*` ablation/manual experiment |

Each checkpoint now stores a `data_manifest` with the primitive label, source
demo files, total sample count, samples by file, samples by arm, samples by
block, samples by stack slot, and label-source counts. The same information is
printed during training so it is explicit which data trained each DS.

The standard `scripts/train_all.sh` trains per-arm checkpoints, not pooled
`both_*` checkpoints.

## Primitives

Each pick-and-stack sequence decomposes into 5 primitives:

| Primitive | Cartesian goal | Joint goal |
|---|---|---|
| `reach`     | hover above source block | IK of hover pose |
| `grasp`     | descend to block surface | IK of grasp pose |
| `lift`      | raise to transport altitude | IK of lift pose |
| `transport` | move above stacking goal with dynamic stack clearance | IK of transport pose |
| `place`     | descend onto stack | IK of place pose |

Only `reach` and `transport` have learned `f_theta` and `V_phi` networks. At a
primitive transition, Lula IK is queried once to compute the new `q_goal`.
`reach` and `transport` use the learned DS to drive `q -> q_goal`; `grasp`,
`lift`, and `place` use the same `q_goal` with a clamped Lula joint-space
controller. Scripted primitive completion is checked with Cartesian tolerances,
with a tighter default for `place`.

## Dual-Arm Coordination

The coordinator remains deliberately slim: it chooses block order, primitive
order, stack-slot reservations, priority/yield decisions near the shared stack,
and return-home transitions. Collision avoidance is handled by DS modulation.
A sampled-link hold can be enabled with `--link_safety_hold` for conservative
debugging.

Dual-arm deployment starts both arms together by default:
`coordination.start_stagger_steps: 0`. For ablations, add a right-arm launch
delay with `--stagger_steps`.

Shared stack layers are reserved before `transport/place`, so both arms cannot
target the same stack height when they arrive at the goal area concurrently.
The reserved height is the desired block center height. During kinematic-carry
debug deployment, the carried block is snapped to that reserved stack slot when
`place` opens the gripper; otherwise the visual block pose can reflect the EE
carry offset instead of the coordinator's stack layer.

Transport targets use dynamic stack clearance rather than a fixed carry height.
As the stack grows, the target rises above the current top of the stack before
the arm descends for `place`. When an arm finishes its assigned blocks, it
returns to its initial home pose so it does not remain beside the stack and
block the other arm.

Dual-arm deployment defaults are usually loaded from
`configs/deploy_neural_physical.yaml`. That preset uses per-arm checkpoints,
safe velocity projection, protected-point modulation, lateral-order modulation,
return-home parking, and no sampled-link hold unless explicitly enabled.

If a newly trained DS does not converge at deployment, inspect
`cos->goal` in `deploy_single_arm.py`. Negative values mean the learned vector
field is pointing away from `q_goal`. The learned-DS setting for `reach` and
`transport` is `--ds_scale 1.0` with the code-level `DS_GOAL_GAIN` left at
`0.0`.

Single-arm deployment aborts on primitive timeout by default. This prevents a
failed `reach` from cascading into a false `grasp`. Use `--advance_on_timeout`
only for phase-flow debugging.

## Stability

Per-primitive: `V(e)` is positive-definite around `e = q - q_goal = 0` by construction;
training enforces `dV/dt + α·V <= 0` on the data distribution; inference can
optionally project `f(x)` onto the half-space where this holds exactly
(`--use_safe`).

Across primitives: the system is hybrid. We use learned stable DS motion for
`reach` and `transport`, then switch to scripted Lula joint-space moves for
`grasp`, `lift`, and `place`. We report empirical primitive convergence and
stack completion rather than claiming one global Lyapunov proof for the whole
task.

Across arms: protected-point modulation preserves the nominal DS behavior when
arms are separated and deflects approaching motion near the other arm. The
lateral-order modulation, optional sampled-link safety hold, and return-home
parking are discrete/pragmatic deployment guards for the real geometry of the
two Frankas; they should be reported separately from the continuous modulation
guarantee.
